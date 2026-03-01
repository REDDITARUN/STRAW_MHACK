from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, set_seed

from experiments.common.data_utils_gen import GEN_DATASETS, load_dataset_split
from experiments.common.observability import (
    finish_wandb_run,
    init_wandb_run,
    log_file_artifact,
    log_metrics,
)
from experiments.common.prompt_utils import build_prompt_messages
from experiments.straw.apply_vproj_adapter import DynamicVProjInjector, hypernet_to_layer_lora
from experiments.straw.hypernet_bert import build_hyper_lora_generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train STRAW hypernetwork for dynamic v_proj adaptation.")
    parser.add_argument("--config", default="experiments/configs/lora_config_b_straw.yaml")
    parser.add_argument("--datasets", nargs="+", default=list(GEN_DATASETS))
    parser.add_argument("--data-root", default="data/processed_gen")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--max-train-samples-per-dataset", type=int, default=0)
    parser.add_argument("--max-eval-samples-per-dataset", type=int, default=0)
    parser.add_argument("--output-dir", default="models/checkpoints/straw/all_tasks")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="straw")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    parser.add_argument("--wandb-log-artifacts", action="store_true")
    return parser.parse_args()


def cap_rows(rows: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    return rows[:max_samples] if max_samples > 0 else rows


def preprocess_row(row: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, Any] | None:
    prompt_messages = build_prompt_messages(row)
    answer = str(row.get("target", "")).strip()
    if not answer:
        return None

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    answer_ids = tokenizer(" " + answer, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        answer_ids = answer_ids + [int(tokenizer.eos_token_id)]
    reserve = max(4, len(answer_ids))
    prompt_max_len = max(1, max_length - reserve)
    prompt_ids = tokenizer(prompt_text, truncation=True, max_length=prompt_max_len)["input_ids"]

    input_ids = (prompt_ids + answer_ids)[:max_length]
    attention_mask = [1] * len(input_ids)

    labels = [-100] * len(prompt_ids) + answer_ids
    labels = labels[: len(input_ids)]
    if len(labels) < len(input_ids):
        labels += [-100] * (len(input_ids) - len(labels))
    if not any(x != -100 for x in labels):
        return None
    mask_len = min(len(prompt_ids), len(input_ids))
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prefix_length": mask_len,
    }


def to_hf_dataset(rows: list[dict[str, Any]], tokenizer: Any, max_length: int) -> Dataset:
    data: list[dict[str, Any]] = []
    skipped = 0
    for r in rows:
        item = preprocess_row(r, tokenizer, max_length=max_length)
        if item is None:
            skipped += 1
            continue
        data.append(item)
    print(f"Prepared {len(data)} rows (skipped {skipped})")
    if not data:
        raise ValueError("No valid supervised samples after preprocessing.")
    return Dataset.from_list(data)


def read_config(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def collate_with_prefix(tokenizer: Any):
    base_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")

    def _collate(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        prefix_lengths = [int(f.pop("prefix_length")) for f in features]
        batch = base_collator(features)
        batch["prefix_length"] = torch.tensor(prefix_lengths, dtype=torch.long)
        return batch

    return _collate


def get_prefix_hidden(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prefix_length: torch.Tensor,
) -> torch.Tensor:
    """
    Extract hidden states and pool only instruction-prefix tokens per layer.
    Returns: [batch, num_layers, hidden_dim]
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden_states = outputs.hidden_states
    layer_hiddens = hidden_states[1:]  # one tensor per transformer layer
    pooled_per_layer: list[torch.Tensor] = []
    batch_size, seq_len, _ = layer_hiddens[0].shape
    for b in range(batch_size):
        pe = int(prefix_length[b].item())
        pe = max(1, min(pe, seq_len))
        per_layer_b: list[torch.Tensor] = []
        for layer_hidden in layer_hiddens:
            per_layer_b.append(layer_hidden[b, :pe, :].mean(dim=0))
        pooled_per_layer.append(torch.stack(per_layer_b, dim=0))
    return torch.stack(pooled_per_layer, dim=0)


def mean_loss(
    model: Any,
    hypernet: Any,
    injector: DynamicVProjInjector,
    loader: Any,
    device: torch.device,
) -> float:
    model.eval()
    hypernet.eval()
    total = 0.0
    steps = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            # Prevent leakage from previous sample's dynamic state.
            injector.clear_state()
            prefix_hidden = get_prefix_hidden(
                model=model,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                prefix_length=batch["prefix_length"],
            ).detach()
            hyper_dtype = next(hypernet.parameters()).dtype
            prefix_hidden = prefix_hidden.to(dtype=hyper_dtype)

            h_out = hypernet(prefix_hidden)
            injector.set_state(hypernet_to_layer_lora(h_out))
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
            )
            total += float(out.loss.item())
            steps += 1
            injector.clear_state()
    return total / max(steps, 1)


def main() -> None:
    args = parse_args()
    cfg = read_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    base_model = cfg["base_model"]
    max_length = int(cfg.get("max_length", 1024))
    straw_rank = int(cfg.get("straw_rank", 8))
    train_cfg = cfg.get("training", {})
    hyper_cfg = cfg.get("hypernet", {})

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        train_rows.extend(
            cap_rows(load_dataset_split(dataset, args.train_split, args.data_root), args.max_train_samples_per_dataset)
        )
        eval_rows.extend(
            cap_rows(load_dataset_split(dataset, args.eval_split, args.data_root), args.max_eval_samples_per_dataset)
        )

    train_ds = to_hf_dataset(train_rows, tokenizer, max_length=max_length)
    eval_ds = to_hf_dataset(eval_rows, tokenizer, max_length=max_length)

    collator = collate_with_prefix(tokenizer)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("per_device_train_batch_size", 2)),
        shuffle=True,
        collate_fn=collator,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_ds,
        batch_size=int(train_cfg.get("per_device_train_batch_size", 2)),
        shuffle=False,
        collate_fn=collator,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.train()

    num_layers = int(model.config.num_hidden_layers)
    hidden_size = int(model.config.hidden_size)
    first_layer = model.model.layers[0]
    v_proj = first_layer.self_attn.v_proj
    v_proj_in = int(v_proj.in_features)
    v_proj_out = int(v_proj.out_features)

    hypernet = build_hyper_lora_generator(
        residual_dim=hidden_size,
        num_layers=num_layers,
        v_proj_in=v_proj_in,
        v_proj_out=v_proj_out,
        rank=straw_rank,
        hyper_cfg=hyper_cfg,
    ).to(device)
    run_name = args.wandb_run_name or f"straw_r{straw_rank}_tinybert"
    run = init_wandb_run(
        enabled=args.use_wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        tags=(args.wandb_tags or []) + ["straw", "hypernetwork"],
        config={
            "script": "train_straw.py",
            "base_model": base_model,
            "datasets": args.datasets,
            "train_split": args.train_split,
            "eval_split": args.eval_split,
            "max_train_samples_per_dataset": args.max_train_samples_per_dataset,
            "max_eval_samples_per_dataset": args.max_eval_samples_per_dataset,
            "straw_rank": straw_rank,
            "num_layers": num_layers,
            "v_proj_in": v_proj_in,
            "v_proj_out": v_proj_out,
            "hypernet": hyper_cfg,
            "training": train_cfg,
            "seed": int(cfg.get("seed", 42)),
        },
    )
    trainable_params = sum(p.numel() for p in hypernet.parameters() if p.requires_grad)
    log_metrics(run, {"params/trainable": trainable_params})

    optimizer = torch.optim.AdamW(
        hypernet.parameters(),
        lr=float(train_cfg.get("learning_rate", 1.0e-4)),
    )
    grad_acc_steps = int(train_cfg.get("gradient_accumulation_steps", 8))
    num_epochs = int(train_cfg.get("num_train_epochs", 1))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_dir / "best_hypernet.pt"

    injector = DynamicVProjInjector(model)
    injector.install()

    global_step = 0
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        hypernet.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = batch_to_device(batch, device)

            with torch.no_grad():
                # Prevent leakage from previous sample's dynamic state.
                injector.clear_state()
                prefix_hidden = get_prefix_hidden(
                    model=model,
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    prefix_length=batch["prefix_length"],
                ).detach()
            hyper_dtype = next(hypernet.parameters()).dtype
            prefix_hidden = prefix_hidden.to(dtype=hyper_dtype)

            h_out = hypernet(prefix_hidden)
            injector.set_state(hypernet_to_layer_lora(h_out))

            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
            )
            loss = out.loss / grad_acc_steps
            loss.backward()
            running_loss += float(loss.item() * grad_acc_steps)

            if step % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(hypernet.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 10 == 0:
                    train_loss = running_loss / max(step, 1)
                    print(f"epoch={epoch+1} step={global_step} train_loss={train_loss:.4f}")
                    log_metrics(
                        run,
                        {
                            "train/loss": train_loss,
                            "train/epoch": epoch + 1,
                            "train/step": global_step,
                        },
                        step=global_step,
                    )

        val_loss = mean_loss(
            model,
            hypernet,
            injector,
            eval_loader,
            device=device,
        )
        print(f"epoch={epoch+1} validation_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(hypernet.state_dict(), best_ckpt_path)
            print(f"Saved best hypernet checkpoint: {best_ckpt_path} (val_loss={val_loss:.4f})")
        log_metrics(
            run,
            {
                "eval/loss": val_loss,
                "eval/epoch": epoch + 1,
                "eval/best_loss": best_val_loss,
            },
            step=global_step,
        )

    torch.save(hypernet.state_dict(), out_dir / "hypernet.pt")
    tokenizer.save_pretrained(out_dir / "tokenizer")
    with (out_dir / "straw_config_used.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config_path": args.config,
                "base_model": base_model,
                "datasets": args.datasets,
                "train_split": args.train_split,
                "eval_split": args.eval_split,
                "straw_rank": straw_rank,
                "num_layers": num_layers,
                "v_proj_in": v_proj_in,
                "v_proj_out": v_proj_out,
                "hypernet": hyper_cfg,
                "training": train_cfg,
            },
            f,
            indent=2,
        )
    if args.wandb_log_artifacts:
        log_file_artifact(
            run,
            name="straw-hypernet-best-pt",
            artifact_type="model",
            file_path=str(best_ckpt_path),
            metadata={"strategy": "straw", "kind": "best"},
        )
        log_file_artifact(
            run,
            name="straw-hypernet-pt",
            artifact_type="model",
            file_path=str(out_dir / "hypernet.pt"),
            metadata={"strategy": "straw"},
        )
        log_file_artifact(
            run,
            name="straw-config-used",
            artifact_type="config",
            file_path=str(out_dir / "straw_config_used.json"),
            metadata={"strategy": "straw"},
        )
    injector.remove()
    finish_wandb_run(run)
    print(f"Saved STRAW checkpoint to: {out_dir}")


if __name__ == "__main__":
    main()
