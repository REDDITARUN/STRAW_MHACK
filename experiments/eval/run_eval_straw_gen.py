from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.analysis.ba_capture import dynamic_state_to_ba_heatmaps, running_mean_update
from experiments.common.data_utils_gen import GEN_DATASETS, load_dataset_split
from experiments.common.observability import (
    finish_wandb_run,
    init_wandb_run,
    log_file_artifact,
    log_metrics,
)
from experiments.common.prompt_utils import build_prompt_messages
from experiments.common.text_metrics import metric_name_from_sample, score_sample
from experiments.straw.apply_vproj_adapter import DynamicVProjInjector, hypernet_to_layer_lora
from experiments.straw.hypernet_bert import build_hyper_lora_generator


DEFAULT_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate STRAW checkpoint on generative datasets.")
    parser.add_argument("--base-model", default="")
    parser.add_argument("--straw-config", default="experiments/configs/lora_config_b_straw.yaml")
    parser.add_argument("--hypernet-ckpt", required=True, help="Path to hypernet.pt")
    parser.add_argument("--datasets", nargs="+", default=list(GEN_DATASETS), choices=list(GEN_DATASETS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-root", default="data/processed_gen")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--output-path", default="experiments/results/straw_eval_gen.json")
    parser.add_argument("--save-ba-path", default=None, help="Optional path to save STRAW dynamic BA heatmaps (.pt).")
    parser.add_argument("--ba-downsample", type=int, default=64, help="Heatmap size for BA compression.")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="straw")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    parser.add_argument("--wandb-log-artifacts", action="store_true")
    return parser.parse_args()


def load_yaml(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_components(args: argparse.Namespace) -> tuple[Any, Any, Any, DynamicVProjInjector]:
    cfg = load_yaml(args.straw_config)
    base_model = (args.base_model or "").strip() or str(cfg.get("base_model", DEFAULT_BASE_MODEL))
    args.base_model = base_model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    hidden_size = int(model.config.hidden_size)
    num_layers = int(model.config.num_hidden_layers)
    v_proj = model.model.layers[0].self_attn.v_proj
    v_proj_in = int(v_proj.in_features)
    v_proj_out = int(v_proj.out_features)
    hyper_cfg = cfg.get("hypernet", {})
    hypernet = build_hyper_lora_generator(
        residual_dim=hidden_size,
        num_layers=num_layers,
        v_proj_in=v_proj_in,
        v_proj_out=v_proj_out,
        rank=int(cfg.get("straw_rank", 8)),
        hyper_cfg=hyper_cfg,
    )
    state_dict = torch.load(args.hypernet_ckpt, map_location="cpu")
    hypernet.load_state_dict(state_dict)
    hypernet.to(model.device)
    hypernet.eval()

    injector = DynamicVProjInjector(model)
    injector.install()
    return model, tokenizer, hypernet, injector


def get_prefix_hidden(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden_states = outputs.hidden_states[1:]
    pooled = [h.mean(dim=1) for h in hidden_states]
    return torch.stack(pooled, dim=1)


@torch.no_grad()
def infer_text(
    model: Any,
    tokenizer: Any,
    hypernet: Any,
    injector: DynamicVProjInjector,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    ba_downsample: int,
) -> tuple[str, dict[int, torch.Tensor]]:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    injector.clear_state()
    prefix_hidden = get_prefix_hidden(
        model=model,
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
    )
    hyper_dtype = next(hypernet.parameters()).dtype
    prefix_hidden = prefix_hidden.to(dtype=hyper_dtype)
    h_out = hypernet(prefix_hidden)
    dynamic_state = hypernet_to_layer_lora(h_out)
    injector.set_state(dynamic_state)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = output_ids[0][inputs["input_ids"].shape[1] :]
    decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
    injector.clear_state()
    return decoded, dynamic_state_to_ba_heatmaps(
        dynamic_state.layer_a,
        dynamic_state.layer_b,
        downsample=ba_downsample,
    )


def evaluate_dataset(
    model: Any,
    tokenizer: Any,
    hypernet: Any,
    injector: DynamicVProjInjector,
    dataset: str,
    split: str,
    data_root: str,
    max_new_tokens: int,
    temperature: float,
    limit: int,
    ba_downsample: int,
) -> dict[str, Any]:
    rows = load_dataset_split(dataset, split, data_root)
    if limit > 0:
        rows = rows[:limit]
    scores: list[float] = []
    metric_name = None
    ba_running: dict[int, torch.Tensor] = {}
    ba_count = 0

    for row in tqdm(rows, desc=f"Eval STRAW {dataset}"):
        target = str(row.get("target", ""))
        pred, ba_map = infer_text(
            model=model,
            tokenizer=tokenizer,
            hypernet=hypernet,
            injector=injector,
            messages=build_prompt_messages(row),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            ba_downsample=ba_downsample,
        )
        metric_name = metric_name_from_sample(row)
        scores.append(score_sample(pred, target, metric_name))
        if ba_map:
            ba_running = running_mean_update(ba_running, ba_map, ba_count)
            ba_count += 1

    return {
        "dataset": dataset,
        "split": split,
        "metric": metric_name or "token_f1",
        "score": mean(scores) if scores else 0.0,
        "num_examples": len(scores),
        "ba_mean_by_layer": {k: v.cpu() for k, v in ba_running.items()},
        "ba_num_samples": ba_count,
    }


def main() -> None:
    args = parse_args()
    if not (args.base_model or "").strip():
        cfg = load_yaml(args.straw_config)
        args.base_model = str(cfg.get("base_model", DEFAULT_BASE_MODEL))
    run_name = args.wandb_run_name or f"eval_straw_gen_{args.split}"
    run = init_wandb_run(
        enabled=args.use_wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        tags=(args.wandb_tags or []) + ["eval", "straw", "generative"],
        config=vars(args),
    )
    model, tokenizer, hypernet, injector = load_components(args)

    results: dict[str, Any] = {
        "base_model": args.base_model,
        "hypernet_ckpt": args.hypernet_ckpt,
        "split": args.split,
        "datasets": [],
    }
    ba_payload: dict[str, Any] = {
        "strategy": "straw_dynamic_gen",
        "base_model": args.base_model,
        "hypernet_ckpt": args.hypernet_ckpt,
        "split": args.split,
        "ba_downsample": args.ba_downsample,
        "datasets": {},
    }

    for dataset in args.datasets:
        metrics = evaluate_dataset(
            model=model,
            tokenizer=tokenizer,
            hypernet=hypernet,
            injector=injector,
            dataset=dataset,
            split=args.split,
            data_root=args.data_root,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            limit=args.limit,
            ba_downsample=args.ba_downsample,
        )
        ba_payload["datasets"][dataset] = {
            "ba_mean_by_layer": metrics.pop("ba_mean_by_layer"),
            "ba_num_samples": metrics.pop("ba_num_samples"),
        }
        results["datasets"].append(metrics)
        pref = f"eval/{dataset}"
        log_metrics(
            run,
            {
                f"{pref}/score": metrics["score"],
                f"{pref}/metric": metrics["metric"],
                f"{pref}/num_examples": metrics["num_examples"],
            },
        )

    if results["datasets"]:
        results["macro_avg_score"] = mean(x["score"] for x in results["datasets"])

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if args.save_ba_path:
        ba_out = Path(args.save_ba_path)
        ba_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ba_payload, ba_out)
        if args.wandb_log_artifacts:
            log_file_artifact(run, "straw-ba-maps-gen", "analysis", str(ba_out))

    if args.wandb_log_artifacts:
        log_file_artifact(run, "straw-eval-results-gen", "evaluation", str(out_path))
    injector.remove()
    finish_wandb_run(run)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
