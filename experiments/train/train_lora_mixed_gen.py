from __future__ import annotations

import argparse
import inspect
import random
from collections import defaultdict
from typing import Any

from datasets import Dataset
from peft import get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from experiments.common.data_utils_gen import GEN_DATASETS, load_dataset_split
from experiments.common.lora_utils import build_lora_config, load_config
from experiments.common.observability import (
    finish_wandb_run,
    init_wandb_run,
    log_dir_artifact,
    log_metrics,
)
from experiments.common.prompt_utils import build_prompt_messages


def build_training_args_compatible(**kwargs: Any) -> TrainingArguments:
    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters
    if "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy", "steps")
    elif "eval_strategy" in params:
        kwargs["eval_strategy"] = kwargs.pop("eval_strategy", "steps")
    else:
        kwargs.pop("eval_strategy", None)
        kwargs.pop("evaluation_strategy", None)
    return TrainingArguments(**kwargs)


def build_trainer_compatible(**kwargs: Any) -> Trainer:
    sig = inspect.signature(Trainer.__init__)
    params = sig.parameters
    tokenizer_obj = kwargs.pop("tokenizer_obj", None)
    if "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer_obj
    elif "processing_class" in params:
        kwargs["processing_class"] = tokenizer_obj
    return Trainer(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one mixed LoRA on all generative datasets.")
    parser.add_argument("--config", default="experiments/configs/lora_config_a.yaml")
    parser.add_argument("--datasets", nargs="+", default=list(GEN_DATASETS), choices=list(GEN_DATASETS))
    parser.add_argument("--data-root", default="data/processed_gen")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--max-train-samples-per-dataset", type=int, default=1000)
    parser.add_argument("--max-eval-samples-per-dataset", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="straw")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    parser.add_argument("--wandb-log-artifacts", action="store_true")
    return parser.parse_args()


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
    reserve = max(8, len(answer_ids))
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
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_hf_dataset(rows: list[dict[str, Any]], tokenizer: Any, max_length: int) -> Dataset:
    data: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        item = preprocess_row(row, tokenizer, max_length=max_length)
        if item is None:
            skipped += 1
            continue
        data.append(item)
    print(f"Prepared {len(data)} rows (skipped {skipped})")
    if not data:
        raise ValueError("No valid supervised samples after preprocessing.")
    return Dataset.from_list(data)


def load_mixed_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    random.seed(args.seed)
    for dataset in args.datasets:
        tr = load_dataset_split(dataset, args.train_split, args.data_root)
        ev = load_dataset_split(dataset, args.eval_split, args.data_root)
        random.shuffle(tr)
        random.shuffle(ev)
        train_rows.extend(tr[: args.max_train_samples_per_dataset])
        eval_rows.extend(ev[: args.max_eval_samples_per_dataset])
    random.shuffle(train_rows)
    random.shuffle(eval_rows)
    return train_rows, eval_rows


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows, eval_rows = load_mixed_rows(args)
    by_ds = defaultdict(int)
    for r in train_rows:
        by_ds[str(r.get("dataset", "unknown"))] += 1
    print(f"Mixed train distribution: {dict(by_ds)}")

    train_ds = build_hf_dataset(train_rows, tokenizer, cfg.max_length)
    eval_ds = build_hf_dataset(eval_rows, tokenizer, cfg.max_length)

    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, device_map="auto")
    model = get_peft_model(model, build_lora_config(cfg))

    output_dir = f"{cfg.output_root}/lora_mixed_gen"
    run_name = args.wandb_run_name or f"lora_mixed_gen_r{cfg.lora_r}"
    run = init_wandb_run(
        enabled=args.use_wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        tags=(args.wandb_tags or []) + ["lora", "mixed", "generative"],
        config={
            "script": "train_lora_mixed_gen.py",
            "datasets": args.datasets,
            "base_model": cfg.base_model,
            "max_train_samples_per_dataset": args.max_train_samples_per_dataset,
            "max_eval_samples_per_dataset": args.max_eval_samples_per_dataset,
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
            "learning_rate": cfg.learning_rate,
            "lr_scheduler_type": cfg.lr_scheduler_type,
            "num_train_epochs": cfg.num_train_epochs,
        },
    )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_metrics(run, {"params/trainable": trainable_params})

    training_args = build_training_args_compatible(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        bf16=True,
        report_to="wandb" if args.use_wandb else "none",
        run_name=run_name if args.use_wandb else None,
        seed=cfg.seed,
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    trainer = build_trainer_compatible(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer_obj=tokenizer,
    )

    trainer.train()
    metrics = trainer.evaluate()
    log_metrics(run, {f"final/{k}": v for k, v in metrics.items()})
    model.save_pretrained(f"{output_dir}/adapter")
    tokenizer.save_pretrained(f"{output_dir}/adapter")
    print(f"Saved mixed adapter: {output_dir}/adapter")

    if args.wandb_log_artifacts:
        log_dir_artifact(
            run,
            name="lora-mixed-gen-adapter",
            artifact_type="model",
            dir_path=f"{output_dir}/adapter",
            metadata={"datasets": args.datasets, "strategy": "mixed_lora_generative"},
        )
    finish_wandb_run(run)


if __name__ == "__main__":
    main()
