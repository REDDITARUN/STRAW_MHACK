from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.common.data_utils_gen import GEN_DATASETS, load_dataset_split
from experiments.common.observability import (
    finish_wandb_run,
    init_wandb_run,
    log_file_artifact,
    log_metrics,
)
from experiments.common.prompt_utils import build_prompt_messages
from experiments.common.text_metrics import metric_name_from_sample, score_sample

DEFAULT_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base or LoRA model on generative datasets.")
    parser.add_argument("--base-model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter. Omit for base model.")
    parser.add_argument("--datasets", nargs="+", default=list(GEN_DATASETS), choices=list(GEN_DATASETS))
    parser.add_argument("--data-root", default="data/processed_gen")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--output", default="experiments/results/eval_results_gen.json")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="straw")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    parser.add_argument("--wandb-log-artifacts", action="store_true")
    return parser.parse_args()


def load_model_and_tokenizer(base_model: str, adapter: str | None):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_text(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_ids = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def evaluate_dataset(model: Any, tokenizer: Any, rows: list[dict[str, Any]], args: argparse.Namespace):
    rows = rows[: args.max_samples] if args.max_samples > 0 else rows
    metric_scores: list[float] = []
    metric_name = None
    for row in tqdm(rows, desc=f"Eval {rows[0]['dataset'] if rows else 'dataset'}"):
        pred = generate_text(
            model=model,
            tokenizer=tokenizer,
            messages=build_prompt_messages(row),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        target = str(row.get("target", ""))
        metric_name = metric_name_from_sample(row)
        metric_scores.append(score_sample(pred, target, metric_name))

    return {
        "metric": metric_name or "token_f1",
        "score": mean(metric_scores) if metric_scores else 0.0,
        "num_samples": len(metric_scores),
    }


def main() -> None:
    args = parse_args()
    args.base_model = (args.base_model or "").strip() or DEFAULT_BASE_MODEL
    run = init_wandb_run(
        enabled=args.use_wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or "eval_gen",
        tags=(args.wandb_tags or []) + ["eval", "generative"],
        config=vars(args),
    )

    model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter)
    results: dict[str, Any] = {}
    for ds in args.datasets:
        rows = load_dataset_split(ds, args.split, args.data_root)
        ds_result = evaluate_dataset(model, tokenizer, rows, args)
        results[ds] = ds_result
        log_metrics(
            run,
            {
                f"eval/{ds}/score": ds_result["score"],
                f"eval/{ds}/num_samples": ds_result["num_samples"],
            },
        )

    if results:
        results["macro_avg_score"] = mean(v["score"] for v in results.values() if isinstance(v, dict))
        log_metrics(run, {"eval/macro_avg_score": results["macro_avg_score"]})
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote results: {out_path}")

    if args.wandb_log_artifacts:
        log_file_artifact(run, "eval-results-generative", "eval", str(out_path))
    finish_wandb_run(run)


if __name__ == "__main__":
    main()
