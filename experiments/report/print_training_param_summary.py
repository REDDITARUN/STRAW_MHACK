from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from transformers import AutoConfig

from experiments.straw.hypernet_bert import build_hyper_lora_generator


@dataclass
class LoraStats:
    layers: int
    rank: int
    in_features: int
    out_features: int
    trainable_per_adapter: int
    num_domain_adapters: int
    trainable_domain_total: int
    trainable_mixed_total: int


@dataclass
class StrawStats:
    trainable_total: int
    residual_dim: int
    num_layers: int
    v_proj_in: int
    v_proj_out: int
    rank: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print trainable parameter summary for Base/LoRA/STRAW strategies."
    )
    parser.add_argument(
        "--lora-config",
        default="experiments/configs/lora_config_a.yaml",
        help="Path to LoRA config A YAML.",
    )
    parser.add_argument(
        "--straw-config",
        default="experiments/configs/lora_config_b_straw.yaml",
        help="Path to STRAW config B YAML.",
    )
    parser.add_argument(
        "--num-domain-datasets",
        type=int,
        default=3,
        help="Number of per-domain LoRA adapters (e.g. 3 for code/summarization/chat).",
    )
    return parser.parse_args()


def load_yaml(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_trainable_params(module: Any) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def infer_vproj_dims_from_config(model_config: Any) -> tuple[int, int, int]:
    hidden_size = int(model_config.hidden_size)
    num_layers = int(model_config.num_hidden_layers)
    num_heads = int(model_config.num_attention_heads)
    num_kv_heads = int(getattr(model_config, "num_key_value_heads", num_heads))
    head_dim = hidden_size // num_heads
    v_proj_in = hidden_size
    v_proj_out = num_kv_heads * head_dim
    return num_layers, v_proj_in, v_proj_out


def lora_stats_from_config(lora_cfg: dict[str, Any], num_domain_datasets: int) -> LoraStats:
    base_model = lora_cfg["base_model"]
    rank = int(lora_cfg["lora_r"])

    model_config = AutoConfig.from_pretrained(base_model)
    layers, v_proj_in, v_proj_out = infer_vproj_dims_from_config(model_config)

    # One target module per layer (v_proj) with params: r*in + out*r.
    trainable_per_adapter = layers * (rank * v_proj_in + v_proj_out * rank)
    trainable_domain_total = trainable_per_adapter * num_domain_datasets
    trainable_mixed_total = trainable_per_adapter

    return LoraStats(
        layers=layers,
        rank=rank,
        in_features=v_proj_in,
        out_features=v_proj_out,
        trainable_per_adapter=trainable_per_adapter,
        num_domain_adapters=num_domain_datasets,
        trainable_domain_total=trainable_domain_total,
        trainable_mixed_total=trainable_mixed_total,
    )


def straw_stats_from_config(straw_cfg: dict[str, Any]) -> StrawStats:
    base_model = straw_cfg["base_model"]
    rank = int(straw_cfg["straw_rank"])
    hyper_cfg = straw_cfg.get("hypernet", {})

    model_config = AutoConfig.from_pretrained(base_model)
    layers, v_proj_in, v_proj_out = infer_vproj_dims_from_config(model_config)
    residual_dim = int(model_config.hidden_size)

    hypernet = build_hyper_lora_generator(
        residual_dim=residual_dim,
        num_layers=layers,
        v_proj_in=v_proj_in,
        v_proj_out=v_proj_out,
        rank=rank,
        hyper_cfg=hyper_cfg,
    )
    trainable = count_trainable_params(hypernet)

    return StrawStats(
        trainable_total=trainable,
        residual_dim=residual_dim,
        num_layers=layers,
        v_proj_in=v_proj_in,
        v_proj_out=v_proj_out,
        rank=rank,
    )


def pretty(n: int) -> str:
    return f"{n:,}"


def main() -> None:
    args = parse_args()
    lora_cfg = load_yaml(args.lora_config)
    straw_cfg = load_yaml(args.straw_config)

    lora = lora_stats_from_config(lora_cfg, num_domain_datasets=args.num_domain_datasets)
    straw = straw_stats_from_config(straw_cfg)

    print("=== Training Parameter Summary ===")
    print(f"Base model: {lora_cfg['base_model']}")
    print("")
    print("[Shared inferred dimensions]")
    print(f"  layers: {lora.layers}")
    print(f"  v_proj: {lora.in_features} -> {lora.out_features}")
    print("")
    print("[Strategy: Base model eval only]")
    print("  trainable params: 0")
    print("")
    print("[Strategy: Domain LoRA (separate adapter per dataset)]")
    print(f"  rank: {lora.rank}")
    print(f"  trainable per adapter: {pretty(lora.trainable_per_adapter)}")
    print(f"  num adapters: {lora.num_domain_adapters}")
    print(f"  trainable total across all domain adapters: {pretty(lora.trainable_domain_total)}")
    print("")
    print("[Strategy: Mixed LoRA (single adapter for all datasets)]")
    print(f"  rank: {lora.rank}")
    print(f"  trainable total: {pretty(lora.trainable_mixed_total)}")
    print("")
    print("[Strategy: STRAW (configurable hypernet)]")
    print(f"  rank: {straw.rank}")
    print(f"  residual_dim: {straw.residual_dim}")
    print(f"  layers: {straw.num_layers}")
    print(f"  v_proj: {straw.v_proj_in} -> {straw.v_proj_out}")
    print(f"  trainable total: {pretty(straw.trainable_total)}")


if __name__ == "__main__":
    main()
