from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from peft import LoraConfig, TaskType


@dataclass
class LoraTrainConfig:
    base_model: str
    output_root: str
    seed: int
    max_length: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: int
    warmup_ratio: float
    logging_steps: int
    save_steps: int
    eval_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    lr_scheduler_type: str = "cosine"


def load_config(path: str) -> LoraTrainConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return LoraTrainConfig(**raw)


def build_lora_config(cfg: LoraTrainConfig) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
    )
