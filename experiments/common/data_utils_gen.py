from __future__ import annotations

import json
from pathlib import Path
from typing import Any


GEN_DATASETS = ("samsum_gen", "dolly_gen", "codealpaca_gen")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_split_path(dataset: str, split: str, data_root: str = "data/processed_gen") -> Path:
    return Path(data_root) / dataset / f"{split}.jsonl"


def load_dataset_split(dataset: str, split: str, data_root: str = "data/processed_gen") -> list[dict[str, Any]]:
    path = get_split_path(dataset, split, data_root)
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return read_jsonl(path)
