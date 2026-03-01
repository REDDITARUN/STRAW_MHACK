from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm


class GenerativeBaseProcessor(ABC):
    dataset_key: str
    source_id: str
    domain: str
    metric: str
    system_prompt: str

    def load(self) -> DatasetDict:
        ds = load_dataset(self.source_id)
        if isinstance(ds, DatasetDict):
            return ds
        # Some datasets expose a single split dataset object.
        return DatasetDict({"train": ds})  # type: ignore[arg-type]

    @abstractmethod
    def convert(self, row: dict[str, Any], split: str, index: int) -> dict[str, Any] | None:
        raise NotImplementedError

    def process_split(self, split_data: Dataset, split: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx, row in enumerate(tqdm(split_data, desc=f"{self.dataset_key}:{split}")):
            out = self.convert(row, split, idx)
            if out is not None:
                rows.append(out)
        return rows

    def save_jsonl(self, records: list[dict[str, Any]], out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _maybe_cap_train(self, split_data: Dataset, max_train_samples: int, seed: int) -> Dataset:
        if max_train_samples > 0 and len(split_data) > max_train_samples:
            return split_data.shuffle(seed=seed).select(range(max_train_samples))
        return split_data

    def run(
        self,
        output_root: Path,
        *,
        max_train_samples: int = 0,
        seed: int = 42,
        validation_size: int = 1000,
        test_size: int = 1000,
    ) -> None:
        ds = self.load()
        if "validation" not in ds and "test" not in ds and "train" in ds:
            # Build deterministic train/validation/test splits for single-split datasets.
            train_full = ds["train"].shuffle(seed=seed)
            total = len(train_full)
            test_n = min(test_size, max(1, total // 20))
            val_n = min(validation_size, max(1, total // 20))
            train_n = max(1, total - test_n - val_n)
            ds = DatasetDict(
                {
                    "train": train_full.select(range(0, train_n)),
                    "validation": train_full.select(range(train_n, train_n + val_n)),
                    "test": train_full.select(range(train_n + val_n, train_n + val_n + test_n)),
                }
            )

        for split, split_data in ds.items():
            effective_split = split_data
            if split == "train":
                effective_split = self._maybe_cap_train(split_data, max_train_samples, seed=seed)
            out = output_root / self.dataset_key / f"{split}.jsonl"
            self.save_jsonl(self.process_split(effective_split, split), out)

    @staticmethod
    def s(value: Any) -> str:
        return "" if value is None else str(value)
