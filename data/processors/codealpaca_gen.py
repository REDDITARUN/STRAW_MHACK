from __future__ import annotations

from typing import Any

from data.processors.generative_base import GenerativeBaseProcessor


class CodeAlpacaGenerativeProcessor(GenerativeBaseProcessor):
    dataset_key = "codealpaca_gen"
    source_id = "sahil2801/CodeAlpaca-20k"
    domain = "code_generation"
    metric = "exact_match_norm"
    system_prompt = (
        "You are a coding assistant. Return a correct, runnable answer for the instruction."
    )

    def convert(self, row: dict[str, Any], split: str, index: int) -> dict[str, Any] | None:
        instruction = self.s(row.get("instruction")).strip()
        inp = self.s(row.get("input")).strip()
        output = self.s(row.get("output")).strip()
        if not instruction or not output:
            return None
        user = instruction if not inp else f"{instruction}\n\nInput:\n{inp}"
        return {
            "id": f"{self.dataset_key}-{split}-{index}",
            "dataset": self.dataset_key,
            "domain": self.domain,
            "split": split,
            "task_type": "generative",
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user},
            ],
            "target": output,
            "metadata": {
                "metric": self.metric,
                "source_dataset": self.source_id,
                "original_id": self.s(row.get("id", f"{split}-{index}")),
            },
        }
