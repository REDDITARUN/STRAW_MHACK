from __future__ import annotations

from typing import Any

from data.processors.generative_base import GenerativeBaseProcessor


class DollyGenerativeProcessor(GenerativeBaseProcessor):
    dataset_key = "dolly_gen"
    source_id = "databricks/databricks-dolly-15k"
    domain = "chat_instruction"
    metric = "token_f1"
    system_prompt = "You are a helpful assistant. Follow the instruction clearly."

    def convert(self, row: dict[str, Any], split: str, index: int) -> dict[str, Any] | None:
        instruction = self.s(row.get("instruction")).strip()
        context = self.s(row.get("context")).strip()
        response = self.s(row.get("response")).strip()
        if not instruction or not response:
            return None

        user = instruction if not context else f"Instruction: {instruction}\n\nContext: {context}"
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
            "target": response,
            "metadata": {
                "metric": self.metric,
                "source_dataset": self.source_id,
                "category": self.s(row.get("category")),
                "original_id": self.s(row.get("id", f"{split}-{index}")),
            },
        }
