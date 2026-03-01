from __future__ import annotations

from typing import Any

from data.processors.generative_base import GenerativeBaseProcessor


class SamsumGenerativeProcessor(GenerativeBaseProcessor):
    dataset_key = "samsum_gen"
    source_id = "knkarthick/samsum"
    domain = "dialogue_summarization"
    metric = "rougeL"
    system_prompt = "You are a concise assistant. Summarize the conversation faithfully."

    def convert(self, row: dict[str, Any], split: str, index: int) -> dict[str, Any] | None:
        dialogue = self.s(row.get("dialogue")).strip()
        summary = self.s(row.get("summary")).strip()
        if not dialogue or not summary:
            return None
        return {
            "id": f"{self.dataset_key}-{split}-{index}",
            "dataset": self.dataset_key,
            "domain": self.domain,
            "split": split,
            "task_type": "generative",
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Summarize this dialogue:\n\n{dialogue}"},
            ],
            "target": summary,
            "metadata": {
                "metric": self.metric,
                "source_dataset": self.source_id,
                "original_id": self.s(row.get("id", f"{split}-{index}")),
            },
        }
