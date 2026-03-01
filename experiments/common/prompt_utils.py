from __future__ import annotations

import re
from typing import Any


OPTION_LABEL_PATTERN = re.compile(r"\b([A-H])\b")


def extract_option_label(text: str) -> str:
    """
    Extract the first option label (A-H) from model output.
    """
    if not text:
        return ""
    upper = text.strip().upper()
    match = OPTION_LABEL_PATTERN.search(upper)
    return match.group(1) if match else ""


def build_prompt_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    messages = sample.get("messages", [])
    return [m for m in messages if isinstance(m, dict) and "role" in m and "content" in m]


def build_train_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    messages = build_prompt_messages(sample)
    answer = str(sample.get("target", "")).strip()
    return messages + [{"role": "assistant", "content": answer}]
