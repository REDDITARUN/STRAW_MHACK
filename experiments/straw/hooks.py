from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass
class ResidualCapture:
    """
    Stores per-layer residual-like activations captured from attention block outputs.
    """

    per_layer: dict[int, torch.Tensor]


def _to_tensor(output: Any) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    return None


def register_attention_output_hooks(model: Any) -> tuple[ResidualCapture, list[Any]]:
    """
    Register hooks on each transformer layer self-attention module.
    For Mistral-style models, layers are expected at model.model.layers.
    """
    capture = ResidualCapture(per_layer={})
    handles: list[Any] = []

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("Expected model.model.layers for Mistral-style model.")

    for layer_idx, layer in enumerate(layers):
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is None:
            continue

        def make_hook(idx: int) -> Callable[..., None]:
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                tensor = _to_tensor(output)
                if tensor is not None:
                    capture.per_layer[idx] = tensor.detach()

            return hook

        handles.append(self_attn.register_forward_hook(make_hook(layer_idx)))

    return capture, handles


def clear_hooks(handles: list[Any]) -> None:
    for handle in handles:
        handle.remove()
