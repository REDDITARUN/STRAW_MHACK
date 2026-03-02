from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from experiments.straw.hypernet_bert import HypernetOutput


@dataclass
class DynamicAdapterState:
    """
    Per-layer dynamic LoRA weights.
    layer_a[layer_idx] has shape [batch, rank, v_proj_in].
    layer_b[layer_idx] has shape [batch, v_proj_out, rank].
    """

    layer_a: dict[int, torch.Tensor]
    layer_b: dict[int, torch.Tensor]
    scale: float = 1.0


def hypernet_to_layer_lora(h: HypernetOutput, lora_alpha: float | None = None) -> DynamicAdapterState:
    """
    Convert hypernetwork output to per-layer LoRA matrices for v_proj.
    A: [B, L, R, in], B: [B, L, out, R]
    scale = alpha/rank (standard LoRA scaling); defaults to 1.0 if alpha not given.
    """
    batch, num_layers, rank, v_proj_in = h.a.shape
    _b2, _l2, v_proj_out, _r2 = h.b.shape
    if (batch, num_layers, rank) != (_b2, _l2, _r2):
        raise ValueError("A/B shape mismatch from hypernetwork output.")

    if v_proj_in <= 0 or v_proj_out <= 0:
        raise ValueError("Invalid v_proj dimensions in hypernetwork output.")

    scale = float(lora_alpha) / rank if lora_alpha is not None else 1.0

    layer_a: dict[int, torch.Tensor] = {}
    layer_b: dict[int, torch.Tensor] = {}
    for layer_idx in range(num_layers):
        layer_a[layer_idx] = h.a[:, layer_idx]
        layer_b[layer_idx] = h.b[:, layer_idx]

    return DynamicAdapterState(layer_a=layer_a, layer_b=layer_b, scale=scale)


class DynamicVProjInjector:
    """
    Registers hooks on each layer's v_proj and adds dynamic LoRA output.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.state: DynamicAdapterState | None = None
        self.handles: list[Any] = []

    def set_state(self, state: DynamicAdapterState) -> None:
        self.state = state

    def clear_state(self) -> None:
        self.state = None

    def install(self) -> None:
        layers = getattr(getattr(self.model, "model", None), "layers", None)
        if layers is None:
            raise ValueError("Expected model.model.layers for Mistral-style model.")

        for layer_idx, layer in enumerate(layers):
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is None:
                continue
            v_proj = getattr(self_attn, "v_proj", None)
            if v_proj is None:
                continue

            def make_hook(idx: int):
                def hook(_module: Any, inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
                    if self.state is None:
                        return output
                    a = self.state.layer_a.get(idx)
                    b = self.state.layer_b.get(idx)
                    if a is None or b is None:
                        return output
                    x = inputs[0]
                    if not isinstance(x, torch.Tensor):
                        return output
                    if a.device != x.device or a.dtype != x.dtype:
                        a = a.to(device=x.device, dtype=x.dtype)
                    if b.device != x.device or b.dtype != x.dtype:
                        b = b.to(device=x.device, dtype=x.dtype)
                    # x: [B,S,in], A: [B,R,in], B: [B,out,R] -> [B,S,out]
                    lora_out = torch.einsum("bsi,bri,bor->bso", x, a, b)
                    return output + self.state.scale * lora_out

                return hook

            self.handles.append(v_proj.register_forward_hook(make_hook(layer_idx)))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []
