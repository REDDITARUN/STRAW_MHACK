from __future__ import annotations

import re
from typing import Any

import torch
import torch.nn.functional as F


def _extract_layer_idx(module_name: str) -> int:
    match = re.search(r"\.layers\.(\d+)\.", module_name)
    return int(match.group(1)) if match else -1


def compress_2d(matrix: torch.Tensor, out_size: int = 64) -> torch.Tensor:
    """
    Compress [out, in] matrix to [out_size, out_size] for visualization.
    """
    if matrix.dim() != 2:
        raise ValueError("Expected 2D matrix for compression.")
    x = matrix.unsqueeze(0).unsqueeze(0).float()
    x = F.interpolate(x, size=(out_size, out_size), mode="area")
    return x.squeeze(0).squeeze(0).detach().cpu()


def get_static_lora_ba(
    model: Any,
    *,
    downsample: int = 64,
) -> dict[int, torch.Tensor]:
    """
    Extract effective BA (= scaling * B@A) from PEFT LoRA-wrapped v_proj layers.
    Returns compressed heatmap tensors per layer index.
    """
    result: dict[int, torch.Tensor] = {}

    for name, module in model.named_modules():
        if "v_proj" not in name:
            continue
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        if not hasattr(module, "active_adapter"):
            continue

        active = module.active_adapter
        if isinstance(active, list):
            adapter_name = active[0] if active else None
        else:
            adapter_name = active
        if adapter_name is None:
            continue
        if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
            continue

        a_weight = module.lora_A[adapter_name].weight.detach()  # [r, in]
        b_weight = module.lora_B[adapter_name].weight.detach()  # [out, r]
        scaling = float(module.scaling.get(adapter_name, 1.0))
        ba = scaling * torch.matmul(b_weight, a_weight)  # [out, in]
        layer_idx = _extract_layer_idx(name)
        if layer_idx >= 0:
            result[layer_idx] = compress_2d(ba, out_size=downsample)

    return result


def running_mean_update(
    current: dict[int, torch.Tensor],
    sample: dict[int, torch.Tensor],
    n_prev: int,
) -> dict[int, torch.Tensor]:
    """
    Update running mean maps: new_mean = (prev*n + sample)/(n+1)
    """
    if n_prev < 0:
        raise ValueError("n_prev must be >= 0")
    out = dict(current)
    for layer_idx, v in sample.items():
        if layer_idx not in out:
            out[layer_idx] = v.clone()
        else:
            out[layer_idx] = (out[layer_idx] * n_prev + v) / float(n_prev + 1)
    return out


def dynamic_state_to_ba_heatmaps(
    layer_a: dict[int, torch.Tensor],
    layer_b: dict[int, torch.Tensor],
    *,
    downsample: int = 64,
) -> dict[int, torch.Tensor]:
    """
    Convert dynamic layer A/B to compressed BA heatmaps.
    Expects batch size 1 during eval generation.
    """
    out: dict[int, torch.Tensor] = {}
    for layer_idx in layer_a.keys():
        a = layer_a[layer_idx]  # [B, r, in]
        b = layer_b[layer_idx]  # [B, out, r]
        if a.shape[0] < 1 or b.shape[0] < 1:
            continue
        ba = torch.matmul(b[0], a[0])  # [out, in]
        out[layer_idx] = compress_2d(ba, out_size=downsample)
    return out
