from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


@dataclass
class HypernetOutput:
    """
    A and B tensors shaped for LoRA-style low-rank update.
    """

    a: torch.Tensor  # [batch, num_layers, rank, v_proj_in]
    b: torch.Tensor  # [batch, num_layers, v_proj_out, rank]


class TinyBertLayerGenerator(nn.Module):
    """
    Tiny BERT-based generator for one transformer layer.
    """

    def __init__(
        self,
        residual_dim: int,
        rank: int,
        v_proj_in: int,
        v_proj_out: int,
        bert_hidden_size: int,
        bert_layers: int,
        bert_heads: int,
        bert_dropout: float,
    ) -> None:
        super().__init__()
        if bert_hidden_size % bert_heads != 0:
            raise ValueError("bert_hidden_size must be divisible by bert_heads.")

        self.rank = rank
        self.v_proj_in = v_proj_in
        self.v_proj_out = v_proj_out

        self.input_proj = nn.Linear(residual_dim, bert_hidden_size)
        config = BertConfig(
            vocab_size=1,  # unused because we pass inputs_embeds; keep tiny to save params
            hidden_size=bert_hidden_size,
            intermediate_size=bert_hidden_size * 4,
            num_hidden_layers=bert_layers,
            num_attention_heads=bert_heads,
            hidden_dropout_prob=bert_dropout,
            attention_probs_dropout_prob=bert_dropout,
            max_position_embeddings=8,
        )
        self.bert = BertModel(config)
        self.head_a = nn.Linear(bert_hidden_size, rank * v_proj_in)
        self.head_b = nn.Linear(bert_hidden_size, v_proj_out * rank)

    def forward(self, layer_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(layer_input).unsqueeze(1)  # [B,1,H]
        encoded = self.bert(inputs_embeds=x).last_hidden_state
        pooled = encoded.mean(dim=1)  # [B,H]

        batch_size = layer_input.shape[0]
        a = self.head_a(pooled).view(batch_size, self.rank, self.v_proj_in)
        b = self.head_b(pooled).view(batch_size, self.v_proj_out, self.rank)
        return a, b


class BertHyperLoraGenerator(nn.Module):
    """
    Per-layer hypernetwork generator.
    Each transformer layer has its own small network that maps that layer's
    prefix-pooled residual state to LoRA A/B for that same layer.
    """

    def __init__(
        self,
        residual_dim: int,
        num_layers: int,
        v_proj_in: int,
        v_proj_out: int,
        rank: int,
        bert_hidden_size: int = 64,
        bert_layers: int = 1,
        bert_heads: int = 2,
        bert_dropout: float = 0.0,
        layer_stride: int = 1,
    ) -> None:
        super().__init__()
        if layer_stride <= 0:
            raise ValueError("layer_stride must be >= 1.")
        self.num_layers = num_layers
        self.v_proj_in = v_proj_in
        self.v_proj_out = v_proj_out
        self.rank = rank
        self.layer_stride = layer_stride
        self.active_layers = [i for i in range(num_layers) if (i % layer_stride) == 0]
        if not self.active_layers:
            raise ValueError("No active layers selected for hypernetwork generation.")

        self.layer_generators = nn.ModuleDict(
            {
                str(layer_idx): TinyBertLayerGenerator(
                    residual_dim=residual_dim,
                    rank=rank,
                    v_proj_in=v_proj_in,
                    v_proj_out=v_proj_out,
                    bert_hidden_size=bert_hidden_size,
                    bert_layers=bert_layers,
                    bert_heads=bert_heads,
                    bert_dropout=bert_dropout,
                )
                for layer_idx in self.active_layers
            }
        )

    def forward(self, prefix_hidden: torch.Tensor) -> HypernetOutput:
        """
        prefix_hidden:
          - [batch, num_layers, residual_dim] preferred
          - [batch, residual_dim] accepted; repeated across layers
        """
        if prefix_hidden.dim() == 2:
            prefix_hidden = prefix_hidden.unsqueeze(1).repeat(1, self.num_layers, 1)
        if prefix_hidden.dim() != 3:
            raise ValueError("Expected prefix_hidden with shape [B, L, H] or [B, H].")
        if prefix_hidden.shape[1] != self.num_layers:
            raise ValueError(
                f"Expected num_layers={self.num_layers}, got {prefix_hidden.shape[1]}."
            )

        a_list: list[torch.Tensor] = []
        b_list: list[torch.Tensor] = []

        batch_size = prefix_hidden.shape[0]
        zero_a = prefix_hidden.new_zeros(batch_size, self.rank, self.v_proj_in)
        zero_b = prefix_hidden.new_zeros(batch_size, self.v_proj_out, self.rank)
        active_layer_set = set(self.active_layers)

        for layer_idx in range(self.num_layers):
            if layer_idx in active_layer_set:
                layer_input = prefix_hidden[:, layer_idx, :]
                a_layer, b_layer = self.layer_generators[str(layer_idx)](layer_input)
            else:
                # Inactive layers receive zero LoRA delta.
                a_layer = zero_a
                b_layer = zero_b
            a_list.append(a_layer)
            b_list.append(b_layer)

        a = torch.stack(a_list, dim=1)
        b = torch.stack(b_list, dim=1)
        return HypernetOutput(a=a, b=b)


class CnnHyperLoraGenerator(nn.Module):
    """
    CNN-based hypernetwork over the layer axis.
    Uses shared conv blocks + shared heads, with per-layer embeddings.
    """

    def __init__(
        self,
        residual_dim: int,
        num_layers: int,
        v_proj_in: int,
        v_proj_out: int,
        rank: int,
        conv_hidden_size: int = 64,
        conv_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.0,
        layer_stride: int = 1,
    ) -> None:
        super().__init__()
        if layer_stride <= 0:
            raise ValueError("layer_stride must be >= 1.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd positive integer.")
        if conv_layers <= 0:
            raise ValueError("conv_layers must be >= 1.")

        self.num_layers = num_layers
        self.v_proj_in = v_proj_in
        self.v_proj_out = v_proj_out
        self.rank = rank
        self.layer_stride = layer_stride
        self.active_layers = [i for i in range(num_layers) if (i % layer_stride) == 0]
        if not self.active_layers:
            raise ValueError("No active layers selected for hypernetwork generation.")

        self.input_proj = nn.Linear(residual_dim, conv_hidden_size)
        self.layer_embed = nn.Embedding(num_layers, conv_hidden_size)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        conv_hidden_size,
                        conv_hidden_size,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(conv_layers)
            ]
        )
        self.norm = nn.LayerNorm(conv_hidden_size)
        self.head_a = nn.Linear(conv_hidden_size, rank * v_proj_in)
        self.head_b = nn.Linear(conv_hidden_size, v_proj_out * rank)

    def forward(self, prefix_hidden: torch.Tensor) -> HypernetOutput:
        if prefix_hidden.dim() == 2:
            prefix_hidden = prefix_hidden.unsqueeze(1).repeat(1, self.num_layers, 1)
        if prefix_hidden.dim() != 3:
            raise ValueError("Expected prefix_hidden with shape [B, L, H] or [B, H].")
        if prefix_hidden.shape[1] != self.num_layers:
            raise ValueError(
                f"Expected num_layers={self.num_layers}, got {prefix_hidden.shape[1]}."
            )

        # [B, L, H] -> [B, C, L]
        x = self.input_proj(prefix_hidden).transpose(1, 2)
        for block in self.blocks:
            x = x + block(x)
        x = x.transpose(1, 2)  # [B, L, C]

        batch_size = x.shape[0]
        zero_a = x.new_zeros(batch_size, self.rank, self.v_proj_in)
        zero_b = x.new_zeros(batch_size, self.v_proj_out, self.rank)
        active_layer_set = set(self.active_layers)
        a_list: list[torch.Tensor] = []
        b_list: list[torch.Tensor] = []

        for layer_idx in range(self.num_layers):
            if layer_idx in active_layer_set:
                layer_idx_tensor = torch.full(
                    (batch_size,),
                    layer_idx,
                    device=x.device,
                    dtype=torch.long,
                )
                feat = self.norm(x[:, layer_idx, :] + self.layer_embed(layer_idx_tensor))
                a_layer = self.head_a(feat).view(batch_size, self.rank, self.v_proj_in)
                b_layer = self.head_b(feat).view(batch_size, self.v_proj_out, self.rank)
            else:
                a_layer = zero_a
                b_layer = zero_b
            a_list.append(a_layer)
            b_list.append(b_layer)

        a = torch.stack(a_list, dim=1)
        b = torch.stack(b_list, dim=1)
        return HypernetOutput(a=a, b=b)


def build_hyper_lora_generator(
    *,
    residual_dim: int,
    num_layers: int,
    v_proj_in: int,
    v_proj_out: int,
    rank: int,
    hyper_cfg: dict[str, Any],
) -> nn.Module:
    model_type = str(hyper_cfg.get("model_type", "small_bert")).strip().lower()
    layer_stride = int(hyper_cfg.get("layer_stride", 1))

    if model_type in {"small_bert", "bert", "tiny_bert"}:
        return BertHyperLoraGenerator(
            residual_dim=residual_dim,
            num_layers=num_layers,
            v_proj_in=v_proj_in,
            v_proj_out=v_proj_out,
            rank=rank,
            bert_hidden_size=int(hyper_cfg.get("hidden_size", 64)),
            bert_layers=int(hyper_cfg.get("num_hidden_layers", 1)),
            bert_heads=int(hyper_cfg.get("num_attention_heads", 2)),
            bert_dropout=float(hyper_cfg.get("dropout", 0.0)),
            layer_stride=layer_stride,
        )

    if model_type in {"cnn", "conv", "conv1d"}:
        return CnnHyperLoraGenerator(
            residual_dim=residual_dim,
            num_layers=num_layers,
            v_proj_in=v_proj_in,
            v_proj_out=v_proj_out,
            rank=rank,
            conv_hidden_size=int(hyper_cfg.get("hidden_size", 64)),
            conv_layers=int(hyper_cfg.get("num_hidden_layers", 2)),
            kernel_size=int(hyper_cfg.get("kernel_size", 3)),
            dropout=float(hyper_cfg.get("dropout", 0.0)),
            layer_stride=layer_stride,
        )

    raise ValueError(f"Unsupported hypernet.model_type='{model_type}'. Use 'small_bert' or 'cnn'.")
