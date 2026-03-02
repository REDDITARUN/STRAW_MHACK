from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize domain-specific STRAW BA maps with comparable scales."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to BA payload (.pt) produced by run_eval_straw_gen --save-ba-path.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[0, 8, 16, 24, 31],
        help="Layer indices to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results/ba_compare",
        help="Directory to write PNG outputs.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Percentile used to set fixed color range for signed/diff maps.",
    )
    return parser.parse_args()


def as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_domain_maps(payload_path: str) -> dict[str, dict[int, np.ndarray]]:
    payload = torch.load(payload_path, map_location="cpu")
    datasets = payload.get("datasets", {})
    out: dict[str, dict[int, np.ndarray]] = {}
    for ds_name, ds_data in datasets.items():
        layer_map = ds_data.get("ba_mean_by_layer", {})
        out[ds_name] = {int(k): as_numpy(v) for k, v in layer_map.items()}
    if not out:
        raise ValueError("No dataset BA maps found in payload.")
    return out


def color_limit(arrays: list[np.ndarray], percentile: float) -> float:
    vals = np.concatenate([np.abs(a).ravel() for a in arrays if a.size > 0], axis=0)
    if vals.size == 0:
        return 1.0
    v = float(np.percentile(vals, percentile))
    return v if v > 0 else 1.0


def save_abs_grid(
    domain_maps: dict[str, dict[int, np.ndarray]],
    layer: int,
    out_path: Path,
) -> None:
    domains = sorted(domain_maps.keys())
    arrays = [np.abs(domain_maps[d][layer]) for d in domains if layer in domain_maps[d]]
    if not arrays:
        return
    vmax = color_limit(arrays, percentile=99.0)
    fig, axes = plt.subplots(1, len(domains), figsize=(4 * len(domains), 4), dpi=160)
    if len(domains) == 1:
        axes = [axes]
    for ax, d in zip(axes, domains):
        if layer not in domain_maps[d]:
            ax.set_axis_off()
            continue
        arr = np.abs(domain_maps[d][layer])
        im = ax.imshow(arr, cmap="viridis", aspect="auto", vmin=0.0, vmax=vmax)
        ax.set_title(f"{d} | abs(BA) | L{layer}")
        ax.set_xlabel("v_proj_in (compressed)")
        ax.set_ylabel("v_proj_out (compressed)")
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_diff_grid(
    domain_maps: dict[str, dict[int, np.ndarray]],
    layer: int,
    out_path: Path,
    percentile: float,
) -> None:
    domains = sorted(domain_maps.keys())
    pairs = [(a, b) for a, b in combinations(domains, 2)]
    diffs: list[np.ndarray] = []
    valid_pairs: list[tuple[str, str]] = []
    for a, b in pairs:
        if layer in domain_maps[a] and layer in domain_maps[b]:
            diffs.append(domain_maps[a][layer] - domain_maps[b][layer])
            valid_pairs.append((a, b))
    if not diffs:
        return

    vmax = color_limit(diffs, percentile=percentile)
    fig, axes = plt.subplots(1, len(diffs), figsize=(4 * len(diffs), 4), dpi=160)
    if len(diffs) == 1:
        axes = [axes]
    for ax, diff, (a, b) in zip(axes, diffs, valid_pairs):
        im = ax.imshow(diff, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{a} - {b} | L{layer}")
        ax.set_xlabel("v_proj_in (compressed)")
        ax.set_ylabel("v_proj_out (compressed)")
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_layer_norms(domain_maps: dict[str, dict[int, np.ndarray]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
    for ds_name in sorted(domain_maps.keys()):
        layer_items = sorted(domain_maps[ds_name].items(), key=lambda x: x[0])
        if not layer_items:
            continue
        xs = [idx for idx, _ in layer_items]
        ys = [float(np.linalg.norm(arr)) for _, arr in layer_items]
        ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3.5, label=ds_name)
    ax.set_title("Per-layer BA Frobenius norm by domain")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("||BA||_F (compressed map)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    domain_maps = load_domain_maps(args.input)
    for layer in args.layers:
        save_abs_grid(
            domain_maps=domain_maps,
            layer=layer,
            out_path=out_dir / f"layer_{layer}_abs.png",
        )
        save_diff_grid(
            domain_maps=domain_maps,
            layer=layer,
            out_path=out_dir / f"layer_{layer}_diff.png",
            percentile=args.percentile,
        )
    save_layer_norms(domain_maps=domain_maps, out_path=out_dir / "layer_norms.png")
    print(f"Saved comparison figures to: {out_dir}")


if __name__ == "__main__":
    main()
