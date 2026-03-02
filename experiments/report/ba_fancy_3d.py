from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create minimalist 3D BA visualizations for domain comparison."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to BA payload (.pt) generated via --save-ba-path.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[0, 8, 16, 24, 31],
        help="Layer indices to render.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results/ba_fancy_3d",
        help="Directory for output plots.",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=35.0,
        help="3D elevation angle.",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=-55.0,
        help="3D azimuth angle.",
    )
    return parser.parse_args()


def to_np(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_domain_maps(path: str) -> dict[str, dict[int, np.ndarray]]:
    payload = torch.load(path, map_location="cpu")
    datasets = payload.get("datasets", {})
    out: dict[str, dict[int, np.ndarray]] = {}
    for ds_name, ds_data in datasets.items():
        layer_map = ds_data.get("ba_mean_by_layer", {})
        out[ds_name] = {int(k): to_np(v) for k, v in layer_map.items()}
    if not out:
        raise ValueError("No BA maps found in payload.")
    return out


def clean_3d_axis(ax: Any, title: str) -> None:
    ax.set_title(title, pad=12, fontsize=10, fontweight="semibold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False


def make_surface(ax: Any, arr: np.ndarray, cmap: Any, elev: float, azim: float) -> None:
    h, w = arr.shape
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(x, y)
    zz = arr
    ax.plot_surface(
        xx,
        yy,
        zz,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1,
        shade=True,
        alpha=0.95,
    )
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1.0, 1.0, 0.35))


def save_domain_surfaces(
    domain_maps: dict[str, dict[int, np.ndarray]],
    layer: int,
    out_path: Path,
    elev: float,
    azim: float,
) -> None:
    domains = sorted(domain_maps.keys())
    arrays: list[np.ndarray] = [domain_maps[d][layer] for d in domains if layer in domain_maps[d]]
    if not arrays:
        return
    vmax = float(np.percentile(np.abs(np.concatenate([a.ravel() for a in arrays])), 99))
    if vmax <= 0:
        vmax = 1.0

    fig = plt.figure(figsize=(4.6 * len(domains), 4.6), dpi=170)
    for i, ds_name in enumerate(domains, start=1):
        if layer not in domain_maps[ds_name]:
            continue
        arr = np.clip(domain_maps[ds_name][layer], -vmax, vmax) / vmax
        ax = fig.add_subplot(1, len(domains), i, projection="3d")
        make_surface(ax, arr, cm.magma, elev=elev, azim=azim)
        clean_3d_axis(ax, f"{ds_name} | L{layer}")

    fig.suptitle("STRAW BA Surfaces (normalized, shared scale)", fontsize=12, y=0.96)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_diff_surfaces(
    domain_maps: dict[str, dict[int, np.ndarray]],
    layer: int,
    out_path: Path,
    elev: float,
    azim: float,
) -> None:
    domains = sorted(domain_maps.keys())
    pairs = [(a, b) for a, b in combinations(domains, 2)]
    diffs: list[tuple[str, np.ndarray]] = []
    for a, b in pairs:
        if layer in domain_maps[a] and layer in domain_maps[b]:
            diffs.append((f"{a} - {b}", domain_maps[a][layer] - domain_maps[b][layer]))
    if not diffs:
        return

    vmax = float(np.percentile(np.abs(np.concatenate([d.ravel() for _, d in diffs])), 99))
    if vmax <= 0:
        vmax = 1.0

    fig = plt.figure(figsize=(4.8 * len(diffs), 4.8), dpi=170)
    for i, (name, diff) in enumerate(diffs, start=1):
        arr = np.clip(diff, -vmax, vmax) / vmax
        ax = fig.add_subplot(1, len(diffs), i, projection="3d")
        make_surface(ax, arr, cm.coolwarm, elev=elev, azim=azim)
        clean_3d_axis(ax, f"{name} | L{layer}")

    fig.suptitle("Domain Difference Surfaces (normalized, shared scale)", fontsize=12, y=0.96)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    domain_maps = load_domain_maps(args.input)
    for layer in args.layers:
        save_domain_surfaces(
            domain_maps=domain_maps,
            layer=layer,
            out_path=out_dir / f"layer_{layer}_surfaces.png",
            elev=args.elev,
            azim=args.azim,
        )
        save_diff_surfaces(
            domain_maps=domain_maps,
            layer=layer,
            out_path=out_dir / f"layer_{layer}_diff_surfaces.png",
            elev=args.elev,
            azim=args.azim,
        )
    print(f"Saved fancy 3D BA plots to: {out_dir}")


if __name__ == "__main__":
    main()
