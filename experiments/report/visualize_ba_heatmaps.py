from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create BA heatmap PNGs/GIFs from saved BA payloads.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Paths to BA .pt payload files.")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=[],
        help="Optional labels for inputs (same count as --inputs).",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[0],
        help="Layer indices to visualize.",
    )
    parser.add_argument("--output-dir", default="experiments/results/ba_viz")
    parser.add_argument("--fps", type=int, default=1)
    return parser.parse_args()


def tensor_to_np(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def frames_from_payload(payload: dict[str, Any], source_label: str, layer_idx: int) -> list[tuple[str, np.ndarray]]:
    frames: list[tuple[str, np.ndarray]] = []
    if "layer_ba" in payload:
        layer_map = payload["layer_ba"]
        if layer_idx in layer_map:
            frames.append((f"{source_label} | static", tensor_to_np(layer_map[layer_idx])))
        return frames

    datasets = payload.get("datasets", {})
    for dataset_name, ds_data in datasets.items():
        layer_map = ds_data.get("ba_mean_by_layer", {})
        if layer_idx in layer_map:
            frames.append((f"{source_label} | {dataset_name}", tensor_to_np(layer_map[layer_idx])))
    return frames


def save_heatmap(arr: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    im = ax.imshow(arr, cmap="magma", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("v_proj_in (compressed)")
    ax.set_ylabel("v_proj_out (compressed)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.labels and len(args.labels) != len(args.inputs):
        raise ValueError("--labels must be empty or have same length as --inputs")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    payloads: list[tuple[str, dict[str, Any]]] = []
    for i, path_str in enumerate(args.inputs):
        label = args.labels[i] if args.labels else Path(path_str).stem
        payloads.append((label, torch.load(path_str, map_location="cpu")))

    for layer_idx in args.layers:
        layer_dir = out_root / f"layer_{layer_idx}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        frames_for_gif: list[np.ndarray] = []
        frame_idx = 0
        for label, payload in payloads:
            frames = frames_from_payload(payload, label, layer_idx)
            for frame_title, arr in frames:
                png_path = layer_dir / f"{frame_idx:03d}.png"
                save_heatmap(arr, frame_title, png_path)
                frames_for_gif.append(imageio.imread(png_path))
                frame_idx += 1

        if frames_for_gif:
            gif_path = layer_dir / f"layer_{layer_idx}.gif"
            imageio.mimsave(gif_path, frames_for_gif, fps=args.fps)
            print(f"Saved GIF: {gif_path}")
        else:
            print(f"No frames found for layer {layer_idx}.")


if __name__ == "__main__":
    main()
