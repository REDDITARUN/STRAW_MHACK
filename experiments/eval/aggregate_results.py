from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate experiment result JSON files.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="List of result JSON files (from eval scripts).",
    )
    parser.add_argument("--output-path", default="experiments/results/aggregate_results.json")
    return parser.parse_args()


def load_json(path: str) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["_source_file"] = str(p)
    return data


def main() -> None:
    args = parse_args()
    runs = [load_json(path) for path in args.inputs]

    summary: dict[str, Any] = {"runs": runs}
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved aggregate file: {out}")


if __name__ == "__main__":
    main()
