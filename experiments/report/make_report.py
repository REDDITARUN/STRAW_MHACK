from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create markdown comparison report from aggregate results.")
    parser.add_argument("--aggregate-path", default="experiments/results/aggregate_results.json")
    parser.add_argument("--output-path", default="experiments/report/comparison_report.md")
    return parser.parse_args()


def read_json(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def run_name(run: dict[str, Any]) -> str:
    if run.get("adapter_path"):
        return f"LoRA({run['adapter_path']})"
    if run.get("hypernet_ckpt"):
        return f"STRAW({run['hypernet_ckpt']})"
    return f"Base({run.get('model_name_or_path', run.get('base_model', 'model'))})"


def to_rows(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in aggregate.get("runs", []):
        name = run_name(run)
        for ds in run.get("datasets", []):
            rows.append(
                {
                    "run": name,
                    "dataset": ds.get("dataset"),
                    "split": ds.get("split"),
                    "accuracy": ds.get("accuracy"),
                    "num_scored": ds.get("num_scored"),
                    "num_unlabeled": ds.get("num_unlabeled"),
                    "source": run.get("_source_file", ""),
                }
            )
    return rows


def format_accuracy(v: Any) -> str:
    if v is None:
        return "NA"
    return f"{100.0 * float(v):.2f}%"


def write_markdown(rows: list[dict[str, Any]], out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# STRAW Experiment Comparison Report",
        "",
        "| Run | Dataset | Split | Accuracy | Scored | Unlabeled |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['dataset']} | {row['split']} | "
            f"{format_accuracy(row['accuracy'])} | {row['num_scored']} | {row['num_unlabeled']} |"
        )

    lines.extend(["", "## Sources", ""])
    for i, row in enumerate(rows, start=1):
        lines.append(f"- {i}. `{row['source']}`")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved report: {out}")


def main() -> None:
    args = parse_args()
    aggregate = read_json(args.aggregate_path)
    rows = to_rows(aggregate)
    write_markdown(rows, args.output_path)


if __name__ == "__main__":
    main()
