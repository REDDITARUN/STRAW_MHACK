from __future__ import annotations

import argparse
from pathlib import Path

from data.processors.codealpaca_gen import CodeAlpacaGenerativeProcessor
from data.processors.dolly_gen import DollyGenerativeProcessor
from data.processors.samsum_gen import SamsumGenerativeProcessor


PROCESSORS = {
    "samsum_gen": SamsumGenerativeProcessor,
    "dolly_gen": DollyGenerativeProcessor,
    "codealpaca_gen": CodeAlpacaGenerativeProcessor,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build generative-domain dataset JSONL files.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(PROCESSORS.keys()),
        help="Datasets to build. Omit to build all.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed_gen",
        help="Output folder for processed JSONL files.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=10000,
        help="Cap train split size per dataset. Set 0 to keep full train split.",
    )
    parser.add_argument(
        "--smoke-train-samples",
        type=int,
        default=1000,
        help="Train cap when --smoke is used.",
    )
    parser.add_argument(
        "--smoke-eval-samples",
        type=int,
        default=500,
        help="Validation/test synthetic split sizes for single-split datasets in smoke mode.",
    )
    parser.add_argument(
        "--validation-size",
        type=int,
        default=1000,
        help="Validation split size when source has only train split.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=1000,
        help="Test split size when source has only train split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = args.datasets if args.datasets else list(PROCESSORS.keys())
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    max_train = args.smoke_train_samples if args.smoke else args.max_train_samples
    val_size = args.smoke_eval_samples if args.smoke else args.validation_size
    test_size = args.smoke_eval_samples if args.smoke else args.test_size

    for key in selected:
        print(f"Building {key}...")
        processor = PROCESSORS[key]()
        processor.run(
            out_root,
            max_train_samples=max_train,
            seed=args.seed,
            validation_size=val_size,
            test_size=test_size,
        )
        print(f"Done: {key}")

    print(f"All done. Files written in: {out_root}")


if __name__ == "__main__":
    main()
