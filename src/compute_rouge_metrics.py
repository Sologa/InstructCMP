#!/usr/bin/env python3
"""Compute ROUGE metrics for InstructCMP JSONL outputs."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import nltk

# Ensure project root (two levels up) is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate_utils import generated_output_post_processing, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute ROUGE-1/2/L metrics from a JSONL output file."
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        required=True,
        help="Path to the JSONL file containing source, target, prompt, and completion entries.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the text file where the computed metrics will be written.",
    )
    return parser.parse_args()


def ensure_tokenizer() -> None:
    """Download the NLTK punkt tokenizer if it's missing."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def load_records(jsonl_path: Path) -> Tuple[List[str], List[str], List[str], int]:
    sources: List[str] = []
    targets: List[str] = []
    completions: List[str] = []
    skipped = 0

    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            completion = row.get("completion", "")
            source = row.get("source", "")
            target = row.get("target", "")

            if completion == "error" or not source.strip() or not target.strip():
                skipped += 1
                continue

            sources.append(source)
            targets.append(target)
            completions.append(completion)

    return sources, targets, completions, skipped


def main() -> None:
    args = parse_args()

    if not args.input_jsonl.exists():
        raise FileNotFoundError(f"Could not find input JSONL file: {args.input_jsonl}")

    sources, targets, completions, skipped = load_records(args.input_jsonl)
    if not completions:
        raise ValueError("No valid completions found in the input JSONL file.")

    ensure_tokenizer()

    post_processed = generated_output_post_processing(completions)
    rouge_report = evaluate(targets, sources, post_processed).strip()

    report_lines = [rouge_report]
    if skipped:
        skip_line = f"Skipped records: {skipped}"
        report_lines.append(skip_line)
        print(skip_line)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
