#!/usr/bin/env python3
"""Compute length-based metrics for InstructCMP JSONL outputs."""

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute MALD and length exact-match rate from a JSONL output file."
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


def extract_summary(sentence: str) -> str:
    """Mimic evaluation post-processing to locate the first non-empty summary line."""
    parts = sentence.strip().split("\n")
    while parts:
        summary = parts.pop(0).strip()
        cleaned = (
            summary.replace("\\'", "")
            .replace('\"', "")
            .replace("`", "")
            .replace("”", "")
            .replace("’ s", "")
            .strip()
        )
        if cleaned not in {"", "Sentence:"}:
            return summary
    return ""


def load_rows(jsonl_path: Path) -> Iterable[Tuple[str, str]]:
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            yield row["target"], extract_summary(row["completion"])


def word_count(text: str) -> int:
    return len(text.split())


def compute_metrics(pairs: Iterable[Tuple[str, str]]) -> Tuple[int, float, float]:
    length_diffs = []
    exact_matches = 0

    for target, hypothesis in pairs:
        lt = word_count(target)
        lh = word_count(hypothesis)
        length_diffs.append(abs(lh - lt))
        if lh == lt:
            exact_matches += 1

    count = len(length_diffs)
    if count == 0:
        raise ValueError("No valid rows found in the input JSONL file.")

    mald = sum(length_diffs) / count
    em_rate = exact_matches / count
    return count, mald, em_rate


def main() -> None:
    args = parse_args()

    if not args.input_jsonl.exists():
        raise FileNotFoundError(f"Could not find input JSONL file: {args.input_jsonl}")

    count, mald, em_rate = compute_metrics(list(load_rows(args.input_jsonl)))

    output_lines = [
        f"Length statistics for {args.input_jsonl}",
        f"Example count: {count}",
        f"Mean absolute length deviation (MALD): {mald:.4f}",
        f"Length exact match rate (EM): {em_rate:.4f}",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
