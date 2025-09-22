#!/usr/bin/env python3
"""Convert MT-Bench LI questions JSONL to DUC2004-style summarization format."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


SOURCE_PATTERN = re.compile(r"\[Source\]\s*:(.*?)(?=\s*\[Prompt\])", re.DOTALL)


def extract_source(block: str, record_id: str | int, line_no: int) -> str:
    """Pull the source passage between the [Source] and [Prompt] markers."""
    match = SOURCE_PATTERN.search(block)
    if not match:
        location = f"id={record_id!r}" if record_id is not None else f"line {line_no}"
        raise ValueError(f"Unable to locate source text for {location}; got: {block[:80]!r}")
    return match.group(1).strip()


def convert_line(raw: str, line_no: int, include_target: bool) -> dict[str, object]:
    data = json.loads(raw)

    question_id = data.get("question_id")
    record_id = str(question_id) if question_id is not None else f"record_{line_no}"

    turns = data.get("turns") or []
    if not turns:
        raise ValueError(f"Missing turns array for id={record_id}")

    source_text = extract_source(turns[0], record_id=record_id, line_no=line_no)

    reference = data.get("reference_answer")
    summaries = [reference] if reference is not None else []

    record: dict[str, object] = {
        "id": record_id,
        "text": source_text,
        "summaries": summaries,
    }

    if include_target and "target_length" in data:
        record["target_length"] = data["target_length"]

    return record


def convert_file(input_path: Path, output_path: Path, include_target: bool) -> None:
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_no, raw in enumerate(src, 1):
            raw = raw.strip()
            if not raw:
                continue
            converted = convert_line(raw, line_no=line_no, include_target=include_target)
            json.dump(converted, dst, ensure_ascii=False)
            dst.write("\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to the MT-Bench LI questions JSONL file")
    parser.add_argument("output", type=Path, help="Destination path for the converted JSONL file")
    parser.add_argument(
        "--include-target-length",
        action="store_true",
        help="Carry the target_length field into the output records",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)

    convert_file(args.input, args.output, include_target=args.include_target_length)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
