#!/usr/bin/env bash
set -euo pipefail

# INPUT_JSONL=${1:-outputs/gpt-4.1-nano_XSUM_test.jsonl}
# LENGTH_OUTPUT=${2:-${INPUT_JSONL%.jsonl}_length_metrics.txt}
# ROUGE_OUTPUT=${3:-${INPUT_JSONL%.jsonl}_rouge_metrics.txt}

# python3 src/compute_length_metrics.py \
#   --input-jsonl "${INPUT_JSONL}" \
#   --output "${LENGTH_OUTPUT}"

# python3 src/compute_rouge_metrics.py \
#   --input-jsonl "${INPUT_JSONL}" \
#   --output "${ROUGE_OUTPUT}"

# Example alternative dataset invocation:
#   bash run_eval.sh outputs/gpt-4.1_DUC2004_test.jsonl


python3 src/compute_length_metrics.py \
  --input-jsonl outputs/gpt-4.1_XSUM_test.jsonl \
  --output outputs/gpt-4.1_XSUM_test_length_metrics.txt

python3 src/compute_length_metrics.py \
  --input-jsonl outputs/gpt-4.1-mini_XSUM_test.jsonl \
  --output outputs/gpt-4.1-mini_XSUM_test_length_metrics.txt

python3 src/compute_length_metrics.py \
  --input-jsonl outputs/gpt-4.1-nano_XSUM_test.jsonl \
  --output outputs/gpt-4.1_XSUM-nano_test_length_metrics.txt

python3 src/compute_rouge_metrics.py \
  --input-jsonl outputs/gpt-4.1_XSUM_test.jsonl \
  --output outputs/gpt-4.1_XSUM_test_rouge_metrics.txt

python3 src/compute_rouge_metrics.py \
  --input-jsonl outputs/gpt-4.1-mini_XSUM_test.jsonl \
  --output outputs/gpt-4.1-mini_XSUM_test_rouge_metrics.txt

python3 src/compute_rouge_metrics.py \
  --input-jsonl outputs/gpt-4.1-nano_XSUM_test.jsonl \
  --output outputs/gpt-4.1_XSUM-nano_test_rouge_metrics.txt