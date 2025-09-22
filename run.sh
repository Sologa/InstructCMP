#!/usr/bin/env bash
set -euo pipefail

: "${OPENAI_API_KEY:?OPENAI_API_KEY is not set. Export it before running.}"
: "${OPENAI_ORGANIZATION_ID:?OPENAI_ORGANIZATION_ID is not set. Export it before running.}"

MODEL=${MODEL:-gpt-4.1}
DATA_NAME=${DATA_NAME:-DUC2004}
SPLIT=${SPLIT:-test}
LIMIT=${LIMIT:-25}
MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS:-150}
TEMPERATURE=${TEMPERATURE:-0.0}
OUTPUT_JSONL=${OUTPUT_JSONL:-outputs/${MODEL}_${DATA_NAME}_${SPLIT}.jsonl}
REQUEST_INTERVAL=${REQUEST_INTERVAL:-1.0}
MAX_RETRIES=${MAX_RETRIES:-3}
TIMEOUT=${TIMEOUT:-60}
RETRY_INITIAL_DELAY=${RETRY_INITIAL_DELAY:-1.0}
RETRY_BACKOFF=${RETRY_BACKOFF:-2.0}
CONCURRENCY=${CONCURRENCY:-1}
SYSTEM_PROMPT=${SYSTEM_PROMPT:-"You are a precise sentence compression assistant. Follow the instructions and respond with only the compressed sentence."}

# Ensure NLTK punkt tokenizer is available for evaluation.
python -m nltk.downloader punkt >/dev/null 2>&1 || true



python src/run_openai.py \
  --model "${MODEL}" \
  --data_name "${DATA_NAME}" \
  --split "${SPLIT}" \
  --limit "${LIMIT}" \
  --max_output_tokens "${MAX_OUTPUT_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --request_interval "${REQUEST_INTERVAL}" \
  --concurrency "${CONCURRENCY}" \
  --max_retries "${MAX_RETRIES}" \
  --timeout "${TIMEOUT}" \
  --retry_initial_delay "${RETRY_INITIAL_DELAY}" \
  --retry_backoff "${RETRY_BACKOFF}" \
  --output_jsonl "${OUTPUT_JSONL}" \
  --system_prompt "${SYSTEM_PROMPT}"
