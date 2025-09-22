#!/usr/bin/env bash
source ~/.zshrc
conda activate instructcmp

set -euo pipefail

CONFIG_FILE=${RUN_CONFIG_FILE:-config/api_keys.sh}
if [[ -f "${CONFIG_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
elif [[ -n "${RUN_CONFIG_FILE:-}" ]]; then
  echo "[run.sh] Warning: RUN_CONFIG_FILE='${CONFIG_FILE}' not found." >&2
fi

API_PROVIDER=${API_PROVIDER:-openai}
RUN_DRIVER=${RUN_DRIVER:-}
API_EXTRA_ARGS=(--api_provider "${API_PROVIDER}")

case "${API_PROVIDER}" in
  openai)
    : "${OPENAI_API_KEY:?OPENAI_API_KEY is not set. Configure it in ${CONFIG_FILE} or the environment.}"
    RUN_DRIVER=${RUN_DRIVER:-src/run_openai.py}
    API_EXTRA_ARGS+=(--api_key_env OPENAI_API_KEY)

    ORG_ENV=""
    if [[ -n "${OPENAI_ORGANIZATION_ID:-}" ]]; then
      ORG_ENV=OPENAI_ORGANIZATION_ID
    fi
    if [[ "${REQUIRE_OPENAI_ORG:-0}" != "0" ]]; then
      : "${OPENAI_ORGANIZATION_ID:?OPENAI_ORGANIZATION_ID is not set. Configure it in ${CONFIG_FILE} or the environment.}"
      ORG_ENV=OPENAI_ORGANIZATION_ID
    fi
    if [[ -n "${ORG_ENV}" ]]; then
      API_EXTRA_ARGS+=(--organization_env "${ORG_ENV}")
    fi

    if [[ -n "${OPENAI_BASE_URL:-}" ]]; then
      API_EXTRA_ARGS+=(--api_base_url "${OPENAI_BASE_URL}")
    fi
    if [[ -n "${OPENAI_API_ENDPOINT:-}" ]]; then
      API_EXTRA_ARGS+=(--api_endpoint "${OPENAI_API_ENDPOINT}")
    fi
    ;;
  deepseek)
    : "${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY is not set. Configure it in ${CONFIG_FILE} or the environment.}"
    RUN_DRIVER=${RUN_DRIVER:-src/run_openai.py}
    DEEPSEEK_BASE_URL=${DEEPSEEK_BASE_URL:-https://api.deepseek.com/v1}
    API_EXTRA_ARGS+=(
      --api_key_env DEEPSEEK_API_KEY
      --api_base_url "${DEEPSEEK_BASE_URL}"
      --api_endpoint chat/completions
    )
    ;;
  *)
    echo "Unsupported API_PROVIDER: ${API_PROVIDER}" >&2
    exit 1
    ;;
esac

if [[ ! -f "${RUN_DRIVER}" ]]; then
  echo "Execution driver '${RUN_DRIVER}' not found. Set RUN_DRIVER to a valid script path." >&2
  exit 1
fi

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

python "${RUN_DRIVER}" \
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
  --system_prompt "${SYSTEM_PROMPT}" \
  "${API_EXTRA_ARGS[@]}"
