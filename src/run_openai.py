import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import requests
from requests.exceptions import RequestException
from tqdm import tqdm

try:
    import aiohttp
    from aiohttp import ClientError
except ImportError:  # pragma: no cover - optional dependency
    aiohttp = None
    ClientError = Exception

# Ensure project root is importable when executing as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_template, apply_template, get_dataset
from src.evaluate_utils import generated_output_post_processing, evaluate


def build_prompts(sources, targets):
    template = get_template()
    instances = []
    for src, tgt in zip(sources, targets):
        instances.append(
            {
                "src": src,
                "del_len": len(src.split()) - len(tgt.split()),
            }
        )
    return apply_template(instances, template)


def extract_text_from_response(payload: dict) -> str:
    text = payload.get("output_text")
    if text:
        return text.strip()

    collected = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                collected.append(content.get("text", ""))
    return "".join(collected).strip()


def build_inputs(prompt: str, system_prompt: str):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": system_prompt,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": prompt,
                }
            ],
        },
    ]


def _run_inference_sync(
    prompts,
    *,
    model,
    max_tokens,
    temperature,
    system_prompt,
    request_interval,
    api_key,
    organization_id=None,
    max_retries=3,
    timeout=60.0,
    retry_initial_delay=1.0,
    retry_backoff=2.0,
):
    outputs = []
    error_count = 0
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if organization_id:
        headers["OpenAI-Organization"] = organization_id
    url = "https://api.openai.com/v1/responses"

    for prompt in tqdm(prompts, desc="Querying OpenAI"):
        body = {
            "model": model,
            "input": build_inputs(prompt, system_prompt),
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        success = False
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(url, headers=headers, json=body, timeout=timeout)
            except RequestException as exc:
                last_error = exc
            else:
                if response.status_code >= 400:
                    try:
                        error_payload = response.json()
                    except json.JSONDecodeError:
                        error_payload = {"error": response.text}
                    last_error = RuntimeError(
                        "OpenAI request failed with status "
                        f"{response.status_code}: {json.dumps(error_payload, ensure_ascii=False)}"
                    )
                else:
                    try:
                        data = response.json()
                    except json.JSONDecodeError as exc:
                        last_error = RuntimeError("Failed to decode OpenAI response as JSON")
                    else:
                        if "error" in data and data["error"]:
                            last_error = RuntimeError(
                                "OpenAI API returned an error: "
                                f"{data['error'].get('message', data['error'])}"
                            )
                        else:
                            outputs.append(extract_text_from_response(data))
                            success = True
                            break

            if success:
                break

            if attempt < max_retries:
                delay = retry_initial_delay * (retry_backoff ** (attempt - 1))
                if delay > 0:
                    time.sleep(delay)

        if not success:
            error_count += 1
            outputs.append("error")
            if last_error is not None:
                tqdm.write(f"Failed to fetch response after {max_retries} attempts: {last_error}")

        if request_interval and success:
            time.sleep(request_interval)

    return outputs, error_count


async def _run_inference_async(
    prompts,
    *,
    model,
    max_tokens,
    temperature,
    system_prompt,
    request_interval,
    api_key,
    organization_id=None,
    max_retries=3,
    timeout=60.0,
    retry_initial_delay=1.0,
    retry_backoff=2.0,
    concurrency=1,
):
    if aiohttp is None:
        raise ImportError(
            "aiohttp is required for asynchronous inference. Install it with `pip install aiohttp`."
        )

    total = len(prompts)
    outputs = ["error"] * total
    error_count = 0
    error_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if organization_id:
        headers["OpenAI-Organization"] = organization_id
    url = "https://api.openai.com/v1/responses"

    timeout_cfg = aiohttp.ClientTimeout(total=timeout)

    async with aiohttp.ClientSession(headers=headers, timeout=timeout_cfg) as session:
        async def fetch_single(idx, prompt):
            nonlocal error_count
            body = {
                "model": model,
                "input": build_inputs(prompt, system_prompt),
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            }
            last_error = None

            for attempt in range(1, max_retries + 1):
                completion = None
                success = False

                try:
                    async with semaphore:
                        async with session.post(url, json=body) as response:
                            if response.status >= 400:
                                try:
                                    error_payload = await response.json()
                                except (aiohttp.ContentTypeError, json.JSONDecodeError):
                                    error_text = await response.text()
                                    error_payload = {"error": error_text}
                                last_error = RuntimeError(
                                    "OpenAI request failed with status "
                                    f"{response.status}: {json.dumps(error_payload, ensure_ascii=False)}"
                                )
                            else:
                                try:
                                    data = await response.json()
                                except (aiohttp.ContentTypeError, json.JSONDecodeError):
                                    last_error = RuntimeError("Failed to decode OpenAI response as JSON")
                                else:
                                    if "error" in data and data["error"]:
                                        last_error = RuntimeError(
                                            "OpenAI API returned an error: "
                                            f"{data['error'].get('message', data['error'])}"
                                        )
                                    else:
                                        completion = extract_text_from_response(data)
                                        success = True
                except (ClientError, asyncio.TimeoutError) as exc:
                    last_error = exc

                if success and completion is not None:
                    outputs[idx] = completion
                    if request_interval:
                        await asyncio.sleep(request_interval)
                    return

                if attempt < max_retries:
                    delay = retry_initial_delay * (retry_backoff ** (attempt - 1))
                    if delay > 0:
                        await asyncio.sleep(delay)

            async with error_lock:
                error_count += 1
            if last_error is not None:
                tqdm.write(
                    f"Prompt index {idx} failed after {max_retries} attempts: {last_error}"
                )

        tasks = [asyncio.create_task(fetch_single(idx, prompt)) for idx, prompt in enumerate(prompts)]

        for coro in tqdm(asyncio.as_completed(tasks), total=total, desc="Querying OpenAI"):
            await coro

    return outputs, error_count


def run_inference(
    prompts,
    *,
    model,
    max_tokens,
    temperature,
    system_prompt,
    request_interval,
    api_key,
    organization_id=None,
    max_retries=3,
    timeout=60.0,
    retry_initial_delay=1.0,
    retry_backoff=2.0,
    concurrency=1,
):
    concurrency = max(1, int(concurrency))
    if concurrency > 1:
        return asyncio.run(
            _run_inference_async(
                prompts,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
                request_interval=request_interval,
                api_key=api_key,
                organization_id=organization_id,
                max_retries=max_retries,
                timeout=timeout,
                retry_initial_delay=retry_initial_delay,
                retry_backoff=retry_backoff,
                concurrency=concurrency,
            )
        )

    return _run_inference_sync(
        prompts,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        request_interval=request_interval,
        api_key=api_key,
        organization_id=organization_id,
        max_retries=max_retries,
        timeout=timeout,
        retry_initial_delay=retry_initial_delay,
        retry_backoff=retry_backoff,
    )


def main():
    parser = argparse.ArgumentParser(description="Run InstructCMP prompting via OpenAI GPT models")
    parser.add_argument("--model", default="gpt-4.1", help="OpenAI model name, e.g., gpt-4.1 or gpt-4.1-mini")
    parser.add_argument("--data_name", default="DUC2004", help="Dataset folder name")
    parser.add_argument("--split", default="test", help="Dataset split suffix")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of examples (0 means all)")
    parser.add_argument("--max_output_tokens", type=int, default=150, help="Maximum tokens generated per sample")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for OpenAI model")
    parser.add_argument(
        "--system_prompt",
        default="You are a precise sentence compression assistant. Follow the instructions and respond with only the compressed sentence.",
        help="System prompt passed to the OpenAI model.",
    )
    parser.add_argument(
        "--request_interval",
        type=float,
        default=1.0,
        help="Seconds to wait between OpenAI calls to avoid rate limits.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent OpenAI requests (requires aiohttp when > 1).",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum number of attempts per prompt when contacting OpenAI.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for each OpenAI HTTP request.",
    )
    parser.add_argument(
        "--retry_initial_delay",
        type=float,
        default=1.0,
        help="Initial delay in seconds before retrying after a failure.",
    )
    parser.add_argument(
        "--retry_backoff",
        type=float,
        default=2.0,
        help="Multiplicative factor applied to the retry delay after each failure.",
    )
    parser.add_argument(
        "--output_jsonl",
        default="outputs/gpt4.1_DUC2004_test.jsonl",
        help="Where to store raw responses.",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip automatic evaluation metrics.",
    )

    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable must be set.")
    organization_id = os.getenv("OPENAI_ORGANIZATION_ID")

    dataset_file = Path("dataset") / args.data_name / f"{args.data_name}_{args.split}.jsonl"
    if not dataset_file.exists():
        raise FileNotFoundError(f"Could not locate dataset file: {dataset_file}")

    os.environ["dataset_path"] = dataset_file.as_posix()
    sources, targets = get_dataset()

    if args.limit and args.limit > 0:
        sources = sources[: args.limit]
        targets = targets[: args.limit]

    prompts = build_prompts(sources, targets)

    outputs, error_count = run_inference(
        prompts=prompts,
        model=args.model,
        max_tokens=args.max_output_tokens,
        temperature=args.temperature,
        system_prompt=args.system_prompt,
        request_interval=args.request_interval,
        api_key=api_key,
        organization_id=organization_id,
        max_retries=args.max_retries,
        timeout=args.timeout,
        retry_initial_delay=args.retry_initial_delay,
        retry_backoff=args.retry_backoff,
        concurrency=args.concurrency,
    )

    os.makedirs(Path(args.output_jsonl).parent, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for src, tgt, prompt, completion in zip(sources, targets, prompts, outputs):
            record = {
                "source": src,
                "target": tgt,
                "prompt": prompt,
                "completion": completion,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if not args.skip_eval:
        valid_sources = []
        valid_targets = []
        valid_outputs = []
        for src, tgt, completion in zip(sources, targets, outputs):
            if completion == "error":
                continue
            valid_sources.append(src)
            valid_targets.append(tgt)
            valid_outputs.append(completion)

        if valid_outputs:
            post_processed = generated_output_post_processing(valid_outputs)
            evaluate(valid_targets, valid_sources, post_processed)
        else:
            print("No successful model outputs available for evaluation.")

    print(f"Number of samples with error: {error_count}")


if __name__ == "__main__":
    main()
