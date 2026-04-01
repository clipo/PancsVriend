#!/usr/bin/env python3
"""Check whether OpenAI-compatible chat-completions URLs return logprobs.

This utility sends a minimal OpenAI-style chat/completions request with:
- logprobs=True
- top_logprobs=<k>

It then verifies whether the response contains supported OpenAI-style
logprob fields under choices[0].logprobs.

Usage examples:

  python llm_utility_approximation/check_openai_logprob_support.py \
      --urls https://api.openai.com/v1/chat/completions \
      --model gpt-4o-mini

  python llm_utility_approximation/check_openai_logprob_support.py \
      --urls https://chat.binghamton.edu/api/chat/completions \
             https://chat.binghamton.edu/v1/chat/completions \
      --model mixtral:8x22b-instruct
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import requests

try:
    import config as cfg
except Exception:
    cfg = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check OpenAI-compatible URL(s) for logprob payload support"
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        required=True,
        help="One or more OpenAI-compatible /chat/completions URLs to test",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=(getattr(cfg, "OLLAMA_MODEL", None) if cfg else None),
        help="Model name to query",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=(
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("OLLAMA_API_KEY")
            or (getattr(cfg, "OLLAMA_API_KEY", None) if cfg else None)
        ),
        help="Bearer API key (defaults: OPENAI_API_KEY, OLLAMA_API_KEY, config.py)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8,
        help="Max generated tokens",
    )
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=5,
        help="Requested top_logprobs for each generated token",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Respond with exactly one word: STAY",
        help="Prompt to send",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write JSON results",
    )
    return parser.parse_args()


def _extract_openai_logprob_status(response_json: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or len(choices) == 0 or not isinstance(choices[0], dict):
        return False, "missing choices[0]", {"response_top_keys": sorted(response_json.keys())}

    choice = choices[0]
    logprobs_obj = choice.get("logprobs")
    diagnostics: dict[str, Any] = {
        "choice_keys": sorted(choice.keys()),
    }

    if not isinstance(logprobs_obj, dict):
        return False, "missing choices[0].logprobs", diagnostics

    diagnostics["logprobs_keys"] = sorted(logprobs_obj.keys())

    content_items = logprobs_obj.get("content")
    if isinstance(content_items, list) and len(content_items) > 0:
        return True, "found choices[0].logprobs.content", diagnostics

    tokens = logprobs_obj.get("tokens")
    token_logprobs = logprobs_obj.get("token_logprobs") or logprobs_obj.get("logprobs")
    if isinstance(tokens, list) and isinstance(token_logprobs, list) and len(tokens) > 0 and len(tokens) == len(token_logprobs):
        return True, "found legacy choices[0].logprobs token arrays", diagnostics

    return False, "logprobs object present but no supported token fields", diagnostics


def check_url(
    session: requests.Session,
    url: str,
    model: str,
    api_key: str | None,
    timeout: int,
    temperature: float,
    max_tokens: int,
    top_logprobs: int,
    prompt: str,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "logprobs": True,
        "top_logprobs": top_logprobs,
    }

    started = time.time()
    try:
        response = session.post(url, headers=headers, json=payload, timeout=timeout)
        elapsed = time.time() - started
    except Exception as exc:
        return {
            "url": url,
            "ok": False,
            "supports_openai_logprobs": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(elapsed if 'elapsed' in locals() else 0.0, 4),
        }

    result: dict[str, Any] = {
        "url": url,
        "http_status": response.status_code,
        "ok": bool(response.ok),
        "elapsed_seconds": round(elapsed, 4),
    }

    if not response.ok:
        result["supports_openai_logprobs"] = False
        result["error"] = f"HTTP {response.status_code}: {response.text[:500]}"
        return result

    try:
        data = response.json()
    except Exception as exc:
        result["supports_openai_logprobs"] = False
        result["error"] = f"Invalid JSON response: {type(exc).__name__}: {exc}"
        return result

    supports, reason, diagnostics = _extract_openai_logprob_status(data)
    result["supports_openai_logprobs"] = supports
    result["reason"] = reason
    result["diagnostics"] = diagnostics
    result["served_model"] = data.get("model")
    return result


def main() -> None:
    args = parse_args()

    if not args.model:
        raise ValueError("--model is required (or set OLLAMA_MODEL in config.py)")

    results: list[dict[str, Any]] = []
    with requests.Session() as session:
        for url in args.urls:
            results.append(
                check_url(
                    session=session,
                    url=url,
                    model=args.model,
                    api_key=args.api_key,
                    timeout=args.timeout,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    top_logprobs=args.top_logprobs,
                    prompt=args.prompt,
                )
            )

    passed = sum(1 for item in results if item.get("supports_openai_logprobs") is True)
    failed = len(results) - passed

    summary = {
        "model": args.model,
        "tested_urls": args.urls,
        "num_urls": len(results),
        "num_supporting_openai_logprobs": passed,
        "num_not_supporting_openai_logprobs": failed,
        "results": results,
    }

    print(json.dumps(summary, indent=2))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
