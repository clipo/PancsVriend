#!/usr/bin/env python3
"""Shared utilities for logprob consistency scripts.

These helpers centralize repeated plumbing (bool parsing, progress logging,
CSV writing, endpoint/model discovery, and role-level consistency metrics)
without changing each script's domain-specific behavior.
"""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any
from urllib.parse import urlparse

import requests


def str_to_bool(value: str) -> bool:
    raw = str(value).strip().lower()
    if raw in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def emit_progress(message: str, mode: str = "timestamp") -> None:
    if mode == "timestamp":
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}", flush=True)
        return
    if mode == "progress":
        print(f"[progress] {message}", flush=True)
        return
    raise ValueError(f"Unsupported progress mode: {mode}")


def resolve_url_list_and_api_key(
    *,
    use_online_api: bool,
    llm_url: str | None,
    llm_api_key: str | None,
    online_default_url: str,
    offline_default_url: str,
    default_online_api_key: str | None,
    use_default_api_key_when_url_unspecified: bool = True,
) -> tuple[list[str], str | None]:
    if bool(use_online_api):
        urls = [str(llm_url)] if llm_url else [str(online_default_url)]
        if llm_api_key is not None:
            api_key = llm_api_key
        elif (not llm_url) and bool(use_default_api_key_when_url_unspecified):
            api_key = default_online_api_key
        else:
            api_key = None
        return urls, api_key

    urls = [str(llm_url)] if llm_url else [str(offline_default_url)]
    return urls, None


def resolve_url_and_api_key(
    *,
    use_online_api: bool,
    llm_url: str | None,
    llm_api_key: str | None,
    online_default_url: str,
    offline_default_url: str,
    default_online_api_key: str | None,
    use_default_api_key_when_url_unspecified: bool = True,
) -> tuple[str, str | None]:
    urls, api_key = resolve_url_list_and_api_key(
        use_online_api=use_online_api,
        llm_url=llm_url,
        llm_api_key=llm_api_key,
        online_default_url=online_default_url,
        offline_default_url=offline_default_url,
        default_online_api_key=default_online_api_key,
        use_default_api_key_when_url_unspecified=use_default_api_key_when_url_unspecified,
    )
    return urls[0], api_key


def role_consistency_summary(
    role_rows: list[dict[str, Any]],
    std_threshold: float,
    agreement_threshold: float,
) -> dict[str, Any]:
    move_probs = [float(row["move_probability"]) for row in role_rows]
    stay_probs = [float(row["stay_probability"]) for row in role_rows]
    label_votes = [str(row["response_label"]) for row in role_rows]

    move_std = stdev(move_probs) if len(move_probs) > 1 else 0.0
    stay_std = stdev(stay_probs) if len(stay_probs) > 1 else 0.0

    move_count = int(sum(1 for label in label_votes if label == "MOVE"))
    stay_count = int(sum(1 for label in label_votes if label == "STAY"))
    majority_count = max(move_count, stay_count)
    majority_label = "MOVE" if move_count >= stay_count else "STAY"
    agreement = majority_count / len(label_votes) if len(label_votes) > 0 else 0.0

    is_consistent = (move_std <= std_threshold) and (agreement >= agreement_threshold)

    return {
        "num_trials": len(role_rows),
        "mean_move_probability": mean(move_probs) if move_probs else math.nan,
        "mean_stay_probability": mean(stay_probs) if stay_probs else math.nan,
        "min_move_probability": min(move_probs) if move_probs else math.nan,
        "max_move_probability": max(move_probs) if move_probs else math.nan,
        "min_stay_probability": min(stay_probs) if stay_probs else math.nan,
        "max_stay_probability": max(stay_probs) if stay_probs else math.nan,
        "std_move_probability": move_std,
        "std_stay_probability": stay_std,
        "move_votes": move_count,
        "stay_votes": stay_count,
        "majority_label": majority_label,
        "majority_agreement": agreement,
        "consistent": bool(is_consistent),
    }


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if len(rows) == 0:
        with path.open("w", newline="", encoding="utf-8") as handle:
            handle.write("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_json_object_from_stdout(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        raise ValueError("Empty stdout from subprocess")
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("Could not locate JSON object in stdout")
    return dict(__import__("json").loads(text[start : end + 1]))


def auth_headers(api_key: str | None) -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def candidate_model_list_endpoints(llm_url: str) -> list[tuple[str, str]]:
    parsed = urlparse(llm_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid --llm-url: {llm_url}")

    base = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path.rstrip("/")
    prefix = path
    if path.endswith("/api/chat"):
        prefix = path[: -len("/api/chat")]
    elif path.endswith("/v1/chat/completions"):
        prefix = path[: -len("/v1/chat/completions")]
    elif path.endswith("/chat/completions"):
        prefix = path[: -len("/chat/completions")]
    elif path.endswith("/v1"):
        prefix = path[: -len("/v1")]

    candidates: list[tuple[str, str]] = []
    if prefix:
        candidates.append((f"{base}{prefix}/api/tags", "ollama_tags"))
        candidates.append((f"{base}{prefix}/v1/models", "openai_models"))
    candidates.append((f"{base}/api/tags", "ollama_tags"))
    candidates.append((f"{base}/v1/models", "openai_models"))

    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for endpoint, endpoint_type in candidates:
        if endpoint not in seen:
            seen.add(endpoint)
            unique.append((endpoint, endpoint_type))
    return unique


def discover_models_from_provider(
    llm_url: str,
    api_key: str | None,
    timeout: int,
) -> tuple[list[str], str]:
    headers = auth_headers(api_key)
    last_error = "No endpoint attempted"

    with requests.Session() as session:
        for endpoint, endpoint_type in candidate_model_list_endpoints(llm_url):
            try:
                resp = session.get(endpoint, headers=headers, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                models: list[str] = []

                if endpoint_type == "ollama_tags":
                    items = data.get("models", []) if isinstance(data, dict) else []
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict):
                                name = item.get("name")
                                if isinstance(name, str) and name.strip():
                                    models.append(name.strip())
                else:
                    items = data.get("data", []) if isinstance(data, dict) else []
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict):
                                name = item.get("id")
                                if isinstance(name, str) and name.strip():
                                    models.append(name.strip())

                models = sorted(set(models))
                if len(models) > 0:
                    return models, endpoint
                last_error = f"No models found at {endpoint}"
            except Exception as exc:  # noqa: BLE001
                last_error = f"{endpoint}: {exc}"

    raise RuntimeError(f"Failed to discover models from provider. Last error: {last_error}")


def role_delta_metrics(
    left_role: dict[str, Any],
    right_role: dict[str, Any],
    left_label_key: str,
    right_label_key: str,
) -> dict[str, Any]:
    mean_delta = abs(float(right_role.get("mean_move_probability", 0.0)) - float(left_role.get("mean_move_probability", 0.0)))
    std_delta = abs(float(right_role.get("std_move_probability", 0.0)) - float(left_role.get("std_move_probability", 0.0)))
    agreement_delta = abs(float(right_role.get("majority_agreement", 0.0)) - float(left_role.get("majority_agreement", 0.0)))

    left_majority = str(left_role.get("majority_label", ""))
    right_majority = str(right_role.get("majority_label", ""))
    labels_match = left_majority == right_majority

    return {
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "agreement_delta": agreement_delta,
        "labels_match": labels_match,
        left_label_key: left_majority,
        right_label_key: right_majority,
    }
