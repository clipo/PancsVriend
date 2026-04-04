import importlib.util
import csv
from pathlib import Path
import sys

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_config_spec = importlib.util.spec_from_file_location("config", PROJECT_ROOT / "config.py")
assert _config_spec is not None and _config_spec.loader is not None
_config = importlib.util.module_from_spec(_config_spec)
_config_spec.loader.exec_module(_config)

OLLAMA_URL =  "https://chat.binghamton.edu/ollama/v1/models"
SHOW_URL = "https://chat.binghamton.edu/ollama/api/show"
OLLAMA_API_KEY = _config.OLLAMA_API_KEY
OUTPUT_CSV = SCRIPT_DIR / "hosted_model_defaults.csv"

FIELDS = [
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repeat_penalty",
    "frequency_penalty",
    "presence_penalty",
    "mirostat",
    "tfs_z",
    "typical_p",
    "num_ctx",
    "num_predict",
    "stop",
]


def parse_parameters(raw_text: str | None) -> dict[str, str]:
    if not raw_text:
        return {}

    parsed: dict[str, str] = {}
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split(maxsplit=1)
        if len(parts) == 1:
            key, value = parts[0], ""
        else:
            key, value = parts[0], parts[1].strip()

        # Keep all stop tokens if there are multiple lines.
        if key in parsed and key == "stop":
            parsed[key] = f"{parsed[key]} | {value}" if value else parsed[key]
        else:
            parsed[key] = value

    return parsed


def fetch_model_ids() -> list[str]:
    response = requests.get(
        OLLAMA_URL,
        headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
        timeout=30,
    )
    response.raise_for_status()
    return [m["id"] for m in response.json().get("data", [])]


def fetch_model_parameters(model_id: str) -> dict[str, str]:
    response = requests.post(
        SHOW_URL,
        headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
        json={"model": model_id},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return parse_parameters(payload.get("parameters"))


def main() -> None:
    model_ids = fetch_model_ids()
    print(f"Found {len(model_ids)} models")

    rows: list[dict[str, str]] = []
    for i, model_id in enumerate(model_ids, start=1):
        print(f"[{i}/{len(model_ids)}] {model_id}")
        row = {"model": model_id}

        try:
            params = fetch_model_parameters(model_id)
            for field in FIELDS:
                row[field] = params.get(field, "")
            row["error"] = ""
        except requests.RequestException as exc:
            for field in FIELDS:
                row[field] = ""
            row["error"] = str(exc)

        rows.append(row)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", *FIELDS, "error"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()