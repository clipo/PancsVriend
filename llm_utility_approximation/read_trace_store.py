#!/usr/bin/env python3
"""Inspect and export trace records from trace_store.sqlite.

This utility helps with:
- listing records in trace_records
- inspecting a specific trace payload
- exporting a payload JSON to disk
- decoding and exporting full-logits blobs (schema 3.0, payload_mode full_logits_compressed)
"""

from __future__ import annotations

import argparse
import base64
import json
import sqlite3
import zlib
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DB = (
    Path(__file__).resolve().parent.parent
    / "llm_log_probs"
    / "gemma-3-4b-it-q4_0"
    / "trace_store.sqlite"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and export traces from trace_store.sqlite")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to trace_store.sqlite")

    parser.add_argument("--list", action="store_true", help="List trace rows")
    parser.add_argument("--list-limit", type=int, default=20, help="Max rows to show in list mode")

    parser.add_argument("--trace-id", type=int, default=None, help="Trace row id to inspect")
    parser.add_argument("--scenario", type=str, default=None, help="Filter selector: scenario")
    parser.add_argument("--agent-role", type=str, default=None, help="Filter selector: agent_role")
    parser.add_argument("--sample-index", type=int, default=None, help="Filter selector: sample_index")

    parser.add_argument("--show-summary", action="store_true", help="Print compact payload summary")
    parser.add_argument("--show-payload", action="store_true", help="Print full payload JSON")

    parser.add_argument("--export-json", type=Path, default=None, help="Write payload JSON to file")
    parser.add_argument(
        "--export-logits-dir",
        type=Path,
        default=None,
        help="Export decoded full logits arrays into this folder as .npy files",
    )
    return parser.parse_args()


def _connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite file not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _list_rows(conn: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    query = """
    SELECT
      id,
      scenario,
      agent_role,
      arrangement_code,
      sample_index,
      repeat_index,
      parsed_decision,
      move_probability,
      stay_probability,
      unknown_probability,
      trace_schema_version,
      trace_payload_mode,
      capture_temperature,
      payload_compressed_bytes,
      created_at_utc
    FROM trace_records
    ORDER BY id
    LIMIT ?
    """
    return conn.execute(query, (int(max(1, limit)),)).fetchall()


def _selector_where_and_params(args: argparse.Namespace) -> tuple[str, list[Any]]:
    filters: list[str] = []
    params: list[Any] = []

    if args.trace_id is not None:
        filters.append("id = ?")
        params.append(int(args.trace_id))
    if args.scenario is not None:
        filters.append("scenario = ?")
        params.append(str(args.scenario))
    if args.agent_role is not None:
        filters.append("agent_role = ?")
        params.append(str(args.agent_role))
    if args.sample_index is not None:
        filters.append("sample_index = ?")
        params.append(int(args.sample_index))

    if not filters:
        # default to first row when no selector is provided
        return "", []

    return "WHERE " + " AND ".join(filters), params


def _fetch_one_record(conn: sqlite3.Connection, args: argparse.Namespace) -> sqlite3.Row | None:
    where_sql, params = _selector_where_and_params(args)
    query = (
        "SELECT * FROM trace_records "
        + where_sql
        + " ORDER BY id LIMIT 1"
    )
    return conn.execute(query, params).fetchone()


def _decompress_payload(blob: bytes) -> dict[str, Any]:
    text = zlib.decompress(blob).decode("utf-8")
    return json.loads(text)


def _payload_summary(payload: dict[str, Any]) -> dict[str, Any]:
    steps = payload.get("steps", [])
    total_states = 0
    states_with_full_logits = 0

    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            states = step.get("states", [])
            if not isinstance(states, list):
                continue
            total_states += len(states)
            for state in states:
                if isinstance(state, dict) and isinstance(state.get("full_logits"), dict):
                    states_with_full_logits += 1

    return {
        "schema_version": payload.get("schema_version"),
        "payload_mode": payload.get("payload_mode"),
        "capture_mode": payload.get("capture_mode"),
        "num_steps": payload.get("num_steps"),
        "total_states": total_states,
        "states_with_full_logits": states_with_full_logits,
        "final_mass": payload.get("final_mass"),
        "trace_quality_flags": payload.get("trace_quality_flags", []),
    }


def _decode_full_logits_blob(full_logits_payload: dict[str, Any]) -> np.ndarray:
    encoding = str(full_logits_payload.get("encoding", ""))
    if encoding != "zlib+base64":
        raise ValueError(f"Unsupported full_logits encoding: {encoding}")

    dtype_name = str(full_logits_payload.get("dtype", "")).lower()
    if dtype_name == "float16":
        dtype = np.float16
    elif dtype_name == "float32":
        dtype = np.float32
    else:
        raise ValueError(f"Unsupported full_logits dtype: {dtype_name}")

    b64 = full_logits_payload.get("compressed_b64", "")
    if not isinstance(b64, str) or len(b64) == 0:
        raise ValueError("Missing compressed_b64 in full_logits payload")

    compressed = base64.b64decode(b64.encode("ascii"), validate=True)
    raw = zlib.decompress(compressed)
    arr = np.frombuffer(raw, dtype=dtype)

    expected = full_logits_payload.get("vocab_size")
    if isinstance(expected, int) and expected > 0 and int(arr.shape[0]) != int(expected):
        raise ValueError(f"Decoded logits size mismatch: got {arr.shape[0]} expected {expected}")

    return arr


def _export_full_logits(payload: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    exports: list[dict[str, Any]] = []
    steps = payload.get("steps", [])
    if not isinstance(steps, list):
        return {"exported": 0, "files": []}

    for step in steps:
        if not isinstance(step, dict):
            continue
        step_index = int(step.get("step_index", -1))
        states = step.get("states", [])
        if not isinstance(states, list):
            continue

        for state_idx, state in enumerate(states):
            if not isinstance(state, dict):
                continue
            full_logits_payload = state.get("full_logits")
            if not isinstance(full_logits_payload, dict):
                continue

            arr = _decode_full_logits_blob(full_logits_payload)
            path = out_dir / f"step_{step_index:03d}_state_{state_idx:04d}.npy"
            np.save(str(path), arr)

            exports.append(
                {
                    "step_index": step_index,
                    "state_index": int(state_idx),
                    "path": str(path),
                    "dtype": str(arr.dtype),
                    "size": int(arr.shape[0]),
                    "min": float(arr.min(initial=0.0)),
                    "max": float(arr.max(initial=0.0)),
                }
            )

    manifest = {
        "exported": len(exports),
        "files": exports,
    }
    (out_dir / "logits_export_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def _print_rows(rows: list[sqlite3.Row]) -> None:
    if len(rows) == 0:
        print("No rows found.")
        return

    for row in rows:
        print(
            json.dumps(
                {
                    "id": row["id"],
                    "scenario": row["scenario"],
                    "agent_role": row["agent_role"],
                    "arrangement_code": row["arrangement_code"],
                    "sample_index": row["sample_index"],
                    "parsed_decision": row["parsed_decision"],
                    "move_probability": row["move_probability"],
                    "stay_probability": row["stay_probability"],
                    "unknown_probability": row["unknown_probability"],
                    "trace_schema_version": row["trace_schema_version"],
                    "trace_payload_mode": row["trace_payload_mode"],
                    "capture_temperature": row["capture_temperature"],
                    "payload_compressed_bytes": row["payload_compressed_bytes"],
                    "created_at_utc": row["created_at_utc"],
                },
                indent=2,
            )
        )


def main() -> None:
    args = parse_args()
    conn = _connect(args.db)

    try:
        if args.list:
            rows = _list_rows(conn, args.list_limit)
            _print_rows(rows)
            if (
                args.trace_id is None
                and args.scenario is None
                and args.agent_role is None
                and args.sample_index is None
                and not args.show_summary
                and not args.show_payload
                and args.export_json is None
                and args.export_logits_dir is None
            ):
                return

        row = _fetch_one_record(conn, args)
        if row is None:
            raise RuntimeError("No matching trace row found")

        payload_blob = row["payload_json_zlib"]
        if not isinstance(payload_blob, (bytes, bytearray)):
            raise RuntimeError("Invalid payload_json_zlib type in row")
        payload = _decompress_payload(bytes(payload_blob))

        print(
            json.dumps(
                {
                    "selected_row": {
                        "id": row["id"],
                        "scenario": row["scenario"],
                        "agent_role": row["agent_role"],
                        "sample_index": row["sample_index"],
                        "trace_schema_version": row["trace_schema_version"],
                        "trace_payload_mode": row["trace_payload_mode"],
                        "capture_temperature": row["capture_temperature"],
                        "payload_compressed_bytes": row["payload_compressed_bytes"],
                        "payload_uncompressed_bytes": row["payload_uncompressed_bytes"],
                    }
                },
                indent=2,
            )
        )

        if args.show_summary:
            print(json.dumps({"payload_summary": _payload_summary(payload)}, indent=2))

        if args.show_payload:
            print(json.dumps(payload, indent=2))

        if args.export_json is not None:
            args.export_json.parent.mkdir(parents=True, exist_ok=True)
            args.export_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(json.dumps({"export_json": str(args.export_json)}, indent=2))

        if args.export_logits_dir is not None:
            manifest = _export_full_logits(payload, args.export_logits_dir)
            print(json.dumps({"export_logits": manifest}, indent=2))

    finally:
        conn.close()


if __name__ == "__main__":
    main()
