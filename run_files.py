"""A run's on-disk record: per-step move counts and per-step grid frames.

Every run writes two files under its experiment directory:

    move_logs/step_moves_run_<id>.csv   one row per step (STEP_LOG_COLUMNS)
    states/states_run_<id>.npz          'states': frame 0 is the initial grid,
                                        frame k+1 is the grid after step k

That is the whole record for value-function and mechanical runs. Runs are
seeded by run_id and deterministic, so anything finer (which agent moved
where, in what order) is regenerated exactly by re-running with
FULL_MOVE_LOG=1 / --full-move-log, which switches to the FULL format:

    move_logs/agent_moves_run_<id>.json.gz   one record per agent decision
    states/states_run_<id>.npz               one frame per record

Live-LLM runs always use the full format, because each record carries the
agent's raw LLM reply and nothing per-step could hold that. The full format is
also what every run before 2026-09-02 wrote (12k of them on disk), so the
readers here accept both and hand back the same per-step tables either way:
callers never see a per-move record unless they ask for one via
load_move_log_json().
"""

from __future__ import annotations

import ast
import gzip
import json
import os

import numpy as np
import pandas as pd

# Outcomes of an agent decision, as named in Simulation.update_agents().
REASONS = ("successful_move", "target_occupied", "invalid_target",
           "chose_to_stay", "same_position")

# `decisions` = agents that decided this step, `moved` = how many relocated,
# `parse_failed` = LLM replies that did not parse (0 for value-function runs).
STEP_LOG_COLUMNS = ("step", "decisions", "moved", "parse_failed") + REASONS


def step_log_path(output_dir, run_id):
    return os.path.join(output_dir, "move_logs", f"step_moves_run_{run_id}.csv")


def move_log_path(output_dir, run_id):
    return os.path.join(output_dir, "move_logs", f"agent_moves_run_{run_id}.json.gz")


def states_path(output_dir, run_id):
    return os.path.join(output_dir, "states", f"states_run_{run_id}.npz")


def new_step_row(step):
    row = {column: 0 for column in STEP_LOG_COLUMNS}
    row["step"] = int(step)
    return row


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

_LOG_SUFFIXES = (".json.gz", ".json", ".csv.gz", ".csv")


def list_run_ids(output_dir):
    """Sorted run ids that have a move log of any format."""
    move_dir = os.path.join(output_dir, "move_logs")
    if not os.path.isdir(move_dir):
        return []
    ids = set()
    for name in os.listdir(move_dir):
        for prefix in ("step_moves_run_", "agent_moves_run_"):
            if not name.startswith(prefix):
                continue
            for suffix in _LOG_SUFFIXES:
                if name.endswith(suffix):
                    try:
                        ids.add(int(name[len(prefix):-len(suffix)]))
                    except ValueError:
                        pass
                    break
    return sorted(ids)


# ---------------------------------------------------------------------------
# Full-format (per-move) readers — legacy runs and --full-move-log runs
# ---------------------------------------------------------------------------

def load_move_log_json(output_dir, run_id):
    """Per-move records (list of dicts) or None when there is no JSON log."""
    for path in (move_log_path(output_dir, run_id),
                 os.path.join(output_dir, "move_logs", f"agent_moves_run_{run_id}.json")):
        if not os.path.exists(path):
            continue
        opener = gzip.open if path.endswith(".gz") else open
        try:
            with opener(path, "rt", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            print(f"[run_files] Warning: could not read {path} ({exc})")
            return None
    return None


def load_move_log_csv(output_dir, run_id):
    """Per-move records as a DataFrame from the oldest (CSV) log format."""
    for suffix in (".csv.gz", ".csv"):
        path = os.path.join(output_dir, "move_logs", f"agent_moves_run_{run_id}{suffix}")
        if not os.path.exists(path):
            continue
        try:
            return pd.read_csv(path, compression="infer")
        except Exception as exc:
            print(f"[run_files] Warning: could not read {path} ({exc})")
            return None
    return None


def _per_move_records(output_dir, run_id):
    """Per-move records as a DataFrame from whichever full-format log exists."""
    records = load_move_log_json(output_dir, run_id)
    if records is not None:
        return pd.DataFrame(records)
    return load_move_log_csv(output_dir, run_id)


def _truthy(series):
    """`moved` as recorded by JSON (bool) or CSV (bool/int/str) logs."""
    if series.dtype == object:
        return series.map(lambda v: str(v).strip().lower() in ("true", "1", "yes"))
    return series.fillna(0).astype(bool)


def _aggregate_per_move(df):
    """Per-step table from per-move records; None if there are no steps."""
    if df is None or df.empty or "step" not in df.columns:
        return None
    df = df.copy()
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df.dropna(subset=["step"])
    reason = df["reason"].astype(str) if "reason" in df.columns else pd.Series("", index=df.index)
    df = df[reason != "initial_state"]      # the bookkeeping record, not a decision
    if df.empty:
        return None
    reason = reason.loc[df.index]
    step = df["step"].astype(int)
    moved = _truthy(df["moved"]) if "moved" in df.columns else pd.Series(False, index=df.index)
    if "llm_parse_status" in df.columns:
        status = df["llm_parse_status"]
        parse_failed = status.notna() & (status.astype(str) != "OK")
    else:
        parse_failed = pd.Series(False, index=df.index)

    table = pd.DataFrame({"step": step, "moved": moved.astype(int),
                          "parse_failed": parse_failed.astype(int)})
    for name in REASONS:
        table[name] = (reason == name).astype(int)
    table["decisions"] = 1
    out = table.groupby("step", sort=True).sum().reset_index()
    return out[list(STEP_LOG_COLUMNS)]


# ---------------------------------------------------------------------------
# Per-step readers — the interface everything downstream uses
# ---------------------------------------------------------------------------

def load_step_log(output_dir, run_id):
    """One row per step (STEP_LOG_COLUMNS), from whichever format the run has.

    None when the run has no usable move log.
    """
    path = step_log_path(output_dir, run_id)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"[run_files] Warning: could not read {path} ({exc})")
            return None
        return df.reindex(columns=list(STEP_LOG_COLUMNS), fill_value=0)
    return _aggregate_per_move(_per_move_records(output_dir, run_id))


def step_moves(step_log):
    """{step: agents moved} — the input convergence_from_step_moves() takes."""
    if step_log is None or step_log.empty:
        return {}
    return {int(s): int(m) for s, m in zip(step_log["step"], step_log["moved"])}


def load_frames(output_dir, run_id):
    """Raw 'states' array of the run's npz, or None."""
    path = states_path(output_dir, run_id)
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as payload:
            return payload["states"]
    except Exception as exc:
        print(f"[run_files] Warning: could not read {path} ({exc})")
        return None


def load_step_frames(output_dir, run_id):
    """(steps, grids): grids[i] is the int grid at the END of steps[i].

    Per-step runs: frame k+1 of the npz is the grid after step k. Full-format
    runs: the last frame belonging to each step, paired with the move log
    (record i <-> frame i; records before 2026-09-01 also carry an inline
    'grid' copy, used when the npz is absent). None when nothing is readable.
    """
    frames = load_frames(output_dir, run_id)

    if os.path.exists(step_log_path(output_dir, run_id)):
        step_log = load_step_log(output_dir, run_id)
        if frames is None or step_log is None or step_log.empty:
            return None
        steps = [int(s) for s in step_log["step"]]
        first = steps[0]
        # frame 0 is the initial grid (the grid at the start of the first
        # logged step, which is not 0 for a resumed run).
        grids = [frames[s - first + 1] for s in steps if s - first + 1 < len(frames)]
        return steps[:len(grids)], grids

    records = load_move_log_json(output_dir, run_id)
    if records is None:
        return None
    last_index, inline = {}, {}
    for index, record in enumerate(records):
        try:
            step = int(record["step"])
        except (KeyError, TypeError, ValueError):
            continue
        last_index[step] = index
        if "grid" in record:
            inline[step] = record["grid"]
    steps, grids = [], []
    for step in sorted(last_index):
        grid = None
        if frames is not None and last_index[step] < len(frames):
            grid = frames[last_index[step]]
        elif step in inline:
            grid = inline[step]
            if isinstance(grid, str):
                grid = ast.literal_eval(grid)
            grid = np.asarray(grid)
        if grid is not None:
            steps.append(step)
            grids.append(grid)
    return (steps, grids) if steps else None


def metrics_history_path(output_dir):
    """metrics_history.csv.gz (written since 2026-09-03; 3x smaller, pandas
    reads it transparently) or the older uncompressed metrics_history.csv.
    Returns the .gz path when neither exists, so callers can write to it."""
    csv = os.path.join(str(output_dir), "metrics_history.csv")
    gz = csv + ".gz"
    return csv if os.path.exists(csv) and not os.path.exists(gz) else gz


def load_final_grid(output_dir, run_id):
    """The int grid as the run ended, or None."""
    loaded = load_step_frames(output_dir, run_id)
    if loaded is None:
        return None
    return loaded[1][-1]
