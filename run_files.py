"""A run's on-disk record: per-step move counts and per-step grid frames.

Every run writes two files under its experiment directory:

    move_logs/step_moves_run_<id>.csv   one row per step (STEP_LOG_COLUMNS)
    states/states_run_<id>.npz          'states': frame 0 is the initial grid,
                                        frame k+1 is the grid after step k

PACKED LAYOUT (2026-09-05). Once an experiment is complete, pack_run_record()
folds those per-run files into two containers holding the same arrays and
rows and deletes the originals:

    move_logs/step_moves_packed.csv.gz  every step log, with a run_id column
    states/states_packed.npz            member run_<id> = that run's frames

A 10k-run experiment is ~20k files of a few hundred bytes each, and on a
4 KiB-block filesystem that is 84 MB on disk for 10 MB of data, plus one
seek per file to read: packed it is ~5 MB and one open. The readers below
look for a run's own files first, then the packed containers, so a
directory can hold both (runs added after packing land as per-run files
until the next pack), and every caller — resume, analysis, the format
checker — is unchanged. Per-run files stay the WRITE format: workers write
them in parallel and an aborted campaign keeps every finished run.

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
import csv
import gzip
import io
import json
import os
import zipfile

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


PACKED_STEP_LOG = "step_moves_packed.csv.gz"
PACKED_STATES = "states_packed.npz"


def packed_step_log_path(output_dir):
    return os.path.join(output_dir, "move_logs", PACKED_STEP_LOG)


def packed_states_path(output_dir):
    return os.path.join(output_dir, "states", PACKED_STATES)


# One parse of each packed container per process, keyed by path and
# invalidated by (mtime, size): the analysis tools call the per-run readers
# 10k times per experiment, and the container's directory must not be
# re-read each time.
_packed_cache = {}


def _cached(path, loader):
    try:
        stamp = (os.path.getmtime(path), os.path.getsize(path))
    except OSError:
        return None
    key = os.path.abspath(path)
    hit = _packed_cache.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    if hit is not None and hasattr(hit[1], "close"):
        hit[1].close()
    value = loader(path)
    _packed_cache[key] = (stamp, value)
    return value


def _load_packed_table(path):
    """(frame, {run_id: (start, stop)}) — the row block of every run, found
    once, so the 10k per-run lookups an analysis makes are slices rather
    than 10k scans of a million-row frame."""
    frame = pd.read_csv(path)
    if "run_id" not in frame.columns:
        return frame, {}
    frame = frame.sort_values(["run_id", "step"], kind="stable").reset_index(drop=True)
    ids = frame["run_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    stops = np.r_[starts[1:], len(ids)]
    return frame, {int(ids[a]): (int(a), int(b)) for a, b in zip(starts, stops)}


def _packed_step_logs(output_dir):
    """All packed step logs as one frame (run_id first), or None."""
    table = _packed_table(output_dir)
    return None if table is None else table[0]


def _packed_table(output_dir):
    path = packed_step_log_path(output_dir)
    if not os.path.exists(path):
        return None
    return _cached(path, _load_packed_table)


def _packed_states(output_dir):
    """The packed states container (lazy NpzFile), or None."""
    path = packed_states_path(output_dir)
    if not os.path.exists(path):
        return None
    return _cached(path, np.load)


def _packed_member(run_id):
    return f"run_{run_id}"


def new_step_row(step):
    row = {column: 0 for column in STEP_LOG_COLUMNS}
    row["step"] = int(step)
    return row


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

_LOG_SUFFIXES = (".json.gz", ".json", ".csv.gz", ".csv")


def _per_run_log_ids(output_dir):
    """{run_id: filename} for every per-run move log of any format."""
    move_dir = os.path.join(output_dir, "move_logs")
    if not os.path.isdir(move_dir):
        return {}
    ids = {}
    for name in os.listdir(move_dir):
        for prefix in ("step_moves_run_", "agent_moves_run_"):
            if not name.startswith(prefix):
                continue
            for suffix in _LOG_SUFFIXES:
                if name.endswith(suffix):
                    try:
                        ids[int(name[len(prefix):-len(suffix)])] = name
                    except ValueError:
                        pass
                    break
    return ids


def list_run_ids(output_dir):
    """Sorted run ids that have a move log of any format, per-run or packed."""
    ids = set(_per_run_log_ids(output_dir))
    packed = _packed_step_logs(output_dir)
    if packed is not None and "run_id" in packed.columns:
        ids.update(int(r) for r in packed["run_id"].unique())
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

def _packed_step_log(output_dir, run_id):
    table = _packed_table(output_dir)
    if table is None:
        return None
    frame, blocks = table
    block = blocks.get(int(run_id))
    if block is None:
        return None
    return frame.iloc[block[0]:block[1]].drop(columns=["run_id"]).reset_index(drop=True)


def has_per_step_log(output_dir, run_id):
    """Whether the run's move log is in the per-step format (own file or packed)."""
    return os.path.exists(step_log_path(output_dir, run_id)) or \
        _packed_step_log(output_dir, run_id) is not None


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
    packed = _packed_step_log(output_dir, run_id)
    if packed is not None:
        return packed.reindex(columns=list(STEP_LOG_COLUMNS), fill_value=0)
    return _aggregate_per_move(_per_move_records(output_dir, run_id))


def step_moves(step_log):
    """{step: agents moved} — the input convergence_from_step_moves() takes."""
    if step_log is None or step_log.empty:
        return {}
    return {int(s): int(m) for s, m in zip(step_log["step"], step_log["moved"])}


def load_frames(output_dir, run_id):
    """Raw 'states' array of the run's npz (own file or packed member), or None."""
    path = states_path(output_dir, run_id)
    if os.path.exists(path):
        try:
            with np.load(path) as payload:
                return payload["states"]
        except Exception as exc:
            print(f"[run_files] Warning: could not read {path} ({exc})")
            return None
    packed = _packed_states(output_dir)
    member = _packed_member(run_id)
    if packed is None or member not in packed.files:
        return None
    try:
        return packed[member]
    except Exception as exc:
        print(f"[run_files] Warning: could not read {member} from {packed_states_path(output_dir)} ({exc})")
        return None


def load_step_frames(output_dir, run_id):
    """(steps, grids): grids[i] is the int grid at the END of steps[i].

    Per-step runs: frame k+1 of the npz is the grid after step k. Full-format
    runs: the last frame belonging to each step, paired with the move log
    (record i <-> frame i; records before 2026-09-01 also carry an inline
    'grid' copy, used when the npz is absent). None when nothing is readable.
    """
    frames = load_frames(output_dir, run_id)

    if has_per_step_log(output_dir, run_id):
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


# ---------------------------------------------------------------------------
# Packing a complete experiment's per-run files into the two containers
# ---------------------------------------------------------------------------

def _read_step_log_rows(path):
    """Rows of a per-run step CSV as a list of dicts (csv module: fast)."""
    with open(path, newline="") as fh:
        return [{k: int(v) for k, v in row.items()} for row in csv.DictReader(fh)]


def pack_run_record(output_dir, verbose=True):
    """Fold every per-run, per-step record into the packed containers and
    delete the per-run files. Returns the number of runs packed.

    Full-format runs (agent_moves_run_<id>.json.gz) are left alone: their
    records carry LLM replies and are read by other tools per file. Runs
    already in the containers are kept; a run present both ways (re-run
    after packing) takes its per-run files. Both containers are written to a
    temporary name and verified member by member against what was read
    before anything is renamed or deleted, so an interrupted pack leaves the
    directory as it was.
    """
    per_run = {rid: name for rid, name in _per_run_log_ids(output_dir).items()
               if name.startswith("step_moves_run_")}
    if not per_run:
        return 0
    frames, logs = {}, {}
    for rid in sorted(per_run):
        frame = load_frames(output_dir, rid) if os.path.exists(states_path(output_dir, rid)) else None
        if frame is None:
            if verbose:
                print(f"[pack] {output_dir}: run {rid} has no states file; left as is")
            continue
        try:
            rows = _read_step_log_rows(step_log_path(output_dir, rid))
        except (OSError, ValueError) as exc:
            if verbose:
                print(f"[pack] {output_dir}: run {rid} step log unreadable ({exc}); left as is")
            continue
        if not rows:
            continue
        frames[rid], logs[rid] = np.ascontiguousarray(frame), rows
    if not frames:
        return 0

    # Merge with what is already packed (repacking after new runs).
    old_states, old_logs = _packed_states(output_dir), _packed_step_logs(output_dir)
    keep = []
    if old_logs is not None and "run_id" in old_logs.columns:
        keep = sorted(set(int(r) for r in old_logs["run_id"].unique()) - set(frames))

    states_out, logs_out = packed_states_path(output_dir), packed_step_log_path(output_dir)
    os.makedirs(os.path.dirname(states_out), exist_ok=True)
    os.makedirs(os.path.dirname(logs_out), exist_ok=True)
    tmp_states, tmp_logs = states_out + ".tmp", logs_out + ".tmp"
    try:
        # Stream the npz member by member (np.savez_compressed needs every
        # array in memory at once; a max-steps-heavy 10k-run experiment is
        # gigabytes of int8).
        with zipfile.ZipFile(tmp_states, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for rid in keep:
                with zf.open(f"{_packed_member(rid)}.npy", "w", force_zip64=True) as fh:
                    np.lib.format.write_array(fh, np.ascontiguousarray(old_states[_packed_member(rid)]))
            for rid in sorted(frames):
                with zf.open(f"{_packed_member(rid)}.npy", "w", force_zip64=True) as fh:
                    np.lib.format.write_array(fh, frames[rid])
        columns = ["run_id", *STEP_LOG_COLUMNS]
        table = pd.DataFrame(
            [{"run_id": rid, **{c: row.get(c, 0) for c in STEP_LOG_COLUMNS}}
             for rid in sorted(frames) for row in logs[rid]], columns=columns)
        if keep:
            table = pd.concat([old_logs[old_logs["run_id"].isin(keep)].reindex(columns=columns), table],
                              ignore_index=True)
        table = table.sort_values(["run_id", "step"], kind="stable").reset_index(drop=True)
        table.to_csv(tmp_logs, index=False, compression="gzip")

        # Verify before touching anything.
        with np.load(tmp_states) as check:
            for rid, frame in frames.items():
                if not np.array_equal(check[_packed_member(rid)], frame):
                    raise RuntimeError(f"packed frames for run {rid} do not match")
        back = pd.read_csv(tmp_logs, compression="gzip")
        for rid, rows in logs.items():
            got = back[back["run_id"] == rid].drop(columns=["run_id"]).reset_index(drop=True)
            want = pd.DataFrame(rows).reindex(columns=list(STEP_LOG_COLUMNS), fill_value=0)
            if not got.reindex(columns=list(STEP_LOG_COLUMNS)).astype(int).equals(want.astype(int)):
                raise RuntimeError(f"packed step log for run {rid} does not match")
    except BaseException:
        for tmp in (tmp_states, tmp_logs):
            if os.path.exists(tmp):
                os.remove(tmp)
        raise

    for key in (os.path.abspath(states_out), os.path.abspath(logs_out)):
        hit = _packed_cache.pop(key, None)
        if hit is not None and hasattr(hit[1], "close"):
            hit[1].close()
    os.replace(tmp_states, states_out)
    os.replace(tmp_logs, logs_out)
    for rid in frames:
        os.remove(states_path(output_dir, rid))
        os.remove(step_log_path(output_dir, rid))
    if verbose:
        print(f"[pack] {output_dir}: packed {len(frames)} run(s) "
              f"({len(keep)} already packed) into {PACKED_STATES} + {PACKED_STEP_LOG}")
    return len(frames)


def pack_enabled():
    """Runners pack a completed experiment unless PACK_RUN_RECORD is off."""
    return os.environ.get("PACK_RUN_RECORD", "1").strip().lower() not in ("0", "false", "no")
