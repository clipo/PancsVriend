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

# The seven per-step metrics Metrics.calculate_all_metrics writes to
# metrics_history, in its order. A stored history is only complete when every
# row carries all of them (runs before 2026-08-25 lack dissimilarity_index).
METRIC_COLUMNS = ("clusters", "switch_rate", "distance", "mix_deviation",
                  "share", "ghetto_rate", "dissimilarity_index")


def step_log_path(output_dir, run_id):
    return os.path.join(output_dir, "move_logs", f"step_moves_run_{run_id}.csv")


def move_log_path(output_dir, run_id):
    return os.path.join(output_dir, "move_logs", f"agent_moves_run_{run_id}.json.gz")


def states_path(output_dir, run_id):
    return os.path.join(output_dir, "states", f"states_run_{run_id}.npz")


# Parquet since 2026-11-23 (speedup review item 5): a binary column table,
# written ~85x and read ~40x faster than the gzipped CSV, 2.8x smaller, and
# readable a few columns or one run at a time. Directories written before
# keep their CSV: every reader below accepts both, and a writer keeps the
# format the directory already has, so an old experiment is never converted.
PACKED_STEP_LOG = "step_moves_packed.parquet"
PACKED_STEP_LOG_CSV = "step_moves_packed.csv.gz"           # before 2026-11-23
PACKED_STATES = "states_packed.npz"
METRICS_HISTORY = "metrics_history.parquet"
# Rows per Parquet row group (whole runs only, so a run never spans two).
PARQUET_ROW_GROUP_ROWS = 1_000_000


def _legacy_folder(output_dir):
    """Whether the experiment folder was written before 2026-11-23: it holds a
    CSV metrics_history or CSV packed step log. Such a folder keeps CSV for
    any container it does not have yet, so it never mixes formats."""
    out = str(output_dir)
    return any(os.path.exists(os.path.join(out, *parts)) for parts in (
        ("metrics_history.csv.gz",), ("metrics_history.csv",),
        ("move_logs", PACKED_STEP_LOG_CSV)))


def packed_step_log_path(output_dir):
    """The packed step log that exists (Parquet, else the older csv.gz), or
    the path to write when there is none: Parquet, unless the folder was
    written before 2026-11-23 (then csv.gz, see _legacy_folder)."""
    move_dir = os.path.join(output_dir, "move_logs")
    parquet = os.path.join(move_dir, PACKED_STEP_LOG)
    legacy = os.path.join(move_dir, PACKED_STEP_LOG_CSV)
    if os.path.exists(parquet):
        return parquet
    if os.path.exists(legacy) or _legacy_folder(output_dir):
        return legacy
    return parquet


def is_parquet(path):
    return str(path).endswith(".parquet")


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
    # One entry per (file, loader): a Parquet step log is opened for its
    # footer, its run ids and (legacy callers) its whole table separately.
    key = (os.path.abspath(path), loader.__name__)
    hit = _packed_cache.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    if hit is not None:
        stale = hit[1][0] if isinstance(hit[1], tuple) and hit[1] else hit[1]
        if hasattr(stale, "close"):
            stale.close()
    value = loader(path)
    _packed_cache[key] = (stamp, value)
    return value


def _load_packed_table(path):
    """(frame, {run_id: (start, stop)}) — the row block of every run, found
    once, so the 10k per-run lookups an analysis makes are slices rather
    than 10k scans of a million-row frame."""
    frame = pd.read_parquet(path) if is_parquet(path) else pd.read_csv(path)
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


def _open_packed_parquet(path):
    """(ParquetFile, [(first run_id, last run_id) per row group]) — row groups
    hold whole runs in run order, so a run is found from the footer alone."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    col = pf.schema_arrow.get_field_index("run_id")
    ranges = []
    for i in range(pf.metadata.num_row_groups):
        st = pf.metadata.row_group(i).column(col).statistics
        ranges.append((int(st.min), int(st.max)) if st is not None and st.has_min_max
                      else (None, None))
    return pf, ranges


_row_group_cache = {}      # path -> (stamp, row group index, frame, blocks)


def _packed_parquet_run(path, run_id):
    """One run's rows from a Parquet packed step log, decoding only the row
    group that holds it (the last one decoded is kept: per-run readers mostly
    walk runs in order). No process ever holds the whole table."""
    opened = _cached(path, _open_packed_parquet)
    if opened is None:
        return None
    pf, ranges = opened
    rid = int(run_id)
    stamp = (os.path.getmtime(path), os.path.getsize(path))
    for i, (lo, hi) in enumerate(ranges):
        if lo is not None and not (lo <= rid <= hi):
            continue
        hit = _row_group_cache.get(path)
        if hit is not None and hit[0] == stamp and hit[1] == i:
            frame, blocks = hit[2], hit[3]
        else:
            frame = pf.read_row_group(i).to_pandas()
            ids = frame["run_id"].to_numpy()
            starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
            stops = np.r_[starts[1:], len(ids)]
            blocks = {int(ids[a]): (int(a), int(b)) for a, b in zip(starts, stops)}
            _row_group_cache[path] = (stamp, i, frame, blocks)
        block = blocks.get(rid)
        if block is not None:
            return frame.iloc[block[0]:block[1]]
    return None


def _read_packed_run_id_column(path):
    return pd.read_parquet(path, columns=["run_id"])["run_id"].unique()


def _drop_cached(path):
    """Forget every cached view of `path` (after it is replaced)."""
    absolute = os.path.abspath(path)
    for key in [k for k in _packed_cache if k[0] == absolute]:
        hit = _packed_cache.pop(key)
        if hasattr(hit[1], "close"):
            hit[1].close()
        elif isinstance(hit[1], tuple) and hit[1] and hasattr(hit[1][0], "close"):
            hit[1][0].close()
    _row_group_cache.pop(path, None)
    _row_group_cache.pop(absolute, None)


def _packed_run_ids(output_dir):
    """Run ids in the packed step log (only the run_id column is read)."""
    path = packed_step_log_path(output_dir)
    if not os.path.exists(path):
        return None
    if is_parquet(path):
        ids = _cached(path, _read_packed_run_id_column)
        return None if ids is None else [int(r) for r in ids]
    packed = _packed_step_logs(output_dir)
    if packed is None or "run_id" not in packed.columns:
        return None
    return [int(r) for r in packed["run_id"].unique()]


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
    ids.update(_packed_run_ids(output_dir) or [])
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
    path = packed_step_log_path(output_dir)
    if is_parquet(path):
        if not os.path.exists(path):
            return None
        rows = _packed_parquet_run(path, run_id)
        return None if rows is None else rows.drop(columns=["run_id"]).reset_index(drop=True)
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
    """The experiment's stored metrics_history: metrics_history.parquet
    (written since 2026-11-23), else the older metrics_history.csv.gz
    (2026-09-03) or plain metrics_history.csv. Returns the Parquet path when
    none exists, so writers create Parquet; a directory that already has a
    CSV keeps it (writers follow the suffix of this path)."""
    out = str(output_dir)
    parquet = os.path.join(out, METRICS_HISTORY)
    csv = os.path.join(out, "metrics_history.csv")
    gz = csv + ".gz"
    if os.path.exists(parquet):
        return parquet
    if os.path.exists(gz):
        return gz
    if os.path.exists(csv):
        return csv
    return gz if _legacy_folder(output_dir) else parquet


def read_metrics_history(output_dir, columns=None, float_precision=None):
    """metrics_history as a DataFrame, or None when there is none.

    `columns` reads only those columns (the ones present; Parquet reads only
    their bytes), returned in the file's column order as a CSV read with
    usecols would. `float_precision` applies to a CSV only (the caller's
    existing parser setting). Parquet gives the exact stored float64 values
    whatever it says -- the same as a 'round_trip' CSV read; pandas' default
    CSV parser could be an ulp off (user decision 2026-11-24: exact values).
    """
    path = metrics_history_path(output_dir)
    if not os.path.exists(path):
        return None
    if is_parquet(path):
        return _read_metrics_parquet(path, columns)
    if columns is None:
        return pd.read_csv(path, float_precision=float_precision)
    header = pd.read_csv(path, nrows=0).columns
    return pd.read_csv(path, usecols=[c for c in header if c in set(columns)],
                       float_precision=float_precision)


def _read_metrics_parquet(path, columns=None):
    """Exact stored values; `columns` in the file's column order."""
    if columns is None:
        return pd.read_parquet(path)
    import pyarrow.parquet as pq
    names = pq.read_schema(path).names
    return pd.read_parquet(path, columns=[c for c in names if c in set(columns)])


_PARQUET_UNSUPPORTED = ("chunksize", "iterator", "skiprows", "converters", "na_values",
                        "parse_dates", "index_col", "header", "names")


def read_metrics_file(path, **csv_kwargs):
    """Read a metrics_history file named by metrics_history_path(): Parquet by
    suffix, otherwise exactly pd.read_csv(path, **csv_kwargs) as callers did
    before 2026-11-23 (so a stored CSV is parsed the way it always was).

    For Parquet the pd.read_csv arguments in use here are honoured: usecols
    (columns in file order), dtype (applied after reading) and nrows.
    float_precision has no effect: Parquet holds the exact float64 values (as
    a 'round_trip' CSV read gives them; user decision 2026-11-24). Anything
    else raises rather than being silently ignored.
    """
    if not is_parquet(path):
        return pd.read_csv(path, **csv_kwargs)
    bad = [k for k in csv_kwargs if k in _PARQUET_UNSUPPORTED]
    if bad:
        raise NotImplementedError(f"read_metrics_file: {bad} not supported for Parquet ({path})")
    usecols = csv_kwargs.get("usecols")
    frame = _read_metrics_parquet(path, list(usecols) if usecols is not None else None)
    nrows = csv_kwargs.get("nrows")
    if nrows is not None:
        frame = frame.head(nrows)
    dtype = csv_kwargs.get("dtype")
    if dtype is not None:
        frame = frame.astype({c: t for c, t in dtype.items() if c in frame.columns}
                             if isinstance(dtype, dict) else dtype)
    return frame


def write_metrics_history(frame, path):
    """Write a whole metrics_history frame in the format `path` names."""
    if is_parquet(path):
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def load_final_grid(output_dir, run_id):
    """The int grid as the run ended, or None."""
    loaded = load_step_frames(output_dir, run_id)
    if loaded is None:
        return None
    return loaded[1][-1]


# ---------------------------------------------------------------------------
# Completeness of the stored metrics_history against the run record
# ---------------------------------------------------------------------------

def step_log_extent(output_dir, run_id):
    """(n_steps, first_step, last_step) of the run's step log, or None.

    Reads the per-step CSV with the csv module rather than pandas: the
    completeness check below calls this once per run, and pandas' fixed
    per-file cost made 10k calls take ~20 s against ~1 s this way. Full-format
    runs go through load_step_log (they are the legacy ones, and rare).
    """
    path = step_log_path(output_dir, run_id)
    if os.path.exists(path):
        try:
            with open(path, newline="") as fh:
                reader = csv.reader(fh)
                col = next(reader).index("step")
                steps = [int(row[col]) for row in reader if row]
        except (OSError, ValueError, StopIteration, IndexError) as exc:
            print(f"[run_files] Warning: could not read {path} ({exc})")
            return None
        if not steps:
            return None
        return len(steps), min(steps), max(steps)
    packed = _packed_step_log(output_dir, run_id)
    log = packed if packed is not None else load_step_log(output_dir, run_id)
    if log is None or log.empty:
        return None
    steps = log["step"].astype(int)
    return int(len(steps)), int(steps.min()), int(steps.max())


def stale_metrics_runs(output_dir, run_ids=None, metrics=None):
    """Run ids whose stored metrics_history rows do not match their step log.

    None when there is no usable metrics_history at all (no file, or missing
    run_id/step/METRIC_COLUMNS), meaning every run needs rebuilding; an empty
    list when every run is complete. A run is complete when the file holds
    exactly one row per logged step — same count, same first and last step —
    with every metric present. Count and first step matter, not just the last:
    a live-LLM run resumed after an abort restarts with an empty in-memory
    history, so its stored rows begin at the resume step while its step log
    (kept by preload_record) begins at 0; only the front-truncated history
    gives it away (2026-09-05, seen in 20+ pre-2026-09 experiment dirs).

    `run_ids` limits the check (default: every run with a move log);
    `metrics` is an already-loaded metrics_history frame, to spare a re-read.
    """
    if metrics is None:
        path = metrics_history_path(output_dir)
        if not os.path.exists(path):
            return None
        try:
            metrics = read_metrics_history(output_dir)
        except Exception as exc:
            print(f"[run_files] Warning: could not read {path} ({exc})")
            return None
    needed = {"run_id", "step", *METRIC_COLUMNS}
    if metrics is None or not needed <= set(metrics.columns):
        return None
    run_ids = list_run_ids(output_dir) if run_ids is None else sorted(run_ids)
    by_run = metrics.groupby("run_id")
    stored = pd.DataFrame({
        "n": by_run.size(),
        "lo": by_run["step"].min(),
        "hi": by_run["step"].max(),
        "clean": metrics[list(METRIC_COLUMNS)].notna().all(axis=1)
                 .groupby(metrics["run_id"]).all(),
    })
    stale = []
    for run_id in run_ids:
        if run_id not in stored.index:
            stale.append(run_id)
            continue
        extent = step_log_extent(output_dir, run_id)
        if extent is None:        # no record to rebuild from; keep the rows
            continue
        row = stored.loc[run_id]
        if not bool(row["clean"]) or \
                (int(row["n"]), int(row["lo"]), int(row["hi"])) != extent:
            stale.append(run_id)
    return stale


# ---------------------------------------------------------------------------
# Packing a complete experiment's per-run files into the two containers
# ---------------------------------------------------------------------------

def _read_step_log_rows(path):
    """Rows of a per-run step CSV as a list of dicts (csv module: fast)."""
    with open(path, newline="") as fh:
        return [{k: int(v) for k, v in row.items()} for row in csv.DictReader(fh)]


def _parquet_batches(path, batch_size=1_000_000):
    """A Parquet file as DataFrames of at most batch_size rows, in order."""
    import pyarrow.parquet as pq
    with pq.ParquetFile(path) as pf:
        for batch in pf.iter_batches(batch_size=batch_size):
            yield batch.to_pandas()


def _write_parquet_parts(parts, path, row_group_rows=None):
    """Write DataFrames (same columns and dtypes, whole runs each) to one
    Parquet file, cutting a row group only between parts, so a run never
    spans two row groups and the reader can find it from the footer."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    limit = row_group_rows or PARQUET_ROW_GROUP_ROWS
    writer, buffer, n = None, [], 0

    def flush():
        nonlocal writer, buffer, n
        if not buffer:
            return
        table = pa.Table.from_pandas(pd.concat(buffer, ignore_index=True), preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema)
        writer.write_table(table, row_group_size=len(table))
        buffer, n = [], 0

    try:
        for part in parts:
            buffer.append(part)
            n += len(part)
            if n >= limit:
                flush()
        flush()
    finally:
        if writer is not None:
            writer.close()


def _step_log_frame(rows, rid, columns):
    """One run's step-log rows as the packed table's DataFrame (int64 columns,
    run_id first, missing columns 0), in stable step order — the slice the
    one-shot table held for this run before 2026-11-23."""
    table = pd.DataFrame(
        [{"run_id": rid, **{c: row.get(c, 0) for c in STEP_LOG_COLUMNS}} for row in rows],
        columns=columns)
    order = np.argsort(table["step"].to_numpy(), kind="stable")
    return table.iloc[order]


def pack_run_record(output_dir, verbose=True):
    """Fold every per-run, per-step record into the packed containers and
    delete the per-run files. Returns the number of runs packed.

    Full-format runs (agent_moves_run_<id>.json.gz) are left alone: their
    records carry LLM replies and are read by other tools per file. Runs
    already in the containers are kept; a run present both ways (re-run
    after packing) takes its per-run files. Both containers are written to a
    temporary name and verified member by member against the per-run files
    before anything is renamed or deleted, so an interrupted pack leaves the
    directory as it was.

    ONE RUN AT A TIME (2026-11-23). This used to load every run's frames and
    step rows (as dicts) before writing: at 10,000 runs x 10,000 steps that is
    ~40 GB of frames plus ~86 GB of row dicts. Now each run is read, written
    and released, and verification re-reads the per-run files one at a time;
    the containers hold the same arrays and the same table text as before.
    """
    per_run = {rid: name for rid, name in _per_run_log_ids(output_dir).items()
               if name.startswith("step_moves_run_")}
    if not per_run:
        return 0

    # Merge with what is already packed (repacking after new runs).
    old_states, old_logs = _packed_states(output_dir), _packed_step_logs(output_dir)
    old_ids = set()
    if old_logs is not None and "run_id" in old_logs.columns:
        old_ids = set(int(r) for r in old_logs["run_id"].unique())

    def read_rows(rid):
        """The run's rows, or None when it is not packable (same rules as
        before: no or unreadable states file, unreadable or empty step log)."""
        frame = load_frames(output_dir, rid) if os.path.exists(states_path(output_dir, rid)) else None
        if frame is None:
            if verbose:
                print(f"[pack] {output_dir}: run {rid} has no states file; left as is")
            return None, None
        try:
            rows = _read_step_log_rows(step_log_path(output_dir, rid))
        except (OSError, ValueError) as exc:
            if verbose:
                print(f"[pack] {output_dir}: run {rid} step log unreadable ({exc}); left as is")
            return None, None
        if not rows:
            return None, None
        return np.ascontiguousarray(frame), rows

    states_out, logs_out = packed_states_path(output_dir), packed_step_log_path(output_dir)
    os.makedirs(os.path.dirname(states_out), exist_ok=True)
    os.makedirs(os.path.dirname(logs_out), exist_ok=True)
    tmp_states, tmp_logs = states_out + ".tmp", logs_out + ".tmp"
    columns = ["run_id", *STEP_LOG_COLUMNS]
    packed = []
    try:
        # Pass 1: frames. Kept members first, then the new runs in run order
        # (the member order the one-shot writer used). Runs that turn out not
        # to be packable are skipped before anything is written for them.
        keep = None
        with zipfile.ZipFile(tmp_states, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            candidates = []
            for rid in sorted(per_run):
                frame, rows = read_rows(rid)
                if frame is None:
                    continue
                candidates.append(rid)
                del frame, rows
            if not candidates:
                pass
            else:
                keep = sorted(old_ids - set(candidates))
                for rid in keep:
                    with zf.open(f"{_packed_member(rid)}.npy", "w", force_zip64=True) as fh:
                        np.lib.format.write_array(fh, np.ascontiguousarray(old_states[_packed_member(rid)]))
                for rid in candidates:
                    frame, _ = read_rows(rid)
                    with zf.open(f"{_packed_member(rid)}.npy", "w", force_zip64=True) as fh:
                        np.lib.format.write_array(fh, frame)
                    packed.append(rid)
                    del frame
        if not packed:
            os.remove(tmp_states)
            return 0

        # Pass 2: step-log table in (run_id, step) order. The one-shot table
        # was int64 throughout unless kept rows brought a float column (a
        # column missing from an older pack reads as NaN); concat then made
        # that column float64 for every row, so the same is done here.
        dtypes = {c: np.int64 for c in columns}
        old_keep = keep_rows = None
        if keep:
            old_keep = old_logs[old_logs["run_id"].isin(keep)].reindex(columns=columns)
            for c in columns:
                if not pd.api.types.is_integer_dtype(old_keep[c].dtype):
                    dtypes[c] = old_keep[c].dtype
            keep_rows = old_keep.groupby("run_id", sort=False).indices   # one pass
        new_set = set(packed)

        def parts():
            for rid in sorted(set(packed) | set(keep or [])):
                if rid in new_set:
                    _, rows = read_rows(rid)
                    part = _step_log_frame(rows, rid, columns)
                else:
                    part = old_keep.iloc[keep_rows[rid]]
                    part = part.iloc[np.argsort(part["step"].to_numpy(), kind="stable")]
                yield part.astype(dtypes)

        if is_parquet(logs_out):
            _write_parquet_parts(parts(), tmp_logs)
        else:                                   # a directory packed before 2026-11-23
            with gzip.open(tmp_logs, "wt", compresslevel=9, newline="", encoding="utf-8") as fh:
                first = True
                for part in parts():
                    fh.write(part.to_csv(index=False, header=first))
                    first = False

        # Verify before touching anything, one run at a time.
        with np.load(tmp_states) as check:
            for rid in packed:
                frame, _ = read_rows(rid)
                if not np.array_equal(check[_packed_member(rid)], frame):
                    raise RuntimeError(f"packed frames for run {rid} do not match")
        seen = set()

        def verify(rid, got):
            if rid not in new_set:
                return
            _, rows = read_rows(rid)
            want = pd.DataFrame(rows).reindex(columns=list(STEP_LOG_COLUMNS), fill_value=0)
            got = got.drop(columns=["run_id"]).reset_index(drop=True)
            if not got.reindex(columns=list(STEP_LOG_COLUMNS)).astype(int).equals(want.astype(int)):
                raise RuntimeError(f"packed step log for run {rid} does not match")
            seen.add(rid)

        pending = None
        if is_parquet(logs_out):
            chunk_iter = _parquet_batches(tmp_logs)
        else:
            chunk_iter = pd.read_csv(tmp_logs, compression="gzip", chunksize=1_000_000)
        for chunk in chunk_iter:
            if pending is not None:
                chunk = pd.concat([pending, chunk], ignore_index=True)
            ids = chunk["run_id"].to_numpy()
            starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
            stops = np.r_[starts[1:], len(ids)]
            for a, b in zip(starts[:-1], stops[:-1]):       # complete runs
                verify(int(ids[a]), chunk.iloc[a:b])
            pending = chunk.iloc[starts[-1]:].reset_index(drop=True)
        if pending is not None and len(pending):
            verify(int(pending["run_id"].iloc[0]), pending)
        if seen != new_set:
            missing = sorted(new_set - seen)[:5]
            raise RuntimeError(f"packed step log lacks run(s) {missing}")
    except BaseException:
        for tmp in (tmp_states, tmp_logs):
            if os.path.exists(tmp):
                os.remove(tmp)
        raise

    for path in (states_out, logs_out):
        _drop_cached(path)
    os.replace(tmp_states, states_out)
    os.replace(tmp_logs, logs_out)
    for rid in packed:
        os.remove(states_path(output_dir, rid))
        os.remove(step_log_path(output_dir, rid))
    if verbose:
        print(f"[pack] {output_dir}: packed {len(packed)} run(s) "
              f"({len(keep or [])} already packed) into {PACKED_STATES} + "
              f"{os.path.basename(logs_out)}")
    return len(packed)


# ---------------------------------------------------------------------------
# Per-run metric rows, staged by the worker (2026-11-23)
# ---------------------------------------------------------------------------
# A worker used to return its whole per-step metric history (one dict per
# step) and the parent kept every run's history until analyze_results built
# metrics_history.csv.gz in one DataFrame: 432 bytes per row, ~43 GB for a
# 10,000-run x 10,000-step scenario whose runs hit the cap. Now the worker
# saves its rows here as one array per column and returns only the final row;
# analyze_results writes the CSV one run at a time and removes the stage.
#
# Exactness. pandas infers a column's dtype from ALL rows it is given: a
# column of Python ints is int64 and prints "3"; ints mixed with floats (the
# metrics return a plain int 0 in degenerate cases, a float otherwise) are
# float64 and print "0.0". So each run records, per column, whether every
# value was an int; the writer types a column int64 only if it was all-int in
# EVERY run, which is exactly what the one-shot DataFrame did.

METRICS_STAGE_DIR = "metrics_runs"


def metrics_stage_path(output_dir, run_id):
    return os.path.join(output_dir, METRICS_STAGE_DIR, f"metrics_run_{run_id}.npz")


def _is_int_value(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def stage_metrics(output_dir, run_id, history):
    """Save a run's per-step metric dicts as column arrays.

    Returns (path, columns, all_int): `columns` is the key order of the rows
    (every row must have the same keys, which calculate_all_metrics + step +
    run_id guarantee), `all_int[i]` whether every value of column i was an int.
    """
    columns = list(history[0].keys())
    for row in history:
        if list(row.keys()) != columns:
            raise ValueError(f"run {run_id}: metric rows do not share one key order")
    arrays, all_int = {}, []
    for k, col in enumerate(columns):
        values = [row[col] for row in history]
        whole = all(_is_int_value(v) for v in values)
        all_int.append(whole)
        if whole:
            arrays[f"c{k}"] = np.asarray(values, dtype=np.int64)
        else:
            arrays[f"c{k}"] = np.asarray([np.nan if v is None else v for v in values],
                                         dtype=np.float64)
    path = metrics_stage_path(output_dir, run_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, columns=np.asarray(columns), all_int=np.asarray(all_int), **arrays)
    os.replace(tmp, path)
    return path, columns, all_int


def load_staged_metrics(path):
    """(columns, [array per column], all_int) from stage_metrics()."""
    with np.load(path) as payload:
        columns = [str(c) for c in payload["columns"]]
        arrays = [payload[f"c{k}"] for k in range(len(columns))]
        all_int = [bool(x) for x in payload["all_int"]]
    return columns, arrays, all_int


def staged_metrics_rows(path):
    """The staged rows as dicts again: ints as int, the rest as float — the
    same dtype inference pandas makes from the original dicts."""
    columns, arrays, _ = load_staged_metrics(path)
    lists = [a.tolist() for a in arrays]
    return [dict(zip(columns, values)) for values in zip(*lists)]


def remove_metrics_stage(output_dir):
    stage = os.path.join(output_dir, METRICS_STAGE_DIR)
    if os.path.isdir(stage):
        for name in os.listdir(stage):
            os.remove(os.path.join(stage, name))
        os.rmdir(stage)


def pack_enabled():
    """Runners pack a completed experiment unless PACK_RUN_RECORD is off."""
    return os.environ.get("PACK_RUN_RECORD", "1").strip().lower() not in ("0", "false", "no")
