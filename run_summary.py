"""One row per run: convergence step + every segregation metric at the final step.

`metrics_history.csv` carries every metric at every step, so the
per-run picture the analysis actually starts from — "run 7 of the race
scenario converged at step 42, and here is its dissimilarity index and the
other six metrics at the last simulated step" — had to be re-derived by hand
(or by a groupby-idxmax in each downstream script) every time. This module
writes it once, as run_summary.csv, next to the other two files, and
Simulation.analyze_results calls it, so every experiment directory carries it
without anyone remembering to ask (2026-09-01).

Column contract (order is the file's order):

    run_id, scenario, converged, convergence_step, dissimilarity_index,
    clusters, switch_rate, distance, mix_deviation, share, ghetto_rate,
    final_step, n_steps, stop_reason, experiment, llm_model, metrics_source

* convergence_step — the FIRST of the NO_MOVE_THRESHOLD consecutive zero-move
  steps that mark convergence (see base_simulation.convergence_from_step_moves);
  empty for runs that never converged.
* final_step — always the last step actually SIMULATED: the lower of the
  convergence-implied last step and the step the run stopped at. For a
  converged run that is convergence_step + NO_MOVE_THRESHOLD - 1 (= +4 at the
  default threshold of 5) BY CONSTRUCTION; for a run that hit the cap it is
  max_steps - 1, and for one aborted mid-flight it is the last step with data.
  A run can stop mid-streak (three no-move steps at step 999), which is not
  convergence: converged is False and convergence_step is empty, but
  final_step is still 999. Note the live loop leaves Simulation.step one PAST
  the last simulated step in that case, so this column is anchored on the
  metrics history rather than on that value.
* the seven metric columns — the metrics_history row with the largest step for
  that run. For a converged run these equal the values at convergence_step,
  since by definition nothing moved in between.
* stop_reason — converged / max_steps / incomplete.
* metrics_source — live (read straight from metrics_history.csv) or rebuilt
  (reconstructed from the run's move log because the stored rows were missing
  or were resume placeholders).
"""

import json
import os
import re

import numpy as np
import pandas as pd

import run_files

import config as cfg
from DissimilarityIndex import compute_dissimilarity_from_int_grid

METRIC_COLUMNS = [
    "dissimilarity_index",
    "clusters",
    "switch_rate",
    "distance",
    "mix_deviation",
    "share",
    "ghetto_rate",
]

SUMMARY_COLUMNS = (
    ["run_id", "scenario", "converged", "convergence_step"]
    + METRIC_COLUMNS
    + ["final_step", "n_steps", "stop_reason", "experiment", "llm_model", "metrics_source"]
)

RUN_SUMMARY_FILENAME = "run_summary.csv"

_PLACEHOLDER_FINAL_STEP = "unknown"


def _to_int_or_none(value):
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _load_config(output_dir):
    path = os.path.join(output_dir, "config.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[run_summary] Warning: could not read {path} ({exc})")
        return {}


def _scenario_from_experiment_name(experiment):
    """'llm_race_white_black_20251119_142602' -> 'race_white_black'.

    Only a fallback: the mechanical baseline's config.json predates the
    'scenario' key, so without this its rows would carry a blank scenario.
    """
    name = re.sub(r"_\d{8}_\d{6}$", "", experiment or "")
    return re.sub(r"^llm_", "", name)


def _read_csv_if_present(path):
    """Round-trip float parsing, so rows read back for reuse are the bits
    that were written (pandas' default parser can be an ulp off)."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, float_precision='round_trip')
    except Exception as exc:
        print(f"[run_summary] Warning: could not read {path} ({exc}); treating as empty")
        return pd.DataFrame()


def _rebuild_from_move_log(output_dir, run_id, threshold):
    """Per-step move counts + the final grid, straight from the run's files.

    The fallback for runs whose result is a resume placeholder
    (final_step == 'unknown') or whose metrics rows never made it to disk. Returns (converged, convergence_step, final_step, int_grid) with
    None entries when the log is unusable. run_files reads the per-step and
    the older per-move formats alike.
    """
    from base_simulation import convergence_from_step_moves

    step_moves = run_files.step_moves(run_files.load_step_log(output_dir, run_id))
    if not step_moves:
        return None, None, None, None
    converged, convergence_step, _ = convergence_from_step_moves(step_moves, threshold)
    return converged, convergence_step, max(step_moves), run_files.load_final_grid(output_dir, run_id)


def _final_grid_from_states(output_dir, run_id):
    """Last frame of states_run_<id>.npz — the grid as the run ended."""
    frames = run_files.load_frames(output_dir, run_id)
    if frames is None or len(frames) == 0:
        return None
    return frames[-1]


def _metrics_from_final_grid(output_dir, run_id, move_log_grid, columns=None):
    """Recompute final-step metrics from the run's final grid.

    Two cases need this. Runs whose metrics_history rows never made it to disk
    (the .npz resume path writes placeholder convergence rows and no metrics)
    would otherwise be an all-NaN row. And every experiment predating
    2026-08-25 lacks dissimilarity_index, since calculate_all_metrics only
    started emitting it then. Every metric is a pure function of the grid, so
    recompute from the last frame instead of leaving the row blank.

    `columns` limits the work to the metrics actually missing.
    """
    grid = move_log_grid if move_log_grid is not None else _final_grid_from_states(output_dir, run_id)
    if grid is None:
        # The .npz is the cheap source, but a run killed mid-write leaves a
        # corrupt one ("invalid distance too far back"); pre-2026-09-01 move
        # logs carry the grid on every record, so try them before giving up.
        grid = _rebuild_from_move_log(output_dir, run_id, 1)[3]
    if grid is None:
        return {}

    wanted = set(columns) if columns else set(METRIC_COLUMNS)
    int_grid = np.asarray(grid)
    try:
        if wanted == {"dissimilarity_index"}:
            # The common case (an old experiment missing only this column):
            # skip rebuilding the agent grid the other six metrics need.
            return {"dissimilarity_index": compute_dissimilarity_from_int_grid(int_grid)}

        from Metrics import calculate_all_metrics

        computed = calculate_all_metrics(int_grid)
        return {k: v for k, v in computed.items() if k in wanted}
    except Exception as exc:
        print(f"[run_summary] Warning: could not recompute metrics for run {run_id} ({exc})")
        return {}


def _metrics_by_run(output_dir, results):
    """Final-step metric rows per run_id, preferring in-memory over on-disk.

    A just-executed run carries its metrics_history in the result dict; runs
    that were not re-executed only exist in metrics_history.csv.
    """
    final_rows = {}

    disk = _read_csv_if_present(run_files.metrics_history_path(output_dir))
    if not disk.empty and {"run_id", "step"} <= set(disk.columns):
        idx = disk.groupby("run_id")["step"].idxmax()
        # to_dict('records'), not iterrows(): a row Series upcasts the integer
        # metrics (clusters, ghetto_rate) to float, so the same run was written
        # as 87 when it came from memory and 87.0 when it came from disk.
        for row in disk.loc[idx].to_dict("records"):
            final_rows[_to_int_or_none(row["run_id"])] = row

    for result in results or []:
        history = result.get("metrics_history") or []
        if not history:
            continue
        best = max(history, key=lambda row: _to_int_or_none(row.get("step")) or 0)
        final_rows[_to_int_or_none(result.get("run_id"))] = dict(best)

    return final_rows


def _convergence_by_run(output_dir, results):
    """Convergence rows per run_id from the in-memory results; real rows win,
    placeholders never do. Runs not in `results` are served by their existing
    run_summary.csv row (_cached_rows) or rebuilt from their step log. A
    legacy convergence_summary.csv is read only when there is no
    run_summary.csv yet (it stopped being written 2026-09-05)."""
    rows = {}

    if not os.path.exists(os.path.join(output_dir, RUN_SUMMARY_FILENAME)):
        disk = _read_csv_if_present(os.path.join(output_dir, "convergence_summary.csv"))
        if not disk.empty and "run_id" in disk.columns:
            for row in disk.to_dict("records"):
                rows[_to_int_or_none(row["run_id"])] = row

    for result in results or []:
        if str(result.get("final_step")) == _PLACEHOLDER_FINAL_STEP:
            continue
        rows[_to_int_or_none(result.get("run_id"))] = {
            "run_id": result.get("run_id"),
            "converged": result.get("converged"),
            "convergence_step": result.get("convergence_step"),
            "final_step": result.get("final_step"),
        }

    return rows


def _cached_rows(output_dir, reuse_existing):
    """Previously written rows, keyed by run_id, for reuse on rebuild.

    Rebuilding a run from its move log means a full gzip+JSON parse of a file
    that stores a whole grid per agent decision — a few hundred MB compressed
    for a 100-run experiment whose convergence rows are all resume
    placeholders. That work is deterministic, so once it is on disk there is
    no reason for the analysis pipeline to redo it on every pass.
    """
    if not reuse_existing:
        return {}
    df = _read_csv_if_present(os.path.join(output_dir, RUN_SUMMARY_FILENAME))
    if df.empty or "run_id" not in df.columns:
        return {}
    rows = {}
    for row in df.to_dict("records"):
        # An incomplete row is one nothing could be recovered for; retry it
        # rather than caching the failure.
        if row.get("stop_reason") == "incomplete":
            continue
        if _to_int_or_none(row.get("final_step")) is None:
            continue
        rows[_to_int_or_none(row.get("run_id"))] = row
    return rows


def build_run_summary(output_dir, results=None, reuse_existing=True):
    """DataFrame of one row per run for an experiment directory.

    `results` is the in-memory result list when called from a live run; when
    omitted everything is read back from the directory, which is what the
    retrofit CLI (analysis_tools/build_run_summary.py) does.

    `reuse_existing` lets a run whose stored bookkeeping is unusable fall back
    to its row in an existing run_summary.csv instead of re-parsing its move
    log. Pass False (the CLI's --force) to rebuild from the raw logs.
    """
    config = _load_config(output_dir)
    threshold = _to_int_or_none(config.get("no_move_threshold")) or getattr(cfg, "NO_MOVE_THRESHOLD", 5)
    max_steps = _to_int_or_none(config.get("max_steps"))
    experiment = os.path.basename(os.path.normpath(output_dir))
    config_scenario = config.get("scenario") or _scenario_from_experiment_name(experiment)
    llm_model = config.get("llm_model")

    metrics_rows = _metrics_by_run(output_dir, results)
    convergence_rows = _convergence_by_run(output_dir, results)
    cached = _cached_rows(output_dir, reuse_existing)
    scenario_by_run = {
        _to_int_or_none(r.get("run_id")): r.get("scenario")
        for r in (results or [])
        if r.get("scenario")
    }

    # Every run the directory knows about: in-memory results, stored metrics,
    # the run record itself, and rows of an existing summary (even when they
    # are not reused, a run that was summarised before is still a run).
    known = set(metrics_rows) | set(convergence_rows) | set(cached)
    known.update(run_files.list_run_ids(output_dir))
    if not reuse_existing:
        known.update(_cached_rows(output_dir, True))
    run_ids = sorted(rid for rid in known if rid is not None)

    summary_rows = []
    for run_id in run_ids:
        conv = convergence_rows.get(run_id, {})
        metrics = metrics_rows.get(run_id)

        converged = _to_bool(conv.get("converged"))
        convergence_step = _to_int_or_none(conv.get("convergence_step"))
        final_step = _to_int_or_none(conv.get("final_step"))
        metrics_source = "live"
        move_log_grid = None

        # Placeholder or incomplete bookkeeping: go back to the move log, the
        # canonical record of what actually happened in the run.
        needs_rebuild = final_step is None or (converged and convergence_step is None) or metrics is None
        if needs_rebuild and run_id in cached:
            # Already rebuilt on an earlier pass; that answer is deterministic.
            summary_rows.append({column: cached[run_id].get(column) for column in SUMMARY_COLUMNS})
            continue
        if needs_rebuild:
            rebuilt_converged, rebuilt_step, rebuilt_final, move_log_grid = _rebuild_from_move_log(
                output_dir, run_id, threshold
            )
            if rebuilt_final is not None:
                converged = bool(rebuilt_converged)
                convergence_step = rebuilt_step
                final_step = rebuilt_final
                metrics_source = "rebuilt"

        # Legacy rows from the pre-2026-09-01 live loop stored the LAST step of
        # the no-move window; they are recognisable because that made
        # convergence_step equal final_step (or left it empty when the row came
        # from the .npz resume placeholder). Shift them onto the first step so
        # every row in this file means the same thing.
        if converged and final_step is not None and threshold > 1:
            if convergence_step is None or convergence_step == final_step:
                convergence_step = final_step - (threshold - 1)

        # `final_step` = the last step actually SIMULATED. The live loop
        # increments self.step past the last simulated step when a run stops at
        # the cap, so its stored value is one too high for non-converged runs
        # (10 for --max-steps 10, where step 9 was the last simulated). The
        # metrics history has exactly one row per simulated step, so its
        # largest step is the authoritative answer; for converged runs the
        # definition pins it directly.
        if converged and convergence_step is not None:
            final_step = convergence_step + threshold - 1
        elif metrics is not None:
            metrics_step = _to_int_or_none(metrics.get("step"))
            if metrics_step is not None:
                final_step = metrics_step

        if converged:
            stop_reason = "converged"
        elif final_step is not None and max_steps is not None and final_step + 1 >= max_steps:
            stop_reason = "max_steps"
        else:
            stop_reason = "incomplete"

        row = {
            "run_id": run_id,
            "scenario": scenario_by_run.get(run_id) or config_scenario or "",
            "converged": converged,
            "convergence_step": convergence_step,
            "final_step": final_step,
            "n_steps": None if final_step is None else final_step + 1,
            "stop_reason": stop_reason,
            "experiment": experiment,
            "llm_model": llm_model or "",
            "metrics_source": metrics_source,
        }

        for column in METRIC_COLUMNS:
            value = None if metrics is None else metrics.get(column)
            if isinstance(value, float) and np.isnan(value):
                value = None
            row[column] = value

        missing = [column for column in METRIC_COLUMNS if row[column] is None]
        if missing:
            recomputed = _metrics_from_final_grid(output_dir, run_id, move_log_grid, columns=missing)
            for column, value in recomputed.items():
                row[column] = value
            if recomputed and metrics is None:
                metrics_source = row["metrics_source"] = "rebuilt"

        # A row nobody could put a final-step value on is not a usable
        # observation; say so rather than shipping a silently blank one.
        if row["dissimilarity_index"] is None:
            row["stop_reason"] = "incomplete"

        summary_rows.append(row)

    if not summary_rows:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    df = pd.DataFrame(summary_rows)[SUMMARY_COLUMNS]
    return df.sort_values("run_id", kind="stable").reset_index(drop=True)


def write_run_summary(output_dir, results=None, verbose=True, reuse_existing=True):
    """Write run_summary.csv into an experiment directory; return the DataFrame."""
    try:
        df = build_run_summary(output_dir, results=results, reuse_existing=reuse_existing)
    except Exception as exc:
        # Never let the summary take an experiment down with it: the run's
        # real outputs are already on disk by the time this is called.
        print(f"[run_summary] Warning: could not build run summary for {output_dir}: {exc}")
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    path = os.path.join(output_dir, RUN_SUMMARY_FILENAME)
    df.to_csv(path, index=False)
    if verbose:
        incomplete = int((df["stop_reason"] == "incomplete").sum()) if not df.empty else 0
        note = f" ({incomplete} incomplete)" if incomplete else ""
        print(f"[run_summary] Wrote {len(df)} run(s) to {path}{note}")
    return df


def combine_run_summaries(output_dirs, out_path=None, verbose=True):
    """Concatenate per-experiment run summaries into one campaign-level CSV.

    Reads each directory's run_summary.csv, building it first if absent, so a
    campaign aggregate can be produced from experiment folders alone.
    """
    frames = []
    for output_dir in output_dirs:
        if not output_dir or not os.path.isdir(output_dir):
            continue
        path = os.path.join(output_dir, RUN_SUMMARY_FILENAME)
        df = _read_csv_if_present(path)
        if df.empty:
            df = write_run_summary(output_dir, verbose=verbose)
        if not df.empty:
            frames.append(df)

    if not frames:
        combined = pd.DataFrame(columns=SUMMARY_COLUMNS)
    else:
        combined = pd.concat(frames, ignore_index=True)
        sort_cols = [c for c in ("scenario", "experiment", "run_id") if c in combined.columns]
        if sort_cols:
            combined = combined.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    if out_path:
        out_path = str(out_path)
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        combined.to_csv(out_path, index=False)
        if verbose:
            n_scenarios = combined["scenario"].nunique() if not combined.empty else 0
            print(f"[run_summary] Wrote combined summary: {out_path} "
                  f"({len(combined)} runs across {n_scenarios} scenario(s))")

    return combined
