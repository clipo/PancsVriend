#!/usr/bin/env python3
"""Regenerate stored value-function runs on the other simulation path and compare bit for bit.

Speed review item 4 (vf_fastpath.py). Each experiment's config.json records
which path produced it (simulation_path: 'fast' = vf_fastpath.FastVFSimulation,
'reference' = llm_runner.LLMSimulation; folders made before 2026-11-24 have no
entry and were all made on the reference path). By default every run is
regenerated on the OTHER path, so a fast-path production run is checked
against the reference simulation and an old reference run against the fast
path; --path fast|reference forces one. For every selected run:

  * frames: the stored states (int8, frame 0 = initial grid, k+1 = after step k)
    against the regenerated ones, cell for cell, dtype and shape included;
  * the step log: every STEP_LOG_COLUMNS value of every step;
  * the per-step metrics: every metric's float64 bit pattern at every step
    (CSV stores are read with float_precision='round_trip', which gives the
    written bits; Parquet stores the float64 itself);
  * run_summary.csv: converged, convergence_step, final_step.

Stops at the FIRST difference and prints where it is (experiment, run, what,
step, cell or column). Read-only on the stored experiments.

    python verify_vf_fastpath.py EXPERIMENT_DIR... [--runs all|N] [--path other|fast|reference]
                                 [--out results.jsonl]

--runs N checks N runs per experiment: the longest runs first (capped runs
to their full step limit), then an evenly spaced spread of run ids.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config as cfg                    # noqa: E402
import llm_runner                       # noqa: E402
import run_files                        # noqa: E402
import vf_fastpath                      # noqa: E402
from Metrics import calculate_all_metrics  # noqa: E402

METRICS = ("clusters", "switch_rate", "distance", "mix_deviation", "share",
           "ghetto_rate", "dissimilarity_index")


class Mismatch(Exception):
    pass


def _bits(values):
    return np.asarray(values, dtype=np.float64).view(np.uint64)


def select_runs(summary, how):
    ids = sorted(int(r) for r in summary["run_id"])
    if how == "all":
        return ids
    n = int(how)
    if n >= len(ids):
        return ids
    longest = summary.sort_values(["final_step", "run_id"], ascending=[False, True])["run_id"]
    chosen = [int(r) for r in longest[: n // 2]]
    rest = n - len(chosen)
    spread = [ids[int(i)] for i in np.linspace(0, len(ids) - 1, rest + len(chosen))]
    for r in spread:
        if len(chosen) >= n:
            break
        if r not in chosen:
            chosen.append(r)
    return sorted(chosen)


def stored_metrics(exp, run_ids):
    """{run_id: DataFrame sorted by step} for the selected runs, read in chunks."""
    path = run_files.metrics_history_path(exp)
    wanted = set(run_ids)
    parts = []
    if run_files.is_parquet(path):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        for i in range(pf.num_row_groups):
            df = pf.read_row_group(i).to_pandas()
            parts.append(df[df["run_id"].isin(wanted)])
    else:
        for df in pd.read_csv(path, float_precision="round_trip", chunksize=2_000_000):
            parts.append(df[df["run_id"].isin(wanted)])
    allm = pd.concat(parts, ignore_index=True)
    return {int(r): g.sort_values("step").reset_index(drop=True)
            for r, g in allm.groupby("run_id")}


def first_diff_cell(a, b):
    idx = np.argwhere(a != b)[0]
    return tuple(int(i) for i in idx)


def compare_run(exp, rid, sim, res, metrics, srow, max_steps):
    where = f"{os.path.basename(exp)} run {rid}"
    # frames
    stored = run_files.load_frames(exp, rid)
    new = np.array(sim.states)
    if stored is None:
        raise Mismatch(f"{where}: no stored frames")
    if stored.dtype != new.dtype:
        raise Mismatch(f"{where}: frame dtype {stored.dtype} stored vs {new.dtype} new")
    n = min(len(stored), len(new))
    for k in range(n):
        if not np.array_equal(stored[k], new[k]):
            cell = first_diff_cell(stored[k], new[k])
            raise Mismatch(f"{where}: FIRST DIFFERENCE frame {k} (after step {k - 1}) cell {cell}: "
                           f"stored {stored[k][cell]} new {new[k][cell]}")
    if len(stored) != len(new):
        raise Mismatch(f"{where}: frame count {len(stored)} stored vs {len(new)} new "
                       f"(all {n} common frames identical)")
    # step log
    slog = run_files.load_step_log(exp, rid)
    nlog = pd.DataFrame(sim.step_log, columns=list(run_files.STEP_LOG_COLUMNS))
    if slog is None or len(slog) != len(nlog):
        raise Mismatch(f"{where}: step-log rows {None if slog is None else len(slog)} stored vs {len(nlog)} new")
    for col in run_files.STEP_LOG_COLUMNS:
        a = slog[col].to_numpy(dtype=np.int64)
        b = nlog[col].to_numpy(dtype=np.int64)
        if not np.array_equal(a, b):
            i = int(np.argwhere(a != b)[0][0])
            raise Mismatch(f"{where}: FIRST DIFFERENCE step log step {int(slog['step'].iloc[i])} "
                           f"column {col}: stored {a[i]} new {b[i]}")
    # per-step metrics
    m = metrics.get(rid)
    nm = pd.DataFrame(sim.metrics_history)
    if m is None or len(m) != len(nm):
        raise Mismatch(f"{where}: metric rows {None if m is None else len(m)} stored vs {len(nm)} new")
    if not np.array_equal(m["step"].to_numpy(np.int64), nm["step"].to_numpy(np.int64)):
        raise Mismatch(f"{where}: metric step column differs")
    for col in METRICS:
        a, b = _bits(m[col].to_numpy()), _bits(nm[col].to_numpy())
        if not np.array_equal(a, b):
            i = int(np.argwhere(a != b)[0][0])
            raise Mismatch(f"{where}: FIRST DIFFERENCE metric {col} at step {int(m['step'].iloc[i])}: "
                           f"stored {m[col].iloc[i]!r} new {nm[col].iloc[i]!r}")
    # run summary. run_summary.csv's final_step is the last step SIMULATED
    # (run_summary.py docstring); the in-memory result's final_step is
    # Simulation.step, one past it for a capped run. Compare like with like.
    last_step = int(nm["step"].iloc[-1])
    if bool(srow["converged"]) != bool(res["converged"]) or int(srow["final_step"]) != last_step:
        raise Mismatch(f"{where}: summary converged/final_step {srow['converged']}/{srow['final_step']} "
                       f"stored vs {res['converged']}/{last_step} new")
    if "n_steps" in srow and not pd.isna(srow["n_steps"]) and int(srow["n_steps"]) != len(nm):
        raise Mismatch(f"{where}: n_steps {srow['n_steps']} stored vs {len(nm)} new")
    for col in METRICS:
        if col in srow and _bits([srow[col]])[0] != _bits([nm[col].iloc[-1]])[0]:
            raise Mismatch(f"{where}: run_summary final {col} {srow[col]!r} stored vs {nm[col].iloc[-1]!r} new")
        icol = "initial_" + col
        if icol in srow and not pd.isna(srow[icol]):
            init = calculate_all_metrics(new[0])[col]
            if _bits([srow[icol]])[0] != _bits([init])[0]:
                raise Mismatch(f"{where}: run_summary {icol} {srow[icol]!r} stored vs {init!r} new")
    sc = srow["convergence_step"]
    nc = res["convergence_step"]
    if (pd.isna(sc) and nc is not None) or (not pd.isna(sc) and (nc is None or int(sc) != int(nc))):
        raise Mismatch(f"{where}: convergence_step {sc} stored vs {nc} new")
    return len(sim.step_log)


def regeneration_path(conf, how):
    """The path to regenerate on: the one that did NOT produce the experiment
    (how='other'), or the one asked for."""
    produced = conf.get("simulation_path", "reference")
    if how == "other":
        return "reference" if produced == "fast" else "fast"
    return how


def verify_experiment(exp, how, out, path="other"):
    conf = json.load(open(os.path.join(exp, "config.json")))
    produced = conf.get("simulation_path", "reference")
    regen = regeneration_path(conf, path)
    sim_class = vf_fastpath.FastVFSimulation if regen == "fast" else llm_runner.LLMSimulation
    if conf.get("rng_scheme") != "keyed":
        raise Mismatch(f"{exp}: rng_scheme {conf.get('rng_scheme')!r}, the fast path is keyed only")
    llm_runner.apply_grid_config({"GRID_SIZE": conf["grid_size"], "NUM_TYPE_A": conf["num_type_a"],
                                  "NUM_TYPE_B": conf["num_type_b"]})
    if conf.get("scenario_file"):
        llm_runner.apply_scenario_file(conf["scenario_file"])
    cfg.NO_MOVE_THRESHOLD = int(conf.get("no_move_threshold", cfg.NO_MOVE_THRESHOLD))
    policy = llm_runner.load_value_function_policy(conf["value_function_file"], conf["scenario"])[0]
    summary = pd.read_csv(os.path.join(exp, "run_summary.csv"), float_precision="round_trip")
    runs = select_runs(summary, how)
    srows = summary.set_index("run_id")
    t0 = time.time()
    metrics = stored_metrics(exp, runs)
    t_load = time.time() - t0
    steps = 0
    t_sim = 0.0
    for rid in runs:
        sim = sim_class(
            rid, conf["scenario"], conf["llm_model"], None, None, random_seed=rid,
            temperature=conf["temperature"], value_function_policy=policy,
            value_function_path=conf["value_function_file"])
        t1 = time.perf_counter()
        res = sim.run_single_simulation(output_dir=None, max_steps=int(conf["max_steps"]))
        t_sim += time.perf_counter() - t1
        steps += compare_run(exp, rid, sim, res, metrics, srows.loc[rid], int(conf["max_steps"]))
    rec = {"experiment": exp, "runs_checked": len(runs), "runs_stored": len(summary),
           "steps_checked": steps, "max_final_step": int(summary.loc[summary.run_id.isin(runs), "final_step"].max()),
           "max_steps": int(conf["max_steps"]), "capped_runs_checked": int(
               (summary.loc[summary.run_id.isin(runs), "final_step"] >= int(conf["max_steps"]) - 1).sum()),
           "produced_on": produced, "regenerated_on": regen,
           "ms_per_step": 1e3 * t_sim / max(steps, 1), "metrics_load_s": t_load,
           "wall_s": time.time() - t0, "result": "IDENTICAL"}
    print(json.dumps(rec), flush=True)
    if out:
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--runs", default="all")
    ap.add_argument("--path", choices=["other", "fast", "reference"], default="other",
                    help="path to regenerate on (default: the one that did not produce the run)")
    ap.add_argument("--out")
    args = ap.parse_args()
    for exp in args.experiments:
        try:
            verify_experiment(exp.rstrip("/"), args.runs, args.out, args.path)
        except Mismatch as exc:
            msg = {"experiment": exp, "result": "MISMATCH", "detail": str(exc)}
            print(json.dumps(msg), flush=True)
            if args.out:
                with open(args.out, "a") as f:
                    f.write(json.dumps(msg) + "\n")
            sys.exit(1)


if __name__ == "__main__":
    main()
