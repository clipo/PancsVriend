#!/usr/bin/env python3
"""Multi-split sampling-sufficiency check for value-function simulations.

Supersedes the single even/odd split in vf_simulation_evaluation.py's E3
check. That test compares the full-data artifact against ONE half-data
artifact, so its verdict is a single draw from a distribution — qwen's
2026-08-25 FLAG (political/share, |Δ|/τ = 1.05) could not be distinguished
from an unlucky partition. This script estimates the distribution instead.

    python analysis_tools/vf_multisplit_check.py \
        --config-yaml configs/vf_run_qwen_sanity.yaml --splits 32
    # any SAMPLED-table run config (the check resplits raw samples; the exact
    # -lp tables have none, and the retired _r3 configs are kept only to
    # reproduce the old sampled runs)

The yaml is the production run being certified: label, style, scenarios,
max_steps and the board (grid_size, num_type_a, num_type_b) come from its
production contexts_args, so the arms simulate the same experiment. Any
explicit flag overrides the yaml; --label alone still works.

WHAT IT MEASURES

For split b, every (role, composition) cell independently keeps a random HALF
of its raw samples; the resulting artifact is simulated against the SAME run
seeds as the full artifact, so run k of each arm starts from an identical grid
and consumes an identical random stream. The paired difference of the final
metric values,

    Δ_b = mean_k( full_k - half_k^b ),

is summarised over B splits as RMS(Δ) = sqrt(mean_b Δ_b²) and compared to

(Since 2026-09-04 llm_runner keys every value-function decision's uniform to
(run_id, step, agent_id), so "identical random stream" holds literally: the
half and full arms differ only at decisions whose uniform falls between the
two tables' probabilities. Before that a single differing decision shifted
the shared stream and re-rolled every later draw, so rulers measured under
the old scheme — every multisplit dir dated before 2026-09-04 — carry extra
run noise and are inflated relative to a re-measurement under this one.)

    τ = SD(full finals) / sqrt(n_runs)

the STANDARD ERROR of the full arm's own mean. PASS when RMS(Δ) <= margin·τ.

Until 2026-09-01 τ carried a 1.96 (a 95% CI half-width), which made the ratio
dimensionally mixed — a 1-sigma RMS over a 1.96-sigma band — so ratio = 1 meant
"table error is 1.96x the run-mean SE" rather than anything readable. As a plain
SE both terms are 1-sigma and the ratio says exactly what it looks like: how big
the table's sampling noise is relative to the simulation's own run-to-run noise.
Every historical ratio scales by 1.96 under this change, and the margin moved
0.7 -> 1.0 to match (0.7 old == 1.372 new; 1.0 is stricter).

The interpretable form is the ERROR-BAR INFLATION. Both sources are independent
and add in variance, so the total uncertainty on the reported mean is

    SE_total = SE_runs · sqrt(1 + ratio²)

ratio = 1.0 therefore means your true error bars are sqrt(2) = 1.41x what the
run-to-run spread alone would suggest. The old 0.7 line permitted 70%.

CAUTION — τ scales as 1/sqrt(n_runs) while RMS(Δ) does NOT: the table-induced
displacement is a systematic offset that more simulation runs cannot average
away. So the ratio grows as sqrt(n_runs), and the run count at which the two
noise sources balance is n_runs = n_ref/ratio_ref². Past that point extra runs
shrink a term that is already negligible, and a CI computed from run variance
alone becomes overconfident by roughly the ratio.

WHY HALVES, AND WHY RMS

Keeping a fraction f of each cell gives

    Var(p̂_f - p̂_full) = (σ²/n)·(1/f - 1),      σ² = p(1-p)

so at f = 1/2 the bracket is 1 and the observed displacement has EXACTLY the
variance of the full artifact's own sampling error, σ²/n. Half is the unique
fraction needing no rescaling, which is what lets RMS(Δ) be read directly as
"how far the published table's simulation output plausibly sits from the
truth". (Odd n keeps n//2, marginally below half; negligible at n >= 100.)

KEEP FRACTIONS OTHER THAN HALF (--keep-fraction, 2026-09-05) are a DIAGNOSTIC
of how the ruler scales with table precision, not a sufficiency check. At
fraction f the injected table error has variance (1/f - 1) x the full table's
own: 1 at f=1/2, 1/3 at f=3/4, 1/7 at f=7/8. vf_sampling_plan prices top-ups
on s ∝ (table error), i.e. s(3/4)/s(1/2) = sqrt(1/3) = 0.577; if the
displacement is instead dominated by discrete decision flips amplified by the
dynamics, s ∝ sqrt(table error) and the ratio is 0.76. The exponent alpha in
s ∝ (error)^alpha follows from two fractions: alpha = ln(s_f/s_half) /
ln(sqrt(1/f - 1)). The verdict/ratio columns are still written at f != 1/2
but must not be read as sufficiency (they are optimistic by (1/f - 1)^(-a/2)).

RMS rather than the mean because the sign of Δ_b depends on which half was
kept and would cancel; RMS² = (E[Δ])² + Var(Δ) retains both the systematic
component (the metric is a nonlinear function of the table) and the random one.

HOW MANY SPLITS

RMS from B splits carries a relative standard error of ~1/sqrt(2B), and that
error is paid TWICE: once in the verdict, once in the price of the fix. As a
verdict, B=8 waves through a model whose true ratio is 0.75 about 39% of the
time — useless as a screen. As a price, the plan must be padded by the ratio's
own uncertainty, so a loose estimate buys a needlessly tight (and quadratically
more expensive) resampling target. Hence B=32 by default, not 8: it holds the
miss rate to ~4% at a true ratio of 0.90 and cuts the padding from 1.8x to
1.26x. Raising B further mainly buys a CHEAPER plan rather than a better
verdict, so escalate to 128 when pricing a large resample, not to decide.

COST: no GPU — value-function simulations are table lookups. B+1 arms of
6 scenarios x 100 runs, roughly 3 min and 32 MB of scratch per arm. Arms are
independent and run --arm-workers at a time; the raw ledger is parsed once per
scenario rather than once per (scenario, split), which is what makes B=32
affordable.

Outputs (durable, --dpi/--format aware) in --out-dir:
  vf_multisplit_check.csv   per (scenario, metric): RMS, τ, ratio, verdict
  vf_multisplit_check.<ext> per-panel Δ_b scatter against the ±τ band
"""
import argparse
import gzip
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

_THIS = Path(__file__).resolve().parent
REPO_ROOT = _THIS.parent
for _p in (_THIS, REPO_ROOT, REPO_ROOT / "prompt_refinement"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from build_value_function import _assemble_artifact, _blank_counts  # noqa: E402
from sampling_common import load_value_function  # noqa: E402
from vf_simulation_evaluation import (ALL_METRICS, METRIC_LABELS,  # noqa: E402
                                      SCENARIO_ORDER, load_batch)

VF_DIR = REPO_ROOT / "prompt_refinement" / "results" / "value_functions"
CONTEXTS_SCRIPT = REPO_ROOT / "run_all_contexts.py"


# ---------------------------------------------------------------------------
# Raw ledger -> split artifacts
# ---------------------------------------------------------------------------

def load_raw_samples(vf, vf_path):
    """{(role, (n_sim, n_occ)): [verdict, ...]} from the artifact's raw shards.

    meta.raw_replies lists EVERY shard (pilot + each top-up pass), so the
    ledger reassembled here is the same sample set the artifact was tallied
    from — verified by the caller against the stored per-cell counts.
    """
    prev = vf["meta"].get("raw_replies")
    shards = [prev] if isinstance(prev, str) else list(prev or [])
    if not shards:
        raise ValueError(f"{vf_path}: meta.raw_replies is empty — cannot split")
    samples = {}
    for shard in shards:
        p = Path(shard)
        if not p.exists():                      # tolerate a moved repo
            p = VF_DIR / "raw" / p.name
        if not p.exists():
            raise FileNotFoundError(f"raw shard missing: {shard}")
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("_meta"):
                    continue
                key = (rec["agent_role"], (rec["n_similar"], rec["n_occupied"]))
                samples.setdefault(key, []).append(rec["parse"])
    return samples


def verify_ledger(samples, vf, vf_path):
    """The reassembled ledger must reproduce the artifact's own counts."""
    for role, rows in vf["compositions"].items():
        for c in rows:
            key = (role, (c["n_similar"], c["n_occupied"]))
            got = len(samples.get(key, []))
            if got != c["n_samples"]:
                raise ValueError(
                    f"{vf_path}: cell {role} {c['n_similar']}/{c['n_occupied']} "
                    f"has {c['n_samples']} samples in the artifact but {got} in "
                    f"the raw shards — the ledger is incomplete, refusing to split")


def split_counts(samples, roles, rng, keep_fraction=0.5):
    """Count store keeping a random fraction (default half) of EACH cell's samples.

    Per-cell rather than one global draw: a global draw would leave cells with
    unequal counts by chance, adding variance unrelated to the quantity being
    measured. Per-cell keeps n_kept = int(n_full * keep_fraction) everywhere.
    """
    counts = _blank_counts(roles)
    for key, verdicts in samples.items():
        if key not in counts:                   # role absent from this artifact
            continue
        keep = rng.sample(range(len(verdicts)), int(len(verdicts) * keep_fraction))
        c = counts[key]
        for i in keep:
            v = verdicts[i]
            if v == "MOVE":
                c["move"] += 1
            elif v == "STAY":
                c["stay"] += 1
            else:
                c["bad"] += 1
            c["samples"] += 1
    return counts


def load_ledgers(label, style, scenarios):
    """Read every scenario's artifact and raw ledger ONCE.

    Previously the shards were re-parsed for each (scenario, split), so B=32
    gunzipped and json-decoded the same raw files 32 times over — the dominant
    cost of raising B, and the reason a large B looked unaffordable. The
    per-cell verdict lists are read-only afterwards, so one copy serves every
    split (and every arm thread).

    Dict insertion order is preserved, so split_counts still consumes its RNG
    in exactly the order it did before: a given --seed reproduces bit-identical
    partitions across this change.

    Returns ({scenario: (vf, roles, samples, vf_path)}, scenario_file).
    """
    ledgers, scenario_files = {}, set()
    for scenario in scenarios:
        vf_path = VF_DIR / f"vf_{label}__{scenario}__{style}.json"
        vf = load_value_function(vf_path)
        scenario_files.add(vf["meta"].get("scenario_file") or "context_scenarios.py")
        samples = load_raw_samples(vf, vf_path)
        verify_ledger(samples, vf, vf_path)
        ledgers[scenario] = (vf, list(vf["compositions"].keys()), samples, vf_path)
    if len(scenario_files) != 1:
        raise ValueError(f"artifacts disagree on scenario_file: {sorted(scenario_files)}")
    return ledgers, scenario_files.pop()


def write_split_artifacts(label, style, ledgers, split_id, rng, dest, keep_fraction=0.5):
    """Build one reduced artifact per scenario for a single split.

    Returns the '{scenario}' template path the simulation consumes.
    """
    dest.mkdir(parents=True, exist_ok=True)
    new_label = f"{label}-split{split_id:02d}"
    for scenario, (vf, roles, samples, vf_path) in ledgers.items():
        counts = split_counts(samples, roles, rng, keep_fraction)
        meta = dict(vf["meta"])
        meta["label"] = new_label
        meta["rebuilt_from"] = {"artifact": str(vf_path),
                                "keep": f"random-{keep_fraction:g}",
                                "split_id": split_id,
                                "samples_kept": sum(c["samples"] for c in counts.values())}
        out = dest / f"vf_{new_label}__{scenario}__{style}.json"
        out.write_text(json.dumps(_assemble_artifact(meta, counts, roles), indent=1))
        # Consumer-side validator: a table the simulation would refuse must
        # fail HERE, naming the gap, not mid-run in a worker process.
        load_value_function(out)
    return dest / f"vf_{new_label}__{{scenario}}__{style}.json"


# ---------------------------------------------------------------------------
# Simulation arms
# ---------------------------------------------------------------------------

def simulate_arm(vf_template, arm_dir, scenarios, runs, max_steps, processes,
                 model_label, scenario_file, grid=None):
    """Run one arm's simulations; returns {scenario: experiment dir}.

    Executed with cwd inside arm_dir (the orchestrator's convention) so each
    arm's experiments/ tree is isolated and scenario dirs cannot collide.

    `grid` = (grid_size, num_type_a, num_type_b), and it MATTERS: without it
    run_all_contexts falls back to config.GRID_SIZE (10), so the check would
    certify a 10x10 board while production runs whatever its yaml specifies
    (20x20 with 160+160 since 2026-09-01). Both τ and RMS(Δ) would then come
    from a different experiment than the one being certified, and segregation
    metrics do not compare across grid sizes.
    """
    arm_dir.mkdir(parents=True, exist_ok=True)
    manifest = arm_dir / "manifest.json"
    cmd = [sys.executable, str(CONTEXTS_SCRIPT),
           "--scenarios", *scenarios,
           "--runs", str(runs), "--max-steps", str(max_steps),
           "--value-function", str(vf_template)]
    if grid and grid[0]:
        cmd += ["--grid-size", str(grid[0])]
    if grid and grid[1]:
        cmd += ["--num-type-a", str(grid[1])]
    if grid and grid[2]:
        cmd += ["--num-type-b", str(grid[2])]
    cmd += [
           "--scenario-file", scenario_file,
           "--llm-model", model_label,
           "--processes", str(processes), "--new",
           "--manifest-file", str(manifest)]
    proc = subprocess.run(cmd, cwd=str(arm_dir), stdout=subprocess.DEVNULL,
                          stderr=subprocess.PIPE, text=True,
                          env={**os.environ, "PYTHONPATH": str(REPO_ROOT)})
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-15:])
        raise RuntimeError(f"arm {arm_dir.name} failed (exit {proc.returncode}):\n{tail}")
    man = json.loads(manifest.read_text())
    dirs = {}
    for exp in man["experiments"]:
        if exp.get("status") != "success" or not exp.get("output_dir"):
            raise RuntimeError(f"arm {arm_dir.name}: scenario {exp.get('scenario')} failed")
        dirs[exp["scenario"]] = arm_dir / exp["output_dir"]
    return dirs


# ---------------------------------------------------------------------------
# Statistic
# ---------------------------------------------------------------------------

def compute_table(full, splits, scenarios, margin):
    """Per (scenario, metric): RMS of the paired Δ over splits vs the full
    arm's own mean CI half-width."""
    rows = []
    for sc in scenarios:
        for m in ALL_METRICS:
            f_all = full[sc][m][:, -1]
            deltas = []
            for half in splits:
                h_all = half[sc][m][:, -1]
                n = min(len(f_all), len(h_all))
                f, h = f_all[:n], h_all[:n]          # paired by run seed
                ok = ~(np.isnan(f) | np.isnan(h))
                if not ok.any():
                    continue
                deltas.append(float(np.mean(f[ok] - h[ok])))
            d = np.asarray(deltas)
            fv = f_all[~np.isnan(f_all)]
            # Plain standard error, NOT 1.96*SE: both this and delta_rms are
            # then 1-sigma, so ratio = 1 means table noise == run noise.
            tau = float(np.std(fv, ddof=1) / np.sqrt(len(fv)))
            rms = float(np.sqrt(np.mean(d ** 2))) if len(d) else float("nan")
            ratio = rms / tau if tau > 0 else float("nan")
            rows.append({
                "scenario": sc, "metric": m, "n_splits": len(d),
                "full_mean": float(np.mean(fv)),
                "delta_mean": float(np.mean(d)) if len(d) else float("nan"),
                "delta_rms": rms,
                "delta_absmax": float(np.max(np.abs(d))) if len(d) else float("nan"),
                "tau_full_mean_se": tau,
                "ratio": ratio,
                # How much the table widens the reported error bar, which is
                # what the ratio actually costs you: sources add in variance.
                "inflation_pct": (100.0 * (math.sqrt(1.0 + ratio ** 2) - 1.0)
                                  if np.isfinite(ratio) else float("nan")),
                "verdict": "PASS" if (tau > 0 and rms <= margin * tau) else "FLAG",
                "deltas": d,
            })
    return pd.DataFrame(rows)


def bootstrap_worst_ratio(tab, n_boot=2000, seed=0, q=97.5):
    """Upper confidence bound on the WORST ratio across all checks.

    Resamples SPLITS, not checks. All 42 (scenario, metric) checks are computed
    from the SAME B half-splits, so their errors are correlated; a replicate
    must therefore reuse one set of resampled split indices across the whole
    table. Drawing per-check indices would treat the checks as independent and
    understate the spread of the maximum.

    Taking the max WITHIN each replicate is what prices in selection. The
    observed worst of 42 noisy ratios is biased upward (the winner's curse:
    whichever check drew the luckiest positive error tends to be the one
    reported), so the sampling distribution of "the max" — not of a single
    pre-chosen check — is the thing a plan has to survive.

    Returns the q-th percentile of that distribution, or NaN if no check has a
    usable tau.
    """
    usable = [(np.asarray(r.deltas, dtype=float), float(r.tau_full_mean_se))
              for r in tab.itertuples()
              if r.tau_full_mean_se > 0 and len(r.deltas)]
    if not usable:
        return float("nan")
    B = min(len(d) for d, _ in usable)
    if B < 2:                      # a single split has no spread to resample
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, B, size=(n_boot, B))
    worst = np.zeros(n_boot)
    for d, tau in usable:
        rms = np.sqrt(np.mean(d[:B][idx] ** 2, axis=1))
        worst = np.maximum(worst, rms / tau)
    return float(np.percentile(worst, q))


def fig_multisplit(tab, scenarios, out_path, margin, dpi):
    """One panel per (metric, scenario): the B split displacements against the
    run-to-run precision band they must stay inside."""
    nrows, ncols = len(ALL_METRICS), len(scenarios)
    # Floor the width so the suptitle and the two-column legend still fit when
    # the check is run on a single scenario (the in-loop / debug case).
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, sharex=True,
                             figsize=(max(8.5, 2.0 * ncols + 1.4), 1.25 * nrows + 1.8))
    rng = np.random.default_rng(0)   # fixed: re-running must reproduce the image
    for i, m in enumerate(ALL_METRICS):
        for j, sc in enumerate(scenarios):
            ax = axes[i][j]
            r = tab[(tab.scenario == sc) & (tab.metric == m)].iloc[0]
            tau, d = r.tau_full_mean_se, r["deltas"]
            ax.axhspan(-tau, tau, color="#c8e6c9", alpha=0.55, lw=0)
            for s in (-1, 1):
                ax.axhline(s * margin * tau, color="#2e7d32", lw=0.8, ls="--")
            ax.axhline(0, color="#888", lw=0.6)
            x = rng.uniform(-0.28, 0.28, size=len(d))
            ax.scatter(x, d, s=14, color="#404040", zorder=3,
                       edgecolors="white", linewidths=0.4)
            for s in (-1, 1):                       # ±RMS: the test statistic
                ax.plot([-0.42, 0.42], [s * r.delta_rms] * 2,
                        color="#7b3294", lw=1.4, zorder=4)
            lim = max(tau, np.max(np.abs(d)) if len(d) else tau,
                      r.delta_rms) * 1.35
            ax.set_ylim(-lim, lim); ax.set_xlim(-0.5, 0.5)
            ax.set_xticks([])
            ax.tick_params(axis="y", labelsize=6)
            ax.set_facecolor("#f7fff7" if r.verdict == "PASS" else "#fff0f0")
            if i == 0:
                ax.set_title(sc, fontsize=7.5)
            if j == 0:
                ax.set_ylabel(METRIC_LABELS[m], fontsize=7)
    n_flag = int((tab.verdict == "FLAG").sum())
    n_splits = int(tab.n_splits.max())
    handles = [
        plt.Line2D([], [], color="#404040", marker="o", ls="", ms=5,
                   label="one random half-split: Δ_b = mean paired (full − half)"),
        plt.Line2D([], [], color="#7b3294", lw=1.4,
                   label="±RMS(Δ) — the test statistic"),
        plt.Line2D([], [], color="#2e7d32", lw=0.8, ls="--",
                   label=f"±{margin:g}·τ pass line"),
        plt.Rectangle((0, 0), 1, 1, color="#c8e6c9",
                      label="±τ = full-arm mean standard error (run-to-run noise)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle(f"Multi-split sampling-sufficiency check — {n_splits} "
                 f"random half-splits ({n_flag} flagged of {len(tab)})", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def apply_run_yaml(args):
    """Fill unset args from a production run yaml (configs/vf_run_*.yaml)."""
    payload = yaml.safe_load(Path(args.config_yaml).read_text())
    ca = (payload.get("profiles") or {}).get("production", {}).get("contexts_args") \
        or payload["contexts_args"]
    m = re.fullmatch(r"vf_(.+)__\{scenario\}__(.+)\.json", Path(ca["value_function"]).name)
    if not m:
        raise ValueError(f"cannot read label/style from value_function {ca['value_function']!r}")
    for attr, val in (("label", m[1]), ("style", m[2]), ("scenarios", ca.get("scenarios")),
                      ("max_steps", ca.get("max_steps")), ("grid_size", ca.get("grid_size")),
                      ("num_type_a", ca.get("num_type_a")), ("num_type_b", ca.get("num_type_b"))):
        if getattr(args, attr) is None and val is not None:
            setattr(args, attr, val)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config-yaml",
                    help="production run yaml; supplies label, style, scenarios, "
                         "max-steps and board geometry unless given explicitly")
    ap.add_argument("--label",
                    help="vf artifact label, e.g. qwen3.6-27b-chat-grammar")
    ap.add_argument("--style", default=None, help="default R3_dual_count")
    ap.add_argument("--splits", type=int, default=32,
                    help="B random half-splits. 32 is the standard setting "
                         "(~12.5%% relative SE); 128 when pricing a large "
                         "resample. 8 is NOT enough to decide — it passes a "
                         "truly-0.75 model 39%% of the time")
    ap.add_argument("--keep-fraction", type=float, default=0.5,
                    help="fraction of each cell's samples a split keeps. 0.5 is "
                         "THE sufficiency check (injected error == the table's "
                         "own); other values (3/4, 7/8) are a scaling "
                         "diagnostic, see the docstring")
    ap.add_argument("--runs", type=int, default=100,
                    help="simulations per arm; τ scales as 1/sqrt(runs), so "
                         "changing it moves the bar — keep 100 for comparability")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="must match the simulation runs being certified "
                         "(200 -> 1000 on 2026-08-27); default 1000")
    ap.add_argument("--processes", type=int, default=8,
                    help="parallel sims WITHIN one arm; keep processes x "
                         "arm-workers below cpu_count when a sampling campaign "
                         "is feeding a llama-server on this host")
    ap.add_argument("--arm-workers", type=int, default=None,
                    help="arms to run concurrently (default: cpu_count // "
                         "processes). Arms are independent, so the old serial "
                         "loop left most of the box idle and made a large B "
                         "look far more expensive than it is")
    ap.add_argument("--margin", type=float, default=1.0,
                    help="PASS when RMS(Δ) <= margin*τ. 1.0 == 'table noise may "
                         "not exceed run noise' (41%% error-bar inflation). Was "
                         "0.7 against the old 1.96-scaled τ, which is 1.372 on "
                         "this scale and permitted 70%% inflation.")
    ap.add_argument("--scenarios", nargs="*", default=None)
    # Board geometry MUST match the production run being certified — see
    # simulate_arm. Left unset, run_all_contexts falls back to config.GRID_SIZE.
    ap.add_argument("--grid-size", type=int, default=None)
    ap.add_argument("--num-type-a", type=int, default=None)
    ap.add_argument("--num-type-b", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None,
                    help="default: <vf dir>/multisplit_<label>")
    ap.add_argument("--scratch", default=None,
                    help="working dir for the arms (default: a temp dir)")
    ap.add_argument("--keep-scratch", action="store_true")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    args = ap.parse_args()
    if args.config_yaml:
        apply_run_yaml(args)
    if not args.label:
        ap.error("--label or --config-yaml is required")
    args.style = args.style or "R3_dual_count"
    args.max_steps = args.max_steps or 1000

    scenarios = args.scenarios or [
        sc for sc in SCENARIO_ORDER
        if (VF_DIR / f"vf_{args.label}__{sc}__{args.style}.json").exists()]
    if not scenarios:
        print(f"no artifacts for label {args.label!r} style {args.style!r}")
        return 1

    out_dir = Path(args.out_dir or (VF_DIR / f"multisplit_{args.label}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(args.scratch) if args.scratch else Path(tempfile.mkdtemp(
        prefix=f"vfsplit_{args.label}_"))
    scratch.mkdir(parents=True, exist_ok=True)

    workers = args.arm_workers or max(1, (os.cpu_count() or 1) // max(1, args.processes))
    grid = (args.grid_size, args.num_type_a, args.num_type_b)
    print(f"label={args.label}  scenarios={len(scenarios)}  splits={args.splits}  "
          f"runs={args.runs}x{args.max_steps}  margin={args.margin}")
    print(f"board: grid={args.grid_size or 'config default'} "
          f"a={args.num_type_a or '-'} b={args.num_type_b or '-'}"
          + ("   WARNING: no --grid-size given; arms use config.GRID_SIZE and may"
             " not match the production run being certified"
             if not args.grid_size else ""))
    print(f"arms: {args.splits + 1} total, {workers} concurrent x "
          f"{args.processes} processes")
    print(f"scratch: {scratch}")

    try:
        ledgers, scenario_file = load_ledgers(args.label, args.style, scenarios)
        full_tpl = str(VF_DIR / f"vf_{args.label}__{{scenario}}__{args.style}.json")

        lock = threading.Lock()
        done = [0]

        def note(what):
            with lock:
                done[0] += 1
                print(f"[arm {done[0]}/{args.splits + 1}] {what} done", flush=True)

        def run_full():
            dirs = simulate_arm(full_tpl, scratch / "arm_full", scenarios,
                                args.runs, args.max_steps, args.processes,
                                f"{args.label}-full", scenario_file, grid)
            out = {sc: load_batch(d, args.max_steps) for sc, d in dirs.items()}
            note("full")
            return out

        def run_split(b):
            # One RNG per split, seeded deterministically from (seed, b): a
            # re-run reproduces the identical partitions, so a verdict is
            # auditable rather than a one-off draw. Seeding per-split (not from
            # a shared stream) is also what makes running arms concurrently
            # safe — a split's partition does not depend on execution order.
            rng = random.Random(args.seed * 100003 + b)
            tpl = write_split_artifacts(args.label, args.style, ledgers, b, rng,
                                        scratch / f"vf_split{b:02d}", args.keep_fraction)
            dirs = simulate_arm(str(tpl), scratch / f"arm_split{b:02d}", scenarios,
                                args.runs, args.max_steps, args.processes,
                                f"{args.label}-split{b:02d}", scenario_file, grid)
            out = {sc: load_batch(d, args.max_steps) for sc, d in dirs.items()}
            note(f"split {b + 1}/{args.splits}")
            return out

        # simulate_arm blocks in subprocess.run, which releases the GIL, so
        # threads are the right pool here — the work is in child processes.
        # Results are collected by index, so concurrency cannot reorder the
        # splits and the per-split deltas CSV stays reproducible.
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_full = ex.submit(run_full)
            fut_splits = [ex.submit(run_split, b) for b in range(args.splits)]
            full = fut_full.result()
            splits = [f.result() for f in fut_splits]

        tab = compute_table(full, splits, scenarios, args.margin)
        csv_path = out_dir / "vf_multisplit_check.csv"
        tab.drop(columns=["deltas"]).to_csv(csv_path, index=False)
        # Per-split Δ_b alongside the summary: the verdict rests on the spread
        # of these, so they must be auditable without re-running the arms.
        long = pd.DataFrame([
            {"scenario": r.scenario, "metric": r.metric, "split": b,
             "seed": args.seed, "delta": float(d)}
            for r in tab.itertuples() for b, d in enumerate(r.deltas)])
        long.to_csv(out_dir / "vf_multisplit_deltas.csv", index=False)
        fig_path = out_dir / f"vf_multisplit_check.{args.format}"
        fig_multisplit(tab, scenarios, fig_path, args.margin, args.dpi)
        print(f"\nwrote {csv_path}\nwrote {fig_path}")

        show = tab.drop(columns=["deltas"]).sort_values("ratio", ascending=False)
        with pd.option_context("display.width", 170, "display.max_rows", 200):
            print("\n=== multi-split sufficiency (RMS(Δ) vs τ), worst first ===")
            print(show[["scenario", "metric", "delta_rms", "tau_full_mean_se",
                        "ratio", "verdict"]].to_string(index=False))
        n_flag = int((tab.verdict == "FLAG").sum())
        print(f"\n{'ALL PASS' if n_flag == 0 else f'{n_flag} FLAG(s)'} "
              f"across {len(tab)} (scenario, metric) checks at margin {args.margin:g}")
        # The adaptive uniform step: ratio is linear in the CI target w, so the
        # multiplier that brings the worst ratio to the margin is read directly.
        #
        # Planned against the bootstrap UPPER bound, not the point estimate.
        # Solving w_next exactly onto the margin lands on the pass/fail
        # boundary, where roughly half of re-checks fail by construction, so
        # some headroom is mandatory. This used to be a flat x0.90 — a guess
        # that is only correct in the limit of a perfectly measured ratio, and
        # which under-bought by 1.8x at B=8 and 1.26x at B=32. The bound
        # derives the headroom from the spread actually observed, so it tightens
        # automatically as B rises: more splits => a cheaper plan.
        worst = float(tab.ratio.max())
        worst_row = tab.loc[tab.ratio.idxmax()]
        worst_hi = bootstrap_worst_ratio(tab, seed=args.seed)
        basis = worst_hi if np.isfinite(worst_hi) and worst_hi > 0 else worst
        mult = min(1.0, args.margin / basis) if basis > 0 else 1.0
        if n_flag:
            hi_txt = f"{worst_hi:.2f}" if np.isfinite(worst_hi) else "n/a"
            print(f"worst ratio {worst:.2f} (bootstrap 97.5% upper bound over "
                  f"the max: {hi_txt}) -> suggested uniform precision "
                  f"w_next = w_cur x {mult:.3f}")

        # Machine-readable verdict. The POINT of this check is to decide whether
        # sampling is sufficient, so the answer must be actionable by the
        # pipeline rather than printed and dropped. Deliberately a file, not
        # scraped stdout — scraping is what made the old calibration loop read a
        # missing artifact as converged.
        status = {
            "label": args.label, "style": args.style, "splits": args.splits,
            "keep_fraction": args.keep_fraction,
            "injected_variance_factor": (1.0 / args.keep_fraction - 1.0),
            "is_sufficiency_check": args.keep_fraction == 0.5,
            "seed": args.seed, "margin": args.margin,
            "runs": args.runs, "max_steps": args.max_steps,
            # Board recorded explicitly: the 2026-09-01 rechecks silently used
            # config.GRID_SIZE=10 while production had moved to 20x20, and
            # nothing in the output said so. A 10x10 ratio runs ~2x optimistic.
            "grid_size": args.grid_size, "num_type_a": args.num_type_a,
            "num_type_b": args.num_type_b,
            "n_checks": int(len(tab)), "n_flags": n_flag,
            "verdict": "PASS" if n_flag == 0 else "INSUFFICIENT",
            "worst_ratio": worst,
            "worst_ratio_ci_high": (round(worst_hi, 4)
                                    if np.isfinite(worst_hi) else None),
            "worst_check": f"{worst_row.scenario}/{worst_row.metric}",
            # Scale marker. Ratios are NOT comparable across tau conventions:
            # before 2026-09-01 tau was 1.96*SE, so those ratios are 1/1.96 of
            # these. Consumers must refuse a status file lacking this key
            # rather than silently mix scales.
            "tau_definition": "se",
            # Decision-RNG scheme the arms ran under (llm_runner.VF_RNG_SCHEME:
            # 'keyed' since 2026-09-04, 'shared' before). Rulers measured under
            # the shared stream carry re-rolled run noise and are inflated; the
            # orchestrator's ruler resolver prefers 'keyed' and treats a status
            # file without this key as 'shared'.
            "rng_scheme": os.environ.get("VF_RNG_SCHEME", "keyed"),
            "worst_inflation_pct": (round(100.0 * (math.sqrt(1.0 + worst ** 2) - 1.0), 1)
                                    if np.isfinite(worst) else None),
            "suggested_precision_multiplier": round(mult, 4),
            "multiplier_basis": ("bootstrap_p97.5_of_max"
                                 if np.isfinite(worst_hi) else "point_estimate"),
            "arm_workers": workers,
        }
        (out_dir / "multisplit_status.json").write_text(json.dumps(status, indent=1))
        print(f"wrote {out_dir}/multisplit_status.json  verdict={status['verdict']}")

        # Exit 4 == "ran fine, sampling is NOT sufficient". Distinct from 0
        # (sufficient) and from a crash (non-zero, non-4 == verdict UNKNOWN),
        # because "we could not tell" must never be reported as "fine".
        return 4 if n_flag else 0
    finally:
        if args.keep_scratch:
            print(f"scratch kept: {scratch}")
        else:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
