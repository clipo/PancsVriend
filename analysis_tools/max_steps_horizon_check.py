"""Is max_steps long enough? Mean-trajectory drift and ordering stability.

A run stops early when NO_MOVE_THRESHOLD consecutive steps pass without a
move; otherwise it is capped at max_steps. A capped run is not necessarily
unsettled -- a value function with P(MOVE) > 0 in common cells keeps churning
forever around a stationary level -- so "capped" is only the first filter.
The question is whether the across-run MEAN of a metric is still moving at the
cap, and whether that movement could change the scenario ordering.

For every experiment (scenario) of every run dir given, this script

  * builds the run x step matrix of the metric, forward-filling absorbed runs
    (their grid is frozen; see plot_style.step_stats_forward_filled),
  * reports the capped fraction, the mean at checkpoint steps, and the paired
    per-run drift mean(DI_T - DI_{T-w}) with its standard error for several
    windows w ending at the last step T,
  * for each pair of scenarios adjacent in the final ordering, tracks the
    paired gap through time, its slope over the last window, and a linear
    extrapolation of the step at which the gap would close,
  * decides "settled" by an EQUIVALENCE test, the same rule for every model:
    settled from step s when, for the --window-step span starting at s and
    every later span (starts every --block steps, plus the final span), the 95%
    confidence interval of the paired per-run change lies inside +-delta. A fully absorbed scenario
    passes from the step its last run froze. With too few runs the interval is
    wider than delta and the scenario is reported as NOT settled (None),
  * reports when absorbed runs stopped (max and 99th percentile step),
  * and the ordering at each checkpoint, so a rank change is visible directly.

Writes into --out-dir:
  horizon_scenarios.csv      one row per (model, scenario)
  horizon_gaps.csv           one row per adjacent pair in the final ordering
  horizon_orderings.csv      ordering at each checkpoint step, per model
  horizon_blocks.csv         paired per-run increment (+ SE, t) per block of
                             --block steps, ending at the last step
  horizon_equivalence.csv    paired change over each --window-step span, its
                             95% CI, and whether it lies inside +-delta
  horizon_mean_curves.csv.gz per-step mean, 95% CI half-width, n_active
  horizon_<metric>.png       mean trajectory per model (+ zoom on the tail)

Usage
-----
    # every model's newest full exact-table run (default)
    .venv/bin/python analysis_tools/max_steps_horizon_check.py

    # specific run dirs (e.g. the long-horizon extension runs)
    .venv/bin/python analysis_tools/max_steps_horizon_check.py \
        experiments_with_llama_cpp/run_..._olmo-2-32b-vf-lp \
        --out-dir experiments_with_llama_cpp/horizon_check/olmo_long
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_files  # noqa: E402

DEFAULT_RUN_ROOT = ROOT / "experiments_with_llama_cpp"
DEFAULT_OUT = DEFAULT_RUN_ROOT / "horizon_check" / "production_1000"
WINDOWS = (50, 100, 250, 500, 1000, 2500, 5000)


# ---------------------------------------------------------------- discovery
def _model_of(run_dir: Path) -> str:
    m = re.match(r"run_\d{8}_\d{6}_(.+)$", run_dir.name)
    return m.group(1) if m else run_dir.name


def default_run_dirs(run_root: Path, family: str = "lp") -> list[Path]:
    """Newest run per model among the largest (by n_runs) of that family."""
    best: dict[str, tuple[int, str, Path]] = {}
    for d in sorted(run_root.glob(f"run_*-vf-{family}")):
        exps = sorted((d / "experiments").glob("*/config.json"))
        if not exps:
            continue
        n = json.loads(exps[0].read_text()).get("n_runs", 0)
        key = _model_of(d)
        cand = (int(n), d.name, d)
        if key not in best or cand[:2] > best[key][:2]:
            best[key] = cand
    return [v[2] for v in sorted(best.values(), key=lambda v: v[1])]


def experiment_dirs(run_dir: Path) -> list[Path]:
    return sorted(p.parent for p in (run_dir / "experiments").glob("*/config.json"))


# ------------------------------------------------------------- per-scenario
def load_matrix(exp_dir: Path, metric: str):
    """(run_ids, matrix[run, step] forward-filled, n_active[step], max_steps, capped)."""
    cfg = json.loads((exp_dir / "config.json").read_text())
    max_steps = int(cfg["max_steps"])
    path = run_files.metrics_history_path(exp_dir)
    dtypes = {"run_id": np.int32, "step": np.int32, metric: np.float32}
    df = run_files.read_metrics_file(path, usecols=["run_id", "step", metric],
                                     dtype=dtypes).astype(dtypes)
    run_ids = np.sort(df["run_id"].unique())
    pos = np.searchsorted(run_ids, df["run_id"].to_numpy())
    mat = np.full((len(run_ids), max_steps), np.nan, dtype=np.float32)
    steps = df["step"].to_numpy()
    keep = steps < max_steps
    mat[pos[keep], steps[keep]] = df[metric].to_numpy()[keep]
    n_active = (~np.isnan(mat)).sum(axis=0)
    last = (~np.isnan(mat)).cumsum(axis=1)          # forward fill along steps
    idx = np.where(~np.isnan(mat), np.arange(max_steps), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    mat = np.take_along_axis(mat, idx, axis=1)
    mat[last == 0] = np.nan
    summ = exp_dir / "run_summary.csv"
    capped = np.nan
    if summ.exists():
        rs = pd.read_csv(summ, usecols=["stop_reason"])
        capped = float((rs["stop_reason"] == "max_steps").mean())
    return run_ids, mat, n_active, max_steps, capped, cfg.get("scenario", exp_dir.name)


def absorption_stats(exp_dir: Path) -> dict:
    """When the runs that met the stop rule stopped: the step the LAST of them
    stopped (max) and the 99th percentile, from run_summary.csv. A run that
    stops is frozen, so any cap above `absorbed_max_step` cannot change it."""
    summ = exp_dir / "run_summary.csv"
    if not summ.exists():
        return {}
    rs = pd.read_csv(summ, usecols=["stop_reason", "final_step"])
    ab = pd.to_numeric(rs.loc[rs["stop_reason"] != "max_steps", "final_step"],
                       errors="coerce").dropna()
    return {"absorbed_frac": float(len(ab) / len(rs)),
            "absorbed_max_step": int(ab.max()) if len(ab) else None,
            "absorbed_p99_step": float(ab.quantile(0.99)) if len(ab) else None}


def equivalence_table(mat: np.ndarray, block: int, window: int, delta: float, scen: str):
    """Paired change over every WINDOW-step span starting on a block boundary.

    For s = 0, block, 2*block, ... with s + window <= T-1: per run
    d = metric[s + window] - metric[s]; its mean, standard error (sd/sqrt(n))
    and 95% confidence interval mean +- 1.96 SE. `within` says the WHOLE
    interval lies inside [-delta, +delta]: the data rule out (at 95%) a net
    change larger than delta over that window -- an equivalence test, the right
    way round for "is it flat?". A small t only fails to show change; it does
    not show flatness, and with few runs a real drift hides in a wide interval.
    A fixed window (not "s to the last step") keeps the test equally strict at
    every s: a span that shrinks toward the end would let any slow creep pass.
    A fully absorbed scenario has d == 0 for every run (frozen grids).
    """
    T = mat.shape[1]
    rows = []
    for s in range(0, T - window, block):
        e = s + window
        d = mat[:, e] - mat[:, s]
        d = d[~np.isnan(d)]
        m = float(d.mean())
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
        lo, hi = m - 1.96 * se, m + 1.96 * se
        rows.append({"scenario": scen, "from_step": s, "to_step": e,
                     "change": m, "se": se, "ci95_lo": lo, "ci95_hi": hi,
                     "within_delta": bool(lo >= -delta and hi <= delta)})
    last = T - 1 - window                       # always include the final window
    if last >= 0 and (not rows or rows[-1]["from_step"] != last):
        d = mat[:, T - 1] - mat[:, last]
        d = d[~np.isnan(d)]
        m = float(d.mean()); se = float(d.std(ddof=1) / np.sqrt(len(d)))
        rows.append({"scenario": scen, "from_step": last, "to_step": T - 1,
                     "change": m, "se": se, "ci95_lo": m - 1.96 * se,
                     "ci95_hi": m + 1.96 * se,
                     "within_delta": bool(abs(m) + 1.96 * se <= delta)})
    return pd.DataFrame(rows)


def settled_from(eq: pd.DataFrame):
    """Earliest window start s such that s and every later window pass the
    equivalence test; None when the final window fails (still moving, or too
    few runs to bound the change within delta)."""
    eq = eq.sort_values("from_step", ascending=False)
    ok = eq["within_delta"].to_numpy()
    if not len(ok) or not ok[0]:
        return None
    k = int(np.argmin(ok)) if not ok.all() else len(ok)
    return int(eq["from_step"].to_numpy()[k - 1])


def process_experiment(args):
    exp_dir, metric, checkpoints, delta, block, window = args
    run_ids, mat, n_active, T, capped, scen = load_matrix(Path(exp_dir), metric)
    mean = np.nanmean(mat, axis=0)
    sd = np.nanstd(mat, axis=0, ddof=1)
    n = (~np.isnan(mat)).sum(axis=0)
    row = {"scenario": scen, "n_runs": len(run_ids), "max_steps": T,
           "capped_frac": capped, "mean_final": float(mean[-1]),
           "sd_final": float(sd[-1])}
    eq = equivalence_table(mat, block, window, delta, scen)
    row[f"settled_from_w{window}_d{delta:g}"] = settled_from(eq)
    last = eq.sort_values("from_step").iloc[-1]
    row["final_window"] = f"{int(last['from_step'])}-{int(last['to_step'])}"
    row["final_window_change"] = last["change"]
    row["final_window_ci95"] = f"[{last['ci95_lo']:+.4f}, {last['ci95_hi']:+.4f}]"
    row.update(absorption_stats(Path(exp_dir)))
    for c in checkpoints:
        if c < T:
            row[f"mean@{c}"] = float(mean[c])
    for w in WINDOWS:
        if w < T:
            d = mat[:, -1] - mat[:, -1 - w]
            d = d[~np.isnan(d)]
            row[f"drift_last{w}"] = float(d.mean())
            row[f"drift_last{w}_se"] = float(d.std(ddof=1) / np.sqrt(len(d)))
    # paired per-run increments over consecutive blocks of `block` steps,
    # aligned to END at the last step: the drift and its SE, block by block
    blocks = []
    for b in range(T - 1, block - 1, -block)[::-1]:
        d = mat[:, b] - mat[:, b - block]
        d = d[~np.isnan(d)]
        se = d.std(ddof=1) / np.sqrt(len(d))
        blocks.append({"scenario": scen, "from_step": b - block, "to_step": b,
                       "increment": float(d.mean()), "se": float(se),
                       "t": float(d.mean() / se) if se > 0 else 0.0})
    blocks = pd.DataFrame(blocks)
    curves = pd.DataFrame({"scenario": scen, "step": np.arange(T),
                           "mean": mean, "ci95": 1.96 * sd / np.sqrt(np.maximum(n, 1)),
                           "n_active": n_active})
    # keep the per-run matrix at a few steps for the paired gap analysis
    keep_steps = sorted({s for s in list(checkpoints) + [T - 1 - w for w in WINDOWS] + [T - 1]
                         if 0 <= s < T})
    snaps = pd.DataFrame(mat[:, keep_steps], index=run_ids, columns=keep_steps)
    return row, curves, snaps, blocks, eq


# ---------------------------------------------------------------- per-model
def analyse_model(model, results, gap_window):
    rows = [r[0] for r in results]
    snaps = {r[0]["scenario"]: r[2] for r in results}
    T = max(r["max_steps"] for r in rows)
    order = sorted(rows, key=lambda r: -r["mean_final"])
    gaps = []
    for hi, lo in zip(order, order[1:]):
        a, b = snaps[hi["scenario"]], snaps[lo["scenario"]]
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        last = T - 1
        prev = last - gap_window
        g = (a[last] - b[last]).to_numpy()
        g_prev = (a[prev] - b[prev]).to_numpy() if prev in a.columns else None
        rec = {"model": model, "higher": hi["scenario"], "lower": lo["scenario"],
               "n_paired": len(common), "gap_final": float(g.mean()),
               "gap_final_se": float(g.std(ddof=1) / np.sqrt(len(g))),
               "gap_t": float(g.mean() / (g.std(ddof=1) / np.sqrt(len(g))))}
        if g_prev is not None:
            dg = g - g_prev
            slope = dg.mean() / gap_window
            rec[f"gap_change_last{gap_window}"] = float(dg.mean())
            rec[f"gap_change_last{gap_window}_se"] = float(dg.std(ddof=1) / np.sqrt(len(dg)))
            rec["extrapolated_close_step"] = (float(last + g.mean() / -slope)
                                              if slope < 0 else np.inf)
        gaps.append(rec)
    ords = []
    for c in sorted(next(iter(snaps.values())).columns):
        means = {sc: float(s[c].mean()) for sc, s in snaps.items()}
        ords.append({"model": model, "step": int(c),
                     "ordering": " > ".join(sorted(means, key=lambda k: -means[k]))})
    return gaps, ords


def _scenario_style(scenarios):
    """The paper's scenario colours and short labels (one definition, in
    plot_horizon_combined, copied from analysis_tools/paper/), so a scenario
    has the same colour here as in the paper figures."""
    sys.path.insert(0, str(ROOT / "analysis_tools"))
    from plot_horizon_combined import SCENARIO_COLORS, SCENARIO_LABELS, canon
    import matplotlib.pyplot as plt
    tab = plt.get_cmap("tab10")
    colors = {sc: SCENARIO_COLORS.get(canon(sc), tab(k % 10)) for k, sc in enumerate(scenarios)}
    labels = {sc: SCENARIO_LABELS.get(canon(sc), sc) for sc in scenarios}
    return colors, labels


def plot(curves: pd.DataFrame, rows: pd.DataFrame, metric: str, out: Path,
         zoom: int = 500, old_cap: int = 1000):
    """Mean trajectory per model: full range (left) and the last `zoom` steps
    (right), linear step axis; a strip underneath gives the share of runs
    still moving.

    A run that meets the stop rule (NO_MOVE_THRESHOLD steps without a move) is
    frozen; its last value is carried forward so the mean stays over ALL runs
    (dropping it would make the mean drift from attrition alone). Each line is
    SOLID while at least one run is still moving and DASHED once EVERY run has
    stopped -- from there on the mean is pure carry-forward with no new data.
    The strip underneath gives the share of runs still moving at each step.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(dict.fromkeys(curves["model"]))
    scen_all = sorted(curves["scenario"].unique())
    colors, labels = _scenario_style(scen_all)
    n_runs = {(r.model, r.scenario): int(r.n_runs) for r in rows.itertuples()}
    H = 5.6 * len(models)
    fig = plt.figure(figsize=(18, H))
    outer = fig.add_gridspec(len(models), 1, hspace=0.35, top=1 - 0.6 / H,
                             bottom=0.4 / H, left=0.05, right=0.78)
    for i, m in enumerate(models):
        g = outer[i].subgridspec(2, 2, height_ratios=(4, 1), hspace=0.08,
                                 width_ratios=(1.15, 1), wspace=0.18)
        cm = curves[curves["model"] == m]
        T = int(cm["step"].max()) + 1
        lo_zoom = max(0, T - zoom)
        for j, lo in enumerate((0, lo_zoom)):
            ax = fig.add_subplot(g[0, j])
            sx = fig.add_subplot(g[1, j], sharex=ax)
            for sc in scen_all:
                c = cm[(cm["scenario"] == sc) & (cm["step"] >= lo)]
                if c.empty:
                    continue
                n = n_runs[(m, sc)]
                live = c["n_active"].to_numpy() > 0
                k = int(np.argmin(live)) if not live.all() else len(c)
                all_stop = cm.loc[(cm["scenario"] == sc) & (cm["n_active"] == 0), "step"]
                lab = labels[sc] + (f"  (dashed: all runs stopped by step {int(all_stop.min())})"
                                    if len(all_stop) else "")
                x, y = c["step"].to_numpy(), c["mean"].to_numpy()
                ax.fill_between(x, y - c["ci95"], y + c["ci95"], color=colors[sc],
                                alpha=0.18, lw=0)
                ax.plot(x[:k], y[:k], color=colors[sc], lw=1.6,
                        label=lab if j == 0 else None)
                if k < len(c):
                    k = max(k - 1, 0)
                    ax.plot(x[k:], y[k:], color=colors[sc], lw=1.6, ls=(0, (4, 2)),
                            label=None)
                sx.plot(x, 100 * c["n_active"] / n, color=colors[sc], lw=1.3)
            for a_ in (ax, sx):
                a_.grid(alpha=0.25, lw=0.6)
                a_.spines[["top", "right"]].set_visible(False)
                if lo < old_cap < T:
                    a_.axvline(old_cap, color="#555555", lw=0.9, ls=":")
            if lo < old_cap < T:
                ax.text(old_cap, 1.0, f" previous cap ({old_cap})", transform=ax.get_xaxis_transform(),
                        va="top", ha="left", fontsize=8, color="#555555")
            ax.set_xlim(lo, T - 1)
            ax.tick_params(labelbottom=False)
            ax.set_ylabel(f"mean {metric.replace('_', ' ')}")
            ax.set_title(f"{m}: steps 0-{T - 1}" if j == 0 else
                         f"{m}: last {T - lo} steps (zoom)", fontsize=10, loc="left")
            sx.set_ylim(-5, 105)
            sx.set_ylabel("% runs\nstill moving", fontsize=8)
            sx.set_xlabel("step")
            if j == 1:      # legend outside, right of the zoom panel: never on a curve
                h = [l for l in fig.axes[-4].get_lines() if not l.get_label().startswith("_")]
                ax.legend(h, [l.get_label() for l in h], fontsize=8.5, frameon=False,
                          loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(f"Mean {metric.replace('_', ' ')} per step over all runs "
                 "(band = 95% CI; stopped runs keep their frozen value; dashed = every run stopped)",
                 fontsize=11, y=1 - 0.15 / H, x=0.42)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dirs", nargs="*", type=Path)
    ap.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    ap.add_argument("--family", default="lp")
    ap.add_argument("--metric", default="dissimilarity_index")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--checkpoints", default="100,250,500,750,999,1999,2999,4999,9999")
    ap.add_argument("--delta", type=float, default=0.01,
                    help="equivalence margin: 'settled from s' means the 95%% CI of the "
                         "paired change over each --window-step span lies within +-delta "
                         "(default 0.01 DI)")
    ap.add_argument("--window", type=int, default=500,
                    help="span of the equivalence test windows (default 500 steps)")
    ap.add_argument("--gap-window", type=int, default=250)
    ap.add_argument("--block", type=int, default=250,
                    help="block length for the paired increment table (default 250)")
    ap.add_argument("--only-capped", type=float, default=None,
                    help="plot only models with a scenario capped above this fraction")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)

    run_dirs = [Path(d).resolve() for d in a.run_dirs] or default_run_dirs(a.run_root, a.family)
    checkpoints = [int(x) for x in a.checkpoints.split(",")]
    jobs, owners = [], []
    for rd in run_dirs:
        for ed in experiment_dirs(rd):
            jobs.append((str(ed), a.metric, checkpoints, a.delta, a.block, a.window))
            owners.append(_model_of(rd))
    print(f"[horizon] {len(run_dirs)} runs, {len(jobs)} experiments, metric={a.metric}")
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(process_experiment, jobs))

    a.out_dir.mkdir(parents=True, exist_ok=True)
    by_model: dict[str, list] = {}
    for owner, res in zip(owners, results):
        by_model.setdefault(owner, []).append(res)
    rows, curves, gaps, ords, blocks, eqs = [], [], [], [], [], []
    for model, res in by_model.items():
        for r, c, _, b, e in res:
            rows.append({"model": model, **r})
            curves.append(c.assign(model=model))
            blocks.append(b.assign(model=model))
            eqs.append(e.assign(model=model))
        g, o = analyse_model(model, res, a.gap_window)
        gaps += g
        ords += o
    rows = pd.DataFrame(rows)
    curves = pd.concat(curves, ignore_index=True)
    rows.to_csv(a.out_dir / "horizon_scenarios.csv", index=False)
    pd.DataFrame(gaps).to_csv(a.out_dir / "horizon_gaps.csv", index=False)
    pd.DataFrame(ords).to_csv(a.out_dir / "horizon_orderings.csv", index=False)
    blocks = pd.concat(blocks, ignore_index=True)
    blocks.to_csv(a.out_dir / "horizon_blocks.csv", index=False)
    pd.concat(eqs, ignore_index=True).to_csv(a.out_dir / "horizon_equivalence.csv", index=False)
    curves.to_csv(a.out_dir / "horizon_mean_curves.csv.gz", index=False)
    plot_models = rows["model"].unique() if a.only_capped is None else \
        rows.loc[rows["capped_frac"] > a.only_capped, "model"].unique()
    if len(plot_models):
        plot(curves[curves["model"].isin(plot_models)], rows, a.metric,
             a.out_dir / f"horizon_{a.metric}.png")
    pd.set_option("display.width", 250)
    cols = ["model", "scenario", "capped_frac", "mean_final",
            f"settled_from_w{a.window}_d{a.delta:g}", "final_window", "final_window_change",
            "final_window_ci95",
            "absorbed_frac", "absorbed_max_step", "absorbed_p99_step"]
    print(rows[cols].round(4).to_string(index=False))
    print(pd.DataFrame(gaps).round(4).to_string(index=False))
    capped_models = rows.loc[rows["capped_frac"] > 0.05, "model"].unique()
    bl = blocks[blocks["model"].isin(capped_models)].copy()
    if len(bl):
        bl["cell"] = bl.apply(lambda r: f"{r.increment:+.4f}±{r.se:.4f}", axis=1)
        print(bl.pivot_table(index=["model", "scenario"], columns="to_step",
                             values="cell", aggfunc="first").to_string())
    print(f"[horizon] wrote {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
