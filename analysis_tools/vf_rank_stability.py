#!/usr/bin/env python3
"""Rank-stability classification: THE criterion for whether a value function
needs more sampling (user decision 2026-09-03).

    python analysis_tools/vf_rank_stability.py --label <label> \
        --multisplit <dir with vf_multisplit_check.csv + vf_multisplit_deltas.csv> \
        [--production-experiments <run_dir>/experiments] [--out-dir <dir>]

WHAT REPLACED WHAT

The absolute sufficiency check (vf_multisplit_check.py) asks "is the table's
sampling noise small compared to run noise?" — a uniform precision standard.
The scientific claims, however, are ORDINAL: which scenario segregates more
than which. So the binding question is whether table noise can scramble the
scenario ordering, pair by pair. The multi-split stays as the INSTRUMENT (its
32 half-table redraws measure the noise); this script issues the verdict, and
the absolute ratio is demoted to reporting (error-bar inflation on values).

THE TWO QUANTITIES, AND WHO REDUCES WHAT

For one metric and two scenarios A > B (by the full table's means):

  gap g      = mean_A - mean_B            <- a DISTANCE between simulation
               outcomes. Estimated from n runs with SE_g = sqrt(tau_A^2+tau_B^2),
               tau = SD(finals)/sqrt(n). MORE RUNS (CPU) shrink this.
  ruler s    = SD over the B half-table redraws of the pair's displacement
               (from vf_multisplit_deltas.csv)  <- the VALUE FUNCTION's
               epistemic noise projected onto this pair. Only MORE LLM SAMPLES
               (GPU) shrink this. s is an UPPER bound on pure table noise: the
               redraw arms re-roll some run noise too (same seeds, but a
               perturbed table sends chaotic trajectories down new paths), so
               every classification below errs conservative.

Top-up is required iff a claims-relevant distance is (a) real, (b) measured,
and (c) smaller than the ruler is thick. That gives five terminal states:

  EXACT-TIE   |g| == 0 to double precision (frozen tables produce literally
              identical scenario outcomes). No ranking exists; report the
              scenarios as a tied group. Terminal.
  CERTIFIED   g - z*SE_g > z*s : even the sceptical gap beats even the
              un-decomposed ruler. Order citable as-is. Terminal.
  TIE         (g + z*SE_g)/z < tie_mult * s : even the OPTIMISTIC gap would
              need the ruler tightened below tie_mult (default 0.15, i.e.
              >44x more samples, n ~ 1/w^2). The scientific result is
              "indistinguishable at feasible precision". Terminal — GPU must
              NOT chase these; that is the infinite-cost trap.
  UNMEASURED  |g| < z*SE_g : the gap's own SIGN is inside run noise. No
              verdict is possible yet — and no GPU decision either, because a
              tie and a small real gap look identical. The fix is CPU: more
              runs (the n=10,000 batch shrinks SE_g 10x), then re-classify.
  FIXABLE     the rest: gap real and measured, ruler comparably thick.
              w_pair = w_cur * (g/z)/s tightens ONLY the two scenarios'
              tables until g >= z*s' — priced per pair via the same
              project_total loop the calibrator spends (vf_sampling_plan).

DECISION METRIC: DI ONLY (user decision 2026-09-04)

The dissimilarity index is the paper's headline metric, so it ALONE drives the
verdict, the exit code, and therefore any GPU spend. Every other metric is
still classified and written out — to rank_pairs.csv and to a per-model note
file, RANK_STABILITY_NOTES.md — but purely as reference: an unstable ordering
in switch_rate or mix_deviation never triggers a top-up.

This matters because the two scopes differ by ~7x. Across the eight models
measured on 2026-09-04, acting on all seven metrics would have demanded
1,957 GPU-h; acting on DI alone demands 284, and 4 of the 8 models are then
settled for <= 6 GPU-h each. Widen --decision-metrics only if a second metric
starts carrying claims.

LOOP CONNECTION (supersedes the absolute-margin trigger)

  production batch (n runs)  ->  gaps measured        [CPU]
  multi-split (B splits)     ->  rulers measured      [CPU]
  THIS SCRIPT                ->  classify all metrics, DECIDE on DI
      DI has FIXABLE     -> exit 4: targeted top-up list + prices
      DI has UNMEASURED  -> exit 6: run more sims first (never buy GPU blind)
      else               -> exit 0: DI ordering certified or an honest tie

Exit codes: 0 settled, 4 fixable pairs exist, 6 only unmeasured gaps remain,
2 inputs missing. 4 beats 6 when both occur (there is actionable GPU work).
All three are judged on the decision metrics only.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
for _p in (_THIS, _THIS.parent, _THIS.parent / "prompt_refinement"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from vf_simulation_evaluation import ALL_METRICS, SCENARIO_ORDER  # noqa: E402

Z = 1.96


# ---------------------------------------------------------------------------
# Gap sources
# ---------------------------------------------------------------------------

def gaps_from_check(ms_dir):
    """Reference mode: means/taus straight from the multi-split's full arm
    (n = the check's --runs, normally 100). SE_g is large here — expect most
    small gaps to land UNMEASURED."""
    chk = pd.read_csv(Path(ms_dir) / "vf_multisplit_check.csv")
    n_runs = None
    st = Path(ms_dir) / "multisplit_status.json"
    if st.exists():
        n_runs = json.loads(st.read_text()).get("runs")
    out = {}
    for m in chk.metric.unique():
        c = chk[chk.metric == m].set_index("scenario")
        out[m] = (c.full_mean, c.tau_full_mean_se)
    return out, n_runs


def gaps_from_production(exp_root):
    """Production mode: per-run FINALS from each scenario's run_summary.csv.

    run_summary.csv is written by Simulation.analyze_results and already holds
    one row per run with every metric at final_step, including the
    dissimilarity index. Reading it beats re-deriving from metrics_history:
    that file is stored GZIPPED (metrics_history.csv.gz) and DI would need a
    final-grid load per run — 60,000 npz opens per model at n=10,000, for
    numbers already tabulated. `metrics_source` flags any row whose metrics
    were recomputed rather than recorded live.

    tau = SD/sqrt(n) uses the ACTUAL run count, so a 10,000-run batch shrinks
    SE_g by 10x against the multi-split's 100-run reference — which is the
    whole point of running it before classifying.
    """
    mu_by, se_by, n_by = {}, {}, {}
    for sc in SCENARIO_ORDER:
        dirs = sorted(glob.glob(str(Path(exp_root) / f"llm_{sc}_*")))
        if not dirs:
            continue
        rs = Path(dirs[-1]) / "run_summary.csv"
        if not rs.exists():
            print(f"  [skip] {sc}: no run_summary.csv (batch still running?)")
            continue
        df = pd.read_csv(rs)
        for m in ALL_METRICS:
            if m not in df.columns:
                continue
            vals = df[m].to_numpy(float)
            vals = vals[~np.isnan(vals)]
            if len(vals) < 2:
                continue
            mu_by.setdefault(m, {})[sc] = float(np.mean(vals))
            se_by.setdefault(m, {})[sc] = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            n_by[sc] = len(vals)
    return ({m: (pd.Series(mu_by[m]), pd.Series(se_by[m])) for m in mu_by}, n_by)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(label, style, ms_dir, gaps, tie_mult, w_cur, price):
    dl = pd.read_csv(Path(ms_dir) / "vf_multisplit_deltas.csv")
    rows = []
    for m, (mu, se) in gaps.items():
        piv = dl[dl.metric == m].pivot(index="split", columns="scenario",
                                       values="delta")
        common = [s for s in mu.index if s in piv.columns]
        order = list(mu[common].sort_values(ascending=False).index)
        for a, b in zip(order[:-1], order[1:]):
            g = float(mu[a] - mu[b])
            se_g = float(np.sqrt(se[a] ** 2 + se[b] ** 2))
            noise = piv[a] - piv[b]           # pair displacement per redraw
            s_obs = float(np.std(noise.dropna(), ddof=1))
            flips = float((noise.dropna() > g).mean())   # redraw reverses order
            if g < 1e-9:
                cls, w_pair = "EXACT-TIE", None
            elif s_obs <= 0:
                cls, w_pair = ("CERTIFIED" if g > Z * se_g else "UNMEASURED"), None
            elif g - Z * se_g > Z * s_obs:
                cls, w_pair = "CERTIFIED", None
            elif (g + Z * se_g) / Z < tie_mult * s_obs:
                cls, w_pair = "TIE", None
            elif g < Z * se_g:
                cls, w_pair = "UNMEASURED", None
            else:
                mult = (g / Z) / s_obs
                cls = "FIXABLE"
                w_pair = round(min(w_cur, w_cur * mult), 6)
            rows.append({"metric": m, "hi": a, "lo": b, "gap": g, "se_gap": se_g,
                         "gap_over_se": g / se_g if se_g > 0 else np.inf,
                         "s_obs": s_obs, "flip_frac": flips,
                         "class": cls, "w_pair": w_pair})
    tab = pd.DataFrame(rows)
    if price and (tab["class"] == "FIXABLE").any():
        from vf_sampling_plan import deficit_at, RATES
        rate = next((v for k, v in RATES.items() if k in label), 8000)
        cost = []
        # iterrows, not itertuples: 'class' is a Python keyword, so itertuples
        # silently renames the field and _asdict() loses it.
        for _, r in tab.iterrows():
            if r["class"] != "FIXABLE":
                cost.append(None); continue
            smp, _ = deficit_at(label, style, r["w_pair"],
                                scenarios=[r["hi"], r["lo"]])
            cost.append(round(smp / rate, 2))
        tab["gpu_h"] = cost
    return tab


OPS = {"CERTIFIED": " > ", "FIXABLE": " ?> ", "UNMEASURED": " ~ ",
       "TIE": " ~ ", "EXACT-TIE": " = "}
SHORT = {"baseline": "baseline", "race_white_black": "race",
         "ethnic_asian_hispanic": "ethnic", "income_high_low": "income",
         "political_liberal_conservative": "politics", "green_yellow": "green"}


def write_notes(path, label, tab, decision, src, ms_dir):
    """RANK_STABILITY_NOTES.md — the reference record for EVERY metric.

    The verdict is DI-only, so the other six metrics would otherwise vanish
    from view. They are still worth reading (an ordering that is unstable
    everywhere hints at a scenario pair that simply does not separate), so
    they are written here as rendered rankings plus a per-metric tally,
    clearly marked as non-acting.
    """
    L = [f"# Rank stability — {label}", "",
         f"- gaps: {src}", f"- ruler: {ms_dir}",
         f"- decision metric(s): **{', '.join(decision)}** — only these trigger a top-up",
         "", "Legend: `>` certified · `?>` fixable (top-up resolves) · "
         "`~` unresolved · `=` exact tie", ""]
    for m in [x for x in decision] + [x for x in ALL_METRICS if x not in decision]:
        d = tab[tab.metric == m].reset_index(drop=True)
        if d.empty:
            continue
        chain = [SHORT.get(d.loc[0, "hi"], d.loc[0, "hi"])]
        for i in range(len(d)):
            chain += [OPS[d.loc[i, "class"]], SHORT.get(d.loc[i, "lo"], d.loc[i, "lo"])]
        tag = "  **(DECISION METRIC)**" if m in decision else ""
        cost = d[d["class"] == "FIXABLE"]["gpu_h"].sum() if "gpu_h" in d else 0
        L += [f"## {m}{tag}", "", "```", "".join(chain), "```",
              f"counts: {d['class'].value_counts().to_dict()}"
              + (f" · targeted top-up {cost:.1f} GPU-h" if cost else ""), ""]
        fx = d[d["class"] == "FIXABLE"]
        if len(fx):
            L.append("| pair | gap | gap/SE | ruler | w_pair | GPU-h |")
            L.append("|---|---|---|---|---|---|")
            for _, r in fx.iterrows():
                L.append(f"| {SHORT.get(r['hi'],r['hi'])} > {SHORT.get(r['lo'],r['lo'])} "
                         f"| {r['gap']:.5g} | {r['gap_over_se']:.2f} | {r['s_obs']:.5g} "
                         f"| {r['w_pair']} | {r.get('gpu_h', float('nan')):.1f} |")
            L.append("")
    Path(path).write_text("\n".join(L))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--style", default="R3_dual_count")
    ap.add_argument("--multisplit", required=True,
                    help="dir holding vf_multisplit_check.csv + deltas (the ruler)")
    ap.add_argument("--production-experiments", default=None,
                    help="run_dir/experiments with the big-n batch (the gaps); "
                         "omitted -> gaps from the check's own 100-run arm")
    ap.add_argument("--w-current", type=float, default=0.02)
    ap.add_argument("--tie-mult", type=float, default=0.15,
                    help="below this required tightening factor a pair is an "
                         "honest tie (0.15 -> >44x samples); terminal")
    ap.add_argument("--decision-metrics", nargs="*", default=["dissimilarity_index"],
                    help="metrics whose ordering drives the verdict, exit code "
                         "and any GPU spend (default: dissimilarity_index, the "
                         "paper's headline). Every other metric is still "
                         "classified into rank_pairs.csv and the notes file, "
                         "but never triggers a top-up.")
    ap.add_argument("--no-pricing", action="store_true")
    ap.add_argument("--out-dir", default=None,
                    help="default: <multisplit dir>/../rank_stability")
    args = ap.parse_args()

    ms = Path(args.multisplit)
    if not (ms / "vf_multisplit_deltas.csv").exists():
        print(f"missing {ms}/vf_multisplit_deltas.csv"); return 2
    if args.production_experiments:
        gaps, n_runs = gaps_from_production(args.production_experiments)
        if not gaps:
            print(f"no run_summary.csv found under {args.production_experiments}")
            return 2
        src = (f"production run_summary finals ({args.production_experiments}, "
               f"n={sorted(set(n_runs.values()))})")
    else:
        gaps, n_runs = gaps_from_check(ms)
        src = f"multi-split full arm (n={n_runs})"
    print(f"label={args.label}  gaps from: {src}\nruler from: {ms}")

    tab = classify(args.label, args.style, ms, gaps, args.tie_mult,
                   args.w_current, price=not args.no_pricing)
    out = Path(args.out_dir or (ms.parent / "rank_stability"))
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "rank_pairs.csv", index=False)

    write_notes(out / "RANK_STABILITY_NOTES.md", args.label, tab,
                args.decision_metrics, src, str(ms))

    priced = "gpu_h" in tab.columns
    def block(sub):
        fx = sub[sub["class"] == "FIXABLE"]
        return {
            "counts": sub["class"].value_counts().to_dict(),
            "n_pairs": int(len(sub)),
            "fixable": [{k: r[k] for k in ("metric", "hi", "lo", "gap", "w_pair")}
                        | ({"gpu_h": r["gpu_h"]} if priced else {})
                        for _, r in fx.iterrows()],
            "total_fixable_gpu_h": (float(fx.gpu_h.sum()) if priced and len(fx) else 0.0),
            "verdict": ("FIXABLE" if len(fx) else
                        "UNMEASURED" if (sub["class"] == "UNMEASURED").any() else
                        "SETTLED"),
        }

    dec = tab[tab.metric.isin(args.decision_metrics)]
    if dec.empty:
        print(f"decision metric(s) {args.decision_metrics} absent from the data")
        return 2
    # The DECISION block drives everything; all_metrics is reference only.
    status = {"label": args.label, "gap_source": src, "ruler": str(ms),
              "tie_mult": args.tie_mult, "w_current": args.w_current,
              "decision_metrics": args.decision_metrics,
              "decision": block(dec), "all_metrics": block(tab)}
    status["verdict"] = status["decision"]["verdict"]
    status["total_fixable_gpu_h"] = status["decision"]["total_fixable_gpu_h"]
    (out / "rank_status.json").write_text(json.dumps(status, indent=1))

    with pd.option_context("display.width", 170):
        show = dec[dec["class"] != "CERTIFIED"]
        if len(show):
            print(f"\n=== {'/'.join(args.decision_metrics)}: pairs not certified ===")
            print(show.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    d, a = status["decision"], status["all_metrics"]
    print(f"\nDECISION ({'/'.join(args.decision_metrics)}): {d['counts']}")
    print(f"reference (all {len(ALL_METRICS)} metrics): {a['counts']}"
          + (f"  [would be {a['total_fixable_gpu_h']:.1f} GPU-h if acted on]"
             if a["total_fixable_gpu_h"] else ""))
    print(f"verdict: {status['verdict']}"
          + (f"  (top-up ~{d['total_fixable_gpu_h']:.1f} GPU-h)" if priced
             and status["verdict"] == "FIXABLE" else ""))
    print(f"wrote {out}/rank_pairs.csv, rank_status.json, RANK_STABILITY_NOTES.md")
    return 4 if status["verdict"] == "FIXABLE" else (
        6 if status["verdict"] == "UNMEASURED" else 0)


if __name__ == "__main__":
    sys.exit(main())
