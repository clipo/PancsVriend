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
               outcomes. Estimated from n runs, PAIRED: run k is seeded by
               run_id, so it starts from the same grid in every scenario and
               the per-run finals correlate (r ~ 0.5 for DI at n=10,000);
               SE_g = SD(A_k - B_k)/sqrt(n), smaller than the unpaired
               sqrt(tau_A^2+tau_B^2) by ~sqrt(1-r). Reference mode (no per-run
               finals) still uses the unpaired form. MORE RUNS (CPU) shrink this.
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
  FLOOR-TIE   |g| < gap_floor (user decision 2026-09-05: 0.01 DI). Below the
              practical-significance floor no ordering is CLAIMED, however
              well measured: on the 20x20/160+160 board one agent changing
              tract moves DI by 0.003-0.006, so 0.01 is ~two agents per run;
              it is 0.3 of the random-allocation SD (0.034); and it sits in
              the empty band (0.0082-0.0134) between the pairs that flip on
              re-measurement and those that never do. Checked BEFORE any
              certification or pricing, so a floor-tie is never a GPU quote —
              the 486 GPU-h qwen green>income (gap 0.0019) asked for is the
              case this exists to stop. Terminal.
              A pair is ALSO a floor-tie when both scenarios sit AT THE
              CHANCE LEVEL (user decision 2026-09-05): mean within the floor
              of the metric's random-allocation value for the run's board.
              Such a scenario does not segregate at all, so a difference
              between two of them is not an ordering. Marked ° in the notes,
              listed in rank_status.json ("chance" block), di_levels.csv and
              metric_levels.csv — the paper's level tables.
              THE OTHER SIX METRICS get the same two rules with their OWN
              scale (2026-09-05, "for completeness"): a Monte Carlo null —
              agents placed uniformly at random on the board, all seven
              metrics computed per draw — gives each metric's chance mean
              and SD, and its floor is floor_ratio x SD. The ratio is fixed
              by the DI decision: 0.01 / SD_DI(0.0336) = 0.30, so DI keeps
              exactly the user's floor and every other metric inherits the
              same fraction of its own chance spread. Polarity is honoured
              (clusters, switch_rate: higher = LESS segregation), so "at
              chance" always means "no more segregated than random". The
              null is cached per board in prompt_refinement/results/.
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
  RUNS        g > z*s already (the ruler is thick ENOUGH) but g - z*se_g is
              not: the gap's own measurement error is what blocks certification.
              More LLM samples cannot help — extend the production batch.
              Treated as UNMEASURED for the verdict: a CPU remedy, never GPU.
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
    per_run, n_by = {}, {}
    for sc in SCENARIO_ORDER:
        dirs = sorted(glob.glob(str(Path(exp_root) / f"llm_{sc}_*")))
        if not dirs:
            continue
        rs = Path(dirs[-1]) / "run_summary.csv"
        if not rs.exists():
            print(f"  [skip] {sc}: no run_summary.csv (batch still running?)")
            continue
        df = pd.read_csv(rs).set_index("run_id")
        for m in ALL_METRICS:
            if m in df.columns:
                per_run.setdefault(m, {})[sc] = df[m].astype(float)
        n_by[sc] = len(df)
    # PER-RUN series, not just means: llm_runner seeds every run with
    # random_seed=run_id, so run k of scenario A and run k of scenario B start
    # from the SAME initial grid. The scenario gap is therefore a PAIRED
    # comparison, and keeping the runs lets classify() use the paired SE.
    return ({m: pd.DataFrame(d) for m, d in per_run.items()}, n_by)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def ruler_meta(ms_dir):
    """multisplit_status.json of the ruler, or {} if the dir predates it."""
    st = Path(ms_dir) / "multisplit_status.json"
    try:
        return json.loads(st.read_text())
    except (OSError, ValueError):
        return {}


def check_board(meta, exp_root):
    """Refuse a ruler measured on a different board or step cap than the batch.

    The ruler's displacement scales with the board (a 10x10 ruler is ~2x
    optimistic against a 20x20 batch, run_vf_multisplit_backfill.sh header),
    so pairing them silently would misprice every FIXABLE pair. Compares the
    status file's board with the first scenario's config.json. Returns a list
    of mismatch strings (empty = comparable, or nothing to compare).
    """
    cfgs = sorted(glob.glob(str(Path(exp_root) / "llm_*" / "config.json")))
    if not cfgs:
        return []
    cfg = json.loads(Path(cfgs[0]).read_text())
    bad = []
    if not meta.get("is_sufficiency_check", True):
        bad.append(f"keep_fraction {meta.get('keep_fraction')}: a scaling diagnostic, "
                   "not a sufficiency ruler (use the half-split one)")
    for k in ("grid_size", "num_type_a", "num_type_b", "max_steps"):
        # A ruler that does not RECORD its board is not comparable either: the
        # unrecorded ones are precisely the pre-2026-09-01 10x10 rechecks.
        if meta.get(k) is None:
            bad.append(f"{k}: ruler records no value (status file predates the "
                       f"board fields; re-measure on the production board)")
        elif k in cfg and str(meta[k]) != str(cfg[k]):
            bad.append(f"{k}: ruler {meta[k]} vs batch {cfg[k]}")
    return bad


# Polarity: +1 = higher value means MORE segregation, -1 = LESS (clusters,
# switch_rate). Same table as cross_model_vf_comparison.SEGREGATION_DIRECTION.
DIRECTION = {"dissimilarity_index": +1, "clusters": -1, "switch_rate": -1,
             "distance": +1, "mix_deviation": +1, "share": +1, "ghetto_rate": +1}
NULL_CACHE_DIR = _THIS.parent / "prompt_refinement" / "results"


def metric_null(board, draws=20000, seed=0):
    """Chance mean/SD of all seven metrics on random-allocation grids.

    Agents placed uniformly at random on the board (no dynamics) — the same
    null DissimilarityIndex.random_baseline samples for DI, extended to the
    metrics that need a full grid. Cached per (board, draws, seed). 20,000
    draws (2026-09-05; was 2,000) put the Monte-Carlo standard error of each
    chance mean at 1/141 of its SD — for DI 0.00024, below the scenario
    means' own SE — so the null's sampling error no longer dominates a
    comparison against chance. ~10 s with the vectorised metrics.
    """
    size, na, nb = (int(board[k]) for k in ("grid_size", "num_type_a", "num_type_b"))
    cache = NULL_CACHE_DIR / f"metric_null_g{size}_a{na}_b{nb}_d{draws}_s{seed}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from types import SimpleNamespace
    from Metrics import calculate_all_metrics
    rng = np.random.default_rng(seed)
    cells = size * size
    agents = [SimpleNamespace(type_id=0)] * na + [SimpleNamespace(type_id=1)] * nb
    acc = {m: [] for m in ALL_METRICS}
    for _ in range(draws):
        flat = np.empty(cells, dtype=object)
        idx = rng.permutation(cells)[:na + nb]
        for i, ag in zip(idx, agents):
            flat[i] = ag
        vals = calculate_all_metrics(flat.reshape(size, size))
        for m in ALL_METRICS:
            acc[m].append(float(vals[m]))
    out = {"board": {"grid_size": size, "num_type_a": na, "num_type_b": nb},
           "draws": draws, "seed": seed,
           "metrics": {m: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
                       for m, v in acc.items()}}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, indent=1))
    return out


def chance_levels(gaps, board, gap_floor, floor_ratio=0.3, draws=20000):
    """Per-metric chance level, floor, and the scenarios sitting at chance.

    Returns None when the board is unknown; else
      {"board", "draws", "floor_ratio", "chance_di", "chance_sd", "level_floor",
       "levels", "se", "excess", "at_floor"   (DI; the 2026-09-05 schema),
       "metrics": {m: {"chance", "chance_sd", "floor", "direction",
                       "levels", "se", "excess", "at_floor"}}}
    excess = direction * (mean - chance): positive = more segregated than a
    random grid regardless of the metric's polarity. DI's floor is the user's
    gap_floor; every other metric's is floor_ratio * its chance SD.
    """
    if not all(board.get(k) for k in ("grid_size", "num_type_a", "num_type_b")):
        return None
    null = metric_null(board, draws=draws)
    per = {}
    for m, src in gaps.items():
        if m not in null["metrics"]:
            continue
        if isinstance(src, pd.DataFrame):          # production mode: per-run finals
            mu, se = src.mean(), src.std(ddof=1) / np.sqrt(src.count())
        else:                                      # reference mode: (means, SEs)
            mu, se = src
        c0, sd0 = null["metrics"][m]["mean"], null["metrics"][m]["sd"]
        if m == "dissimilarity_index":
            # exact null (multivariate-hypergeometric tract counts), so the
            # published chance DI stays DissimilarityIndex.random_baseline's
            from DissimilarityIndex import random_baseline
            rb = random_baseline(int(board["grid_size"]), int(board["num_type_a"]),
                                 int(board["num_type_b"]))
            c0, sd0 = float(rb["mean"]), float(rb["sd"])
        d = DIRECTION.get(m, +1)
        floor = gap_floor if m == "dissimilarity_index" else float(f"{floor_ratio * sd0:.3g}")
        excess = d * (mu - c0)
        per[m] = {"chance": c0, "chance_sd": sd0, "floor": floor, "direction": d,
                  "levels": {sc: float(v) for sc, v in mu.items()},
                  "se": {sc: float(v) for sc, v in se.items()},
                  "excess": {sc: float(v) for sc, v in excess.items()},
                  "at_floor": sorted(sc for sc, v in excess.items() if v < floor)}
    if "dissimilarity_index" not in per:
        return None
    di = per["dissimilarity_index"]
    return {"board": null["board"], "draws": draws, "floor_ratio": floor_ratio,
            "chance_di": di["chance"], "chance_sd": di["chance_sd"],
            "level_floor": di["floor"], "levels": di["levels"], "se": di["se"],
            "excess": di["excess"], "at_floor": di["at_floor"], "metrics": per}


def production_board(exp_root):
    """Board of the batch, from the first scenario's config.json."""
    cfgs = sorted(glob.glob(str(Path(exp_root) / "llm_*" / "config.json")))
    if not cfgs:
        return {}
    cfg = json.loads(Path(cfgs[0]).read_text())
    return {k: cfg.get(k) for k in ("grid_size", "num_type_a", "num_type_b")}


def classify(label, style, ms_dir, gaps, tie_mult, w_cur, price, gap_floor=0.01,
             chance=None):
    """ms_dir=None is EXACT mode (2026-09-05): the tables carry no sampling
    error (logprob-derived value functions), so the ruler is identically 0
    and every pair is decided by run noise, the gap floor and the chance rule
    alone (CERTIFIED / UNMEASURED / FLOOR-TIE / EXACT-TIE; never FIXABLE)."""
    dl = pd.read_csv(Path(ms_dir) / "vf_multisplit_deltas.csv") if ms_dir else None
    rows = []
    for m, gap_src in gaps.items():
        # Two input shapes. A DataFrame of per-run values supports the PAIRED
        # standard error; a (means, SEs) tuple (reference mode, from the
        # check's own arm) only supports the unpaired one.
        paired = isinstance(gap_src, pd.DataFrame)
        if paired:
            mu, se = gap_src.mean(), gap_src.std(ddof=1) / np.sqrt(gap_src.count())
        else:
            mu, se = gap_src
        piv = (dl[dl.metric == m].pivot(index="split", columns="scenario", values="delta")
               if dl is not None else None)
        common = [s for s in mu.index if piv is None or s in piv.columns]
        # Polarity (2026-09-05): clusters and switch_rate run the other way
        # (higher = LESS segregation). Sorting on d*mean makes every chain
        # read most-segregated first, and g / the per-run differences / the
        # ruler displacements are sign-corrected the same way, so 'hi' is
        # always the MORE segregated scenario.
        d = DIRECTION.get(m, +1)
        order = list((d * mu[common]).sort_values(ascending=False).index)
        cm = (chance or {}).get("metrics", {}).get(m, {})
        floor_m = cm.get("floor", gap_floor if m == "dissimilarity_index" else 0.0)
        at_floor = set(cm.get("at_floor", []))
        for a, b in zip(order[:-1], order[1:]):
            g = float(d * (mu[a] - mu[b]))
            if paired:
                # PAIRED SE: run k shares an initial grid across scenarios
                # (random_seed=run_id), so Var(A-B) = Var(A)+Var(B)-2Cov(A,B)
                # and the covariance is real — measured 2026-09-04 at r=+0.47
                # to +0.97 on DI, shrinking SE_g by 23-82%. Using the unpaired
                # sqrt(tau_A^2+tau_B^2) discards that for free and pushes pairs
                # into UNMEASURED/RUNS that are in fact certifiable.
                dif = (d * (gap_src[a] - gap_src[b])).dropna()
                se_g = float(dif.std(ddof=1) / np.sqrt(len(dif))) if len(dif) > 1 else np.inf
            else:
                se_g = float(np.sqrt(se[a] ** 2 + se[b] ** 2))
            if piv is not None:
                noise = d * (piv[a] - piv[b])     # pair displacement per redraw
                s_obs = float(np.std(noise.dropna(), ddof=1))
                flips = float((noise.dropna() > g).mean())   # redraw reverses order
            else:
                s_obs, flips = 0.0, 0.0           # exact tables: no ruler
            # Order matters. Past CERTIFIED we know g - Z*se_g <= Z*s, and the
            # binding constraint is whichever term is too large — they have
            # DIFFERENT remedies, so they must not share a class:
            #   se_g too large -> the GAP is poorly measured  -> more RUNS (CPU)
            #   s   too large  -> the RULER is too thick      -> more SAMPLES (GPU)
            # Conflating them sends money at the wrong problem: a pair already
            # past g > Z*s needs no sampling at all, and w_pair would come back
            # >= w_cur (clamped to w_cur), quoting a bill that buys no ordering.
            reason = ""
            if g < 1e-9:
                cls, w_pair = "EXACT-TIE", None
            elif a in at_floor and b in at_floor:
                cls, w_pair, reason = "FLOOR-TIE", None, "both at chance level"
            elif g < floor_m:
                cls, w_pair, reason = "FLOOR-TIE", None, "gap below floor"
            elif s_obs <= 0:
                cls, w_pair = ("CERTIFIED" if g > Z * se_g else "UNMEASURED"), None
            elif g - Z * se_g > Z * s_obs:
                cls, w_pair = "CERTIFIED", None
            elif g < Z * se_g:
                cls, w_pair = "UNMEASURED", None          # sign not established
            elif g > Z * s_obs:
                cls, w_pair = "RUNS", None                # ruler fine; gap needs runs
            elif (g + Z * se_g) / Z < tie_mult * s_obs:
                cls, w_pair = "TIE", None
            else:
                mult = (g / Z) / s_obs
                cls = "FIXABLE"
                w_pair = round(min(w_cur, w_cur * mult), 6)
            rows.append({"metric": m, "hi": a, "lo": b, "gap": g, "se_gap": se_g,
                         "gap_over_se": g / se_g if se_g > 0 else np.inf,
                         "se_paired": bool(paired), "floor": floor_m,
                         "s_obs": s_obs, "flip_frac": flips,
                         "class": cls, "tie_reason": reason, "w_pair": w_pair})
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
       "RUNS": " n> ", "TIE": " ~ ", "EXACT-TIE": " = ", "FLOOR-TIE": " ≈ "}
SHORT = {"baseline": "baseline", "race_white_black": "race",
         "ethnic_asian_hispanic": "ethnic", "income_high_low": "income",
         "political_liberal_conservative": "politics", "green_yellow": "green"}


def write_notes(path, label, tab, decision, src, ms_dir, ruler_scheme="unknown",
                gap_floor=0.01, chance=None):
    """RANK_STABILITY_NOTES.md — the reference record for EVERY metric.

    The verdict is DI-only, so the other six metrics would otherwise vanish
    from view. They are still worth reading (an ordering that is unstable
    everywhere hints at a scenario pair that simply does not separate), so
    they are written here as rendered rankings plus a per-metric tally,
    clearly marked as non-acting.
    """
    L = [f"# Rank stability — {label}", "",
         f"- gaps: {src}", f"- ruler: {ms_dir}",
         f"- ruler rng_scheme: {ruler_scheme}",
         f"- decision metric(s): **{', '.join(decision)}** — only these trigger a top-up",
         f"- gap floor: {gap_floor} DI (pairs below it are `≈`, never priced)"]
    per = (chance or {}).get("metrics", {})
    if chance:
        b = chance["board"]
        L.append(f"- chance level on this board ({b['grid_size']}x{b['grid_size']}, "
                 f"{b['num_type_a']}+{b['num_type_b']}; {chance['draws']} random "
                 f"allocations): DI {chance['chance_di']:.4f} (SD {chance['chance_sd']:.3f}); "
                 f"DI at chance (no segregation): "
                 + (", ".join(SHORT.get(s, s) for s in chance["at_floor"]) or "none"))
        L.append("- other metrics: floor = "
                 f"{chance['floor_ratio']} x chance SD (the DI ratio), polarity-aware: "
                 + ", ".join(f"{m} {cm['floor']:g}" for m, cm in per.items()
                             if m != "dissimilarity_index"))
    L += ["", "Every chain reads MOST segregated first (clusters and switch_rate "
          "are sign-corrected: fewer clusters / lower switch rate = more segregated).",
          "", "Legend: `>` certified · `?>` fixable (GPU top-up resolves) · "
          "`n>` needs more RUNS, not sampling (ruler already fine, gap "
          "under-measured) · `~` unresolved · `=` exact tie · "
          "`≈` below the practical-significance floor · `°` at the chance "
          "level (no more segregated than a random grid; marked as a tie)", ""]
    def name(sc, m):
        return SHORT.get(sc, sc) + ("°" if sc in per.get(m, {}).get("at_floor", []) else "")
    for m in [x for x in decision] + [x for x in ALL_METRICS if x not in decision]:
        d = tab[tab.metric == m].reset_index(drop=True)
        if d.empty:
            continue
        chain = [name(d.loc[0, "hi"], m)]
        for i in range(len(d)):
            chain += [OPS[d.loc[i, "class"]], name(d.loc[i, "lo"], m)]
        tag = "  **(DECISION METRIC)**" if m in decision else ""
        cost = d[d["class"] == "FIXABLE"]["gpu_h"].sum() if "gpu_h" in d else 0
        L += [f"## {m}{tag}", "", "```", "".join(chain), "```",
              f"counts: {d['class'].value_counts().to_dict()}"
              + (f" · targeted top-up {cost:.1f} GPU-h" if cost else ""), ""]
        if m in per:
            cm = per[m]
            L += [f"chance {cm['chance']:.4g} (SD {cm['chance_sd']:.3g}), floor {cm['floor']:g}"
                  + (" — higher = LESS segregation; excess is sign-corrected" if cm['direction'] < 0 else ""),
                  "", "| scenario | mean | SE | excess over chance | at chance |", "|---|---|---|---|---|"]
            for sc, v in sorted(cm["levels"].items(), key=lambda kv: -cm['direction'] * kv[1]):
                L.append(f"| {SHORT.get(sc, sc)} | {v:.4g} | {cm['se'][sc]:.3g} "
                         f"| {cm['excess'][sc]:+.4g} | {'°' if sc in cm['at_floor'] else ''} |")
            L.append("")
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
    ap.add_argument("--multisplit", default=None,
                    help="dir holding vf_multisplit_check.csv + deltas (the ruler); "
                         "omit together with --exact for logprob-derived tables")
    ap.add_argument("--exact", action="store_true",
                    help="the value functions carry no sampling error (exact "
                         "logprob tables): no ruler, s = 0 for every pair")
    ap.add_argument("--production-experiments", default=None,
                    help="run_dir/experiments with the big-n batch (the gaps); "
                         "omitted -> gaps from the check's own 100-run arm")
    ap.add_argument("--w-current", type=float, default=0.02)
    ap.add_argument("--gap-floor", type=float, default=0.01,
                    help="practical-significance floor on the DI gap: pairs "
                         "with |gap| below it are FLOOR-TIE (terminal, never "
                         "priced). 0.01 = ~two agent tract moves on the "
                         "20x20/160+160 board, 0.3 random-allocation SDs; "
                         "user decision 2026-09-05")
    ap.add_argument("--floor-ratio", type=float, default=0.3,
                    help="floor for the non-DI metrics as a fraction of each "
                         "metric's chance SD (0.3 = the DI decision: "
                         "0.01 / 0.0336); a scenario within its floor of the "
                         "chance value is 'at chance' and two such scenarios "
                         "are FLOOR-TIE regardless of gap")
    ap.add_argument("--null-draws", type=int, default=20000,
                    help="random-allocation grids for the chance level of "
                         "every metric (cached per board)")
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

    if args.exact and args.multisplit:
        print("--exact and --multisplit are mutually exclusive"); return 2
    if not args.exact and not args.multisplit:
        print("need --multisplit <ruler dir> or --exact"); return 2
    ms = Path(args.multisplit) if args.multisplit else None
    if ms is not None and not (ms / "vf_multisplit_deltas.csv").exists():
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
    meta = ruler_meta(ms) if ms is not None else {}
    scheme = ("exact" if ms is None else meta.get("rng_scheme", "shared" if meta else "unknown"))
    print(f"label={args.label}  gaps from: {src}\nruler from: "
          f"{ms if ms is not None else 'none (exact tables, s = 0)'}  (rng_scheme={scheme})")
    if args.production_experiments and ms is not None:
        mism = check_board(meta, args.production_experiments)
        if mism:
            print("ruler and production batch are not comparable: " + "; ".join(mism))
            return 2

    board = (production_board(args.production_experiments) if args.production_experiments
             else {k: meta.get(k) for k in ("grid_size", "num_type_a", "num_type_b")})
    chance = chance_levels(gaps, board, args.gap_floor, args.floor_ratio, args.null_draws)
    if chance:
        print(f"chance DI {chance['chance_di']:.4f}; DI at chance: {chance['at_floor'] or 'none'}; "
              "floors: " + ", ".join(f"{m} {cm['floor']:g}" for m, cm in chance["metrics"].items()))
    tab = classify(args.label, args.style, ms, gaps, args.tie_mult,
                   args.w_current, price=not args.no_pricing,
                   gap_floor=args.gap_floor, chance=chance)
    if args.out_dir is None and ms is None:
        print("--exact needs --out-dir"); return 2
    out = Path(args.out_dir or (ms.parent / "rank_stability"))
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "rank_pairs.csv", index=False)

    write_notes(out / "RANK_STABILITY_NOTES.md", args.label, tab,
                args.decision_metrics, src, str(ms), scheme, args.gap_floor, chance)
    if chance:
        # The paper's level table: does each scenario segregate at all?
        lv = pd.DataFrame({"scenario": list(chance["levels"]),
                           "mean_di": list(chance["levels"].values()),
                           "se": [chance["se"][s] for s in chance["levels"]]})
        lv["chance_di"] = chance["chance_di"]
        lv["excess"] = lv.mean_di - lv.chance_di
        lv["ci95_low"] = lv.mean_di - Z * lv.se
        lv["ci95_high"] = lv.mean_di + Z * lv.se
        lv["at_chance"] = lv.scenario.isin(chance["at_floor"])
        lv.sort_values("mean_di", ascending=False).to_csv(out / "di_levels.csv", index=False)
        # All seven metrics, one row per (metric, scenario): the completeness table.
        ml = []
        for m, cm in chance["metrics"].items():
            for sc, v in cm["levels"].items():
                ml.append({"metric": m, "scenario": sc, "mean": v, "se": cm["se"][sc],
                           "chance": cm["chance"], "chance_sd": cm["chance_sd"],
                           "direction": cm["direction"], "excess": cm["excess"][sc],
                           "floor": cm["floor"], "at_chance": sc in cm["at_floor"]})
        pd.DataFrame(ml).to_csv(out / "metric_levels.csv", index=False)

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
                        "UNMEASURED" if sub["class"].isin(["UNMEASURED", "RUNS"]).any()
                        else "SETTLED"),
        }

    dec = tab[tab.metric.isin(args.decision_metrics)]
    if dec.empty:
        print(f"decision metric(s) {args.decision_metrics} absent from the data")
        return 2
    # The DECISION block drives everything; all_metrics is reference only.
    status = {"label": args.label, "gap_source": src,
              "ruler": (str(ms) if ms is not None else None), "exact_tables": ms is None,
              "ruler_rng_scheme": scheme,
              "ruler_board": {k: meta.get(k) for k in
                              ("grid_size", "num_type_a", "num_type_b",
                               "max_steps", "splits", "runs")},
              "tie_mult": args.tie_mult, "w_current": args.w_current,
              "gap_floor": args.gap_floor, "floor_ratio": args.floor_ratio,
              "floors": ({m: cm["floor"] for m, cm in chance["metrics"].items()}
                         if chance else {"dissimilarity_index": args.gap_floor}),
              "chance": chance,
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
