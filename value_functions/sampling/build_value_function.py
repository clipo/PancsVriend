#!/usr/bin/env python3
"""Build the sampled VALUE FUNCTION of an LLM model: P(MOVE | neighbourhood ratio)
for both agent roles, over the 23 distinct opposite-neighbour ratios reachable in
a Moore neighbourhood (+ the no-neighbours point), per prompt style and scenario.

    python value_functions/sampling/build_value_function.py --config configs/value_function_baseline.yaml
    python value_functions/sampling/build_value_function.py --config ... --dry-run
    python value_functions/sampling/build_value_function.py --config ... --from-existing-only --plot

Everything is driven by the YAML config (see configs/value_function_baseline.yaml):
which scenarios, which prompt styles, both roles, and the sampling allocation —
`per_composition` gives every occupied composition the same N, while `per_ratio`
gives every RATIO DATAPOINT the same total N split across the compositions that
alias to it (1/2 pools 1-of-2, 2-of-4, 3-of-6, 4-of-8; 1/8 has only 8 occupied
with 1 opposite). Counts from earlier ratio-sweep runs can be merged in via
`merge_from` (they are sufficient statistics — addition is exact), so already-paid
samples are never re-bought.

Output: one JSON artifact per (scenario, style) with BOTH roles —
    value_functions/results/sampled/vf_<label>__<scenario>__<style>.json   (schema vf-1)
carrying the per-composition counts (ground truth), the 23+1 ratio datapoints
(effective P(MOVE) = equal-weight mean over member compositions, propagated 95% CI; pooled counts kept alongside), and full provenance. The
simulation consumes it directly:  python llm_runner.py --value-function <file>.
Raw replies stream to value_functions/results/sampled/raw/ incrementally (a killed run
keeps everything sampled so far).

All rates are EFFECTIVE move rates n_move/(n_move+n_stay) — the production-
faithful quantity (production retries until parseable). See NOTES.md for why
grammar arms are the trustworthy channel.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from value_functions.paths import SAMPLED_DIR, add_import_paths  # noqa: E402
add_import_paths()

import yaml  # noqa: E402

from sampling_common import (  # noqa: E402
    GRAMMAR,
    NO_NEIGHBORS_KEY,
    RESULTS_DIR,
    aggregate_counts,
    init_slice_state,
    load_value_function,
    parse,
    ratio_groups,
    ratio_key,
    ratio_of,
    request_seed,
    role_keywords,
    sample_batch,
    server_fingerprint,
    slice_ping,
    wilson_ci,
    RawWriter,
)
from ratio_prompt_templates import (  # noqa: E402
    ALL_COMPOSITIONS,
    RATIO_CANDIDATES,
    grid_context,
    mechanical_move,
)
from evaluate_ratio_prompts import render_prompt  # noqa: E402
from analyze_ratio_consistency import parse_label  # noqa: E402
from llm_runner import (  # noqa: E402
    LLM_STYLES,
    SAMPLER_PARAMS,
    resolve_llm_request_url,
)

VF_DIR_DEFAULT = SAMPLED_DIR


# ---------------------------------------------------------------------------
# Sampling allocation
# ---------------------------------------------------------------------------

def allocate_samples(mode, n):
    """{(n_similar, n_occupied): samples} over all 45 compositions.

    per_composition: every occupied composition gets n (a ratio datapoint then
        holds n x its multiplicity — up to 8n at ratio 0 or 1, n at e.g. 1/8).
    per_ratio: every ratio DATAPOINT gets n in total, split as evenly as
        possible across its member compositions (deterministic: the leftover
        goes to the lowest-occupancy members first). Equalizes the CI width
        across the 23-point axis instead of across compositions.
    The (0, 0) no-neighbours cell is its own datapoint and gets n either way.
    """
    alloc = {}
    if mode == "per_composition":
        for cell in ALL_COMPOSITIONS:
            alloc[cell] = n
    elif mode == "per_ratio":
        alloc[(0, 0)] = n
        for frac, members in ratio_groups().items():
            base, extra = divmod(n, len(members))
            for i, cell in enumerate(members):
                alloc[cell] = base + (1 if i < extra else 0)
    else:
        raise ValueError(f"sampling.mode must be per_composition or per_ratio, got {mode!r}")
    return alloc


# ---------------------------------------------------------------------------
# Count store: {(role, (n_sim, n_occ)): {move, stay, bad, samples}}
# ---------------------------------------------------------------------------

def _blank_counts(roles):
    return {(role, cell): {"move": 0, "stay": 0, "bad": 0, "samples": 0}
            for role in roles for cell in ALL_COMPOSITIONS}


def merge_existing(counts, label, style, roles):
    """Fold the counts of results/ratio_comparison_<label>.csv into `counts`.

    Counts are sufficient statistics, so addition is exact — provided the
    prompts were identical. That holds only for the BASELINE scenario (the
    historical sweep's identity labels); the caller enforces that.
    Returns the number of samples folded in.
    """
    path = RESULTS_DIR / f"ratio_comparison_{label}.csv"
    if not path.exists():
        raise FileNotFoundError(f"merge_from label {label!r}: {path} does not exist")
    added = 0
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["candidate"] != style or row["agent_role"] not in roles:
                continue
            key = (row["agent_role"], (int(row["n_similar"]), int(row["n_occupied"])))
            c = counts[key]
            c["move"] += int(row["n_move"])
            c["stay"] += int(row["n_stay"])
            c["bad"] += int(row["n_bad"])
            c["samples"] += int(row["n_samples"])
            added += int(row["n_samples"])
    if added == 0:
        raise ValueError(f"merge_from label {label!r}: no rows for style {style!r} "
                         f"and roles {roles} — wrong label or style name?")
    return added


def sample_into_counts(counts, alloc_fn, roles, kw_by_role, style, tpl, fn,
                       url, model, temperature, grammar_on, concurrency, seed,
                       raw, ping_label, ping_ctx, seed_ctx=None):
    """The sampling loop, shared by the initial build and --top-up.

    alloc_fn(role, cell) -> how many NEW samples that (role, cell) gets —
    a constant per cell for the initial build, a per-role deficit for top-up.
    Counts accumulate in place; every raw reply streams through `raw`.
    Returns the number of new samples taken.

    seed_ctx=(stage, scenario): activates the CLEAN MEASUREMENT PROTOCOL
    (KV_CACHE_SAMPLING_ARTIFACT.md): cache_prompt=false and a deterministic
    request_seed(stage|scenario|style|role|sim|occ|i) per request, recorded in
    each raw record. None keeps the legacy payload (no cache/seed fields).
    """
    # One request queue per ROLE rather than one per (role, cell)
    # (2026-09-05). The 45 cells of a role used to be sampled strictly one
    # after another, each through its own thread pool, so a calibration pass
    # buying the 25-sample floor for most cells kept a server with 8-18
    # slots mostly idle between cells. Prompts are still rendered cell by
    # cell in the same order (the grid_context RNG advances exactly as
    # before), every request carries the same seed, and the raw records,
    # their order and the counts are unchanged — only how many requests are
    # in flight at once differs. NOTE (2026-09-06): that is not harmless on
    # this server — replies depend on the batch they are processed in
    # (KV_CACHE_SAMPLING_ARTIFACT.md §8), so a sampled table is only exact
    # at `concurrency` 1; the exact tables come from logprob_value_function.py.
    n_new = 0
    for role in roles:
        role_kw = kw_by_role[role]
        g0_rng = random.Random(seed) if fn is grid_context else None
        jobs = []                                  # (cell, rendered, seed_fn), cell order
        for cell in ALL_COMPOSITIONS:
            n_sim, n_occ = cell
            k = alloc_fn(role, cell)
            if k == 0:
                continue
            if fn is grid_context:
                rendered = [render_prompt(style, tpl, fn, n_sim, n_occ,
                                          role_kw, rng=g0_rng)
                            for _ in range(k)]
            else:
                rendered = [render_prompt(style, tpl, fn, n_sim, n_occ,
                                          role_kw)] * k
            if seed_ctx is None:
                seed_fn = None
            else:
                stage, scenario = seed_ctx
                # default-arg binding pins the loop variables at definition time
                seed_fn = (lambda i, s=stage, sc=scenario, r=role,
                           ns=n_sim, no=n_occ:
                           request_seed(s, sc, style, r, ns, no, i))
            jobs.append((cell, rendered, seed_fn))
        if jobs:
            prompts = [p for _, rendered, _ in jobs for p, _ in rendered]
            if seed_ctx is None:
                cache_prompt, flat_seed_fn = None, None
            else:
                cache_prompt = False
                seeds = [seed_fn(i) for _, rendered, seed_fn in jobs
                         for i in range(len(rendered))]
                flat_seed_fn = seeds.__getitem__
            replies = sample_batch(url, model, prompts, temperature,
                                   GRAMMAR if grammar_on else None,
                                   concurrency,
                                   cache_prompt=cache_prompt, seed_fn=flat_seed_fn)
            pos = 0
            for cell, rendered, seed_fn in jobs:
                n_sim, n_occ = cell
                cell_replies = replies[pos:pos + len(rendered)]
                pos += len(rendered)
                for i, (r, (_, ctx)) in enumerate(zip(cell_replies, rendered)):
                    rec = {"agent_role": role, "n_occupied": n_occ,
                           "n_similar": n_sim, "sample": i,
                           "context": ctx, "text": r["text"],
                           "finish_reason": r["finish_reason"],
                           "completion_tokens": r["completion_tokens"],
                           "parse": parse(r["text"])}
                    if seed_fn is not None:
                        rec["seed"] = seed_fn(i)
                    raw.write(rec)
                mv, st, bd = aggregate_counts(cell_replies)
                c = counts[(role, cell)]
                c["move"] += mv; c["stay"] += st
                c["bad"] += bd; c["samples"] += len(cell_replies)
                n_new += len(cell_replies)
        slice_ping(ping_label, f"{ping_ctx} [{role}]")
    return n_new


def counts_from_artifact(vf):
    """Rehydrate the count store from a vf-1 artifact's composition rows."""
    counts = {}
    for role, rows in vf["compositions"].items():
        for c in rows:
            counts[(role, (c["n_similar"], c["n_occupied"]))] = {
                "move": c["n_move"], "stay": c["n_stay"],
                "bad": c["n_bad"], "samples": c["n_samples"]}
    return counts


TOPUP_FRACTION = 0.5      # buy this share of the outstanding deficit per pass
TOPUP_FLOOR = 25          # ...but never fewer than this, to bound the pass count


def cell_target(move, valid, w):
    """Samples this cell needs so its move-rate is known to ±w.

    THE single criterion, used for both "is this cell done?" and "how many
    more?" — n = z²·p*(1−p*)/w², with p* the current Wilson-CI bound NEAREST
    0.5 (or 0.5 itself when the interval straddles it).

    Evaluating at p* rather than at p̂ is deliberate and is the whole safety
    argument: a cell showing 100 of 100 MOVE might really sit at p = 0.963,
    and planning against p̂ = 1 would stop ~4x too early. Simulated over cells
    with known true p, trusting p̂ mis-certifies 5.5–14.3% of them (declares
    ±2pp while the estimate is off by more); planning at p* holds that to
    0.1–1.8%, inside the ~5% a 95% interval implies.

    Until 2026-09-01 this function's caller ALSO short-circuited on the
    observed Wilson width ("if (hi−lo)/2 <= w: 0"), which trusted p̂ for the
    stop test while distrusting it for the purchase. Two postures met at
    w = z²/(2n+2z²) — 0.01850 for a saturated cell at n=100 — and the deficit
    jumped 0 -> 305 across that line. The gate existed only because purchases
    were one-shot; incremental buying (see advance_pass) removes the overshoot
    it was patching, so the gate is gone and one criterion governs throughout.
    """
    z2 = 1.96 ** 2
    if valid == 0:
        return math.ceil(z2 * 0.25 / (w * w))
    lo, hi = wilson_ci(move, valid)
    p_star = 0.5 if lo <= 0.5 <= hi else (
        lo if abs(lo - 0.5) < abs(hi - 0.5) else hi)
    return math.ceil(z2 * p_star * (1 - p_star) / (w * w))


def topup_deficits(counts, roles, w):
    """{(role, cell): outstanding shortfall} against cell_target.

    The full remaining gap, NOT the amount to buy this pass — that is
    advance_pass. Used as the convergence test: residual == 0 iff every cell
    meets the criterion.
    """
    deficits = {}
    for role in roles:
        for cell in ALL_COMPOSITIONS:
            c = counts[(role, cell)]
            valid = c["move"] + c["stay"]
            deficits[(role, cell)] = max(0, cell_target(c["move"], valid, w) - valid)
    return deficits


def advance_pass(counts, roles, w, fraction=TOPUP_FRACTION, floor=TOPUP_FLOOR):
    """{(role, cell): samples to buy THIS pass} — one calibration step.

    Buys a FRACTION of the outstanding deficit rather than all of it, because
    p* moves toward p̂ as n grows and the target therefore falls as you
    approach it. A saturated cell at n=100 shows a first-pass deficit of 243
    at ±2pp, but its fixed point is ~180: buying the full deficit overshoots
    by ~1.9x. Halving captures 33 of the 35 available percentage points of
    saving in a median of 2 passes; finer fractions add passes for ~3% more
    (0.35 -> 2,950 samples vs 0.50 -> 3,035, at max 17 passes instead of 13).

    The floor bounds the pass count: without it a cell owing thousands would
    creep up in tiny steps.

    This is the ONE definition of a pass. calibrate() hands the result to the
    sampler; project_total() hands it to a p̂-preserving simulator. A quote and
    the spend it predicts therefore cannot drift.
    """
    buys = {}
    for key, short in topup_deficits(counts, roles, w).items():
        buys[key] = 0 if short <= 0 else min(short, max(floor, math.ceil(fraction * short)))
    return buys


def project_total(counts, roles, w, fraction=TOPUP_FRACTION, floor=TOPUP_FLOOR,
                  max_passes=200):
    """Total extra samples calibrate() would spend to reach ±w, per cell.

    Walks the same advance_pass loop the calibrator will, assuming each cell's
    p̂ does not move as samples arrive. That assumption makes this a FLOOR: if
    sampling reveals a cell is less saturated than it looked, p* drops toward
    0.5 and the real cost rises. Bounded — cell_target never exceeds
    z²·0.25/w² — so the walk always terminates.
    """
    sim = {k: dict(v) for k, v in counts.items()}
    spent = {k: 0 for k in sim}
    for _ in range(max_passes):
        buys = advance_pass(sim, roles, w, fraction, floor)
        if not any(buys.values()):
            break
        for key, n in buys.items():
            if n <= 0:
                continue
            c = sim[key]
            valid = c["move"] + c["stay"]
            p = (c["move"] / valid) if valid else 0.5
            add_move = int(round(p * n))
            c["move"] += add_move
            c["stay"] += n - add_move
            c["samples"] = c.get("samples", 0) + n
            spent[key] += n
    return spent


# ---------------------------------------------------------------------------
# Artifact assembly
# ---------------------------------------------------------------------------

def _rate_ci(move, stay):
    valid = move + stay
    if valid == 0:
        return None, (0.0, 1.0)
    return move / valid, wilson_ci(move, valid)


def build_rows(counts, role):
    """(composition_rows, ratio_rows) for one role from the count store."""
    comp_rows = []
    for n_sim, n_occ in ALL_COMPOSITIONS:
        c = counts[(role, (n_sim, n_occ))]
        p, ci = _rate_ci(c["move"], c["stay"])
        comp_rows.append({
            "n_similar": n_sim, "n_occupied": n_occ, "n_opposite": n_occ - n_sim,
            "ratio": ratio_key(ratio_of(n_sim, n_occ)),
            "n_samples": c["samples"], "n_move": c["move"], "n_stay": c["stay"],
            "n_bad": c["bad"],
            "p_move_effective": None if p is None else round(p, 6),
            "ci95": [round(ci[0], 6), round(ci[1], 6)],
            "mechanical": mechanical_move(n_sim, n_occ),
        })

    ratio_rows = []
    groups = [(None, [(0, 0)])] + [(f, m) for f, m in ratio_groups().items()]
    for frac, members in groups:
        move = stay = bad = samples = 0
        cell_rates, cell_vars = [], []
        for cell in members:
            c = counts[(role, cell)]
            move += c["move"]; stay += c["stay"]
            bad += c["bad"]; samples += c["samples"]
            valid = c["move"] + c["stay"]
            if valid > 0:
                pi = c["move"] / valid
                cell_rates.append(pi)
                # Variance from the cell's WILSON interval, not the Wald form
                # pi*(1-pi)/valid. Wald is identically ZERO at pi = 0 or 1, so
                # a ratio datapoint whose members are all saturated used to
                # report a zero-width CI: on qwen/baseline the median ratio-row
                # half-width was 0.0000 while every underlying cell sat at
                # 0.0185. The lineplot therefore drew invisible error bars over
                # precisely the points whose uncertainty matters most.
                # (hw/1.96)^2 treats the Wilson interval as a normal SE — an
                # approximation, since Wilson is asymmetric near the boundary,
                # but one that does not degenerate and that is derived from the
                # SAME number printed per cell, so the two renderings agree.
                lo_i, hi_i = wilson_ci(c["move"], valid)
                cell_vars.append(((hi_i - lo_i) / 2 / 1.96) ** 2)
        # THE value-function datapoint = SIMPLE (equal-weight) average of the
        # member compositions (user decision 2026-08-23): the pooled
        # sample-weighted rate silently re-weights members by our sampling
        # budget after top-ups (a topped-up member can carry 24x the vote).
        # CI by variance propagation: Var(mean) = (1/k^2) * sum p_i q_i / n_i.
        if cell_rates:
            k = len(cell_rates)
            p_mean = sum(cell_rates) / k
            hw = 1.96 * math.sqrt(sum(cell_vars)) / k
            mean_ci = (max(0.0, p_mean - hw), min(1.0, p_mean + hw))
        else:
            p_mean, mean_ci = None, (0.0, 1.0)
        p_pooled, pooled_ci = _rate_ci(move, stay)
        ratio_rows.append({
            "ratio": ratio_key(frac),
            "ratio_float": None if frac is None else round(float(frac), 6),
            "members": [list(m) for m in members],
            "n_samples": samples, "n_move": move, "n_stay": stay, "n_bad": bad,
            "p_move_effective": None if p_mean is None else round(p_mean, 6),
            "ci95": [round(mean_ci[0], 6), round(mean_ci[1], 6)],
            "aggregation": "equal_member_weights",
            # Sample-weighted pooled rate retained for reference/audit only.
            "p_move_pooled": None if p_pooled is None else round(p_pooled, 6),
            "ci95_pooled": [round(pooled_ci[0], 6), round(pooled_ci[1], 6)],
            "mechanical": mechanical_move(*members[0]),
        })
    return comp_rows, ratio_rows


# ---------------------------------------------------------------------------
# Plot: the value function itself
# ---------------------------------------------------------------------------

def draw_precision_panel(ax, vf):
    """CI half-width per ratio datapoint against the calibration target.

    Two series per role, and the distinction matters:
      * the ratio datapoint's OWN half-width (line) — what the curve above
        draws as error bars;
      * the WORST member composition cell behind it (crosses) — because ±w is
        a per-CELL target, and a ratio point is an equal-weight mean of up to
        5 members, so it is inherently tighter than any of them. Judging
        calibration from the ratio series alone would read as comfortably
        inside target while individual cells sit outside it.
    """
    from plot_value_functions import ROLE_COLORS
    from sampling_common import NO_NEIGHBORS_KEY
    hw = lambda ci: (ci[1] - ci[0]) / 2 if ci else float("nan")
    comps = vf.get("compositions", {})
    for role, rows in vf["ratios"].items():
        col = ROLE_COLORS.get(role, "black")
        by_ratio = {}
        for c in comps.get(role, []):
            if c.get("ci95"):
                by_ratio.setdefault(c["ratio"], []).append(hw(c["ci95"]))
        pts = sorted((-0.06 if r["ratio"] == NO_NEIGHBORS_KEY else r["ratio_float"],
                      hw(r.get("ci95")), max(by_ratio.get(r["ratio"], [float("nan")])))
                     for r in rows if r.get("ci95") is not None
                     and (r["ratio"] == NO_NEIGHBORS_KEY or r["ratio_float"] is not None))
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ax.plot(xs, [p[1] for p in pts], color=col, lw=1.2, marker="o", ms=2.5)
        ax.scatter(xs, [p[2] for p in pts], color=col, marker="x", s=14, lw=0.9)
    cal = (vf.get("meta") or {}).get("calibration") or {}
    tgt = cal.get("requested_w")
    if tgt:
        ax.axhline(tgt, color="#2e7d32", ls="--", lw=1.0)
        ax.annotate(f"target ±{tgt:g}", (1.0, tgt), xytext=(-2, 2),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=6.5, color="#2e7d32")
    ax.set_xlim(-0.12, 1.02)
    ax.set_ylabel("CI ½-width", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.grid(alpha=0.25, axis="y")
    ax.set_axisbelow(True)


def plot_vf(vf, out_path, dpi=300):
    """Single-artifact figure; panel drawing shared with the cross-scenario
    comparison (plot_value_functions.draw_value_function_axes / draw_n_bars)
    so the two renderings of a value function cannot drift apart. Curve on
    top, per-datapoint N bar chart, then achieved precision (shared x)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_value_functions import draw_value_function_axes, draw_n_bars

    fig, (ax, axn, axp) = plt.subplots(3, 1, figsize=(9, 8.4), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1, 1],
                                                    "hspace": 0.08})
    draw_value_function_axes(ax, vf, legend_roles=True)
    draw_n_bars(axn, vf)
    draw_precision_panel(axp, vf)
    axp.set_xlabel("opposite / occupied neighbors (23 reachable ratios)")
    ax.plot([], [], color="black", ls="--", lw=1, label="mechanical (>0.5 moves)")
    ax.annotate("no\nneighbors", (-0.06, 0.02), ha="center", fontsize=7, color="#555")
    m = vf["meta"]
    # Report the ACTUAL per-datapoint sample counts in the artifact, not the
    # config target: a --from-existing-only build carries only the merged
    # counts, and mixed merge+sample builds vary by datapoint.
    ns = [r["n_samples"] for rows in vf["ratios"].values() for r in rows]
    n_txt = f"{min(ns)}" if min(ns) == max(ns) else f"{min(ns)}–{max(ns)}"
    ax.set_ylabel("effective P(MOVE), equal-weight member mean  [95% CI]")
    # Calibration status in the title: whether the table met the precision it
    # was asked for has to be visible when inspecting the figure, not only in a
    # log line. achieved_w is the RMS over cells; the max and the short count
    # say whether a shortfall is one outlier or systemic.
    cal = m.get("calibration") or {}
    if cal:
        st = ("converged" if cal.get("converged") else
              f"SHORT — {cal.get('n_cells_short', '?')} of {cal.get('n_cells', '?')} cells over target")
        cal_txt = (f"\nprecision: asked ±{cal.get('requested_w')}, achieved "
                   f"±{cal.get('achieved_w')} rms / ±{cal.get('achieved_w_max')} max — {st}")
    else:
        cal_txt = "\nprecision: not recorded (built before calibration provenance)"
    ax.set_title(f"Value function — {m['model']} / {m['style']} / {m['scenario']} / {m['arm']}\n"
                 f"(samples per ratio datapoint: {n_txt}, see N panel; T={m['temperature']})"
                 f"{cal_txt}", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Mode: --top-up (stage 2 of the two-stage design)
# ---------------------------------------------------------------------------

def _assemble_artifact(vf_meta, counts, roles):
    """Rebuild the compositions/ratios sections from a count store, keeping meta."""
    vf = {"schema": "vf-1", "meta": vf_meta, "compositions": {}, "ratios": {}}
    for role in roles:
        comp_rows, ratio_rows = build_rows(counts, role)
        vf["compositions"][role] = comp_rows
        vf["ratios"][role] = ratio_rows
    return vf


def _load_vf_tolerant(vf_path):
    """load_value_function, but accept an artifact with undefined datapoints.

    The validator refuses any artifact whose ratio rows contain a None
    p_move_effective — but that is EXACTLY the artifact that needs topping up
    (a cell where every pilot sample was unparseable). The pilot itself only
    warns in that case, so --calibrate must be able to open what the pilot was
    willing to write. Only --calibrate passes tolerant=True; --top-up keeps the
    strict load so its documented behaviour is unchanged.
    """
    try:
        return load_value_function(vf_path)
    except ValueError as exc:
        print(f"[calibrate] {vf_path.name}: incomplete artifact ({exc}); "
              f"topping up anyway")
        with open(vf_path) as fh:
            return json.load(fh)


def achieved_precision(vf):
    """(rms, worst, n_cells) of the composition cells' Wilson CI half-widths.

    What the table ACTUALLY delivers, read off the cells — never the value that
    was requested. RMS is the headline because it is the summary that drives
    the sufficiency ratio: the metric displacement is sqrt(sum_i g_i^2 sigma_i^2)
    with sigma_i = hw_i/1.96, so under equal cell influence it scales with the
    RMS of the half-widths. The max is reported alongside because the ±w target
    is stated per cell, but the max alone is a poor base — one cell whose
    replies never parse would set it arbitrarily high and make the NEXT target
    looser rather than tighter.
    """
    hws = [(c["ci95"][1] - c["ci95"][0]) / 2
           for rows in vf.get("compositions", {}).values() for c in rows
           if c.get("ci95")]
    if not hws:
        return None, None, 0
    rms = math.sqrt(sum(h * h for h in hws) / len(hws))
    return rms, max(hws), len(hws)


def record_calibration_meta(vf_path, requested_w, converged, passes):
    """Stamp requested-vs-achieved precision into an artifact's meta.

    Lets a reader (and run_vf_sufficiency_loop.sh) see what the table is at
    without re-deriving it, and makes a shortfall visible in the artifact
    itself rather than only in a log line that scrolls away.
    """
    if not vf_path.exists():
        return
    with open(vf_path) as fh:
        vf = json.load(fh)
    rms, worst, n_cells = achieved_precision(vf)
    short = sum(1 for rows in vf.get("compositions", {}).values() for c in rows
                if c.get("ci95") and (c["ci95"][1] - c["ci95"][0]) / 2 > requested_w)
    vf["meta"]["calibration"] = {
        "requested_w": requested_w,
        "achieved_w": None if rms is None else round(rms, 6),
        "achieved_w_max": None if worst is None else round(worst, 6),
        "n_cells": n_cells,
        "n_cells_short": short,
        "converged": bool(converged),
        "max_passes": passes,
        "topup_fraction": TOPUP_FRACTION,
    }
    with open(vf_path, "w") as fh:
        json.dump(vf, fh, indent=1)




def vf_table_path(out_dir, label, scenario, style):
    """Where a vf-1 table lives: <out_dir>/tables/vf_<label>__<scenario>__<style>.json.

    Moved out of the store root on 2026-09-07 (102 tables in sampled/ buried
    every other artifact). Relative to out_dir so the sanity arm gets the same
    layout in its own store without a second constant.
    """
    d = Path(out_dir) / TABLES_REL
    d.mkdir(parents=True, exist_ok=True)
    return d / f"vf_{label}__{scenario}__{style}.json"


def vf_plot_path(out_dir, label, scenario, style, fmt):
    """Per-SCENARIO figure path: <out_dir>/vf_mapping_plots/by_scenario/.

    One level below the combined all-scenario figures, which are what normally
    gets looked at. Relative to out_dir on purpose: every store (sampled,
    sampled_small, llm_logprob) gets the same layout without another constant.
    """
    d = Path(out_dir) / VF_MAPPING_PLOTS_REL / BY_SCENARIO_REL
    d.mkdir(parents=True, exist_ok=True)
    return d / f"vf_{label}__{scenario}__{style}.{fmt}"


def residual_deficit(out_dir, label, scenarios, styles, roles, precision):
    """(total_deficit, missing_slices, total_valid) across every configured slice.

    Computed in Python from the artifacts on disk rather than scraped out of
    stdout. The bash loop this replaces summed `-> N new samples` lines with
    grep/bc, which counted a MISSING artifact as zero deficit — so a scenario
    whose pilot had failed read as converged and the campaign pinged
    "±2pp reached" with no value function at all for it.
    """
    total, missing, valid = 0, [], 0
    for scenario in scenarios:
        for style in styles:
            vf_path = vf_table_path(out_dir, label, scenario, style)
            if not vf_path.exists():
                missing.append(f"{scenario}/{style}")
                continue
            with open(vf_path) as fh:
                vf = json.load(fh)
            counts = counts_from_artifact(vf)
            total += sum(topup_deficits(counts, roles, precision).values())
            valid += sum(c["move"] + c["stay"] for c in counts.values())
    return total, missing, valid


def top_up(out_dir, label, scenarios, styles, roles, conf, url, arm,
           temperature, grammar_on, concurrency, seed, precision,
           dry_run, plot, dpi, fmt="png", tolerant=False, stats=None,
           fraction=None, floor=TOPUP_FLOOR):
    """Bring every (role, composition) of the existing artifacts toward the
    target precision, sampling only the deficits (counts merge exactly).

    fraction=None buys the ENTIRE outstanding deficit in this call, which is
    what a bare `--top-up` has always done and what its callers expect.
    --calibrate passes a fraction (0.5) so each pass buys part of the gap and
    re-prices from the updated p̂ — see advance_pass for why that is cheaper.
    """
    init_slice_state(len(scenarios) * len(styles) * len(roles))
    if stats is not None:
        stats["sampled"] = 0
    for scenario in scenarios:
        kw_by_role = role_keywords(scenario, conf.get("scenario_file"))
        for style in styles:
            vf_path = vf_table_path(out_dir, label, scenario, style)
            if not vf_path.exists():
                print(f"[top-up] SKIP {scenario}/{style}: {vf_path.name} not found")
                continue
            vf = _load_vf_tolerant(vf_path) if tolerant else load_value_function(vf_path)
            if vf["meta"].get("arm") != arm:
                sys.exit(f"[top-up] {vf_path.name} is arm {vf['meta'].get('arm')!r} "
                         f"but config llm_style is {arm!r}")
            counts = counts_from_artifact(vf)
            deficits = (topup_deficits(counts, roles, precision) if fraction is None
                        else advance_pass(counts, roles, precision, fraction, floor))
            nonzero = {k: v for k, v in deficits.items() if v > 0}
            total = sum(nonzero.values())
            print(f"[top-up] {scenario}/{style}: {len(nonzero)} of "
                  f"{len(deficits)} (role, composition) cells below ±{precision:.0%} "
                  f"-> {total} new samples")
            for (role, cell), d in sorted(nonzero.items(), key=lambda kv: -kv[1]):
                c = counts[(role, cell)]
                valid = c["move"] + c["stay"]
                print(f"    {role:4s} sim={cell[0]} occ={cell[1]}  "
                      f"p̂={c['move']/max(valid,1):.2f} n={valid:4d}  +{d}")
            if dry_run or not nonzero:
                continue

            # Shard index from the filesystem AND from how many top-ups meta
            # already records. Filesystem alone resets to 1 if raw/ is ever
            # archived, and k feeds the per-request seed
            # (crc32(stage|scenario|style|role|sim|occ|i)) — so a reset would
            # re-issue identical seeds and buy duplicate samples that narrow
            # the CI without adding information.
            k = 1
            while (out_dir / "raw" /
                   f"vf_{label}__{scenario}__{style}_topup{k}_raw.jsonl.gz").exists():
                k += 1
            k = max(k, 1 + sum(1 for s in (vf["meta"].get("sources") or [])
                               if "topup" in str(s.get("label", ""))))
            raw_path = out_dir / "raw" / f"vf_{label}__{scenario}__{style}_topup{k}_raw.jsonl.gz"
            tpl, fn = RATIO_CANDIDATES[style]
            meta_raw = {"label": label, "model": conf["model"], "url": url,
                        "temperature": temperature, "grammar": grammar_on,
                        "seed": seed, "scenario": scenario, "style": style,
                        "stage": f"topup{k}", "precision": precision}
            with RawWriter(raw_path, meta_raw) as raw:
                n_new = sample_into_counts(
                    counts, lambda role, cell: deficits.get((role, cell), 0),
                    roles, kw_by_role, style, tpl, fn, url, conf["model"],
                    temperature, grammar_on, concurrency, seed, raw,
                    ping_label=label, ping_ctx=f"topup {scenario}/{style}",
                    seed_ctx=(f"topup{k}", scenario))

            meta = vf["meta"]
            prev_raw = meta.get("raw_replies")
            raws = ([prev_raw] if isinstance(prev_raw, str) else list(prev_raw or []))
            meta["raw_replies"] = raws + [str(raw_path)]
            meta.setdefault("sources", []).append(
                {"label": f"(topup{k} to ±{precision:.0%})", "samples_added": n_new})
            meta["created"] = datetime.now().isoformat(timespec="seconds")
            vf_new = _assemble_artifact(meta, counts, roles)
            vf_path.write_text(json.dumps(vf_new, indent=1))
            print(f"[top-up] rewrote {vf_path}  (+{n_new} samples)")
            if stats is not None:
                stats["sampled"] += n_new
            if not tolerant:
                load_value_function(vf_path)
            if plot:
                fig_path = vf_plot_path(out_dir, label, scenario, style, fmt)
                plot_vf(vf_new, fig_path, dpi=dpi)
                print(f"[top-up] rewrote {fig_path}")
    return 0


def calibrate(out_dir, label, scenarios, styles, roles, conf, url, arm,
              temperature, grammar_on, concurrency, seed, precision,
              max_passes, plot, dpi, fmt,
              fraction=TOPUP_FRACTION, floor=TOPUP_FLOOR):
    """Top up repeatedly until every cell is within ±precision.

    Replaces the four-pass bash loop that lived in run_vf_model_campaign.sh and
    run_vf_clean_campaign.sh, which decided convergence by grepping `-> N new
    samples` out of stdout and summing it with bc. Two failure modes motivated
    moving it here: a MISSING artifact printed "SKIP ... not found", matched no
    line, summed to zero and read as CONVERGED; and any change to the wording
    (or a missing bc) made the pipeline non-zero, which under `set -euo
    pipefail` silently aborted the whole campaign after the expensive pass.

    Multiple passes are genuinely required: topup_deficits sizes each purchase
    from the CURRENT p̂, so buying samples moves p̂ toward 0.5 and re-prices the
    cell upward. Total deficit is therefore NOT monotone, which is why the stop
    test is `== 0` and not "no change since last pass". Termination is bounded:
    the per-cell ceiling is ceil(1.96²·0.25/w²) (2401 at ±2pp) and at that n the
    worst-case Wilson half-width is 0.019984 <= 0.02, so the ceiling absorbs.

    Exit codes: 0 converged; 2 missing artifacts (the pilot did not complete —
    fatal, and the case the old loop got wrong); 3 no sampling progress (a dead
    server returns empty text that parses as `bad`, never as `valid`, so
    deficits would never shrink while each pass still writes a shard — that
    would loop until the budget with nothing to show for it); 5 budget spent
    with a residual remaining.
    """
    total, missing, valid = residual_deficit(
        out_dir, label, scenarios, styles, roles, precision)
    if missing:
        print(f"[calibrate] ABORT: no artifact for {len(missing)} slice(s): "
              f"{', '.join(missing)}\n[calibrate] run the pilot first — "
              f"calibrate deliberately does NOT build it (the pilot's raw shard "
              f"has a fixed name and is opened truncating, so re-running it "
              f"would destroy the existing ledger for that slice).")
        return 2
    print(f"[calibrate] start: residual={total} samples, {valid} valid on disk, "
          f"target ±{precision:.0%}, budget {max_passes} passes")

    for p in range(1, max_passes + 1):
        if total == 0:
            break
        print(f"[calibrate] === pass {p}/{max_passes} (residual {total}) ===")
        stats = {}
        top_up(out_dir, label, scenarios, styles, roles, conf, url, arm,
               temperature, grammar_on, concurrency, seed, precision,
               dry_run=False, plot=False, dpi=dpi, fmt=fmt,
               tolerant=True, stats=stats, fraction=fraction, floor=floor)
        new_total, new_missing, new_valid = residual_deficit(
            out_dir, label, scenarios, styles, roles, precision)
        print(f"[calibrate] pass {p}: sampled {stats.get('sampled', 0)}, "
              f"valid {valid} -> {new_valid}, residual {total} -> {new_total}")
        if new_valid <= valid:
            print(f"[calibrate] ABORT: pass {p} added no VALID samples "
                  f"(server down, or every reply unparseable). Not retrying — "
                  f"further passes would write shards and buy nothing.")
            return 3
        total, valid = new_total, new_valid

    if plot:  # once, at the end: per-pass figures are immediately superseded
        for scenario in scenarios:
            for style in styles:
                vf_path = vf_table_path(out_dir, label, scenario, style)
                if vf_path.exists():
                    with open(vf_path) as fh:
                        plot_vf(json.load(fh),
                                vf_plot_path(out_dir, label, scenario, style, fmt),
                                dpi=dpi)
        print(f"[calibrate] figures written for {len(scenarios) * len(styles)} slice(s)")

    # Provenance: what was ASKED for and what was actually achieved, written
    # into every artifact so neither the loop nor a reader has to infer it.
    for scenario in scenarios:
        for style in styles:
            record_calibration_meta(
                vf_table_path(out_dir, label, scenario, style),
                precision, converged=(total == 0), passes=max_passes)

    if total == 0:
        print(f"[calibrate] CONVERGED: every cell within ±{precision:.0%} "
              f"({valid} valid samples)")
        return 0
    print(f"[calibrate] GIVING UP: residual={total} after {max_passes} passes "
          f"(artifacts are usable but NOT at ±{precision:.0%})")
    # Exit 5, not 0. Until 2026-09-01 this returned 0, indistinguishable from
    # CONVERGED — harmless while one-shot purchasing converged in a single pass,
    # but incremental buying makes budget exhaustion a live branch at the same
    # moment run_vf_sufficiency_loop.sh started reasoning from this exit code.
    # A loop that reads "gave up" as "reached ±w" records a precision the table
    # does not have, and every later plan multiplies from that false base.
    # The campaign treats 5 as a warning and carries on — the artifacts are
    # valid, just less precise than requested, and aborting after the pilot has
    # been paid for is the wrong trade.
    return 5


# ---------------------------------------------------------------------------
# Mode: --rebuild-from-raw (half-data artifact for the sufficiency check)
# ---------------------------------------------------------------------------

def rebuild_from_raw(out_dir, label, scenarios, styles, keep, suffix):
    """Re-count an artifact from its raw reply files, keeping every 2nd sample
    per (role, composition) for keep=even/odd. Pure re-parse, zero LLM calls;
    with keep=all this must reproduce the original counts exactly."""
    import gzip
    for scenario in scenarios:
        for style in styles:
            vf_path = vf_table_path(out_dir, label, scenario, style)
            if not vf_path.exists():
                print(f"[rebuild] SKIP {scenario}/{style}: {vf_path.name} not found")
                continue
            vf = load_value_function(vf_path)
            prev_raw = vf["meta"].get("raw_replies")
            raws = ([prev_raw] if isinstance(prev_raw, str) else list(prev_raw or []))
            if not raws:
                print(f"[rebuild] SKIP {scenario}/{style}: no raw_replies in meta")
                continue
            roles = list(vf["compositions"].keys())
            counts = _blank_counts(roles)
            seen = {}   # (role, cell) -> running sample index across ALL raw files
            kept = 0
            for rp in raws:
                with gzip.open(rp, "rt", encoding="utf-8") as f:
                    for line in f:
                        rec = json.loads(line)
                        if rec.get("_meta"):
                            continue
                        key = (rec["agent_role"],
                               (rec["n_similar"], rec["n_occupied"]))
                        idx = seen.get(key, 0)
                        seen[key] = idx + 1
                        if keep == "even" and idx % 2 != 0:
                            continue
                        if keep == "odd" and idx % 2 != 1:
                            continue
                        c = counts[key]
                        verdict = rec["parse"]
                        if verdict == "MOVE":
                            c["move"] += 1
                        elif verdict == "STAY":
                            c["stay"] += 1
                        else:
                            c["bad"] += 1
                        c["samples"] += 1
                        kept += 1
            new_label = label + ("" if keep == "all" else suffix)
            meta = dict(vf["meta"])
            meta["label"] = new_label
            meta["rebuilt_from"] = {"artifact": str(vf_path), "keep": keep,
                                    "samples_kept": kept,
                                    "date": datetime.now().isoformat(timespec="seconds")}
            vf_new = _assemble_artifact(meta, counts, roles)
            out_path = vf_table_path(out_dir, new_label, scenario, style)
            if keep == "all":
                # Round-trip identity check, NOT an overwrite of the original.
                same = vf_new["compositions"] == vf["compositions"]
                print(f"[rebuild] keep=all identity vs {vf_path.name}: "
                      f"{'OK' if same else 'MISMATCH'}")
                if not same:
                    return 1
                continue
            out_path.write_text(json.dumps(vf_new, indent=1))
            print(f"[rebuild] wrote {out_path}  ({kept} samples kept, keep={keep})")
            load_value_function(out_path)
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="YAML config; see configs/value_function_baseline.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the sampling allocation and example prompts; no requests")
    ap.add_argument("--from-existing-only", action="store_true",
                    help="build purely from merge_from counts; zero LLM calls")
    ap.add_argument("--plot", action="store_true", help="also write a P(MOVE)-vs-ratio figure per artifact")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    ap.add_argument("--top-up", action="store_true",
                    help="two-stage mode: load each existing artifact, compute the "
                         "per-(role, composition) sample DEFICIT to reach --precision, "
                         "sample only the deficits, merge counts, rewrite. "
                         "With --dry-run: print the deficit table only.")
    ap.add_argument("--precision", type=float, default=0.05,
                    help="top-up target: 95%% CI half-width per composition (default 0.05)")
    ap.add_argument("--rebuild-from-raw", action="store_true",
                    help="re-count an artifact from its raw reply files (no LLM): "
                         "with --keep even/odd this builds the half-data artifact "
                         "for the sufficiency check, written as <label><suffix>.")
    ap.add_argument("--keep", default="even", choices=["even", "odd", "all"],
                    help="which samples per (role, composition) the rebuild keeps")
    ap.add_argument("--rebuild-suffix", default="-half",
                    help="label suffix for the rebuilt artifact (default -half)")
    ap.add_argument("--calibrate", action="store_true",
                    help="run --top-up passes REPEATEDLY until every cell is within "
                         "--precision (or --max-passes is spent), deciding convergence "
                         "in Python instead of grepping stdout. Requires the pilot to "
                         "have run: it will not build missing artifacts. Exit 0 "
                         "converged, 2 missing artifact, 3 no progress, "
                         "5 budget spent with a residual (warning for a "
                         "campaign, fatal for the sufficiency loop).")
    ap.add_argument("--max-passes", type=int, default=12,
                    help="pass budget for --calibrate. Raised 6 -> 12 on "
                         "2026-09-01 with incremental top-up: buying half the "
                         "deficit per pass needs more of them (p50 1, p99 3, "
                         "p100 9 over the 4,860 real cells). At 6, 19 cells "
                         "would be left short; at 12, none.")
    ap.add_argument("--topup-fraction", type=float, default=TOPUP_FRACTION,
                    help=f"share of each cell's outstanding deficit to buy per "
                         f"--calibrate pass (default {TOPUP_FRACTION}). 1.0 "
                         f"restores one-shot buying, which overshoots the fixed "
                         f"point by ~1.9x on saturated cells.")
    ap.add_argument("--topup-floor", type=int, default=TOPUP_FLOOR,
                    help=f"never buy fewer than this per cell per pass "
                         f"(default {TOPUP_FLOOR}); bounds the pass count")
    args = ap.parse_args()
    if args.top_up and args.rebuild_from_raw:
        sys.exit("--top-up and --rebuild-from-raw are mutually exclusive")
    if args.calibrate and (args.top_up or args.rebuild_from_raw
                           or args.from_existing_only):
        sys.exit("--calibrate cannot be combined with --top-up, "
                 "--rebuild-from-raw or --from-existing-only")
    if args.calibrate and args.dry_run:
        sys.exit("--calibrate --dry-run is not supported; use "
                 "--top-up --dry-run for a one-shot residual report")

    cfg_path = Path(args.config)
    conf = yaml.safe_load(cfg_path.read_text())
    label = conf["label"]
    roles = list(conf.get("roles", ["red", "blue"]))
    styles = list(conf["styles"])
    scenarios = list(conf.get("scenarios", ["baseline"]))
    samp = conf.get("sampling", {})
    mode, n_per = samp.get("mode", "per_ratio"), int(samp.get("samples", 200))
    # Endpoint arm comes from llm_style (production vocabulary: llm_runner
    # LLM_STYLES) — chat vs completions and grammar on/off in ONE key, so the
    # config can't say one thing while the URL path says another.
    style_arm = conf.get("llm_style")
    if style_arm is None:
        sys.exit("config needs llm_style: one of "
                 f"{list(LLM_STYLES)} (e.g. chat+grammar)")
    if style_arm not in LLM_STYLES:
        sys.exit(f"llm_style {style_arm!r} not in {list(LLM_STYLES)}")
    grammar_on = style_arm.endswith("+grammar")
    temperature = float(conf.get("temperature", 0.3))
    concurrency = int(conf.get("concurrency", 8))
    seed = int(conf.get("seed", 0))
    merge_labels = list(conf.get("merge_from") or [])
    out_dir = Path(conf.get("out_dir") or VF_DIR_DEFAULT)
    out_dir.mkdir(parents=True, exist_ok=True)

    alloc = allocate_samples(mode, n_per)
    need_llm = not (args.dry_run or args.from_existing_only or args.rebuild_from_raw)
    if need_llm and not (conf.get("model") and conf.get("llm_url")):
        sys.exit("config needs model + llm_url to sample (or use --from-existing-only / --dry-run)")
    # The style rewrites the URL path (sample_once infers chat-vs-completions
    # payload from the URL, same as production's build_llm_request).
    url = resolve_llm_request_url(conf.get("llm_url", ""), style_arm) \
        if conf.get("llm_url") else ""
    arm = style_arm

    # merge_from labels encode their arm (…-chat-grammar etc.); folding counts
    # measured on a DIFFERENT arm into this artifact would silently mix two
    # different measurement channels, so a mismatch is an error, not a warning.
    for ml in merge_labels:
        parsed = parse_label(ml)   # full label incl. ratio-/pilot- prefix
        if parsed and parsed[1] != arm:
            sys.exit(f"merge_from label {ml!r} is arm {parsed[1]!r} but the config "
                     f"llm_style is {arm!r} — merging across arms mixes "
                     f"measurement channels. Use the matching label.")

    if args.rebuild_from_raw:
        return rebuild_from_raw(out_dir, label, scenarios, styles,
                                args.keep, args.rebuild_suffix)
    if args.top_up:
        return top_up(out_dir, label, scenarios, styles, roles, conf, url, arm,
                      temperature, grammar_on, concurrency, seed,
                      args.precision, args.dry_run, args.plot, args.dpi)
    if args.calibrate:
        return calibrate(out_dir, label, scenarios, styles, roles, conf, url, arm,
                         temperature, grammar_on, concurrency, seed,
                         args.precision, args.max_passes, args.plot,
                         args.dpi, args.format,
                         fraction=args.topup_fraction, floor=args.topup_floor)

    if args.dry_run:
        print(f"allocation mode={mode}  N={n_per}  -> "
              f"{sum(alloc.values())} samples/role/style ({len(alloc)} compositions)")
        for frac, members in ratio_groups().items():
            per = [alloc[m] for m in members]
            print(f"  ratio {ratio_key(frac):>4s}: total {sum(per):4d}  over {members} -> {per}")
        print(f"  ratio none: total {alloc[(0, 0)]:4d}  over [(0, 0)]")
        kw = role_keywords(scenarios[0], conf.get("scenario_file"))
        tpl, fn = RATIO_CANDIDATES[styles[0]]
        pr, _ = render_prompt(styles[0], tpl, fn, 3, 8, kw[roles[0]])
        print(f"\nexample prompt ({scenarios[0]}/{styles[0]}/{roles[0]}, 5-of-8 opposite):\n{pr}")
        return 0

    written = []
    total_slices = len(scenarios) * len(styles) * len(roles)
    init_slice_state(total_slices)
    for scenario in scenarios:
        kw_by_role = role_keywords(scenario, conf.get("scenario_file"))
        for style in styles:
            if style not in RATIO_CANDIDATES:
                sys.exit(f"unknown style {style!r}; choose from {list(RATIO_CANDIDATES)}")
            tpl, fn = RATIO_CANDIDATES[style]
            counts = _blank_counts(roles)
            sources = []

            if merge_labels:
                if scenario != "baseline":
                    sys.exit(f"merge_from is only valid for the baseline scenario "
                             f"(the historical sweep's identity labels); got {scenario!r}")
                for ml in merge_labels:
                    added = merge_existing(counts, ml, style, roles)
                    sources.append({"label": ml, "samples_added": added})
                    print(f"[merge] {ml}: +{added} samples for {style}")

            raw_path = out_dir / "raw" / f"vf_{label}__{scenario}__{style}_raw.jsonl.gz"
            n_new = 0
            if not args.from_existing_only:
                meta = {"label": label, "model": conf["model"], "url": url,
                        "temperature": temperature, "grammar": grammar_on,
                        "seed": seed, "scenario": scenario, "style": style,
                        "sampling_mode": mode, "samples_per": n_per}
                with RawWriter(raw_path, meta) as raw:
                    n_new = sample_into_counts(
                        counts, lambda role, cell: alloc[cell], roles, kw_by_role,
                        style, tpl, fn, url, conf["model"], temperature,
                        grammar_on, concurrency, seed, raw,
                        ping_label=label, ping_ctx=f"{scenario}/{style}",
                        seed_ctx=("pilot", scenario))
                sources.append({"label": f"(new sampling {mode} N={n_per})",
                                "samples_added": n_new})

            vf = {
                "schema": "vf-1",
                "meta": {
                    "label": label, "model": conf.get("model"), "url": url,
                    "arm": arm, "style": style, "scenario": scenario,
                    "scenario_file": conf.get("scenario_file"),
                    "role_to_type": {"red": "type_a", "blue": "type_b"},
                    "role_labels": {r: kw_by_role[r]["agent_type"] for r in roles},
                    "temperature": temperature, "sampler_params": SAMPLER_PARAMS,
                    "grammar": grammar_on,
                    "grammar_sha256": (hashlib.sha256(GRAMMAR.encode()).hexdigest()
                                       if grammar_on else None),
                    "samples_mode": mode, "samples_per": n_per, "seed": seed,
                    # Clean measurement protocol (KV_CACHE_SAMPLING_ARTIFACT.md):
                    # no server-side prompt-cache reuse, deterministic per-request
                    # sampling seeds, and the serving build this was measured on.
                    "sampling_protocol": {
                        "cache_prompt": False, "seeded": True,
                        "seed_scheme": "crc32(stage|scenario|style|role|sim|occ|i)",
                        "server": server_fingerprint(url) if url else None,
                    },
                    "created": datetime.now().isoformat(timespec="seconds"),
                    "sources": sources,
                    "raw_replies": str(raw_path) if n_new else None,
                },
                "compositions": {}, "ratios": {},
            }
            for role in roles:
                comp_rows, ratio_rows = build_rows(counts, role)
                vf["compositions"][role] = comp_rows
                vf["ratios"][role] = ratio_rows

            vf_path = out_dir / f"vf_{label}__{scenario}__{style}.json"
            vf_path.write_text(json.dumps(vf, indent=1))
            written.append(vf_path)
            print(f"wrote {vf_path}")

            # Round-trip through the consumer-side validator so an artifact the
            # simulation would refuse is flagged HERE, listing the gaps.
            try:
                load_value_function(vf_path)
                print("  validated: complete 24-point value function per role")
            except ValueError as e:
                print(f"  ⚠️  INCOMPLETE (kept on disk; counts are mergeable): {e}")

            if args.plot:
                fig_path = out_dir / f"vf_{label}__{scenario}__{style}.{args.format}"
                plot_vf(vf, fig_path, dpi=args.dpi)
                print(f"wrote {fig_path}")

    print(f"\n{len(written)} artifact(s) in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
