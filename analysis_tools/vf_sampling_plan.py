#!/usr/bin/env python3
"""Turn an INSUFFICIENT multi-split verdict into a concrete sampling plan.

    python analysis_tools/vf_sampling_plan.py --label qwen3.6-27b-chat-grammar \
        --ratio 0.775 [--margin 0.7] [--current-precision 0.02] [--rate 8000]

The multi-split check answers "is the sampling sufficient?". This answers the
follow-up: "then how much more do I need?" — in samples and in GPU-hours.

THE MATH
--------
1. Each cell is calibrated to a 95% CI half-width w, so its estimate carries a
   standard error of sigma_cell = w / 1.96.

2. A simulation metric is, to first order, a linear functional of the table:
   a perturbation eps_i in cell i moves the metric by g_i·eps_i, with
   g_i = dM/dp_i. Cell errors are independent, so

       SD(metric displacement) = sqrt( sum_i g_i^2 sigma_i^2 )

   With every cell calibrated to the same w, sigma_i = w/1.96 for all i and the
   sum factorises:  SD = (w/1.96)·sqrt(sum_i g_i^2)  —  i.e. PROPORTIONAL to w.

3. RMS(Delta) from the multi-split IS an estimate of that SD (the half-vs-full
   displacement has exactly the variance of the full artifact's own error), and
   tau depends only on run-to-run spread and the run count, NOT on sampling.
   Therefore

       ratio = RMS(Delta)/tau  is proportional to  w

   so to bring an observed ratio r down to a target margin m:

       w_next = w_cur · (m / r)

4. r is ITSELF an estimate, from B splits, with relative SE ~ 1/sqrt(2B). Using
   the point estimate lands w_next exactly on the pass/fail boundary, where
   about half of re-checks fail by construction. So the plan is built against
   the bootstrap 97.5% upper bound of the worst ratio (emitted by
   vf_multisplit_check.py as worst_ratio_ci_high):

       w_next = w_cur · (m / r_97.5%)

   This replaced a flat --safety 0.90, which was an arbitrary constant: it is
   only the correct headroom in the limit B -> inf, and under-bought by 1.8x at
   B=8 and 1.26x at B=32. The bound derives the headroom from the measured
   spread, so raising B makes the plan CHEAPER rather than merely more certain.

5. Cost follows from the sample-size formula n = z^2·p(1-p)/w^2, evaluated per
   cell against what is already on disk. Because n ~ 1/w^2, halving w quadruples
   the samples — which is why the plan is priced, not guessed.

CAVEAT worth stating: step 2 assumes every cell sits AT the target w. Saturated
cells sit far below it (their intervals are much tighter than w), so the
proportionality is approximate and holds best for the near-p=0.5 cells that
dominate the displacement. Treat the result as a first-order plan and RE-MEASURE
after resampling rather than trusting it to land exactly on the margin.
"""
import argparse
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent
REPO_ROOT = _THIS.parent
for _p in (REPO_ROOT, REPO_ROOT / "prompt_refinement", REPO_ROOT / "value_functions" / "sampling"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from build_value_function import project_total  # noqa: E402

VF_DIR = REPO_ROOT / "value_functions" / "results" / "sampled"
SCENARIOS = ["baseline", "race_white_black", "ethnic_asian_hispanic",
             "income_high_low", "political_liberal_conservative", "green_yellow"]

# Measured sampling throughput, requests/hour, from the campaign logs.
RATES = {"qwen": 8000, "gemma": 8700, "llama": 7000, "deepseek": 4972,
         "mistral": 3600, "phi": 18000, "granite": 9000, "olmo": 8000}


def deficit_at(label, style, w, scenarios=SCENARIOS):
    """(total extra samples, per-scenario dict) to reach half-width w.

    Calls project_total — the SAME advance_pass loop --calibrate will walk —
    rather than a single topup_deficits call. A one-shot deficit is what the
    calibrator used to buy, but with incremental top-up it converges to a
    fixed point roughly half that (qwen at w=0.0158: 234k one-shot vs 111k
    incremental). Quoting the one-shot figure would over-price by ~2.1x and
    make the loop's GPU-hour gate refuse work that is comfortably affordable.
    Sharing the function means the quote cannot drift from the spend.
    """
    total, per = 0, {}
    for sc in scenarios:
        p = VF_DIR / f"vf_{label}__{sc}__{style}.json"
        if not p.exists():
            continue
        vf = json.loads(p.read_text())
        counts = {(role, (c["n_similar"], c["n_occupied"])):
                  {"move": c["n_move"], "stay": c["n_stay"],
                   "samples": c["n_samples"]}
                  for role, rows in vf["compositions"].items() for c in rows}
        d = sum(project_total(counts, list(vf["compositions"]), w).values())
        per[sc] = d
        total += d
    return total, per


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--style", default="R3_dual_count")
    ap.add_argument("--ratio", type=float,
                    help="observed worst RMS(Delta)/tau; read from "
                         "multisplit_status.json when --status is given")
    ap.add_argument("--status", type=Path,
                    help="path to multisplit_status.json (supplies --ratio/--margin)")
    ap.add_argument("--margin", type=float, default=1.0,
                    help="target ratio. 1.0 == table noise may not exceed run "
                         "noise, on the tau-as-SE scale (see vf_multisplit_check)")
    ap.add_argument("--allow-legacy-scale", action="store_true",
                    help="accept a status file with no tau_definition marker "
                         "(pre-2026-09-01, tau = 1.96*SE). Its ratios are 1/1.96 "
                         "of the current scale, so the default margin would be "
                         "applied to the wrong convention.")
    ap.add_argument("--current-precision", type=float, default=0.02)
    ap.add_argument("--safety", type=float, default=1.0,
                    help="extra shrink on top of the bootstrap bound. Default 1.0"
                         " — the bound already supplies the headroom; set <1 only"
                         " to be deliberately more conservative than 97.5%%")
    ap.add_argument("--point-estimate", action="store_true",
                    help="plan against the worst ratio itself rather than its "
                         "upper bound (lands ON the pass/fail line — expect "
                         "roughly a coin flip on the re-check)")
    ap.add_argument("--rate", type=float, default=None,
                    help="sampling throughput req/hour (default: measured per model)")
    ap.add_argument("--json", type=Path, default=None,
                    help="also write the plan as JSON here. The sufficiency "
                         "loop consumes this file — never this script's stdout. "
                         "Scraping stdout is what made the old bash calibration "
                         "loop read a missing artifact as converged.")
    args = ap.parse_args()

    ratio, margin = args.ratio, args.margin
    basis = "supplied on the command line"
    if args.status and args.status.exists():
        st = json.loads(args.status.read_text())
        # w_next = w_cur * margin/ratio is scale-invariant ONLY when margin and
        # ratio come from the same tau convention. Mixing a pre-2026-09-01
        # status (tau = 1.96*SE) with the current margin would silently plan
        # against a target 1.96x off, so refuse rather than guess.
        if st.get("tau_definition") != "se" and not args.allow_legacy_scale:
            ap.error(f"{args.status} has no tau_definition=='se' marker — it "
                     f"predates the 2026-09-01 tau change and its ratios are on "
                     f"the old 1.96-scaled convention. Re-run "
                     f"vf_multisplit_check.py, or pass --allow-legacy-scale "
                     f"together with an explicit --margin on that scale.")
        margin = st.get("margin", margin)
        print(f"status: {st['label']}  verdict={st['verdict']}  "
              f"{st['n_flags']}/{st['n_checks']} flags  worst={st['worst_check']}  "
              f"splits={st.get('splits', '?')}")
        if ratio is None:
            point = float(st["worst_ratio"])
            hi = st.get("worst_ratio_ci_high")
            if args.point_estimate or hi is None:
                ratio = point
                basis = ("point estimate (--point-estimate)" if args.point_estimate
                         else "point estimate — status has no CI, re-run the "
                              "check to get one")
            else:
                ratio = float(hi)
                basis = (f"bootstrap 97.5% upper bound over the max "
                         f"(point estimate {point:.3f})")
    if ratio is None:
        ap.error("need --ratio or a --status file")

    rate = args.rate or next((v for k, v in RATES.items() if k in args.label), 8000)
    w0 = args.current_precision
    have, _ = deficit_at(args.label, args.style, w0)

    def emit(payload):
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(payload, indent=1))
            print(f"\nwrote {args.json}")

    print(f"\nplanning ratio {ratio:.3f} vs margin {margin:.2f} "
          f"-> {'INSUFFICIENT' if ratio > margin else 'already sufficient'}")
    print(f"  basis: {basis}")
    if ratio <= margin:
        emit({"label": args.label, "action": "none", "ratio": ratio,
              "margin": margin, "basis": basis, "w_current": w0,
              "w_next": w0, "extra_samples": 0, "gpu_hours": 0.0})
        return 0
    w_next = w0 * (margin / ratio) * args.safety
    safety_txt = f" x {args.safety}" if args.safety != 1.0 else ""
    print(f"required precision: w_next = {w0} x ({margin}/{ratio:.3f}){safety_txt}"
          f" = {w_next:.5f}")
    print(f"(residual at the CURRENT target {w0}: {have} samples — "
          f"{'already calibrated' if have == 0 else 'NOT yet calibrated'})\n")

    print(f"cost ladder (throughput {rate:,.0f} req/h for this model):")
    print(f"  {'w':>8s} {'extra samples':>14s} {'GPU-h':>7s} {'projected ratio':>16s}")
    cands = sorted({round(w0 * f, 5) for f in (0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5)}
                   | {round(w_next, 5)}, reverse=True)
    for w in cands:
        tot, _ = deficit_at(args.label, args.style, w)
        mark = "  <- plan" if abs(w - round(w_next, 5)) < 1e-9 else ""
        print(f"  {w:8.5f} {tot:14,d} {tot/rate:7.1f} {ratio*w/w0:16.2f}{mark}")

    tot, per = deficit_at(args.label, args.style, w_next)
    print(f"\nplan: recalibrate at ±{w_next:.4f} -> {tot:,} extra samples "
          f"(~{tot/rate:.1f} GPU-h), then RE-RUN the multi-split check.")
    print("  per scenario:")
    for sc, d in sorted(per.items(), key=lambda kv: -kv[1]):
        print(f"    {sc:32s} {d:9,d}")
    print(f"\n  command:\n    ./run_vf_model_campaign.sh <label> <sampling.yaml> <model.gguf> ...\n"
          f"    # or, server already up:\n"
          f"    .venv/bin/python value_functions/sampling/build_value_function.py \\\n"
          f"        --config configs/value_function_scenarios_<m>.yaml \\\n"
          f"        --calibrate --precision {w_next:.4f} --plot")
    emit({"label": args.label, "action": "resample", "ratio": ratio,
          "margin": margin, "basis": basis, "w_current": w0,
          "w_next": round(w_next, 6), "extra_samples": int(tot),
          "gpu_hours": round(tot / rate, 3), "rate_req_per_hour": rate,
          "per_scenario": {k: int(v) for k, v in per.items()}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
