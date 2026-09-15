#!/usr/bin/env python3
"""How much is a 'RESOLVED' cell actually worth?

    python value_functions/comparison/adjudication_power.py --label llama-3.3-70b-chat-grammar

THE FLAW THIS MEASURES
build_consensus_table.py adjudicates two disagreeing extractions by asking
which one falls inside the sanity Wilson CI. That rule picks a winner from the
SAME sample that defines the interval, and its power against the loser is never
stated. When the two candidates are close relative to the CI half-width, "only
one inside" is near a coin flip: a fresh sample of the same size would hand the
verdict to the other candidate a good fraction of the time. A verdict that does
not survive resampling is not evidence, it is noise with a label on it.

WHAT IS COMPUTED, per cell, exactly (no simulation — Binomial(n, p) with
n <= a few thousand is enumerable):

  P(repeat), P(flip), P(inconclusive)
      Take the CI winner as the working hypothesis, p = v_win. Draw a fresh
      k' ~ Binomial(n, v_win), rebuild the Wilson CI from k', re-apply the
      same "exactly one inside" rule, and total the probability of each
      outcome. P(flip) is the chance the rule reverses itself on new data of
      the size already used.

  log10 BF
      The likelihood ratio L(v1)/L(v2) on the counts actually observed, where
      L(v) = v^k (1-v)^(n-k) is the binomial likelihood — the probability of
      the observed data as a function of the candidate value. This is the
      whole of the evidence for one candidate over the other (Neyman-Pearson:
      for two simple hypotheses the LR is the most powerful statistic), and
      unlike CI membership it is a continuous measure that does not depend on
      where an interval boundary happens to land.

  n_decisive
      The smallest sample size at which a PRE-COMMITTED LR test separates the
      two candidates: decide v1 if log BF >= log K, v2 if <= -log K, otherwise
      inconclusive. Chosen so that, under either candidate being true, the
      probability of the wrong call is <= --alpha and of no call <= --beta.
      Computed by exact enumeration, not a normal approximation, because these
      cells are mostly saturated near 0 or 1 where that approximation is worst.

  absolute fit
      The LR is relative: it can crown a winner when BOTH candidates are wrong.
      So the two-sided exact binomial p-value for H0: p = v_win is reported
      alongside. A cell only earns its value if the winner beats the loser AND
      is not itself rejected.

Writes adjudication_power_<label>.csv next to the consensus table. Reads only;
extracts nothing, needs no GPU or server.
"""
import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import SEQCHECK_PLOTS_DIR  # noqa: E402

EPS = 1e-12          # keeps log-likelihoods finite at a candidate of exactly 0 or 1
CI_EPS = 1e-9        # Wilson limits are analytically 0/1 at k=0/k=n but land ~1e-16 inside


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def log_binom_pmf(k, n, p):
    """log P(K = k) for K ~ Binomial(n, p), via lgamma so large n stays stable."""
    p = min(max(p, EPS), 1 - EPS)
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log(1 - p))


def log_lr(k, n, v1, v2):
    """log L(v1)/L(v2) on k successes in n draws. The binomial coefficient cancels."""
    a1, a2 = min(max(v1, EPS), 1 - EPS), min(max(v2, EPS), 1 - EPS)
    return k * math.log(a1 / a2) + (n - k) * math.log((1 - a1) / (1 - a2))


def ci_verdict(k, n, v1, v2):
    """Re-apply the consensus rule to a count: 'v1', 'v2', or None if it cannot call."""
    lo, hi = wilson(k, n)
    in1 = lo - CI_EPS <= v1 <= hi + CI_EPS
    in2 = lo - CI_EPS <= v2 <= hi + CI_EPS
    if in1 and not in2:
        return "v1"
    if in2 and not in1:
        return "v2"
    return None


def verdict_stability(n, v1, v2, v_win_key):
    """Exact P(repeat / flip / inconclusive) for the CI rule under p = the winner."""
    p_true = v1 if v_win_key == "v1" else v2
    rep = flip = inconc = 0.0
    for k in range(n + 1):
        w = math.exp(log_binom_pmf(k, n, p_true))
        v = ci_verdict(k, n, v1, v2)
        if v is None:
            inconc += w
        elif v == v_win_key:
            rep += w
        else:
            flip += w
    return rep, flip, inconc


def lr_error_rates(n, v1, v2, log_k):
    """Worst-case P(wrong call) and P(no call) for the pre-committed LR test at size n."""
    worst_err = worst_inc = 0.0
    for truth, other in (("v1", v1), ("v2", v2)):
        p_true = v1 if truth == "v1" else v2
        err = inc = 0.0
        for k in range(n + 1):
            w = math.exp(log_binom_pmf(k, n, p_true))
            s = log_lr(k, n, v1, v2)
            if s >= log_k:
                call = "v1"
            elif s <= -log_k:
                call = "v2"
            else:
                call = None
            if call is None:
                inc += w
            elif call != truth:
                err += w
        worst_err = max(worst_err, err)
        worst_inc = max(worst_inc, inc)
    return worst_err, worst_inc


def n_decisive(v1, v2, alpha, beta, log_k, n_cap=200000):
    """Smallest n meeting both error targets. KL gives the starting guess; then step."""
    a1, a2 = min(max(v1, EPS), 1 - EPS), min(max(v2, EPS), 1 - EPS)
    kl = (a1 * math.log(a1 / a2) + (1 - a1) * math.log((1 - a1) / (1 - a2)))
    kl2 = (a2 * math.log(a2 / a1) + (1 - a2) * math.log((1 - a2) / (1 - a1)))
    kl = max(min(kl, kl2), 1e-12)
    n = max(2, int(log_k / kl))
    # geometric search up, then bisect, keeping the exact enumeration affordable
    lo = 1
    while n <= n_cap:
        err, inc = lr_error_rates(n, v1, v2, log_k)
        if err <= alpha and inc <= beta:
            break
        lo = n + 1
        n *= 2
    if n > n_cap:
        return None
    hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        err, inc = lr_error_rates(mid, v1, v2, log_k)
        if err <= alpha and inc <= beta:
            hi = mid
        else:
            lo = mid + 1
    return lo


def binom_two_sided_p(k, n, p0):
    """Exact two-sided binomial test of H0: p = p0, by the method of small likelihoods."""
    p0 = min(max(p0, EPS), 1 - EPS)
    obs = log_binom_pmf(k, n, p0)
    tot = 0.0
    for i in range(n + 1):
        lp = log_binom_pmf(i, n, p0)
        if lp <= obs + 1e-9:
            tot += math.exp(lp)
    return min(1.0, tot)


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--alpha", type=float, default=0.01, help="max P(wrong call)")
    ap.add_argument("--beta", type=float, default=0.05, help="max P(no call)")
    ap.add_argument("--bf", type=float, default=100.0, help="likelihood-ratio threshold K")
    ap.add_argument("--n-cap", type=int, default=200000)
    args = ap.parse_args()

    cpath = os.path.join(SEQCHECK_PLOTS_DIR, f"consensus_{args.label}.csv")
    spath = os.path.join(SEQCHECK_PLOTS_DIR, f"sanity_vs_exact_{args.label}.csv")
    for p in (cpath, spath):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")

    counts = {}
    for r in read_csv(spath):
        key = (r["scenario"], r["role"], int(r["n_similar"]), int(r["n_occupied"]))
        counts[key] = (int(r["n_move"]), int(r["n_samples"]))

    cons = read_csv(cpath)
    run2_col = next(c for c in cons[0] if c.startswith("p_reextract_"))
    log_k = math.log(args.bf)

    out = []
    for r in cons:
        if r["status"] == "STABLE":
            continue
        key = (r["scenario"], r["role"], int(r["n_similar"]), int(r["n_occupied"]))
        if key not in counts:
            continue
        k, n = counts[key]
        v1, v2 = float(r["p_run1_stored"]), float(r[run2_col])
        lr = log_lr(k, n, v1, v2)                     # evidence for run1 over run2
        lead = "v1" if lr >= 0 else "v2"              # which the LIKELIHOOD favours
        v_lead = v1 if lead == "v1" else v2
        v_trail = v2 if lead == "v1" else v1
        log10_bf = abs(lr) / math.log(10)
        win = ci_verdict(k, n, v1, v2)                # what the CI rule said
        if win is not None:
            rep, flip, inc = verdict_stability(n, v1, v2, win)
        else:
            rep = flip = inc = float("nan")
        nd = n_decisive(v1, v2, args.alpha, args.beta, log_k, args.n_cap)
        out.append(dict(
            scenario=r["scenario"], role=r["role"],
            n_similar=r["n_similar"], n_occupied=r["n_occupied"],
            ci_status=r["status"], k=k, n=n, p_hat=round(k / n, 4),
            v_lead=v_lead, v_trail=v_trail, spread=round(abs(v1 - v2), 6),
            lead_run="run1" if lead == "v1" else "run2",
            ci_pick=("run1" if win == "v1" else "run2" if win == "v2" else "none"),
            agrees=(win is None or win == lead),
            p_repeat=round(rep, 4), p_flip=round(flip, 4), p_inconclusive=round(inc, 4),
            log10_bf=round(log10_bf, 3),
            decisive=bool(log10_bf >= log_k / math.log(10)),
            abs_fit_p=round(binom_two_sided_p(k, n, v_lead), 4),
            n_decisive=nd if nd is not None else f">{args.n_cap}",
        ))

    out.sort(key=lambda d: (d["ci_status"], -d["log10_bf"]))
    opath = os.path.join(SEQCHECK_PLOTS_DIR, f"adjudication_power_{args.label}.csv")
    with open(opath, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    print(f"{args.label}: {len(out)} cells where the two extractions disagree\n")
    hdr = (f"{'CI status':<11}{'scenario':<32}{'role':<6}{'cell':<7}{'lead':>8}{'trail':>8}"
           f"{'k/n':>9}{'log10BF':>9}{'fit p':>7}{'P(flip)':>9}{'n_dec':>8}")
    print(hdr)
    print("-" * len(hdr))
    for d in out:
        pf = "-" if d["p_flip"] != d["p_flip"] else f"{d['p_flip']:.3f}"
        print(f"{d['ci_status']:<11}{d['scenario']:<32}{d['role']:<6}"
              f"{d['n_similar']+'/'+d['n_occupied']:<7}{d['v_lead']:>8.4f}{d['v_trail']:>8.4f}"
              f"{str(d['k'])+'/'+str(d['n']):>9}{d['log10_bf']:>9.2f}{d['abs_fit_p']:>7.3f}"
              f"{pf:>9}{str(d['n_decisive']):>8}")

    res = [d for d in out if d["ci_status"] == "RESOLVED"]
    unr = [d for d in out if d["ci_status"] == "UNRESOLVED"]
    thr = log_k / math.log(10)
    print(f"\nCI-rule RESOLVED  {len(res):>3}  of which BF >= {args.bf:g}: {sum(d['decisive'] for d in res)}"
          f"   BF < 10: {sum(d['log10_bf'] < 1 for d in res)}"
          f"   CI pick != likelihood lead: {sum(not d['agrees'] for d in res)}")
    print(f"CI-rule UNRESOLVED {len(unr):>3}  of which BF >= {args.bf:g}: {sum(d['decisive'] for d in unr)}"
          f"   (the CI rule could not call these, the likelihood already can)")
    print(f"leader itself rejected p<.05 : {sum(d['abs_fit_p'] < 0.05 for d in out)}/{len(out)}")
    budget = [d["n_decisive"] for d in out if isinstance(d["n_decisive"], int)]
    print(f"\nfresh draws for a pre-committed LR test on every disagreeing cell "
          f"(alpha={args.alpha}, beta={args.beta}, K={args.bf:g}):")
    print(f"   {len(budget)} cells, {sum(budget):,} draws, "
          f"~{sum(budget)*1.065/3600:.1f} h at 1.065 s/draw"
          + (f"   ({len(out)-len(budget)} cells exceed --n-cap)" if len(out) > len(budget) else ""))
    print(f"\nwrote {opath}")


if __name__ == "__main__":
    main()
