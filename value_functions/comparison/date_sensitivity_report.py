#!/usr/bin/env python3
"""How much does the exact value function — and the scenario ordering built on
it — depend on the date llama.cpp bakes into the chat prompt?

    python value_functions/comparison/date_sensitivity_report.py --model llama

Inputs (run_date_sensitivity.sh; layout in paths.DATE_SENSITIVITY_DIR):
  results/date_sensitivity/<model>/tables/<date>/tables/vf_<label>[-d<date>]-lp__*.json
  results/date_sensitivity/<model>/simulations/<date>/run_*/        that date's simulation(s)
  llm_logprob/tables/vf_<label>-lp__*.json                          the canonical table
  experiments_with_llama_cpp/run_*_<sim>-vf-lp/                     the canonical 10k run
Canonical dates: llama 2024-07-26 (template default), mistral 2026-03-16 (release).

Reports
  1. table sensitivity: per cell, spread of P(MOVE) across dates and the
     logit-gap shift |Δ(z_MOVE − z_STAY)| = T·|logit p_date − logit p_canon|
     (T = 0.3), which is the physically meaningful scale; counts of cells
     with spread > 0.01 / 0.05 / 0.1.
  2. ordering by date: per date the scenario means of the final
     dissimilarity index, the induced ordering, and for each adjacent pair the
     gap, Cohen's d (pooled SD) and the rank-stability class if the pipeline
     wrote rank_pairs.csv. Ordering robustness = do the certified pairs keep
     their sign and class on every date.
Writes date_sensitivity_<model>.csv (per date x scenario) and
date_sensitivity_cells_<model>.csv next to the tables, and a figure.
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import DATE_SENSITIVITY_DIR, LOGPROB_TABLES_DIR, REPO_ROOT  # noqa: E402

T = 0.3
STYLE = "R3_dual_count"
SCEN = ["baseline", "race_white_black", "ethnic_asian_hispanic", "income_high_low",
        "political_liberal_conservative", "green_yellow"]
LABELS = {"llama": ("llama-3.3-70b-chat-grammar", "llama-3.3-70b"),
          "mistral": ("mistral-small-4-119b-chat-grammar", "mistral-small-4-119b")}


def load_table(path):
    vf = json.load(open(path))
    out = {}
    for role, rows in vf["compositions"].items():
        for c in rows:
            out[(role, c["n_similar"], c["n_occupied"])] = c["p_move_effective"]
    return out


def tables_for(label, ddir):
    """One table per scenario in ddir, whatever label suffix it carries (the
    06/11 Sep 2026 tables were extracted under the canonical label, the study
    dates under label-d<date>)."""
    out = {}
    for s in SCEN:
        hits = (glob.glob(os.path.join(ddir, "tables", f"vf_{label}*-lp__{s}__{STYLE}.json"))
                or glob.glob(os.path.join(ddir, f"vf_{label}*-lp__{s}__{STYLE}.json")))
        if hits:
            for k, v in load_table(sorted(hits)[0]).items():
                out[(s,) + k] = v
    return out


def logit(p, eps=1e-9):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def finals(run):
    out = {}
    for d in glob.glob(f"{run}/experiments/llm_*"):
        s = os.path.basename(d).split("_", 1)[1].rsplit("_", 2)[0]
        out[s] = pd.read_csv(f"{d}/run_summary.csv")["dissimilarity_index"].to_numpy()
    return out


def rank_classes(run):
    f = glob.glob(f"{run}/**/rank_pairs.csv", recursive=True)
    if not f:
        return {}
    t = pd.read_csv(f[0]); t = t[t.metric == "dissimilarity_index"]
    return {(r.hi, r.lo): r["class"] for _, r in t.iterrows()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama", choices=list(LABELS))
    args = ap.parse_args()
    label, sim = LABELS[args.model]
    ds = DATE_SENSITIVITY_DIR / args.model
    canon_date = {"llama": "2024-07-26", "mistral": "2026-03-16"}[args.model]
    canon = tables_for(label, str(LOGPROB_TABLES_DIR))
    dates = {f"{canon_date} (canonical)": canon}
    for d in sorted(glob.glob(str(ds / "tables" / "*"))):
        dd = os.path.basename(d)
        if dd == canon_date:
            continue
        t = tables_for(label, d)
        if t:
            dates[dd] = t
    if len(dates) < 2:
        sys.exit("need at least one date-pinned table besides the canonical one")
    cells = sorted(canon)

    # 1. table sensitivity
    rows = []
    for c in cells:
        vals = {d: t[c] for d, t in dates.items() if c in t}
        spread = max(vals.values()) - min(vals.values())
        dz = max(T * abs(logit(v) - logit(canon[c])) for v in vals.values())
        rows.append({"scenario": c[0], "role": c[1], "n_similar": c[2], "n_occupied": c[3],
                     "p_canonical": canon[c], "spread_p": spread, "max_dz_nats": dz,
                     **{f"p_{d.split()[0]}": v for d, v in vals.items()}})
    cdf = pd.DataFrame(rows)
    (ds / "figures").mkdir(parents=True, exist_ok=True)
    cdf.to_csv(ds / "figures" / f"date_sensitivity_cells_{args.model}.csv", index=False)
    print(f"{args.model}: {len(dates)} dates x {len(cells)} cells")
    print(f"  cells with P(MOVE) spread > 0.01: {(cdf.spread_p > 0.01).sum()}   > 0.05: {(cdf.spread_p > 0.05).sum()}"
          f"   > 0.10: {(cdf.spread_p > 0.10).sum()}   max spread {cdf.spread_p.max():.3f}")
    print(f"  logit-gap shift vs canonical: median {cdf.max_dz_nats.median():.3f} nats, p90 {cdf.max_dz_nats.quantile(.9):.3f}, max {cdf.max_dz_nats.max():.3f}")

    # 2. ordering by date
    runs = {}
    for d in dates:
        dd = d.split()[0]; tag = "d" + dd.replace("-", "")
        pat = (str(REPO_ROOT / f"experiments_with_llama_cpp/run_*_{sim}-vf-lp") if "canonical" in d
               else str(ds / "simulations" / dd / f"run_*_{sim}-vf-lp*"))
        cands = sorted(glob.glob(pat))
        cands = [r for r in cands if len(glob.glob(f"{r}/experiments/llm_*")) == 6]
        if cands:                                   # the fullest run for that date (10k beats 1k)
            runs[d] = max(cands, key=lambda r: min(len(v) for v in finals(r).values()))
    print("\nordering by date (final dissimilarity index):")
    orows = []
    ref_order = None
    for d, r in runs.items():
        f = finals(r); order = sorted(f, key=lambda s: -f[s].mean()); cls = rank_classes(r)
        n = min(len(v) for v in f.values())
        line = " > ".join(f"{s[:8]} {f[s].mean():.3f}" for s in order)
        same = "" if ref_order is None else ("  (same order)" if order == ref_order else "  (ORDER DIFFERS)")
        ref_order = ref_order or order
        print(f"  {d:<24} n={n:<6} {line}{same}")
        for a, b in zip(order, order[1:]):
            x, y = f[a][:n], f[b][:n]
            dd_ = (x.mean() - y.mean()) / math.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
            orows.append({"date": d, "n_runs": n, "hi": a, "lo": b, "gap": x.mean() - y.mean(),
                          "cohen_d": dd_, "class": cls.get((a, b), "")})
            print(f"      {a[:14]:<14} > {b[:14]:<14} gap {x.mean()-y.mean():+.4f}  d {dd_:5.2f}  {cls.get((a, b), '')}")
    odf = pd.DataFrame(orows)
    odf.to_csv(ds / "figures" / f"date_sensitivity_{args.model}.csv", index=False)

    # figure: cells vs canonical | shift histogram | levels by date | rank bump chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    for d, t in dates.items():
        if "canonical" in d:
            continue
        xs = [canon[c] for c in cells if c in t]; ys = [t[c] for c in cells if c in t]
        ax.scatter(xs, ys, s=8, alpha=.6, label=d)
    ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.set_xlabel(f"P(MOVE), canonical ({canon_date})"); ax.set_ylabel("P(MOVE), other pinned date")
    ax.set_title("exact table vs the date in the prompt, every cell"); ax.legend(fontsize=7)
    ax = axes[0, 1]
    ax.hist(cdf.max_dz_nats[cdf.max_dz_nats > 0], bins=40, color="C0")
    ax.set_xlabel("max |Δ(z_MOVE − z_STAY)| across dates, nats (T=0.3)"); ax.set_ylabel("cells")
    ax.set_title(f"logit-gap shift per cell (cells with any shift: {(cdf.max_dz_nats > 0).sum()}/{len(cdf)})")
    ax = axes[1, 0]
    piv = {}
    for d, r in runs.items():
        f = finals(r); piv[d] = {s_: f[s_].mean() for s_ in f}
    pdf = pd.DataFrame(piv)
    if not pdf.empty:
        pdf = pdf.loc[sorted(pdf.index, key=lambda s_: -pdf.iloc[:, 0][s_])]
        cols = sorted(pdf.columns, key=lambda c: c.split()[0])
        for s_ in pdf.index:
            ax.plot(range(len(cols)), pdf.loc[s_, cols].values, marker="o", label=s_)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([c.split()[0] for c in cols], rotation=30, fontsize=8)
        ax.set_ylabel("mean final dissimilarity index"); ax.set_title("scenario levels by pinned date"); ax.legend(fontsize=7)
        ax = axes[1, 1]
        ranks = pdf[cols].rank(ascending=False)
        for s_ in ranks.index:
            ax.plot(range(len(cols)), ranks.loc[s_].values, marker="o", label=s_)
        ax.set_yticks(range(1, len(ranks) + 1)); ax.invert_yaxis()
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([c.split()[0] for c in cols], rotation=30, fontsize=8)
        ax.set_ylabel("rank by dissimilarity index"); ax.set_title("scenario ORDER by pinned date (bump chart)")
        for i, c in enumerate(cols):
            cls = odf[odf.date == c]
            for _, r in cls.iterrows():
                if r["class"] and r["class"] != "CERTIFIED":
                    ax.annotate(r["class"].replace("FLOOR-", "").lower(), (i, ranks.loc[r.hi, c] + 0.5), fontsize=6, ha="center", color="gray")
    fig.suptitle(f"{args.model}: date-sensitivity of the exact value function and of the scenario ordering", fontsize=12)
    fig.tight_layout()
    out = ds / "figures" / f"date_sensitivity_{args.model}.png"
    fig.savefig(out, dpi=150); print(f"\nwrote {out}")

    # REPORT.md: the numbers, generated; the narrative sits above the marker and is kept
    rp = ds / "REPORT.md"
    head = ""
    if rp.exists():
        txt = rp.read_text()
        head = txt.split("<!-- generated below -->")[0]
    lines = ["<!-- generated below -->", f"\n_Generated by `value_functions/comparison/date_sensitivity_report.py --model {args.model}`._\n",
             f"## Numbers ({len(dates)} dates)\n",
             f"- Dates: {', '.join(sorted(dates, key=lambda d: d.split()[0]))}",
             f"- Cells with P(MOVE) spread > 0.01: **{(cdf.spread_p > 0.01).sum()}** / {len(cdf)}; > 0.05: {(cdf.spread_p > 0.05).sum()}; > 0.10: {(cdf.spread_p > 0.10).sum()}; max spread {cdf.spread_p.max():.3f}",
             f"- Logit-gap shift vs canonical: median {cdf.max_dz_nats.median():.3f} nats, p90 {cdf.max_dz_nats.quantile(.9):.3f}, max {cdf.max_dz_nats.max():.3f}\n",
             "### Scenario ordering by date\n", "| date | n runs | ordering (mean final DI) |", "|---|---|---|"]
    for d, r in runs.items():
        f = finals(r); order = sorted(f, key=lambda s_: -f[s_].mean()); n = min(len(v) for v in f.values())
        lines.append(f"| {d} | {n} | " + " > ".join(f"{s_} {f[s_].mean():.3f}" for s_ in order) + " |")
    lines += ["", "### Adjacent pairs by date (gap, Cohen's d, rank-stability class)\n", "| date | pair | gap | d | class |", "|---|---|---|---|---|"]
    for _, r in odf.iterrows():
        lines.append(f"| {r.date} | {r.hi} > {r.lo} | {r.gap:+.4f} | {r.cohen_d:.2f} | {r['class']} |")
    lines += ["", f"![figure](figures/date_sensitivity_{args.model}.png)", ""]
    rp.write_text(head + "\n".join(lines))
    print(f"wrote {rp}")


if __name__ == "__main__":
    main()
