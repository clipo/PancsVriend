"""The census escalation: fresh draws, never-issued seeds, one round per census.

    .venv/bin/python -m pytest tests/test_census_escalation.py -q

No server: the model call is faked, and the tables are tiny synthetic ones in
tmp_path (nothing here reads value_functions/results/). Covers:
- sample_into_counts: default seeds unchanged; index_offset_fn moves both the
  seed index and the recorded sample index
- RawWriter(append=True) keeps earlier rounds (one _meta header per round)
- sanity_vs_exact end to end: stage 1 from the census raw only; ONE escalation
  of the census misses with seeds escalate@<census id> starting past the
  cell's draws; fresh counts recorded in the table; a second escalation of the
  same census refused (exit 4) with no draws; a report-only re-run reuses the
  recorded fresh draws; two verdict blocks (goodness_of_fit / sanity_check)
- cell_tests directly: Holm across the family, and BEYOND_CAP for a cell
  whose fresh n cannot reach the target power
- census.py's one-line summary reads the new verdict layout
No test reads scenarios_a2.py: the baseline keywords come from KW_BY_ROLE.
"""
import gzip
import json
import sys
from pathlib import Path

from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "prompt_refinement"), str(REPO / "value_functions" / "sampling"),
          str(REPO / "value_functions" / "comparison")):
    if p not in sys.path:
        sys.path.insert(0, p)

import build_value_function as bvf  # noqa: E402
import sanity_vs_exact as sve  # noqa: E402
from ratio_prompt_templates import RATIO_CANDIDATES  # noqa: E402
from sampling_common import RawWriter, request_seed, role_keywords  # noqa: E402

STYLE, SCEN = "R3_dual_count", "baseline"


class _Raw:
    def __init__(self):
        self.records = []

    def write(self, rec):
        self.records.append(dict(rec))


def _fake_batch(seen):
    """Deterministic in the seed: MOVE for one seed in ten (~10%)."""
    def sample_batch(url, model, prompts, temperature, grammar, concurrency,
                     cache_prompt=None, seed_fn=None):
        seeds = [seed_fn(i) for i in range(len(prompts))]
        seen.extend(seeds)
        return [{"text": "MOVE" if s % 10 == 0 else "STAY", "finish_reason": "stop",
                 "completion_tokens": 1} for s in seeds]
    return sample_batch


@pytest.fixture
def fake_model(monkeypatch):
    seen = []
    monkeypatch.setattr(bvf, "sample_batch", _fake_batch(seen))
    monkeypatch.setattr(bvf, "slice_ping", lambda *a, **k: None)
    return seen


def _sample(offset_fn, k=5):
    tpl, fn = RATIO_CANDIDATES[STYLE]
    counts, raw = bvf._blank_counts(["red"]), _Raw()
    bvf.sample_into_counts(counts, lambda r, c: k if c == (4, 5) else 0, ["red"],
                           role_keywords(SCEN), STYLE, tpl, fn, "u", "m",
                           0.3, True, 1, 0, raw, "t", None, seed_ctx=("escalate", SCEN),
                           index_offset_fn=offset_fn)
    return raw.records


def test_default_seeds_unchanged_and_offset_moves_them(fake_model):
    recs = _sample(None)
    assert [r["sample"] for r in recs] == list(range(5))
    assert [r["seed"] for r in recs] == [request_seed("escalate", SCEN, STYLE, "red", 4, 5, i)
                                         for i in range(5)]
    recs2 = _sample(lambda r, c: 100)
    assert [r["sample"] for r in recs2] == list(range(100, 105))
    assert not {r["seed"] for r in recs} & {r["seed"] for r in recs2}


def test_rawwriter_append_keeps_rounds(tmp_path):
    p = tmp_path / "x.jsonl.gz"
    with RawWriter(p, {"round": 1}) as w:
        w.write({"a": 1})
    with RawWriter(p, {"round": 2}, append=True) as w:
        w.write({"a": 2})
    lines = [json.loads(l) for l in gzip.open(p, "rt")]
    assert [l.get("round") for l in lines if l.get("_meta")] == [1, 2]
    assert [l["a"] for l in lines if not l.get("_meta")] == [1, 2]
    with RawWriter(p, {"round": 3}) as w:                  # default still replaces
        pass
    assert len([json.loads(l) for l in gzip.open(p, "rt")]) == 1


# --------------------------------------------------------------------------- end to end

CELLS = {  # (n_similar, n_occupied): (p_exact, census moves out of 100)
    (2, 4): (0.5, 50),     # consistent
    (4, 5): (0.1, 30),     # census miss -> escalated; fresh draws ~10% -> consistent
                           # (exact interval [0.09, 0.11]: the composite null is exercised)
    (0, 0): (0.0, 0),      # saturated, exact
}


def _write_inputs(root, label):
    exact = {"compositions": {"red": [
        {"n_similar": s, "n_occupied": o, "p_move_effective": p,
         "ci95": [0.09, 0.11] if (s, o) == (4, 5) else [p, p]}
        for (s, o), (p, _) in CELLS.items()]}}
    (root / "exact").mkdir()
    (root / "exact" / f"vf_{label}-lp__{SCEN}__{STYLE}.json").write_text(json.dumps(exact))
    comps = [{"n_similar": s, "n_occupied": o, "n_samples": 100, "n_move": k, "n_stay": 100 - k,
              "n_bad": 0} for (s, o), (_, k) in CELLS.items()]
    table = {"schema": "vf-1", "meta": {"label": label, "model": "fake", "temperature": 0.3,
                                        "seed": 0, "created": "UNIT-CENSUS"},
             "compositions": {"red": comps}, "ratios": {}}
    (root / "tables").mkdir()
    (root / "tables" / f"vf_{label}-sanity__{SCEN}__{STYLE}.json").write_text(json.dumps(table))
    (root / "raw").mkdir()
    with gzip.open(root / "raw" / f"vf_{label}-sanity__{SCEN}__{STYLE}_raw.jsonl.gz", "wt") as fh:
        fh.write(json.dumps({"_meta": True}) + "\n")
        for (s, o), (_, k) in CELLS.items():
            for i in range(100):
                fh.write(json.dumps({"agent_role": "red", "n_similar": s, "n_occupied": o,
                                     "sample": i, "parse": "MOVE" if i < k else "STAY"}) + "\n")


@pytest.fixture
def census(tmp_path, monkeypatch, fake_model):
    label = "unit-model"
    _write_inputs(tmp_path, label)
    monkeypatch.setattr(sve, "LOGPROB_TABLES_DIR", tmp_path / "exact")
    monkeypatch.setattr(sve, "SANITY_TABLES_DIR", tmp_path / "tables")
    monkeypatch.setattr(sve, "SANITY_DIR", tmp_path)

    def run(*extra):
        monkeypatch.setattr(sys, "argv", ["sanity_vs_exact.py", "--label", label, "--scenarios", SCEN,
                                          "--out-dir", str(tmp_path / "out"), *extra])
        rc = sve.main()
        return rc, json.loads((tmp_path / "out" / f"sanity_vs_exact_{label}.json").read_text())
    return tmp_path, label, run, fake_model


def test_escalation_end_to_end(census):
    root, label, run, seeds = census
    table_path = root / "tables" / f"vf_{label}-sanity__{SCEN}__{STYLE}.json"

    rc, v = run()                                          # report only: miss not yet tested
    assert v["sanity_check"]["census_misses"] == 1
    assert v["sanity_check"]["verdict"] == "UNRESOLVED"
    assert v["goodness_of_fit"]["role"] == "reported in the paper"
    # the p printed here must equal census_goodness_of_fit.py's: same reps and seed
    assert (v["goodness_of_fit"]["reps"], v["goodness_of_fit"]["seed"]) == (10000, 0)
    # 30/100 against 0.1 in a 3-cell table: the whole-table test fails on its
    # own, and either check failing makes the combined verdict DISCREPANCY (3)
    assert v["goodness_of_fit"]["verdict"] == "FAIL" and v["verdict"] == "DISCREPANCY" and rc == 3

    rc, v = run("--url", "http://fake", "--escalate", "300", "--max-n", "1200")
    esc = json.loads(table_path.read_text())["meta"]["escalations"]
    assert len(esc) == 1 and esc[0]["seed_stage"] == "escalate@UNIT-CENSUS"
    fresh = esc[0]["cells"]["red|4|5"]
    n = fresh["move"] + fresh["stay"]
    assert n == len(seeds) >= 300 and len(set(seeds)) == n            # every seed new
    assert seeds[0] == request_seed("escalate@UNIT-CENSUS", SCEN, STYLE, "red", 4, 5, 100)
    s = v["sanity_check"]
    assert s["cells_not_tested"] == 0 and s["cells_disagree"] == 0 and s["verdict"] == "PASS"
    raw = root / "raw" / f"vf_{label}-sanity__{SCEN}__{STYLE}_escalate_raw.jsonl.gz"
    hdr = [json.loads(l) for l in gzip.open(raw, "rt") if '"_meta"' in l]
    assert len(hdr) == 1 and hdr[0]["census_id"] == "UNIT-CENSUS"
    assert hdr[0]["escalated_cells"] == ["red:(4, 5)"]

    rc, _ = run("--url", "http://fake", "--escalate", "300")         # same census again
    assert rc == 4 and len(seeds) == n                                # refused, nothing drawn

    rc, v = run()                                                     # report only reuses it
    assert v["sanity_check"]["cells_not_tested"] == 0 and v["sanity_check"]["verdict"] == "PASS"

    # stage 1 never reads the merged table (which now holds ~400 draws for (4, 5))
    assert v["sanity_check"]["census_misses"] == 1 and v["census_draws"] == 300
    cell = [r for r in sve.stage1(label, [SCEN], STYLE) if (r["n_similar"], r["n_occupied"]) == (4, 5)][0]
    assert (cell["n_move"], cell["n_samples"]) == (30, 100)
    merged = [c for c in json.loads(table_path.read_text())["compositions"]["red"]
              if (c["n_similar"], c["n_occupied"]) == (4, 5)][0]
    assert merged["n_samples"] == 100 + n


def _family_row(s, p_exact):
    return {"scenario": SCEN, "role": "red", "n_similar": s, "n_occupied": 5, "p_exact": p_exact,
            "exact_lo": p_exact, "exact_hi": p_exact, "p_sanity": 0.3, "inside": False}


def test_cell_tests_holm_and_beyond_cap():
    import census_stats as cs
    args = SimpleNamespace(alpha=0.05, odds_ratio=2.0, power=0.8)
    F = [_family_row(1, 0.1), _family_row(2, 0.1), _family_row(3, 0.001)]
    fresh = {(SCEN, "red|1|5"): (41, 259), (SCEN, "red|2|5"): (41, 259), (SCEN, "red|3|5"): (0, 300)}
    raw = cs.composite_binom_p(41, 300, 0.1, 0.1, 0.1)
    assert raw <= 0.05 < 2 * raw              # each rejects alone; Holm (x2 or x3) lifts both
    sve.cell_tests(F, fresh, args)
    assert [r["status"] for r in F] == ["CONSISTENT", "CONSISTENT", "BEYOND_CAP"]
    assert all(r["p_holm"] > 0.05 for r in F[:2])
    assert F[2]["power"] < 0.8


def test_cell_tests_use_the_exact_interval():
    # p_exact 0 with mass-bound interval [0, 0.02]: one MOVE in 300 is inside the
    # interval (a point test at 0 would call it a certain disagreement)
    row = dict(_family_row(1, 0.0), exact_lo=0.0, exact_hi=0.02)
    sve.cell_tests([row], {(SCEN, "red|1|5"): (1, 299)},
                   SimpleNamespace(alpha=0.05, odds_ratio=2.0, power=0.8))
    assert row["p_fresh"] == 1.0 and row["status"] != "DISAGREES"


def test_census_summary_line(tmp_path, monkeypatch):
    import census as census_mod
    monkeypatch.setattr(census_mod, "SEQCHECK_PLOTS_DIR", tmp_path)
    assert census_mod.verdict_line("x") == "no verdict"
    (tmp_path / "sanity_vs_exact_x.json").write_text(json.dumps({
        "goodness_of_fit": {"verdict": "FAIL", "p_mc": 0.01},
        "sanity_check": {"verdict": "PASS", "census_misses": 2, "cells_disagree": 0,
                         "cells_consistent": 1, "cells_beyond_cap": 1, "cells_not_tested": 0}}))
    line = census_mod.verdict_line("x")
    assert "goodness of fit (paper) FAIL, p 0.010" in line
    assert "sanity check (internal) PASS" in line
