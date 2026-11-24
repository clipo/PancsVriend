"""The exact table's sequential check: an escalated cell is judged on its fresh draws alone.

    .venv/bin/python -m pytest tests/test_logprob_validation_escalation.py -q

No server: the sampler is faked and the verdict files go to tmp_path. Covers
- a first-stage miss that is a fluctuation (the 2026-11-16 qwen cell: exact
  0.0867, 37/300 then 88/900 fresh) PASSES, and seq_n / seq_move are the fresh
  round, not the pooled 125/1200 that used to fail it;
- a real error still FAILS on the fresh draws;
- every raw reply of both stages is written.
"""
import gzip
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "prompt_refinement"), str(REPO / "value_functions" / "logprob")):
    if p not in sys.path:
        sys.path.insert(0, p)

import logprob_value_function as lv  # noqa: E402

LABEL, STYLE, SCEN = "test-label", "R3_dual_count", "baseline"


class FakeServer:
    """sample_sequential returns scripted (move, stay) counts per call and records one reply per draw."""

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def sample_sequential(self, prompt, n, seed0, record=None):
        move = self.script.pop(0)
        self.calls.append((n, seed0))
        for i in range(n):
            if record:
                record({"i": i, "seed": seed0 + i, "reply": "MOVE" if i < move else "STAY"})
        return move, n - move


def run(tmp_path, monkeypatch, p_exact, script):
    monkeypatch.setattr(lv, "LOGPROB_VALIDATION_DIR", tmp_path)
    monkeypatch.setitem(lv.USER_PROMPTS, (SCEN, "red", (1, 5)), "prompt")
    lp = {SCEN: {"compositions": {"red": [{"n_similar": 1, "n_occupied": 5, "p_move": p_exact,
                                           "mass_bound": 1e-9, "n_requests": 2}]}}}
    server = FakeServer(script)
    summary = lv.validate(LABEL, STYLE, [SCEN], ["red"], lp, tmp_path, server=server,
                          seq_cells=1, seq_samples=300, seq_escalate=3, seq_saturated=0, seq_random=0, plot=False)
    return summary, server, pd.read_csv(tmp_path / f"seqcheck_{LABEL}.csv").iloc[0]


def test_fluctuation_passes_on_fresh_draws(tmp_path, monkeypatch):
    summary, server, row = run(tmp_path, monkeypatch, 0.0867, [37, 88])
    assert [n for n, _ in server.calls] == [300, 900]
    assert not row.inside_stage1 and row.escalated
    assert (row.seq_n, row.seq_move) == (900, 88)            # the fresh round, not the pooled 1200 / 125
    assert (row.seq_n_stage1, row.seq_move_stage1) == (300, 37)
    assert row.inside_seq_ci and summary["pass"]
    assert summary["seq_check_escalation_rule"] == "fresh draws only"


def test_real_error_still_fails(tmp_path, monkeypatch):
    summary, _, row = run(tmp_path, monkeypatch, 0.0867, [60, 180])   # 20% in both rounds
    assert row.escalated and not row.inside_seq_ci and not summary["pass"]


def test_raw_replies_are_kept(tmp_path, monkeypatch):
    run(tmp_path, monkeypatch, 0.0867, [37, 88])
    recs = [json.loads(l) for l in gzip.open(tmp_path / f"seqcheck_{LABEL}_raw.jsonl.gz", "rt")]
    assert recs[0]["_meta"] and recs[0]["escalation_rule"] == "fresh draws only"
    draws = recs[1:]
    assert len(draws) == 1200 and {d["stage"] for d in draws} == {1, 2}
    assert sum(d["reply"] == "MOVE" for d in draws if d["stage"] == 2) == 88
