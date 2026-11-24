"""Tests for value_functions/first_token_fresh/fresh_first_token.py (no server needed).

They pin the fresh validation to the simulation runs: the request it sends is
the extraction's own first-step request with the production settings, the
prompts it sends hash to the stored prompt_sha256, the served file is the one
in the record headers, and the comparison arithmetic is right on small tables
with known answers. Tests that read the stored records skip when the
(gitignored) store is absent.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "value_functions" / "first_token_fresh"))
sys.path.insert(0, str(REPO))

import fresh_first_token as fresh  # noqa: E402
import llm_runner  # noqa: E402
from value_functions.serving import MODELS  # noqa: E402

LABEL = MODELS["phi4"].label
CASES = fresh.load_stored(LABEL)
needs_store = pytest.mark.skipif(not CASES, reason=f"no stored records for {LABEL}")

HEADER = {"label": "toy-chat-grammar", "scenario": "baseline", "style": "R3_dual_count", "scenario_file": "scenarios_a2.py",
          "temperature": 0.3, "n_probs": 64, "max_depth": 8, "mass_floor": 1e-07, "concurrency": 1,
          "sampler_params": dict(llm_runner.SAMPLER_PARAMS),
          "grammar_sha256": hashlib.sha256(llm_runner.MOVE_STAY_GRAMMAR.encode()).hexdigest(),
          "endpoint": "/v1/chat/completions (campaign endpoint); prefix states as assistant prefill",
          "server": {"model_path": "llms/toy.gguf", "build_info": "b1", "total_slots": 4}}
ROOT = [{"token": " ", "prob": 0.15}, {"token": "MOVE", "prob": 0.5}, {"token": "ST", "prob": 0.3}, {"token": "hi", "prob": 0.05}]
CELL = {"role": "red", "n_similar": 1, "n_occupied": 3, "p_move": 0.6, "p_stay": 0.4, "prompt_sha256": "0" * 64,
        "states": [{"prefix": "", "prompt_n": 138, "top": ROOT}]}
PAIRS = [(e["token"], e["prob"]) for e in ROOT]


# ------------------------------------------------- like for like: settings --

def test_check_header_accepts_production_and_rejects_anything_else():
    assert fresh.check_header(HEADER)["temperature"] == 0.3
    for key, value in (("temperature", 1.0), ("n_probs", 20), ("grammar_sha256", "0" * 64),
                       ("sampler_params", dict(llm_runner.SAMPLER_PARAMS, top_p=0.9)),
                       ("endpoint", "/completion")):
        with pytest.raises(ValueError):
            fresh.check_header(dict(HEADER, **{key: value}))


@needs_store
def test_stored_records_are_the_production_settings_and_the_registered_file():
    assert len(CASES) == 540 and len(fresh.stored_files(LABEL)) == 6
    for header, _ in CASES:
        fresh.check_header(header)
        assert header["concurrency"] == 1
        assert Path(header["server"]["model_path"]).name == Path(MODELS["phi4"].gguf).name


@needs_store
def test_every_rebuilt_prompt_hashes_to_the_stored_prompt_sha256():
    prompts = fresh.prompts_for(CASES)                       # raises on any mismatch
    assert len(prompts) == 540 and len(set(prompts)) == 540
    header, cell = CASES[0]
    with pytest.raises(ValueError):
        fresh.prompts_for([(header, dict(cell, n_similar=cell["n_similar"] + 1, n_occupied=8))])


# -------------------------------------------------- like for like: request --

class _FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append((url, json))

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"logprobs": {"content": [{"top_probs": ROOT}]}}], "timings": {"prompt_n": 138}}
        return R()


def test_the_request_is_the_extractions_first_step_with_production_settings():
    server = fresh.lpv.Server("http://127.0.0.1:1", "toy", HEADER["temperature"], HEADER["n_probs"])
    server.session = _FakeSession()
    out = fresh.query_fresh(server, ["PROMPT A", "PROMPT B"], concurrency=1)
    assert [pairs for pairs, _ in out] == [PAIRS, PAIRS] and out[0][1] == 138
    assert len(server.session.calls) == 2                     # one request per case
    url, payload = server.session.calls[0]
    assert url == "http://127.0.0.1:1/v1/chat/completions"
    assert payload["messages"] == [{"role": "user", "content": "PROMPT A"}]     # no assistant prefill: the empty prefix
    assert payload["max_tokens"] == 1                         # first token only
    assert payload["n_probs"] == 64 and payload["post_sampling_probs"] is True
    assert payload["temperature"] == 0.3 and payload["cache_prompt"] is False and payload["stream"] is False
    assert payload["grammar"] == llm_runner.MOVE_STAY_GRAMMAR
    assert hashlib.sha256(payload["grammar"].encode()).hexdigest().startswith("cd9f6867")
    for key, value in llm_runner.SAMPLER_PARAMS.items():
        assert payload[key] == value


def test_concurrent_query_keeps_case_order():
    server = fresh.lpv.Server("http://127.0.0.1:1", "toy", 0.3, 64)
    server.session = _FakeSession()
    out = fresh.query_fresh(server, [f"P{i}" for i in range(8)], concurrency=4)
    assert len(out) == 8 and all(pairs == PAIRS for pairs, _ in out)
    assert sorted(c[1]["messages"][0]["content"] for c in server.session.calls) == [f"P{i}" for i in range(8)]


# ------------------------------------------------------------- comparison --

def test_identical_fresh_table_gives_zero_differences():
    row = fresh.compare_case(HEADER, CELL, PAIRS, prompt_n=138)
    assert row["table_identical"] == 1 and row["table_max_abs_diff"] == 0.0
    assert row["d_mask_vs_stored_first"] == 0.0
    assert math.isclose(row["fresh_ft_mask"], 0.5 / 0.8)       # MOVE share of the deciding mass ("ST" starts STAY)
    assert math.isclose(row["d_mask_vs_path"], abs(0.625 - 0.6))
    assert math.isnan(row["fresh_whole_word"]) and math.isnan(row["d_word_vs_path"])   # no whole-word STAY token
    assert row["unsaturated"] == 1 and row["fresh_prompt_n"] == row["stored_prompt_n"] == 138
    assert math.isclose(row["fresh_m_off"], 0.15 / 0.95)


def test_shifted_fresh_table_gives_the_known_differences():
    shifted = [(" ", 0.15), ("MOVE", 0.4), ("ST", 0.4), ("hi", 0.05)]
    row = fresh.compare_case(HEADER, CELL, shifted)
    assert row["table_identical"] == 0 and math.isclose(row["table_max_abs_diff"], 0.1)
    assert math.isclose(row["d_mask_vs_stored_first"], 0.625 - 0.5)
    assert math.isclose(row["d_mask_vs_path"], 0.6 - 0.5)


def test_whole_word_read_uses_whole_word_tokens_only():
    cell = dict(CELL, states=[{"prefix": "", "top": [{"token": "MOVE", "prob": 0.6}, {"token": "STAY", "prob": 0.2},
                                                     {"token": "ST", "prob": 0.2}]}])
    row = fresh.compare_case(HEADER, cell, [("MOVE", 0.6), ("STAY", 0.2), ("ST", 0.2)])
    assert math.isclose(row["fresh_whole_word"], 0.75) and math.isclose(row["fresh_ft_mask"], 0.6)
    assert math.isclose(row["d_word_vs_path"], 0.15) and row["d_word_vs_stored_word"] == 0.0


def test_table_max_abs_diff_counts_missing_tokens():
    assert math.isclose(fresh.table_max_abs_diff([("a", 0.7), ("b", 0.3)], [("a", 0.7), ("c", 0.3)]), 0.3)


def test_saturation_rule_is_strict_at_the_extractions_eps():
    assert fresh.SAT_EPS == 0.005
    for p, want in ((0.005, 0), (0.0051, 1), (0.5, 1), (0.9949, 1), (0.995, 0), (0.0, 0), (1.0, 0)):
        assert fresh.compare_case(HEADER, dict(CELL, p_move=p), PAIRS)["unsaturated"] == want, p


def test_diff_stats_and_summarise():
    s = fresh.diff_stats([0.0, 0.02, 0.005, math.nan])
    assert (s["n"], s["n_nan"], s["max"], s["median"]) == (3, 1, 0.02, 0.005)
    assert math.isclose(s["share_above_gate"], 1 / 3)
    assert fresh.diff_stats([math.nan])["n"] == 0
    rows = [dict({d: 0.0 for d in fresh.DIFFS}, unsaturated=1), dict({d: 0.5 for d in fresh.DIFFS}, unsaturated=0)]
    out = fresh.summarise(rows)
    assert out["d_mask_vs_path"]["all"]["max"] == 0.5 and out["d_mask_vs_path"]["unsaturated"]["max"] == 0.0
    assert out["d_mask_vs_path"]["all"]["share_above_gate"] == 0.5


def test_pick_subset_spreads_over_the_ordering():
    items = list(range(100))
    picked = fresh.pick_subset(items, 10, key=lambda v: v)
    assert len(picked) == 10 and picked == sorted(picked) and 99 in picked
    assert fresh.pick_subset(items, 500, key=lambda v: v) == items
