"""Ties value_functions/first_token/first_token_vs_path_sum.py to the production code.

The script's own unit tests sit next to it (value_functions/first_token/
test_first_token_vs_path_sum.py, checked against the hand-over manifest and
left unchanged). These three belong to the repo's tests/ tree because they
import llm_runner and read the stored extraction records:

  a. the header settings of the records the comparison uses equal the
     production settings (llm_runner.SAMPLER_PARAMS, T = 0.3, top-64, depth 8,
     mass floor 1e-7, and the sha256 of llm_runner.MOVE_STAY_GRAMMAR computed
     the way logprob_value_function.py computes it);
  b. classify() agrees with MOVE_STAY_GRAMMAR on every case variant and prefix
     of MOVE and STAY behind any leading mix of space, tab and newline;
  c. the ADAPTER reads the real record layout (flat header, a list of states
     with a "top" table, n_similar / n_occupied as the cell).

The records are gitignored results, so (a) and the real-file half of (c) skip
when the store is absent.
"""
import gzip
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "value_functions" / "first_token"))
sys.path.insert(0, str(REPO))

import first_token_vs_path_sum as ft  # noqa: E402
import llm_runner  # noqa: E402

RAW = REPO / "value_functions" / "results" / "llm_logprob" / "raw"
RECORDS = sorted(RAW.glob("vflp_*_states.jsonl.gz"))   # the live store: 9 models x 6 scenarios


def _header(path):
    with gzip.open(path, "rt") as fh:
        return json.loads(fh.readline())


# ------------------------------------------------------------------ (a) --

def test_record_headers_equal_production_settings():
    if not RECORDS:
        pytest.skip(f"no extraction records under {RAW}")
    assert len(RECORDS) == 54, f"expected 9 models x 6 scenarios, found {len(RECORDS)}"
    # the same hashing as logprob_value_function.py: sha256 of the grammar string, utf-8
    grammar_sha = hashlib.sha256(llm_runner.MOVE_STAY_GRAMMAR.encode()).hexdigest()
    want = {"temperature": 0.3, "n_probs": 64, "max_depth": 8, "mass_floor": 1e-7,
            "grammar_sha256": grammar_sha,
            "sampler_params": json.dumps(llm_runner.SAMPLER_PARAMS, sort_keys=True),
            "arm": "chat+grammar"}
    for path in RECORDS:
        assert ft.get_run_params(_header(path)) == want, path.name


# ------------------------------------------------------------------ (b) --

def _grammar_regexes():
    """Full-match and prefix regexes built from MOVE_STAY_GRAMMAR itself."""
    rules = dict(re.findall(r"^(\w+)\s*::=\s*(.+)$", llm_runner.MOVE_STAY_GRAMMAR, flags=re.M))
    assert rules["root"].split() == ["ws", "answer"] and rules["answer"].split() == ["move", "|", "stay"]
    ws = rules["ws"].strip()
    assert ws.endswith("*")
    ws_class = ws[:-1]                                     # "[ \t\n]" in GBNF escapes = a Python regex class
    classes = {name: rules[name].split() for name in ("move", "stay")}   # e.g. ["[Mm]", "[Oo]", "[Vv]", "[Ee]"]
    return ws_class, classes


def test_classify_agrees_with_move_stay_grammar():
    ws_class, classes = _grammar_regexes()
    assert ws_class == r"[ \t\n]" and set(ft.WHITESPACE) == {" ", "\t", "\n"}
    assert classes == {"move": ["[Mm]", "[Oo]", "[Vv]", "[Ee]"], "stay": ["[Ss]", "[Tt]", "[Aa]", "[Yy]"]}
    leads = [""] + ["".join(t) for n in (1, 2, 3) for t in itertools.product(" \t\n", repeat=n)]
    for name, label in (("move", "M"), ("stay", "S")):
        for k in range(1, 5):                              # every non-empty prefix of the word
            pattern = re.compile(f"{ws_class}*" + "".join(classes[name][:k]))
            for chars in itertools.product(*[c[1:-1] for c in classes[name][:k]]):   # every case variant
                for lead in leads:
                    piece = lead + "".join(chars)
                    assert pattern.fullmatch(piece), piece  # the grammar accepts it as a prefix
                    assert ft.classify(piece) == label, repr(piece)
    for lead in leads[1:]:
        assert re.fullmatch(f"{ws_class}+", lead) and ft.classify(lead) == "ws", repr(lead)
    full = re.compile(f"{ws_class}*(?:{''.join(classes['move'])}|{''.join(classes['stay'])})")
    for piece in ["MOVES", "Yes", "STAYED", "MOVE ", " no", "", "M!", "Mv", "St y", "\rMOVE", " STAY", "STOP", "MOVED"]:
        assert not any(full.fullmatch(piece + tail) for tail in ("", "E", "VE", "OVE", "Y", "AY", "TAY")), piece
        assert ft.classify(piece) == "x", repr(piece)


# ------------------------------------------------------------------ (c) --

REAL_HEADER = {"_meta": True, "label": "toy-chat-grammar", "model": "toy-Q8_0", "scenario": "baseline",
               "temperature": 0.3,
               "sampler_params": {"top_k": 0, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0,
                                  "frequency_penalty": 0.0, "presence_penalty": 0.0},
               "grammar_sha256": "cd9f6867af8d41081898d703c322a7b877787348fde5564638013955205b2fa3",
               "n_probs": 64, "max_depth": 8, "mass_floor": 1e-07,
               "endpoint": "/v1/chat/completions (campaign endpoint); prefix states as assistant prefill"}
REAL_CELL = {"role": "red", "n_similar": 1, "n_occupied": 3, "n_opposite": 2, "context": "...",
             "p_move": 0.6, "p_stay": 0.4, "mass_bound": 1e-8, "n_requests": 2, "prompt_sha256": "0" * 64,
             "paths": [{"text": "MOVE", "decision": "MOVE", "mass": 0.6}],
             "states": [{"prefix": "", "mass": 1.0, "captured": 1.0, "allowed_mass": 0.95, "prompt_n": 138,
                         "top": [{"token": " ", "prob": 0.15}, {"token": "MOVE", "prob": 0.5},
                                 {"token": "ST", "prob": 0.3}, {"token": "hi", "prob": 0.05}]},
                        {"prefix": " ", "mass": 0.15, "top": [{"token": "MOVE", "prob": 0.7}, {"token": "STAY", "prob": 0.3}]}]}


def test_adapter_reads_the_real_record_layout():
    params = ft.get_run_params(REAL_HEADER)                # flat header, no "logprob" block, no "arm"
    assert (params["n_probs"], params["max_depth"], params["mass_floor"]) == (64, 8, 1e-07)
    assert params["temperature"] == 0.3 and params["arm"] == "chat+grammar"
    assert ft.get_root_table(REAL_CELL) == [(" ", 0.15), ("MOVE", 0.5), ("ST", 0.3), ("hi", 0.05)]   # the empty prefix, not " "
    assert ft.get_path_p_move(REAL_CELL) == 0.6
    other = dict(REAL_CELL, n_similar=2)                   # a different composition, same role
    keys = {ft.cell_key(REAL_HEADER, c, "f") for c in (REAL_CELL, other, dict(REAL_CELL, role="blue"))}
    assert len(keys) == 3                                  # cells are told apart by (n_similar, n_occupied) and role
    assert ft.cell_key(REAL_HEADER, REAL_CELL, "f")[3] == "[1, 3]"
    renamed = dict(REAL_HEADER, model="toy-Q8_0-00001-of-00002")   # same label, another gguf name (a re-extraction)
    assert ft.cell_key(renamed, REAL_CELL, "f") == ft.cell_key(REAL_HEADER, REAL_CELL, "f")   # one model, keyed by label
    assert ft.cell_key(REAL_HEADER, REAL_CELL, "f")[0] == "toy-chat-grammar"


def test_adapter_on_one_stored_record():
    if not RECORDS:
        pytest.skip(f"no extraction records under {RAW}")
    with gzip.open(RECORDS[0], "rt") as fh:
        meta = json.loads(fh.readline())
        cells = [json.loads(line) for line in fh]
    assert len(cells) == 90                                # 45 compositions x 2 roles
    assert len({ft.cell_key(meta, c, "f") for c in cells}) == 90
    for c in cells:
        table = ft.get_root_table(c)
        assert 0 < len(table) <= 64 and abs(sum(p for _, p in table) - 1.0) < 1e-3
        assert 0.0 <= ft.get_path_p_move(c) <= 1.0
