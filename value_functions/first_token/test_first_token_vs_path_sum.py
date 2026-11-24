"""Tests for first_token_vs_path_sum.py (run with: python3 -m pytest -q).

They check the pieces on small hand-made tables where the right answer is
known, and one end-to-end run on a tiny record written in the same layout
and with the same settings as the simulation (T = 0.3, top-64, chat+grammar).
"""

import csv  # reads the CSV the script writes
import gzip  # writes the tiny test record in the same .jsonl.gz format
import json  # writes the header and cell lines
import math  # isclose() and isnan()
import subprocess  # runs the script the way a user would
import sys  # path to the current Python interpreter
from pathlib import Path  # builds file paths

import pytest  # the test runner

sys.path.insert(0, str(Path(__file__).parent))  # let us import the script that sits next to this file
import first_token_vs_path_sum as e1  # the code under test

SCRIPT = Path(e1.__file__)  # where the script lives, for the end-to-end test

# Header with the production settings recorded by the extraction (method notes, section 1.3).
PROD_HEADER = {
    "model": "toy-model", "scenario": "race_white_black", "arm": "chat+grammar", "temperature": 0.3,
    "sampler_params": {"top_k": 0, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0,
                       "frequency_penalty": 0.0, "presence_penalty": 0.0},
    "grammar_sha256": "cd9f6867af8d41081898d703c322a7b877787348fde5564638013955205b2fa3",
    "logprob": {"temperature": 0.3, "n_probs": 64, "max_depth": 8, "mass_floor": 1e-07},
}


# ------------------------------------------------------------- classify --

@pytest.mark.parametrize("piece", ["M", " M", "Mo", " MOVE", "move", "\nMov", "mOvE"])  # tokens that start MOVE
def test_classify_move(piece):
    assert e1.classify(piece) == "M"  # each one commits to MOVE under [Mm][Oo][Vv][Ee]


@pytest.mark.parametrize("piece", ["S", " St", "STAY", "stay", "\tSta"])  # tokens that start STAY
def test_classify_stay(piece):
    assert e1.classify(piece) == "S"  # each one commits to STAY


@pytest.mark.parametrize("piece", [" ", "\n", "\t\t", " \n "])  # pure whitespace
def test_classify_whitespace(piece):
    assert e1.classify(piece) == "ws"  # whitespace delays the decision


@pytest.mark.parametrize("piece", ["MOVES", "Yes", "M!", " no", "", "  .", "STAYED"])  # not allowed by the grammar
def test_classify_forbidden(piece):
    assert e1.classify(piece) == "x"  # the grammar rejects these


# ------------------------------------------------------------- retemper --

def test_retemper_power_one_is_identity():
    pairs = [("a", 0.7), ("b", 0.3)]  # already normalised
    out = dict(e1.retemper(pairs, 1.0))  # power 1 should change nothing
    assert math.isclose(out["a"], 0.7) and math.isclose(out["b"], 0.3)  # unchanged


def test_retemper_matches_formula():
    pairs = [("a", 0.8), ("b", 0.2)]  # a table read at T = 0.3
    out = dict(e1.retemper(pairs, 0.3))  # move it to T = 1
    expect = 0.8 ** 0.3 / (0.8 ** 0.3 + 0.2 ** 0.3)  # p^T renormalised
    assert math.isclose(out["a"], expect)  # same number


# ------------------------------------------------------------ readouts --

def test_readouts_on_known_table():
    pairs = [(" ", 0.15), ("MOVE", 0.5), ("STAY", 0.3), ("hi", 0.05)]  # one table of each class
    m_off, ft_mask, ft_mask_t1, ft_canon, ft_canon_t1 = e1.first_token_readouts(pairs, 0.3)  # compute
    assert math.isclose(m_off, 0.15 / 0.95)  # whitespace share of allowed mass
    assert math.isclose(ft_mask, 0.5 / 0.8)  # MOVE share of deciding mass
    assert math.isclose(ft_canon, 0.625)  # only one spelling each, so equal to ft_mask here
    t1 = 0.5 ** 0.3 / (0.5 ** 0.3 + 0.3 ** 0.3)  # expected value at T = 1
    assert math.isclose(ft_mask_t1, t1) and math.isclose(ft_canon_t1, t1)  # both at T = 1


def test_canon_uses_best_spelling_and_nan_when_missing():
    pairs = [(" MOVE", 0.4), ("Move", 0.1), ("STAY", 0.5)]  # two MOVE spellings
    assert math.isclose(e1.canon_ratio(pairs), 0.4 / 0.9)  # the larger spelling is used
    assert math.isnan(e1.canon_ratio([("M", 0.6), ("STAY", 0.4)]))  # MOVE only as a fragment -> NaN


def test_bound_identity_holds_for_any_whitespace_continuation():
    # With the grammar ws* (MOVE|STAY), |path - ft_mask| <= m_off, whatever the model does after whitespace.
    pairs = [(" ", 0.2), ("M", 0.48), ("S", 0.32)]  # 20 % whitespace first
    m_off, ft_mask, *_ = e1.first_token_readouts(pairs)  # first-token numbers
    for a in (0.0, 0.3, 1.0):  # P(MOVE | whitespace first) at its extremes and in between
        path = (1 - m_off) * ft_mask + m_off * a  # exact law for this tree
        assert abs(path - ft_mask) <= m_off + 1e-12  # the bound the paper states


# ------------------------------------------------------------ settings --

def test_run_params_read_from_header():
    params = e1.get_run_params(PROD_HEADER)  # read the production header
    assert params["temperature"] == 0.3 and params["n_probs"] == 64 and params["max_depth"] == 8  # same as the runs
    assert params["arm"] == "chat+grammar"  # same arm as the simulation


# ----------------------------------------------------------- end to end --

def _write_record(path, header, cells):
    with gzip.open(path, "wt") as fh:  # same compressed line format as the real records
        fh.write(json.dumps(header) + "\n")  # header line
        for cell in cells:  # one line per (role, cell)
            fh.write(json.dumps(cell) + "\n")


def test_end_to_end_dedupe_and_self_check(tmp_path):
    cell = {"role": "white", "cell": [1, 2], "p_move": 0.6,  # exact path sum stored by the extraction
            "states": {"": {"top": [{"token": " ", "prob": 0.15}, {"token": "MOVE", "prob": 0.5},
                                    {"token": "STAY", "prob": 0.3}, {"token": "hi", "prob": 0.05}]}}}
    a, b = tmp_path / "vflp_a_states.jsonl.gz", tmp_path / "vflp_b_states.jsonl.gz"  # two files holding the same state
    _write_record(a, PROD_HEADER, [cell])
    _write_record(b, PROD_HEADER, [cell])
    out = tmp_path / "cells.csv"  # where the per-cell table goes
    res = subprocess.run([sys.executable, str(SCRIPT), str(a), str(b), "--out", str(out)],
                         capture_output=True, text=True, check=True)  # run the script
    assert "skipped 1 duplicate" in res.stdout  # the copy was not counted twice
    assert "SELF-CHECK" not in res.stdout  # implied a lies in [0, 1]
    assert "'temperature': 0.3" in res.stdout or '"temperature": 0.3' in res.stdout  # settings are printed
    rows = list(csv.DictReader(open(out)))  # read the table back
    assert len(rows) == 1  # one unique state
    assert math.isclose(float(rows[0]["a_implied"]), (0.6 - (1 - 0.15 / 0.95) * 0.625) / (0.15 / 0.95))  # identity
