"""Tests for the multi-option path sum, the first-token reads, the census tests and the tasks.

Run from the repo root:   python -m pytest value_functions/multi_option/tests -q
No server is needed: a fake server with a made-up model stands in, and every
expected number is computed independently by visiting every token path.
"""

import math  # isnan, isinf

import numpy as np  # simulated counts
import pytest  # the test runner

from fake_server import EOS, FakeModel, FakeServer, brute_force  # the stand-in server
from option_language import OptionLanguage, grammar_language, free_language, parse_free, parse_loose
from multi_option_path_sum import (ChatTextBackend, TokenBackend, path_sum, ft_mask, ft_split,
                                   depth_read, free_conditional, tv)
import multi_option_census as mc  # census and tests
import tasks  # the experiments

SAMPLER = {"top_k": 0, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0,
           "frequency_penalty": 0.0, "presence_penalty": 0.0}  # the simulation's neutral settings

# A toy vocabulary where "AB" can be written as one token or as "A" + "B",
# and two labels share the start "AB".
VOCAB = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "AB", 7: " ", 8: ".", 9: "x"}
LABELS = ["ABC", "ABD", "E"]


def backends(model, n_probs=64):
    """A token backend and a text backend on the same fake server."""
    srv = FakeServer(model)  # the stand-in
    tok = TokenBackend("http://fake", "toy", 0.3, n_probs, SAMPLER, end_ids={EOS}, session=srv)
    txt = ChatTextBackend("http://fake", "toy", 0.3, n_probs, SAMPLER, end_ids={EOS}, session=srv)
    return tok, txt, srv


# ----------------------------------------------------------------------
# The reply language.
# ----------------------------------------------------------------------

def test_move_stay_grammar_shape():
    lang = grammar_language(["MOVE", "STAY"])  # the simulation's language
    assert lang.gbnf() == "root ::= [ \\t\\n]* ( [Mm] [Oo] [Vv] [Ee] | [Ss] [Tt] [Aa] [Yy] )\n"
    assert lang.remainder_gbnf("MO") == "root ::= [Vv] [Ee]\n"  # same as the extractor's remainder_grammar
    assert lang.complete(" move") == 0 and lang.complete("STAY") == 1 and lang.complete("MOV") is None


def test_shared_starts_and_prefix_labels():
    lang = grammar_language(["GIVE NOTHING", "GIVE A LITTLE", "GIVE A LOT", "GIVE ALL"])
    assert lang.possible("GIVE") == {0, 1, 2, 3}  # nothing decided yet
    assert lang.possible("GIVE A") == {1, 2, 3}  # "GIVE A" starts three labels
    assert lang.possible("GIVE A L") == {1, 2}
    assert lang.decided("GIVE AL") == 3
    nums = grammar_language([str(i) for i in range(11)])
    assert nums.complete("1") == 1 and nums.can_grow("1")  # "1" is complete and can still become "10"
    assert nums.decided("1") is None and nums.complete("10") == 10
    assert nums.remainder_gbnf("1") == 'root ::= ( "0" )?\n' and nums.remainder_gbnf("2") is None


def test_free_language_and_parsers():
    labs = ["MOVE", "STAY"]
    assert parse_free(labs, " Move.\n") == 0  # whitespace, case and one full stop allowed
    assert parse_free(labs, "I would stay") is None  # off-format under the strict rule
    assert parse_loose(labs, "I would stay") == 1  # the loose parse finds the word
    assert parse_loose(["AGREE", "STRONGLY AGREE"], "strongly agree") == 1  # longest label wins
    assert parse_loose(labs, "move or stay") is None  # two labels: no answer


# ----------------------------------------------------------------------
# The path sum against an independent visit of every token path.
# ----------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["grammar", "free"])
def test_token_backend_is_exact(mode):
    model = FakeModel(VOCAB)
    tok, _, _ = backends(model)
    lang = grammar_language(LABELS) if mode == "grammar" else free_language(LABELS)
    res = path_sum(tok, "q", LABELS, mode, max_depth=6, mass_floor=0.0)  # no floor: follow everything
    p, off, cut = brute_force(model, lang, mode, max_depth=6)
    for a, b in zip(res["p"], p):  # every option
        assert abs(a - b) < 1e-12
    assert abs(res["off_format"] - off) < 1e-12
    assert abs(res["bound"] - cut) < 1e-12  # the whole table is listed, so B is only the depth cut
    assert abs(sum(res["p"]) + res["off_format"] + res["bound"] - 1.0) < 1e-12  # nothing lost


def test_text_backend_merges_token_paths():
    """The text prefill reads 'A'+'B' as if it were 'AB', so it is not exact when they differ."""
    model = FakeModel(VOCAB)
    tok, txt, _ = backends(model)
    exact = path_sum(tok, "q", LABELS, "grammar", 6, 0.0)["p"]
    merged = path_sum(txt, "q", LABELS, "grammar", 6, 0.0)["p"]
    assert tv(exact, merged) > 1e-3  # a real difference in this toy model


def test_token_and_text_agree_when_no_shared_paths():
    """With labels that each have one spelling, both backends give the same numbers."""
    model = FakeModel({1: "M", 2: "S", 3: " "})
    tok, txt, _ = backends(model)
    a = path_sum(tok, "q", ["M", "S"], "grammar", 6, 0.0)["p"]
    b = path_sum(txt, "q", ["M", "S"], "grammar", 6, 0.0)["p"]
    assert tv(a, b) < 1e-12


def test_top_n_cut_is_inside_the_bound():
    """With short tables, the exact value stays within +/- B of the path sum."""
    model = FakeModel(VOCAB)
    tok, _, _ = backends(model, n_probs=5)  # only 5 of 10 tokens listed
    res = path_sum(tok, "q", LABELS, "grammar", 6, 0.0)
    p, _, cut = brute_force(model, grammar_language(LABELS), "grammar", 6)
    for a, b in zip(res["p"], p):
        assert abs(a - b) <= res["bound"] + 1e-12


# ----------------------------------------------------------------------
# First-token and depth-d reads.
# ----------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["grammar", "free"])
def test_depth1_equals_ft_mask(mode):
    model = FakeModel(VOCAB)
    tok, _, _ = backends(model)
    res = path_sum(tok, "q", LABELS, mode, 6, 0.0)
    a, m_a = ft_mask(res["root"], LABELS, mode)
    b, m_b = depth_read(res, 1)
    assert all(abs(x - y) < 1e-12 for x, y in zip(a, b)) and abs(m_a - m_b) < 1e-12


def test_depth_reads_obey_the_bound_with_grammar():
    model = FakeModel(VOCAB)
    tok, _, _ = backends(model)
    res = path_sum(tok, "q", LABELS, "grammar", 6, 0.0)
    for d in (1, 2, 3):
        read, m_off = depth_read(res, d)
        if any(math.isnan(x) for x in read):
            continue
        assert tv(res["p"], read) <= m_off + 2 * res["bound"] + 1e-9


def test_free_first_token_can_miss_even_when_resolved():
    """Without grammar the m_off bound does not hold: a reply that starts an option can still
    leave the format, at a different rate per option. This is the gap question 3 measures."""
    model = FakeModel(VOCAB)
    tok, _, _ = backends(model)
    res = path_sum(tok, "q", LABELS, "free", 6, 0.0)
    read, m_off = depth_read(res, 3)
    assert tv(free_conditional(res), read) > m_off + 2 * res["bound"]  # true for this toy model


def test_ft_mask_undefined_when_nothing_resolves():
    labs = ["GIVE NOTHING", "GIVE ALL"]
    model = FakeModel({1: "GIVE", 2: " NOTHING", 3: " ALL"})
    tok, _, _ = backends(model)
    res = path_sum(tok, "q", labs, "grammar", 6, 0.0)
    probs, m_off = ft_mask(res["root"], labs, "grammar")
    assert all(math.isnan(x) for x in probs) and m_off == 1.0  # every first token is shared
    split = ft_split(res["root"], labs, "grammar")
    assert abs(split[0] - 0.5) < 1e-12  # the shared token is split equally
    d2, _ = depth_read(res, 2)  # one token deeper decides it
    assert tv(d2, res["p"]) < 1e-12


def test_power_retempering():
    root = [(1, "M", 0.8, False), (2, "S", 0.2, False)]  # Entry-like tuples
    from multi_option_path_sum import Entry
    root = [Entry(*e) for e in root]
    p1, _ = ft_mask(root, ["M", "S"], "grammar", power=0.3)  # read at T = 1 from a T = 0.3 table
    expect = 0.8 ** 0.3 / (0.8 ** 0.3 + 0.2 ** 0.3)
    assert abs(p1[0] - expect) < 1e-12


# ----------------------------------------------------------------------
# Like for like with the existing extractor.
# ----------------------------------------------------------------------

def _extractor():
    """The repo's existing extractor, or skip when the repo's code is not on the path."""
    return pytest.importorskip("logprob_value_function")


def test_payload_matches_extractor():
    lvf = _extractor()
    from llm_runner import MOVE_STAY_GRAMMAR, SAMPLER_PARAMS
    model = FakeModel({1: "MOVE", 2: "STAY", 3: "MO", 4: "VE", 5: " "})
    srv = FakeServer(model)
    old = lvf.Server("http://fake", "m", 0.3, 64)
    old.session = srv
    old.next_token_probs("q", "MO", "MO")  # the extractor's request after "MO"
    new = ChatTextBackend("http://fake", "m", 0.3, 64, SAMPLER_PARAMS, end_ids={EOS})
    body = new.payload("q", "MO", grammar_language(["MOVE", "STAY"]).remainder_gbnf("MO"))
    assert body == srv.calls[-1][1]  # byte-for-byte the same request
    old.next_token_probs("q", "", "")  # and at the root
    assert new.payload("q", "", MOVE_STAY_GRAMMAR) == srv.calls[-1][1]


def test_k2_reproduces_extractor():
    lvf = _extractor()
    from llm_runner import MOVE_STAY_GRAMMAR, SAMPLER_PARAMS
    model = FakeModel({1: "MOVE", 2: "STAY", 3: "MO", 4: "VE", 5: " ", 6: "Move", 7: "x"})
    srv = FakeServer(model)
    old = lvf.Server("http://fake", "m", 0.3, 64)
    old.session = srv
    a = lvf.path_sum(old, "q", 8, 1e-7)
    new = ChatTextBackend("http://fake", "m", 0.3, 64, SAMPLER_PARAMS, end_ids={EOS}, session=srv)
    b = path_sum(new, "q", ["MOVE", "STAY"], "grammar", 8, 1e-7, root_grammar=MOVE_STAY_GRAMMAR)
    assert abs(a["p_move"] - b["p"][0]) < 1e-12 and abs(a["p_stay"] - b["p"][1]) < 1e-12


# ----------------------------------------------------------------------
# Census and statistics.
# ----------------------------------------------------------------------

def test_census_matches_path_sum():
    model = FakeModel(VOCAB)
    lang = grammar_language(LABELS)
    srv = FakeServer(model, mask_language=lang)
    tok = TokenBackend("http://fake", "toy", 0.3, 64, SAMPLER, end_ids={EOS}, session=srv)
    exact = path_sum(tok, "q", LABELS, "grammar", 12, 0.0)
    smp = mc.Sampler("http://fake", "toy", 0.3, SAMPLER, session=srv)
    c = mc.census(smp, "q", LABELS, "grammar", n=3000, seed0=1, max_tokens=12)
    P = [exact["p"] + [max(0.0, 1 - sum(exact["p"]))]]  # options plus the cut share as "other"
    res = mc.deviance_test([c["strict"]], P, n_boot=400)
    assert res["p"] > 0.001  # no disagreement detected


def test_classify():
    assert mc.classify(["MOVE", "STAY"], " MOVE", "stop", "grammar") == (0, 0)
    assert mc.classify(["MOVE", "STAY"], "MOVE", "length", "grammar")[0] is None  # cut off: not counted
    assert mc.classify(["MOVE", "STAY"], "I'd stay.", "stop", "free") == (None, 1)


def test_plan_and_lr_test():
    rng = np.random.default_rng(0)
    P_path = [[0.5, 0.5]] * 20  # the truth
    P_alt = [[0.6, 0.4]] * 20  # a read 0.1 away
    plan, info, ok = mc.plan_draws(dict(enumerate(P_path)), dict(enumerate(P_alt)))
    assert ok and info >= 16 and sum(plan.values()) <= 1000  # chi2 = 0.04 per draw -> 400 draws
    n_by_state = [plan.get(s, 0) for s in range(20)]
    keep = [s for s in range(20) if n_by_state[s] > 0]
    counts = mc.simulate_counts([P_path[s] for s in keep], [n_by_state[s] for s in keep], rng)
    res = mc.lr_test(counts, [P_path[s] for s in keep], [P_alt[s] for s in keep], n_boot=2000)
    assert res["decision"] in ("path", "not separated")  # never the wrong read
    zero = mc.lr_test([[3, 1]], [[0.5, 0.5]], [[1.0, 0.0]], n_boot=10)
    assert zero["decision"] == "structural (alt refuted)"  # an impossible reply was seen


def test_holm():
    assert mc.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


# ----------------------------------------------------------------------
# The tasks.
# ----------------------------------------------------------------------

@pytest.mark.parametrize("cls,n", [(tasks.C2, 30), (tasks.T3, 30), (tasks.P4, 40), (tasks.O5, 70), (tasks.A11, 18)])
def test_task_sizes_and_prompts(cls, n):
    t = cls()
    assert len(t.states) == n
    for arm in t.arms:
        for rev in (False, True):
            prompt = t.render(t.states[0], arm, rev)
            assert prompt.endswith("Your response:\n")  # the simulation's cue
            if cls is not tasks.A11:
                for lab in t.labels(arm):
                    assert lab in prompt  # every label is named in the prompt
            lang = t.language(arm, "grammar", rev)
            for i, lab in enumerate(t.labels(arm, rev)):
                assert lang.complete(lab) == i  # every label parses as itself


@pytest.mark.parametrize("cls", [tasks.C2, tasks.T3, tasks.P4])
def test_arms_differ_only_in_the_options(cls):
    t = cls()
    a = t.render(t.states[3], "test").splitlines()
    b = t.render(t.states[3], "control").splitlines()
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert len(a) == len(b) and len(diff) == 1  # only the answer sentence differs


def test_test_arms_share_a_start():
    for cls in (tasks.C2, tasks.T3, tasks.P4, tasks.O5):
        labs = [l.lower() for l in cls.arms["test"]]
        assert any(x[:2] == y[:2] for i, x in enumerate(labs) for y in labs[i + 1:])
        ctrl = [l.lower() for l in cls.arms["control"]]
        assert len({x[0] for x in ctrl}) == len(ctrl)  # control labels all start differently


def test_p1_json_language():
    lang = OptionLanguage(["MOVE", "STAY"], [("lit", ['{"action": "'], False), ("label",),
                                             ("lit", ['"}'], False)], label_ci=False)
    assert lang.complete('{"action": "MOVE"}') == 0 and lang.complete('{"action": "move"}') is None
    assert lang.decided('{"action": "') is None and lang.decided('{"action": "S') == 1
    assert lang.gbnf().startswith('root ::= "{" "\\"" "a"')  # literal characters, upper-case labels


def test_b2_and_p1_prompts_match_the_repo():
    lvf = _extractor()  # needs the repo's prompt code
    b2, p1 = tasks.B2(), tasks.P1()
    assert len(b2.states) == 90
    from ratio_prompt_templates import RATIO_CANDIDATES
    from evaluate_ratio_prompts import render_prompt
    from sampling_common import role_keywords
    st = b2.states[17]
    tpl, fn = RATIO_CANDIDATES["R3_dual_count"]
    ref, _ = render_prompt("R3_dual_count", tpl, fn, st["n_similar"], st["n_occupied"],
                           role_keywords("baseline")[st["role"]])
    assert b2.render(st) == ref  # byte for byte the extractor's prompt
    assert "STAY or MOVE" in b2.render(st, rev=True)
    j = p1.render(st)
    assert '{"action": "MOVE"} or {"action": "STAY"}' in j and tasks.R3_SENTENCE not in j
    lang = p1.language("single", "grammar")
    assert lang.complete('{"action": "STAY"}') == 1 and lang.complete(' {"action": "STAY"}') is None
    assert p1.language("single", "free").complete(' {"action": "STAY"}\n') == 1
