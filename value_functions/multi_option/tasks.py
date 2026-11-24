"""The short multi-option experiments: prompts, option labels and state grids (design v4).

Every task is a memoryless agent's decision in a small, enumerable state
space, written in the simulation's prompt layout (persona and rules, one
decision sentence, the state, then "Your response:").

Each task with a shared-start test has two label arms that differ ONLY in the
option words (and, for O5, in the one sentence that lists them):
  test     labels that share their first token(s)
  control  labels with the same meaning that all start differently
Every exact read can be run in the listed option order or reversed.

A Task gives:
  states                      list of dicts (one per prompt)
  render(state, arm, rev)     the user prompt
  labels(arm, rev)            the labels in prompt order
  language(arm, mode)         the OptionLanguage for that mode
"""

from itertools import combinations_with_replacement  # the P4 and O5 state grids

from option_language import OptionLanguage, grammar_language, free_language  # reply languages


# ----------------------------------------------------------------------
# Small helpers.
# ----------------------------------------------------------------------

def _join_or(items):
    """'A', 'A or B', 'A, B or C': a natural list ending in 'or'."""
    items = list(items)  # a copy we can index
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " or " + items[-1]


def _join_and(items):
    """'A', 'A and B', 'A, B and C'."""
    items = list(items)
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


class Task:
    """One experiment: its labels per arm, its states and its prompt."""

    name = ""  # short id, e.g. "C2"
    arms = {}  # arm name -> labels in the listed order
    glosses = []  # what each option means, same order (shown next to the labels)

    def labels(self, arm, rev=False):
        """The labels as the prompt lists them (reversed if rev)."""
        labs = list(self.arms[arm])  # listed order
        return labs[::-1] if rev else labs

    def option_list(self, arm, rev=False):
        """The answer sentence's list, e.g. 'CONSERVE (take the small catch) or CONSUME (...)'."""
        pairs = list(zip(self.arms[arm], self.glosses))  # each label with its meaning
        if rev:  # reversed order
            pairs = pairs[::-1]
        return _join_or(f"{lab} ({g})" if g else lab for lab, g in pairs)

    def language(self, arm, mode, rev=False):
        """The reply language for this arm: ws label (grammar) or ws label ["."] ws (free)."""
        labs = self.labels(arm, rev)  # labels in prompt order
        return grammar_language(labs) if mode == "grammar" else free_language(labs)


# ----------------------------------------------------------------------
# C2. Common-pool resource: 2 options, shared start inside a word.
# ----------------------------------------------------------------------

class C2(Task):
    """Fishers sharing a lake: take the small or the large catch (Hardin 1968; Ostrom 1990)."""

    name = "C2"
    arms = {"test": ["CONSERVE", "CONSUME"], "control": ["RESTRAIN", "CONSUME"]}
    glosses = ["take the small catch", "take the large catch"]

    FRAME = """You are one of 5 fishers who share a lake. Each season every fisher either takes a small catch (5 fish) or a large catch (15 fish). The fish left after everyone has fished regrow by {regrowth}% before the next season, up to the lake's limit of 100 fish. If the lake is empty, nobody can fish any more.

You will be told how many fish are in the lake now and what the other 4 fishers did last season. After learning this, decide: will you take the small catch or the large catch? Answer with ONLY one word - {options} - with no explanation or other text.

The lake:

The lake now holds {stock} fish. Last season, {n_large} of the other 4 fishers took the large catch and {n_small} took the small catch.

Your response:
"""

    def __init__(self, stocks=(20, 50, 80)):
        self.states = [{"stock": s, "n_large": k, "regrowth": g}  # 3 x 5 x 2 = 30 states
                       for s in stocks for k in range(5) for g in (10, 30)]

    def render(self, st, arm, rev=False):
        return self.FRAME.format(options=self.option_list(arm, rev), stock=st["stock"],
                                 n_large=st["n_large"], n_small=4 - st["n_large"],
                                 regrowth=st["regrowth"])


# ----------------------------------------------------------------------
# T3. Trust game, the trustee's decision: 3 options, shared first word.
# ----------------------------------------------------------------------

class T3(Task):
    """Investment game, trustee's return (Berg, Dickhaut and McCabe 1995)."""

    name = "T3"
    arms = {"test": ["RETURN NOTHING", "RETURN HALF", "RETURN ALL"],
            "control": ["NOTHING", "HALF", "ALL"]}
    glosses = ["send back nothing", "send back half of what you received",
               "send back all of what you received"]

    FRAME = """You are taking part in a one-time exchange with a stranger you will never meet again. The stranger started with 10 coins and chose how many of them to send to you. Every coin sent to you was multiplied by {mult} on the way. You now decide how much of what you received to send back to the stranger; you keep the rest.

You will be told how many coins the stranger sent and how many you received. After learning this, decide: how much will you send back? Answer with ONLY one of these answers - {options} - with no explanation or other text.

The exchange:

The stranger sent you {sent} coins, which became {received} coins. The stranger kept {kept} coins.

Your response:
"""

    def __init__(self):
        self.states = [{"sent": x, "mult": m} for x in range(1, 11) for m in (2, 3, 4)]  # 30 states

    def render(self, st, arm, rev=False):
        return self.FRAME.format(options=self.option_list(arm, rev), mult=st["mult"], sent=st["sent"],
                                 received=st["sent"] * st["mult"], kept=10 - st["sent"])


# ----------------------------------------------------------------------
# P4. Public goods contribution: 4 options, nested shared words.
# ----------------------------------------------------------------------

class P4(Task):
    """Linear public goods game with four contribution levels (Ledyard 1995; Fehr and Gaechter 2000)."""

    name = "P4"
    arms = {"test": ["GIVE NOTHING", "GIVE A LITTLE", "GIVE A LOT", "GIVE ALL"],
            "control": ["NOTHING", "LITTLE", "MUCH", "ALL"]}
    glosses = ["0 tokens", "5 tokens", "15 tokens", "20 tokens"]
    LEVELS = (0, 5, 15, 20)  # the token amounts behind the four options

    FRAME = """You are one of 4 members of a group. Each round every member receives 20 tokens and decides how many to put into a shared project. All tokens put into the project are multiplied and shared equally: every member receives {mpcr} tokens for each token that anyone puts in, whatever they gave themselves. Tokens you do not put in are yours to keep.

You will be told what the other 3 members put in last round. After learning this, decide: how much will you put in this round? Answer with ONLY one of these answers - {options} - with no explanation or other text.

Your group:

Last round, the other 3 members put in {others}.

Your response:
"""

    def __init__(self):
        combos = list(combinations_with_replacement(self.LEVELS, 3))  # the 20 multisets of 3 levels
        self.states = [{"others": list(c), "mpcr": m} for c in combos for m in (0.4, 0.8)]  # 40 states

    def render(self, st, arm, rev=False):
        others = _join_and(f"{x} tokens" for x in st["others"])  # e.g. "0 tokens, 5 tokens and 15 tokens"
        return self.FRAME.format(options=self.option_list(arm, rev), mpcr=st["mpcr"], others=others)


# ----------------------------------------------------------------------
# O5. Opinion on a five-point scale: 5 options, shared first word.
# ----------------------------------------------------------------------

class O5(Task):
    """Discrete opinion dynamics (Castellano, Fortunato and Loreto 2009): state = neighbours' stances."""

    name = "O5"
    STANCES = ["strongly disagree", "disagree", "are neutral", "agree", "strongly agree"]  # how neighbours are described
    arms = {"test": ["STRONGLY DISAGREE", "DISAGREE", "NEUTRAL", "AGREE", "STRONGLY AGREE"],
            "control": ["1", "2", "3", "4", "5"]}
    glosses_by_arm = {"test": ["", "", "", "", ""],  # the words explain themselves
                      "control": ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"]}

    FRAME = """You live in a small town. The town council is deciding whether the public library should stay open two hours later in the evening. You have talked about it with 4 of your neighbors.

You will be told what your 4 neighbors think. After learning this, decide: what is your own view on the later closing time? Answer with ONLY one of these answers - {options} - with no explanation or other text.

Your neighbors:

Of your 4 neighbors, {views}.

Your response:
"""

    def __init__(self):
        combos = combinations_with_replacement(range(5), 4)  # 4 neighbours over 5 levels: 70 states
        self.states = [{"counts": [list(c).count(i) for i in range(5)]} for c in combos]

    def option_list(self, arm, rev=False):
        pairs = list(zip(self.arms[arm], self.glosses_by_arm[arm]))  # label and meaning
        if rev:
            pairs = pairs[::-1]
        return _join_or(f"{lab} ({g})" if g else lab for lab, g in pairs)

    def render(self, st, arm, rev=False):
        parts = [f"{n} {s}" for n, s in zip(st["counts"], self.STANCES) if n > 0]  # only stances someone holds
        return self.FRAME.format(options=self.option_list(arm, rev), views=_join_and(parts))


# ----------------------------------------------------------------------
# A11. Public goods contribution as a whole number 0 to 10.
# ----------------------------------------------------------------------

class A11(Task):
    """Integer contribution: '1' is a complete answer and also the start of '10'."""

    name = "A11"
    arms = {"single": [str(i) for i in range(11)]}  # one arm: relabelling cannot remove the overlap
    glosses = [""] * 11

    FRAME = """You are one of 4 members of a group. Each round every member receives 10 tokens and decides how many to put into a shared project. All tokens put into the project are multiplied and shared equally: every member receives {mpcr} tokens for each token that anyone puts in, whatever they gave themselves. Tokens you do not put in are yours to keep.

You will be told what the other 3 members put in last round. After learning this, decide: how many tokens will you put in this round? Answer with ONLY one whole number from 0 to 10 - the number of tokens you put in - with no explanation or other text.

Your group:

Last round, the other 3 members put in {mean} tokens each on average.

Your response:
"""

    def __init__(self):
        self.states = [{"mean": x, "mpcr": m} for x in (0, 2, 4, 6, 8, 10) for m in (0.4, 0.6, 0.8)]  # 18 states

    def render(self, st, arm="single", rev=False):
        return self.FRAME.format(mpcr=st["mpcr"], mean=st["mean"])  # the order does not apply here


# ----------------------------------------------------------------------
# B2 and P1: the paper's MOVE/STAY task (prompts built by the repo's own code).
# ----------------------------------------------------------------------

R3_SENTENCE = "Answer with ONLY one word - MOVE or STAY - with no explanation or other text."  # in RATIO_FRAME
P1_SENTENCE = ('Provide your output in the following valid JSON format, with no explanation or other text:\n'
               '{"action": "MOVE"} or {"action": "STAY"}')  # prompt_variants.md, P1


class B2(Task):
    """The paper's MOVE/STAY task, 'baseline' scenario, R3_dual_count, both roles: 90 states."""

    name = "B2"
    arms = {"single": ["MOVE", "STAY"]}
    glosses = ["", ""]
    STYLE = "R3_dual_count"  # the style of every exact table in the paper

    def __init__(self, scenario="baseline"):
        # Imported here so the other tasks work without the repo's prompt code on the path.
        from ratio_prompt_templates import ALL_COMPOSITIONS, RATIO_CANDIDATES  # the 45 compositions
        from evaluate_ratio_prompts import render_prompt  # the repo's prompt renderer
        from sampling_common import role_keywords  # role labels per scenario
        self._tpl, self._fn = RATIO_CANDIDATES[self.STYLE]  # template and context sentence
        self._render = render_prompt  # keep the function
        self._kw = role_keywords(scenario)  # {'red': {...}, 'blue': {...}}
        self.states = [{"scenario": scenario, "role": r, "n_similar": s, "n_occupied": o}
                       for r in ("red", "blue") for (s, o) in ALL_COMPOSITIONS]  # 90 states

    def render(self, st, arm="single", rev=False):
        prompt, _ = self._render(self.STYLE, self._tpl, self._fn, st["n_similar"], st["n_occupied"],
                                 self._kw[st["role"]])  # byte for byte the extractor's prompt
        if rev:  # options named in the other order
            assert R3_SENTENCE in prompt
            prompt = prompt.replace(R3_SENTENCE, R3_SENTENCE.replace("MOVE or STAY", "STAY or MOVE"))
        return prompt


class P1(B2):
    """The same 90 states with the strict JSON answer format: {"action": "MOVE"}."""

    name = "P1"
    WRAP_OPEN = '{"action": "'  # the fixed text before the label
    WRAP_CLOSE = '"}'  # the fixed text after it

    def render(self, st, arm="single", rev=False):
        prompt = B2.render(self, st, arm, rev=False)  # the R3 prompt
        assert R3_SENTENCE in prompt  # the sentence we replace must be there
        sent = P1_SENTENCE if not rev else P1_SENTENCE.replace(
            '{"action": "MOVE"} or {"action": "STAY"}', '{"action": "STAY"} or {"action": "MOVE"}')
        return prompt.replace(R3_SENTENCE, sent)

    def language(self, arm, mode, rev=False):
        """Strict JSON, upper-case labels: with grammar exactly the wrapper; free adds outer whitespace and nothing else."""
        labs = self.labels(arm, rev)
        if mode == "grammar":  # root ::= "{\"action\": \"" ("MOVE" | "STAY") "\"}"
            segs = [("lit", [self.WRAP_OPEN], False), ("label",), ("lit", [self.WRAP_CLOSE], False)]
        else:  # free replies: the same JSON, with whitespace allowed around it
            segs = [("ws",), ("lit", [self.WRAP_OPEN], False), ("label",), ("lit", [self.WRAP_CLOSE], False), ("ws",)]
        return OptionLanguage(labs, segs, label_ci=False)


ALL_TASKS = {"C2": C2, "T3": T3, "P4": P4, "O5": O5, "A11": A11, "B2": B2, "P1": P1}  # by name
