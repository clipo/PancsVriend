"""Candidate prompt templates for the MOVE/STAY decision.

WHY THIS EXISTS
---------------
Two independent problems drove this comparison:

1. COST. We serve the raw /v1/completions endpoint on llama.cpp, which caches a
   *prefix* of the prompt across calls. Only the tokens BEFORE the first varying
   token can be reused. `{context}` (the 3x3 neighbourhood grid) is the only part
   that varies, so every token placed AFTER it is recomputed on every single call.
   At ~60 tok/s prompt-eval on the GB10, each recomputed token is real wall-clock:
   the production run makes hundreds of thousands of these calls.

   => Design rule: put all fixed text ABOVE {context}, keep the text below it short.

2. QUALITY. The template in context_scenarios.py buries the grid near the top and
   puts ~145 tokens of legend/instructions after it. That is both maximally
   expensive AND (see results/) yields a decision that barely tracks the
   neighbourhood: P(MOVE) is compressed into ~0.00-0.15 and is NOT monotonic --
   it actually FALLS at maximum hostility, which is backwards.

A prompt is only good if it is cheap AND still elicits a neighbourhood-sensitive,
monotonic decision. Ranking on cost alone would happily select a prompt the model
ignores, so evaluate_prompts.py scores both axes.

Every template must contain the placeholders {agent_type}, {opposite_type} and
{context} so it is a drop-in for context_scenarios.py's prompt_template.
"""

# The symbol key. Fixed text -> free if placed above {context} (lands in the cached
# prefix), ~65 recomputed tokens per call if placed below it.
LEGEND = """- X = Your current position (center)
- S = neighbors who are also {agent_type}s like you
- O = neighbors from the {opposite_type} community
- E = empty houses you could move to
- # = area outside the neighborhood"""


# ---------------------------------------------------------------------------
# 0. CURRENT -- what context_scenarios.py ships today. The baseline to beat.
#    Grid sits near the TOP, so the whole legend + instruction block below it is
#    recomputed every call (145 tokens).
# ---------------------------------------------------------------------------
CURRENT = """You are a {agent_type} living in a neighborhood, considering whether to move to a different house.

{context}

Where:
""" + LEGEND + """

As a real person, you have your own preferences about neighborhood composition, but you also consider practical factors like proximity to friends, community ties, and comfort level.

Based on this neighborhood, do you want to move to a different position or stay where you are?

IMPORTANT: Respond with ONLY one word: MOVE or STAY. Do not provide any explanation, reasoning, or additional text.

Your response:
"""


# ---------------------------------------------------------------------------
# A. BRIEFING -> MAP -> ASK.
#    All fixed text above the grid; only a short question below it.
#    The one-word rule is ANTICIPATORY ("You will be shown a map... after looking
#    at the map, decide"), so it reads as task setup rather than arriving with no
#    context, and the actual question lands last (recency).
# ---------------------------------------------------------------------------
BRIEFING_MAP_ASK = """You are a {agent_type} living in a neighborhood, considering whether to move to a different house.

As a real person, you have your own preferences about neighborhood composition, but you also consider practical factors like proximity to friends, community ties, and comfort level.

You will be shown a map of your immediate neighborhood, where:
""" + LEGEND + """

After looking at the map, decide whether you would move to a different house or stay where you are. Answer with ONLY one word - MOVE or STAY - with no explanation or other text.

Your neighborhood:

{context}

Do you want to move or stay?

Your response:
"""


# ---------------------------------------------------------------------------
# B. As A, but the one-word rule is RESTATED after the question instead of in the
#    briefing. Tests whether ending on a terse format command changes the answer.
#    (It does -- it suppresses MOVE. See results/.)
# ---------------------------------------------------------------------------
BRIEFING_MAP_ASK_RULE = """You are a {agent_type} living in a neighborhood, considering whether to move to a different house.

As a real person, you have your own preferences about neighborhood composition, but you also consider practical factors like proximity to friends, community ties, and comfort level.

You will be shown a map of your immediate neighborhood, where:
""" + LEGEND + """

Your neighborhood:

{context}

Based on this neighborhood, do you want to move to a different house or stay where you are?

Answer with ONLY one word - MOVE or STAY. Do not explain.

Your response:
"""


# ---------------------------------------------------------------------------
# C. LEGEND AFTER THE GRID -- the "figure, then key" convention (show the symbols,
#    then explain them). Reads naturally, but the legend is the biggest fixed block
#    and placing it below {context} means paying for it on every call.
# ---------------------------------------------------------------------------
LEGEND_AFTER_GRID = """You are a {agent_type} living in a neighborhood, considering whether to move to a different house.

As a real person, you have your own preferences about neighborhood composition, but you also consider practical factors like proximity to friends, community ties, and comfort level.

IMPORTANT: Respond with ONLY one word: MOVE or STAY. Do not provide any explanation, reasoning, or additional text.

Your neighborhood:

{context}

Where:
""" + LEGEND + """

Based on this neighborhood, do you want to move to a different position or stay where you are?

Your response:
"""



# ---------------------------------------------------------------------------
# A-VARIANTS -- refinements of the winning template A. A itself is FROZEN as the
# baseline; each variant isolates exactly ONE change so any difference in the
# results is attributable. Naming: A1, A2, ... (see results/ for the comparison).
# ---------------------------------------------------------------------------

# A1: tail shortened. A's post-grid text ("Do you want to move or stay?") repeats
# what the briefing already set up; this trims it to the bare question. Tests
# whether ~8 recomputed tokens can be saved without touching quality.
A1_MIN_TAIL = """You are a {agent_type} living in a neighborhood, considering whether to move to a different house.

As a real person, you have your own preferences about neighborhood composition, but you also consider practical factors like proximity to friends, community ties, and comfort level.

You will be shown a map of your immediate neighborhood, where:
""" + LEGEND + """

After looking at the map, decide whether you would move to a different house or stay where you are. Answer with ONLY one word - MOVE or STAY - with no explanation or other text.

Your neighborhood:

{context}

Move or stay?

Your response:
"""


# A2: the question moves fully ABOVE the grid; the only recomputed text is the
# answer cue. Cheapest possible tail (~5 tokens). Tests whether the model still
# fires the decision without a restated question after the map.
A2_ASK_ABOVE_GRID = """You are a {agent_type} living in a neighborhood, considering whether to move to a different house.

As a real person, you have your own preferences about neighborhood composition, but you also consider practical factors like proximity to friends, community ties, and comfort level.

You will be shown a map of your immediate neighborhood, where:
""" + LEGEND + """

After looking at the map, decide: would you move to a different house, or stay where you are? Answer with ONLY one word - MOVE or STAY - with no explanation or other text.

Your neighborhood:

{context}

Your response:
"""


# A3: final question word order flipped ("stay or move?"). Tests order bias --
# whether MOVE's rate in A partly comes from being named first in the question.
A3_STAY_OR_MOVE = """You are a {agent_type} living in a neighborhood, considering whether to move to a different house.

As a real person, you have your own preferences about neighborhood composition, but you also consider practical factors like proximity to friends, community ties, and comfort level.

You will be shown a map of your immediate neighborhood, where:
""" + LEGEND + """

After looking at the map, decide whether you would move to a different house or stay where you are. Answer with ONLY one word - MOVE or STAY - with no explanation or other text.

Your neighborhood:

{context}

Do you want to stay or move?

Your response:
"""


# A4: the "As a real person..." persona sentence removed. It lives in the cached
# prefix (free), so this is purely a QUALITY probe: does the persona framing
# actually shape the decision, or is it inert filler?
A4_NO_PERSONA = """You are a {agent_type} living in a neighborhood, considering whether to move to a different house.

You will be shown a map of your immediate neighborhood, where:
""" + LEGEND + """

After looking at the map, decide whether you would move to a different house or stay where you are. Answer with ONLY one word - MOVE or STAY - with no explanation or other text.

Your neighborhood:

{context}

Do you want to move or stay?

Your response:
"""


CANDIDATES = {
    "0_current":            CURRENT,
    "A_briefing_map_ask":   BRIEFING_MAP_ASK,
    "B_briefing_map_ask_rule": BRIEFING_MAP_ASK_RULE,
    "C_legend_after_grid":  LEGEND_AFTER_GRID,
    "A1_min_tail":          A1_MIN_TAIL,
    "A2_ask_above_grid":    A2_ASK_ABOVE_GRID,
    "A3_stay_or_move":      A3_STAY_OR_MOVE,
    "A4_no_persona":        A4_NO_PERSONA,
}
