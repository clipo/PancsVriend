"""Ratio-only prompt candidates: the agent is told its neighborhood COMPOSITION,
never the arrangement.

Motivation: the canonical Schelling agent (Agent.py `utility`) depends only on
the fraction of unlike neighbors among OCCUPIED neighbors — it never sees where
anyone lives, and empties/walls are excluded from the denominator. These
candidates give the LLM agent exactly that information set, so that sampling
MOVE frequencies over all (n_similar, n_occupied) compositions yields a
per-model "sampled value function" P(MOVE | composition) that can drive the
Schelling simulation without live LLM calls.

Wording rules (design decisions, see plans/wondrous-bouncing-ladybug.md):
- NEIGHBOR-COUNT-ONLY: prompts state how many neighbors the agent has and their
  composition. They never mention empty houses, walls, or "of the 8 slots":
  (a) that stays truthful for boundary agents, whose missing slots are walls,
  (b) local vacancy is mechanically irrelevant (relocation is global), and
  (c) stated vacancy is itself a MOVE cue that would confound the comparison.
- One shared FRAME (persona + instruction, from the scenarios_a2.py baseline)
  across all ratio candidates; only the composition sentence differs, so the
  manipulated variable is exactly the numeric encoding.
- No verbal quantifiers ("most", "a minority"): they pre-frame majority status
  — the judgment under study — and their wording would switch mid-gradient.

Each candidate is (template, context_fn) where
    context_fn(n_similar, n_occupied, agent_type, opposite_type) -> str
fills the template's {context}. Templates must contain {agent_type},
{opposite_type} and {context}, and keep all fixed text ABOVE {context} so
llama.cpp's prefix cache covers everything up to the varying sentence.
"""

# Shared frame for all ratio candidates. Persona paragraph is verbatim from the
# scenarios_a2.py baseline; the map legend is replaced by the composition
# briefing; A2 structure kept (question above {context}, bare "Your response:").
RATIO_FRAME = """You are {a_agent_type} living in a neighborhood, considering whether to move to a different house.

As a real person, you have your own preferences about neighborhood composition, but you also consider practical factors like proximity to friends, community ties, and comfort level.

You will be told how many neighbors you currently have and their composition. You cannot see where anyone lives, only how many. After learning the composition, decide: would you move to a different house, or stay where you are? Answer with ONLY one word - MOVE or STAY - with no explanation or other text.

Your neighborhood:

{context}

Your response:
"""

# The exact scenarios_a2.py baseline template — the grid anchor arm renders the
# same 3x3 map production uses, tying this sweep to the existing evidence base.
GRID_FRAME = """You are {a_agent_type} living in a neighborhood, considering whether to move to a different house.

As a real person, you have your own preferences about neighborhood composition, but you also consider practical factors like proximity to friends, community ties, and comfort level.

You will be shown a map of your immediate neighborhood, where:
- X = Your current position (center)
- S = neighbors who are also {agent_type_plural} like you
- O = neighbors from the {opposite_type} community
- E = empty houses you could move to
- # = area outside the neighborhood

After looking at the map, decide: would you move to a different house, or stay where you are? Answer with ONLY one word - MOVE or STAY - with no explanation or other text.

Your neighborhood:

{context}

Your response:
"""

_NO_NEIGHBORS = "You currently have no neighbors."


def plural(label: str) -> str:
    """Pluralize an identity label by its LAST word ('white middle class family'
    -> '... families', 'red team resident' -> '... residents').

    Added 2026-08-21 for the scenario sweep: the naive f'{label}s' produced
    'Black familys' for every family-labelled scenario — a typo the model
    plausibly reacts to. Baseline labels end in 'resident', where this rule
    reproduces the old bytes exactly (asserted in sampling_common), so all
    baseline measurements stay comparable.
    """
    head, _, last = label.rpartition(" ")
    if last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
        last = last[:-1] + "ies"
    else:
        last += "s"
    return f"{head} {last}" if head else last


def with_article(label: str) -> str:
    """'a red team resident' / 'an Asian American family' (letter heuristic —
    sufficient for every label in scenarios_a2.py)."""
    return f"an {label}" if label[:1].lower() in "aeiou" else f"a {label}"


def _count_phrase(n: int, singular: str, plural_form: str) -> str:
    """'{n} is <a singular>' / '{n} are <plural>' with number agreement."""
    return f"{n} is {with_article(singular)}" if n == 1 else f"{n} are {plural_form}"


def r1_count_opposite(n_similar, n_occupied, agent_type, opposite_type):
    """Single count, opposite-valence: 'k of your n neighbors are <opposite>s.'"""
    if n_occupied == 0:
        return _NO_NEIGHBORS
    n_opp = n_occupied - n_similar
    if n_occupied == 1:
        return (f"Your only neighbor is {with_article(opposite_type)}." if n_opp == 1
                else f"Your only neighbor is {with_article(agent_type)} like you.")
    phrase = (f"is {with_article(opposite_type)}" if n_opp == 1
              else f"are {plural(opposite_type)}")
    return f"{n_opp} of your {n_occupied} neighbors {phrase}."


def r2_count_similar(n_similar, n_occupied, agent_type, opposite_type):
    """Single count, similar-valence: 'k of your n neighbors are <same>s like you.'"""
    if n_occupied == 0:
        return _NO_NEIGHBORS
    if n_occupied == 1:
        return (f"Your only neighbor is {with_article(agent_type)} like you." if n_similar == 1
                else f"Your only neighbor is {with_article(opposite_type)}.")
    phrase = (f"is {with_article(agent_type)}" if n_similar == 1
              else f"are {plural(agent_type)}")
    return f"{n_similar} of your {n_occupied} neighbors {phrase} like you."


def r3_dual_count(n_similar, n_occupied, agent_type, opposite_type):
    """Both counts explicit, symmetric valence — no arithmetic left to the model."""
    if n_occupied == 0:
        return _NO_NEIGHBORS
    n_opp = n_occupied - n_similar
    if n_occupied == 1:
        return (f"You have 1 neighbor: {with_article(agent_type)} like you." if n_similar == 1
                else f"You have 1 neighbor: {with_article(opposite_type)}.")
    sim_part = (f"{n_similar} is {with_article(agent_type)} like you" if n_similar == 1
                else f"{n_similar} are {plural(agent_type)} like you")
    opp_part = _count_phrase(n_opp, opposite_type, plural(opposite_type))
    return f"You have {n_occupied} neighbors: {sim_part} and {opp_part}."


def r4_percent_opposite(n_similar, n_occupied, agent_type, opposite_type):
    """Percentage of neighbors (occupied only — the mechanical ratio), 1 decimal.

    Deliberately states no absolute count: aliasing across occupancies (2/4 and
    4/8 both read '50.0%') is part of what this numeric-anchoring probe measures.
    """
    if n_occupied == 0:
        return _NO_NEIGHBORS
    pct = 100.0 * (n_occupied - n_similar) / n_occupied
    # Render eighths exactly (62.5%), others to 1 decimal (66.7%).
    pct_str = f"{pct:.1f}".rstrip("0").rstrip(".")
    return f"{pct_str}% of your neighbors are {plural(opposite_type)}."


def r5_percent_similar(n_similar, n_occupied, agent_type, opposite_type):
    """Percentage in similar valence — completes the encoding x valence 2x2
    (count/percent x similar/opposite), so a percent-vs-count divergence can be
    attributed to the encoding rather than to which group is named."""
    if n_occupied == 0:
        return _NO_NEIGHBORS
    pct = 100.0 * n_similar / n_occupied
    pct_str = f"{pct:.1f}".rstrip("0").rstrip(".")
    return f"{pct_str}% of your neighbors are {plural(agent_type)} like you."


def grid_context(n_similar, n_occupied, agent_type, opposite_type, rng=None):
    """3x3 map at the given composition, empties filling the remaining slots.

    Same contract as evaluate_prompts.grid_with but with an E count: `rng`
    shuffles the cells so repeated samples marginalise over geometry; None gives
    a deterministic layout (token counts are arrangement-invariant).
    """
    n_opp = n_occupied - n_similar
    cells = ["S"] * n_similar + ["O"] * n_opp + ["E"] * (8 - n_occupied)
    if rng is not None:
        rng.shuffle(cells)
    c = cells[:4] + ["X"] + cells[4:]
    return "\n".join(" ".join(c[i * 3:i * 3 + 3]) for i in range(3))


# name -> (template, context_fn). G0's context_fn takes an extra rng kwarg,
# which the harness supplies (ratio renderers are deterministic per cell).
RATIO_CANDIDATES = {
    "R1_count_opposite": (RATIO_FRAME, r1_count_opposite),
    "R2_count_similar": (RATIO_FRAME, r2_count_similar),
    "R3_dual_count": (RATIO_FRAME, r3_dual_count),
    "R4_percent_opposite": (RATIO_FRAME, r4_percent_opposite),
    "R5_percent_similar": (RATIO_FRAME, r5_percent_similar),
    "G0_grid_anchor": (GRID_FRAME, grid_context),
}

# All compositions of the 8-cell Moore neighborhood the simulation can visit:
# n_occupied 0..8 x n_similar 0..n_occupied  ->  45 cells.
ALL_COMPOSITIONS = [(n_sim, n_occ)
                    for n_occ in range(9)
                    for n_sim in range(n_occ + 1)]


def mechanical_move(n_similar: int, n_occupied: int) -> int:
    """The canonical reference decision at each cell, per Agent._unlike_ratio
    semantics: MOVE iff the unlike fraction over occupied neighbors exceeds
    1 - SIMILARITY_THRESHOLD (= 0.5); no neighbors -> satisfied -> STAY."""
    if n_occupied == 0:
        return 0
    return 1 if (n_occupied - n_similar) / n_occupied > 0.5 else 0
