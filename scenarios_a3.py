# scenarios_a3.py
#
# All study scenarios re-expressed in the A3 prompt structure, selected by the
# 2026-07 prompt-refinement evaluation (see prompt_refinement/README.md).
#
# The A3 structure (winner on Llama-3.3/Gemma-4/Qwen3.6; cheapest + most
# responsive + cleanest parses):
#   1. briefing ("You are a {agent_type}...")           -- fixed text, cached
#   2. scenario persona paragraph                        -- fixed text, cached
#   3. anticipatory legend ("You will be shown a map")   -- fixed text, cached
#   4. anticipatory one-word format rule                 -- fixed text, cached
#   5. {context} grid                                    -- varies every call
#   6. short flipped question ("stay or move?") + cue    -- ~25 recomputed tokens
#
# Design rules this file preserves (breaking them re-opens the evaluation):
# - ALL fixed text sits ABOVE {context}: llama.cpp caches a prompt PREFIX, so
#   every token after the grid is recomputed on every call.
# - The persona paragraph stays: removing it (variant A4) collapsed high-hostility
#   behaviour on Llama and produced 24% unparseable replies on Gemma.
# - The question is restated after the grid: trimming it (A1) or moving it above
#   (A2) cost 30-50% of the response range on Llama.
# - "stay or move?" word order (A3): equal or better than "move or stay?" (A) on
#   every model tested; better low-hostility calibration on Llama.
#
# Scenario voice: each scenario keeps its own persona paragraph and its own
# S/O/E/# legend wording from context_scenarios.py, so only the STRUCTURE
# changed relative to the originals, not the social framing.
#
# Used via the `scenario_file` key in a run-config YAML (or --scenario-file):
# it REPLACES context_scenarios.py definitions, and only the scenarios listed
# here are run/valid. Keys must match context_scenarios.py naming so tooling
# (resume detection, analysis) keeps working.

CONTEXT_SCENARIOS = {

    'baseline': {
        'type_a': 'red team resident',
        'type_b': 'blue team resident',
        'prompt_template': """You are a {agent_type} living in a neighborhood, considering whether to move to a different house.

As a real person, you have your own preferences about neighborhood composition, but you also consider practical factors like proximity to friends, community ties, and comfort level.

You will be shown a map of your immediate neighborhood, where:
- X = Your current position (center)
- S = neighbors who are also {agent_type}s like you
- O = neighbors from the {opposite_type} community
- E = empty houses you could move to
- # = area outside the neighborhood

After looking at the map, decide whether you would move to a different house or stay where you are. Answer with ONLY one word - MOVE or STAY - with no explanation or other text.

Your neighborhood:

{context}

Do you want to stay or move?

Your response:
"""
    },

    'race_white_black': {
        'type_a': 'white middle class family',
        'type_b': 'Black family',
        'prompt_template': """You are a {agent_type} considering whether to move to a different house in your neighborhood.

As a real person with your own background, experiences, and family considerations, think about where you would genuinely prefer to live. Consider factors like community comfort, children's friendships, cultural connections, safety perceptions, and social dynamics that matter to you.

You will be shown a map of your immediate 3x3 neighborhood, where:
- X = Your current position (center)
- S = neighbors who are {agent_type}s like you
- O = neighbors who are {opposite_type}s
- E = empty houses available to move to
- # = outside the immediate area

After looking at the map, decide whether you would move to a different house or stay where you are. Answer with ONLY one word - MOVE or STAY - with no explanation or other text.

Your neighborhood:

{context}

Do you want to stay or move?

Your response:
"""
    },

    'ethnic_asian_hispanic': {
        'type_a': 'Asian American family',
        'type_b': 'Hispanic/Latino family',
        'prompt_template': """You are a {agent_type} considering moving to a different house in your neighborhood.

As a real person with your own cultural background, family needs, and community connections, consider where you would actually want to live. Think about factors like cultural familiarity, language communities, children's social connections, extended family proximity, and your comfort level in different neighborhood compositions.

You will be shown a map of your immediate 3x3 neighborhood, where:
- X = Your current position (center)
- S = neighbors who are {agent_type}s with similar backgrounds
- O = neighbors who are {opposite_type}s
- E = empty houses you could potentially move to
- # = outside your immediate area

After looking at the map, decide whether you would move to a different house or stay where you are. Answer with ONLY one word - MOVE or STAY - with no explanation or other text.

Your neighborhood:

{context}

Do you want to stay or move?

Your response:
"""
    },

    'income_high_low': {
        'type_a': 'high-income household',
        'type_b': 'low-income household',
        'prompt_template': """You are a {agent_type} considering whether to move to a different house in your neighborhood.

As a real person making housing decisions, consider your economic priorities: affordability, property values, proximity to work, schools, and the kind of community environment that matches your lifestyle and budget.

You will be shown a map of your immediate 3x3 neighborhood, where:
- X = Your current position (center)
- S = neighbors from households similar to yours ({agent_type}s)
- O = neighbors from {opposite_type}s
- E = empty houses you could move to
- # = area outside the neighborhood

After looking at the map, decide whether you would move to a different house or stay where you are. Answer with ONLY one word - MOVE or STAY - with no explanation or other text.

Your neighborhood:

{context}

Do you want to stay or move?

Your response:
"""
    },

    'political_liberal_conservative': {
        'type_a': 'politically liberal household',
        'type_b': 'politically conservative household',
        'prompt_template': """You are a {agent_type} considering whether to move to a different house in your neighborhood.

As a real person with your own political beliefs, social values, and family considerations, think about where you would actually want to live. Consider factors like community discussions, local politics, school board dynamics, social comfort at neighborhood gatherings, shared values around children's upbringing, and day-to-day interactions with neighbors.

You will be shown a map of your immediate 3x3 neighborhood, where:
- X = Your current position (center)
- S = neighbors who are also {agent_type}s with similar political views
- O = neighbors who are {opposite_type}s with different political perspectives
- E = empty houses you could move to
- # = beyond your immediate area

After looking at the map, decide whether you would move to a different house or stay where you are. Answer with ONLY one word - MOVE or STAY - with no explanation or other text.

Your neighborhood:

{context}

Do you want to stay or move?

Your response:
"""
    },

    'green_yellow': {
        'type_a': 'green team resident',
        'type_b': 'yellow team resident',
        'prompt_template': """You are a {agent_type} living in a neighborhood, considering whether to move to a different house.

As a real person, you have your own preferences about neighborhood composition, but you also consider practical factors like proximity to friends, community ties, and comfort level.

You will be shown a map of your immediate neighborhood, where:
- X = Your current position (center)
- S = neighbors who are also {agent_type}s like you
- O = neighbors from the {opposite_type} community
- E = empty houses you could move to
- # = area outside the neighborhood

After looking at the map, decide whether you would move to a different house or stay where you are. Answer with ONLY one word - MOVE or STAY - with no explanation or other text.

Your neighborhood:

{context}

Do you want to stay or move?

Your response:
"""
    },
}
