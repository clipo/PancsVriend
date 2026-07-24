# scenarios_a2.py
#
# All study scenarios in the A2 prompt structure. DERIVED from scenarios_a3.py
# by two mechanical edits (generator: throwaway gen_a2.py); persona paragraphs,
# legends and social framing are preserved verbatim so ONLY the prompt STRUCTURE
# differs from the A3 file.
#
# A2 vs A3 (see prompt_refinement/prompt_templates.py A2_ASK_ABOVE_GRID):
#   - the decision instruction sits fully ABOVE the grid ("decide: would you
#     move to a different house, or stay where you are?");
#   - the restated question after the grid ("Do you want to stay or move?") is
#     REMOVED -- only the "Your response:" cue follows {context}.
# The A3 file records that A2 "cost 30-50% of the response range on Llama"; this
# file exists to run that comparison, not because A2 is preferred.
#
# Used via the `scenario_file` key in a run-config YAML (or --scenario-file):
# it REPLACES context_scenarios.py; only the scenarios here are run/valid.

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

After looking at the map, decide: would you move to a different house, or stay where you are? Answer with ONLY one word - MOVE or STAY - with no explanation or other text.

Your neighborhood:

{context}

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

After looking at the map, decide: would you move to a different house, or stay where you are? Answer with ONLY one word - MOVE or STAY - with no explanation or other text.

Your neighborhood:

{context}

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

After looking at the map, decide: would you move to a different house, or stay where you are? Answer with ONLY one word - MOVE or STAY - with no explanation or other text.

Your neighborhood:

{context}

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

After looking at the map, decide: would you move to a different house, or stay where you are? Answer with ONLY one word - MOVE or STAY - with no explanation or other text.

Your neighborhood:

{context}

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

After looking at the map, decide: would you move to a different house, or stay where you are? Answer with ONLY one word - MOVE or STAY - with no explanation or other text.

Your neighborhood:

{context}

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

After looking at the map, decide: would you move to a different house, or stay where you are? Answer with ONLY one word - MOVE or STAY - with no explanation or other text.

Your neighborhood:

{context}

Your response:
"""
    },

}
