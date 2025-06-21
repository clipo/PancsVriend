# context_scenarios.py
# Contains all CONTEXT_SCENARIOS for LLM and baseline runners

CONTEXT_SCENARIOS = {
    'baseline': {
        'type_a': 'red team resident',
        'type_b': 'blue team resident',
        'prompt_template': """You are a {agent_type} living in a neighborhood, considering whether to move to a different house.

You are looking at your immediate 3x3 neighborhood:
{context}

Where:
- 'S' = neighbors who are also {agent_type}s like you
- 'O' = neighbors from the {opposite_type} community  
- 'E' = empty houses you could move to
- 'X' = area outside the neighborhood

As a real person, you have your own preferences about neighborhood composition, but you also consider practical factors like proximity to friends, community ties, and comfort level.

Think about where you'd genuinely want to live, then respond with ONLY:
- The coordinates (row, col) of the empty house you'd move to, OR
- None (if you prefer to stay where you are)

Do not explain your reasoning. Just give your decision.

Your choice:"""
    },
    'race_white_black': {
        'type_a': 'white middle class family',
        'type_b': 'Black family',
        'prompt_template': """You are a {agent_type} considering whether to move to a different house in your neighborhood.

Your immediate 3x3 neighborhood looks like this:
{context}

Where:
- 'S' = neighbors who are {agent_type}s like you
- 'O' = neighbors who are {opposite_type}s
- 'E' = empty houses available to move to  
- 'X' = outside the immediate area

As a real person with your own background, experiences, and family considerations, think about where you would genuinely prefer to live. Consider factors like community comfort, children's friendships, cultural connections, safety perceptions, and social dynamics that matter to you.

Respond with ONLY:
- The coordinates (row, col) of an empty house you'd move to, OR  
- None (if you'd rather stay put)

No explanation needed. Just your decision:"""
    },
    'ethnic_asian_hispanic': {
        'type_a': 'Asian American family',
        'type_b': 'Hispanic/Latino family',
        'prompt_template': """You are a {agent_type} considering moving to a different house in your neighborhood.

Your immediate 3x3 neighborhood currently looks like:
{context}

Where:
- 'S' = neighbors who are {agent_type}s with similar backgrounds
- 'O' = neighbors who are {opposite_type}s  
- 'E' = empty houses you could potentially move to
- 'X' = outside your immediate area

As a real person with your own cultural background, family needs, and community connections, consider where you would actually want to live. Think about factors like cultural familiarity, language communities, children's social connections, extended family proximity, and your comfort level in different neighborhood compositions.

Respond with ONLY:
- The coordinates (row, col) of an empty house you'd move to, OR
- None (if you prefer staying where you are)

Just your decision:"""
    },
    'income_high_low': {
        'type_a': 'high-income household',
        'type_b': 'working-class household',
        'prompt_template': """You are a {agent_type} considering whether to move to a different house in your neighborhood.

You are looking at your immediate 3x3 neighborhood:
{context}

Where:
- 'S' = neighbors from households similar to yours ({agent_type}s)
- 'O' = neighbors from {opposite_type}s
- 'E' = empty houses you could move to
- 'X' = area outside the neighborhood

As a real person making housing decisions, consider your economic priorities: affordability, property values, proximity to work, schools, and the kind of community environment that matches your lifestyle and budget.

Where would you prefer to move? Respond with ONLY:
- Coordinates like (0,1) for a specific empty house
- None (if you'd rather stay put)

No explanation needed. Just your decision:"""
    },
    'economic_high_working': {
        'type_a': 'high-income household',
        'type_b': 'working-class household',
        'prompt_template': """You are a {agent_type} considering whether to move to a different house in your neighborhood.

Looking at your immediate 3x3 neighborhood:
{context}

Where:
- 'S' = neighbors who are also {agent_type}s in a similar economic situation  
- 'O' = neighbors who are {opposite_type}s with different economic circumstances
- 'E' = empty houses available for you to move to
- 'X' = outside your immediate area

As a real person with your own financial situation, lifestyle preferences, and family priorities, think about where you would genuinely want to live. Consider factors like property values, school quality expectations, social comfort levels, shared community values, and the kind of neighborhood environment you want for your family.

Respond with ONLY:
- The coordinates (row, col) of an empty house you'd move to, OR
- None (if you'd rather stay where you are)

Your decision:"""
    },
    'political_liberal_conservative': {
        'type_a': 'politically liberal household',
        'type_b': 'politically conservative household',
        'prompt_template': """You are a {agent_type} considering whether to move to a different house in your neighborhood.

Your immediate 3x3 neighborhood situation:
{context}

Where:
- 'S' = neighbors who are also {agent_type}s with similar political views
- 'O' = neighbors who are {opposite_type}s with different political perspectives  
- 'E' = empty houses you could move to
- 'X' = beyond your immediate area

As a real person with your own political beliefs, social values, and family considerations, think about where you would actually want to live. Consider factors like community discussions, local politics, school board dynamics, social comfort at neighborhood gatherings, shared values around children's upbringing, and day-to-day interactions with neighbors.

Respond with ONLY:
- The coordinates (row, col) of an empty house you'd move to, OR
- None (if you prefer to stay put)

Your choice:"""
    }
}
