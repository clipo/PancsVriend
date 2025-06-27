# context_scenarios.py
# Contains all CONTEXT_SCENARIOS for LLM and baseline runners

CONTEXT_SCENARIOS = {
    'baseline': {
        'type_a': 'red team resident',
        'type_b': 'blue team resident',
        'prompt_template': """You are a {agent_type} living in a neighborhood, considering whether to move to a different house.

{context}

Where:
- X = Your current position (center)
- S = neighbors who are also {agent_type}s like you
- O = neighbors from the {opposite_type} community  
- E = empty houses you could move to
- # = area outside the neighborhood

As a real person, you have your own preferences about neighborhood composition, but you also consider practical factors like proximity to friends, community ties, and comfort level.

Based on this neighborhood, do you want to move to a different position or stay where you are?

IMPORTANT: Respond with ONLY one word: MOVE or STAY. Do not provide any explanation, reasoning, or additional text.

Your response:"""
    },
    'race_white_black': {
        'type_a': 'white middle class family',
        'type_b': 'Black family',
        'prompt_template': """You are a {agent_type} considering whether to move to a different house in your neighborhood.

Your immediate 3x3 neighborhood looks like this:
{context}

Where:
- X = Your current position (center)
- S = neighbors who are {agent_type}s like you
- O = neighbors who are {opposite_type}s
- E = empty houses available to move to  
- # = outside the immediate area

As a real person with your own background, experiences, and family considerations, think about where you would genuinely prefer to live. Consider factors like community comfort, children's friendships, cultural connections, safety perceptions, and social dynamics that matter to you.

Based on this neighborhood, do you want to move to a different position or stay where you are?

IMPORTANT: Respond with ONLY one word: MOVE or STAY. Do not provide any explanation, reasoning, or additional text.

Your response:"""
    },
    'ethnic_asian_hispanic': {
        'type_a': 'Asian American family',
        'type_b': 'Hispanic/Latino family',
        'prompt_template': """You are a {agent_type} considering moving to a different house in your neighborhood.

Your immediate 3x3 neighborhood currently looks like:
{context}

Where:
- X = Your current position (center)
- S = neighbors who are {agent_type}s with similar backgrounds
- O = neighbors who are {opposite_type}s  
- E = empty houses you could potentially move to
- # = outside your immediate area

As a real person with your own cultural background, family needs, and community connections, consider where you would actually want to live. Think about factors like cultural familiarity, language communities, children's social connections, extended family proximity, and your comfort level in different neighborhood compositions.

Based on this neighborhood, do you want to move to a different position or stay where you are?

IMPORTANT: Respond with ONLY one word: MOVE or STAY. Do not provide any explanation, reasoning, or additional text.

Your response:"""
    },
    'income_high_low': {
        'type_a': 'high-income household',
        'type_b': 'working-class household',
        'prompt_template': """You are a {agent_type} considering whether to move to a different house in your neighborhood.

You are looking at your immediate 3x3 neighborhood:
{context}

Where:
- X = Your current position (center)
- S = neighbors from households similar to yours ({agent_type}s)
- O = neighbors from {opposite_type}s
- E = empty houses you could move to
- # = area outside the neighborhood

As a real person making housing decisions, consider your economic priorities: affordability, property values, proximity to work, schools, and the kind of community environment that matches your lifestyle and budget.

Based on this neighborhood, do you want to move to a different position or stay where you are?

IMPORTANT: Respond with ONLY one word: MOVE or STAY. Do not provide any explanation, reasoning, or additional text.

Your response:"""
    },
    'economic_high_working': {
        'type_a': 'high-income household',
        'type_b': 'working-class household',
        'prompt_template': """You are a {agent_type} considering whether to move to a different house in your neighborhood.

Looking at your immediate 3x3 neighborhood:
{context}

Where:
- X = Your current position (center)
- S = neighbors who are also {agent_type}s in a similar economic situation  
- O = neighbors who are {opposite_type}s with different economic circumstances
- E = empty houses available for you to move to
- # = outside your immediate area

As a real person with your own financial situation, lifestyle preferences, and family priorities, think about where you would genuinely want to live. Consider factors like property values, school quality expectations, social comfort levels, shared community values, and the kind of neighborhood environment you want for your family.

Based on this neighborhood, do you want to move to a different position or stay where you are?

IMPORTANT: Respond with ONLY one word: MOVE or STAY. Do not provide any explanation, reasoning, or additional text.

Your response:"""
    },
    'political_liberal_conservative': {
        'type_a': 'politically liberal household',
        'type_b': 'politically conservative household',
        'prompt_template': """You are a {agent_type} considering whether to move to a different house in your neighborhood.

Your immediate 3x3 neighborhood situation:
{context}

Where:
- X = Your current position (center)
- S = neighbors who are also {agent_type}s with similar political views
- O = neighbors who are {opposite_type}s with different political perspectives  
- E = empty houses you could move to
- # = beyond your immediate area

As a real person with your own political beliefs, social values, and family considerations, think about where you would actually want to live. Consider factors like community discussions, local politics, school board dynamics, social comfort at neighborhood gatherings, shared values around children's upbringing, and day-to-day interactions with neighbors.

Based on this neighborhood, do you want to move to a different position or stay where you are?

IMPORTANT: Respond with ONLY one word: MOVE or STAY. Do not provide any explanation, reasoning, or additional text.

Your response:"""
    }
}
