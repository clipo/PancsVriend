"""
Centralized list of experiments (scenarios), human-friendly labels, and colors
to be reused across analysis scripts.

Usage:
    from analysis_tools.experiment_list_for_analysis import (
        SCENARIOS, SCENARIO_LABELS, SCENARIO_COLORS,
    )

Notes:
    - Keys in SCENARIOS are the canonical scenario identifiers used across the
      codebase (e.g., 'baseline', 'race_white_black', ...).
    - Values in SCENARIOS are the folder names under 'experiments/'.
    - SCENARIO_LABELS provides readable labels for plots.
    - SCENARIO_COLORS provides a consistent color per scenario for plots.
"""

from __future__ import annotations

# with mixtral:8x22b-instruct used for the paper
SCENARIOS = { #with mixtral:8x22b-instruct
    # Baselines
    # '01': "llm_01_20250804_205533",
    # 'AB': "llm_AB_20250804_153344",
    'llm_baseline': 'llm_baseline_20250703_101243',
    'green_yellow': 'llm_green_yellow_20250912_072712',
    'mech_baseline': 'baseline_20250729_174459',

    # Social contexts
    'ethnic_asian_hispanic': 'llm_ethnic_asian_hispanic_20250713_221759',
    'income_high_low': 'llm_income_high_low_20251006_150254',
    # 'economic_high_working': 'llm_economic_high_working_20250728_220134',
    'political_liberal_conservative': 'llm_political_liberal_conservative_20250724_154733',
    'race_white_black': 'llm_race_white_black_20250718_195455',
    # 'gender_man_woman': 'llm_gender_man_woman_20251007_091936',
}

# Human-friendly labels for plots
SCENARIO_LABELS = {
    # 'baseline': 'Baseline (Control)',
    'llm_baseline': 'Color (Red/Blue)',
    'mech_baseline': 'Mechanical Baseline',
    'political_liberal_conservative': 'Political (Liberal/Conservative)',
    'ethnic_asian_hispanic': 'Ethnic (Asian/Hispanic)',
    'race_white_black': 'Racial (White/Black)',
    'income_high_low': 'Economic (High/Low Income)',
    # 'economic_high_working': 'Economic (High/Working)',
    'green_yellow': 'Color (Green/Yellow)',

    '01': '01',
    'AB': 'AB',
    'gender_man_woman': 'Gender (Man/Woman)',
}
SCENARIO_ORDER = [
    'mech_baseline',
    '01',
    'AB',
    'llm_baseline',
    'green_yellow',
    'political_liberal_conservative',
    'race_white_black',
    'ethnic_asian_hispanic',
    # 'economic_high_working',
    'income_high_low',
    'gender_man_woman',
]
# Consistent colors for scenarios (hex codes)
SCENARIO_COLORS = {
    'mech_baseline': '#34495E',       # Dark slate - contrasts with LLM baseline
    'llm_baseline': '#E74C3C',        # Light gray - neutral baseline
    'green_yellow': '#27AE60',        # Emerald green - vibrant, nature-inspired
    'political_liberal_conservative': '#E67E22', # Warm orange - distinctive, accessible
    'race_white_black': '#95A5A6',    # Vivid red - high contrast, attention-grabbing
    'ethnic_asian_hispanic': '#9B59B6', # Purple - distinct from red/blue politics
    # 'economic_high_working': "#445ED1", # Bright blue - professional, clear
    'income_high_low': '#3498DB', # Bright blue - professional, clear
    '01': "#4D4D4D",                  # Turquoise - fresh, modern
    'AB': '#F1C40F',                  # Yellow - energetic, stands out
    'gender_man_woman': '#FF69B4',          # Hot pink - vibrant, easily distinguishable
}

