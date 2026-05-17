# LLM Scenario Coverage Report

Scenarios are filtered using SCENARIOS_TO_PLOT when available; otherwise SCENARIOS keys.

| LLM model | Found | Used | Missing | Extra |
|---|---|---|---|---|
gemma3:27b | 01, AB, ethnic_asian_hispanic, gender_man_woman, green_yellow, income_high_low, llm_baseline, political_liberal_conservative, race_white_black | ethnic_asian_hispanic, green_yellow, income_high_low, llm_baseline, political_liberal_conservative, race_white_black | mech_baseline | 01, AB, gender_man_woman
gemma3:4b | ethnic_asian_hispanic, green_yellow, income_high_low, llm_baseline, political_liberal_conservative, race_white_black | ethnic_asian_hispanic, green_yellow, income_high_low, llm_baseline, political_liberal_conservative, race_white_black | mech_baseline | -
hermes3:latest | ethnic_asian_hispanic, green_yellow, income_high_low, llm_baseline, political_liberal_conservative, race_white_black | ethnic_asian_hispanic, green_yellow, income_high_low, llm_baseline, political_liberal_conservative, race_white_black | mech_baseline | -
llama3.3:latest | 01, AB, ethnic_asian_hispanic, gender_man_woman, green_yellow, income_high_low, llm_baseline, political_liberal_conservative, race_white_black | ethnic_asian_hispanic, green_yellow, income_high_low, llm_baseline, political_liberal_conservative, race_white_black | mech_baseline | 01, AB, gender_man_woman
mixtral:8x22b-instruct | 01, AB, economic_high_working, ethnic_asian_hispanic, gender_man_woman, green_yellow, income_high_low, llm_baseline, political_liberal_conservative, race_white_black | ethnic_asian_hispanic, green_yellow, income_high_low, llm_baseline, political_liberal_conservative, race_white_black | mech_baseline | 01, AB, economic_high_working, gender_man_woman
phi4:latest | 01, AB, ethnic_asian_hispanic, gender_man_woman, green_yellow, income_high_low, llm_baseline, political_liberal_conservative, race_white_black | ethnic_asian_hispanic, green_yellow, income_high_low, llm_baseline, political_liberal_conservative, race_white_black | mech_baseline | 01, AB, gender_man_woman
qwen2.5-coder:32B | 01, AB, ethnic_asian_hispanic, gender_man_woman, green_yellow, income_high_low, llm_baseline, political_liberal_conservative, race_white_black | ethnic_asian_hispanic, green_yellow, income_high_low, llm_baseline, political_liberal_conservative, race_white_black | mech_baseline | 01, AB, gender_man_woman