# TODO: Overall Ordering Metric Tasks

## Task 1: Generate Python Script to Process CSV and Create LaTeX Tables
- Create a Python script that:
  1. Reads `Schelling-Core-results-formatted.csv`
  2. Calculates an 'overall ordering' metric that summarizes orderings from six metrics:
     - -clusters, distance, ghetto_rate, mix_deviation, share, -switch_rate
  3. Uses consensus rules for the overall ordering symbol:
     - `<` if 5 or 6 metrics agree on the ordering (strong consensus)
     - `\geq` if at least 1 metric agrees on the ordering (weak consensus)  
     - `=` if all metrics agree the scenarios are equal
     - `\lessgtr^?` if there's disagreement about the ordering direction
  4. Generates both LaTeX table files:
     - `results_table.tex` (all scenarios with overall ordering as first row)
     - `results_table_llmonly.tex` (excluding mechanical with overall ordering as first row)
  5. Preserves all formatting:
     - Vertical vectors using `\begin{pmatrix}`
     - Proper spacing and alignment
     - Centered inequalities
     - Stars preserved as superscripts
- Make the script reusable for when CSV data changes

## Task 2: Create Legend/Glossary for Tables
- Write a complete legend/glossary that includes:
  - Bullet point definitions of the six segregation metrics:
    - -clusters
    - distance
    - ghetto_rate
    - mix_deviation
    - share
    - -switch_rate
  - Bullet point definitions of the segregation scenario types:
    - green/yellow
    - income
    - llm_baseline
    - race:wht/blk
    - ethnicity
    - mechanical
    - political
- Include the legend/glossary in show_results_table.tex
- The legend should apply to both tables (all scenarios and LLM-only)

## Task 3: [Merged with Task 1]
- This task has been merged with Task 1 since they both involve creating a Python script to process the CSV and generate LaTeX tables

## Completed Tasks
- ✅ Convert CSV to LaTeX formatted tables with math mode inequalities
- ✅ Create two tables: all scenarios and LLM-only scenarios
- ✅ Task 1: Generate Python script to process CSV and create LaTeX tables with overall ordering
- ✅ Implement proper spacing and alignment using array environment
- ✅ Center inequalities and add vertical spacing for readability
- ✅ Add explanatory text below tables