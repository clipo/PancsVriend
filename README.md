# PancsVriend: Revealing the Bias Paradox in Large Language Models

A groundbreaking research framework that uncovers how Large Language Models (LLMs), despite having no explicit programming for discrimination, reproduce human segregation patterns with disturbing accuracy. Using the classic Schelling Segregation Model, we demonstrate the "bias paradox": LLMs' absorption of societal prejudices makes them both concerning perpetuators of bias AND superior tools for studying human social dynamics.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-pending-orange.svg)](https://github.com/clipo/PancsVriend)

## Current Status (last updated 2026-09-28)

**Production runs complete for 9 models** using exact value functions from token log-probabilities (`-vf-lp`). Each model has 10,000 runs × 6 scenarios × 1000 max steps. All runs have rank-stability status `SETTLED`.

| Model | Parameters | Run folder | Mean DI range |
|-------|------------|------------|---------------|
| gemma-4-31b | 31B | `run_20260906_003730_gemma-4-31b-vf-lp` | 0.13 – 0.26 |
| phi-4-14b | 14B | `run_20260906_010130_phi-4-14b-vf-lp` | 0.12 – 0.13 |
| granite-4.2-30b | 30B | `run_20260906_020531_granite-4.2-30b-vf-lp` | 0.12 – 0.13 |
| hermes-4.3-36b | 36B | `run_20260906_043033_hermes-4.3-36b-vf-lp` | 0.16 – 0.48 |
| deepseek-v4-flash | — | `run_20260906_092955_deepseek-v4-flash-vf-lp` | 0.14 – 0.42 |
| qwen3.6-27b | 27B | `run_20260906_124956_qwen3.6-27b-vf-lp` | 0.16 – 0.38 |
| olmo-2-32b | 32B | `run_20260906_131256_olmo-2-32b-vf-lp` | 0.46 – 0.74 |
| llama-3.3-70b | 70B | `run_20260912_200213_llama-3.3-70b-vf-lp` | 0.13 – 0.34 |
| mistral-small-4-119b | 119B | `run_20260913_134219_mistral-small-4-119b-vf-lp` | 0.12 – 0.13 |

### Key findings across models

- **Context differentiation varies dramatically**: Some models (olmo, hermes, deepseek, qwen) show strong scenario effects; others (phi, granite, mistral) show minimal differentiation
- **Racial/ethnic scenarios consistently lowest**: Across most models, `race_white_black` and `ethnic_asian_hispanic` produce the lowest segregation
- **No single "correct" ordering**: Different models produce different scenario rankings, suggesting biases are training artifacts, not universal LLM properties

Cross-model comparison plots and statistical tests are in `experiments_with_llama_cpp/cross_model/`

### Pending tasks for CCS 2026 presentation

The presentation at `pres-overleaf.link/pres/Schelling-llm-social-context.tex` needs to be updated with new results for the CCS 2026 conference (October 12–16, Binghamton, NY). The following tasks are required:

#### Task 1: Update Key Results tables

The "Key Results" table comparing LLM scenario orderings to empirical data must be updated with the new 9-model results using statistically rigorous analysis.

**Two tables needed:**

1. **3-way comparison (for presentation):** Income vs Political vs Racial — matches empirical data categories
2. **6-way comparison (comprehensive):** All scenarios — full picture of model behavior

**Task:** Build ordering strings (e.g., "Economic < Political < Racial") for each model where comparison symbols reflect statistical significance:

| Symbol | Meaning | Threshold |
|--------|---------|-----------|
| `<` or `>` | Significantly different | p < 0.01 (Holm-corrected) AND \|Cohen's d\| ≥ 0.2 |
| `≈` | Not statistically distinguishable | p ≥ 0.01 OR \|Cohen's d\| < 0.2 |

**Star notation:** A comparison is marked with a star (`<*` in LaTeX: `$<^{\star}$`) when it matches the empirical ordering. The empirical ordering is Economic < Political < Racial, so any pairwise comparison where the lower-ranked scenario (in empirical) appears on the left is starred — including transitive comparisons (e.g., Economic < Racial).

**Grouping notation:** When one "end" of the ordering matches empirical against all others but the internal ordering of the remaining pair doesn't match, parentheses indicate the non-matching subgroup:
- `A <* (B < C)` — A being lowest matches empirical, but B < C doesn't (e.g., Olmo: `Economic <* (Racial < Political)`)
- `(A < B) <* C` — C being highest matches empirical, but A < B doesn't

**Model sorting:** Models are sorted by: (1) decreasing pairwise matches to empirical, (2) decreasing number of significant differences, (3) alphabetically. Current order: Olmo (2/3 matches) → Deepseek, Llama, Qwen (1/3 each, 2 sig diffs) → Gemma, Hermes (0/3, 1 sig diff) → Granite, Mistral, Phi (0/3, 0 sig diffs — context-insensitive).

**Data source:** Raw data in `<run>/analysis/run_summary_by_run_all_scenarios.csv` (10,000 rows per scenario per model). The existing `cross_model_pairwise_tests.csv` only contains adjacent-rank comparisons; direct pairwise tests between all scenario pairs must be generated.

**Methodology**: Independent samples t-test (Welch's) comparing scenario distributions, Holm-corrected for multiple comparisons. Requires both p < 0.01 AND |Cohen's d| ≥ 0.2 (small effect) to be considered significant.

**Scripts and outputs:** `analysis_tools/ccs2026_presentation/`
- `compute_scenario_orderings.py` — core script to compute pairwise tests and generate orderings
- `generate_and_deploy_tables.py` — wrapper that runs analysis and deploys to pres folder
- `pairwise_tests_all.csv` — full pairwise test results for all models
- `ordering_3way.csv` — 3-way orderings (Income, Political, Racial) per model
- `ordering_6way.csv` — 6-way orderings (all scenarios) per model
- `key_results_table.tex` — LaTeX code ready for presentation

**Usage:**
```bash
# Generate tables and interactively ask to deploy (on ECON-0FM96LD-L)
python analysis_tools/ccs2026_presentation/generate_and_deploy_tables.py

# Generate and auto-deploy without asking
python analysis_tools/ccs2026_presentation/generate_and_deploy_tables.py --yes

# Generate only, no deployment
python analysis_tools/ccs2026_presentation/generate_and_deploy_tables.py --no
```

**In the presentation:** Use `\input{key_results_table.tex}` to include the table.

#### Task 2: Update Dissimilarity Index grid slide

The presentation slide "Dissimilarity Index: Sakoda-Schelling Model" describes the grid layout used for computing DI. The current slide shows a 10×10 grid, but the new experiments use a 20×20 grid.

**Comparison of current slide vs actual experiments:**

| Property | Current Slide (10×10) | Actual Experiments (20×20) |
|----------|----------------------|---------------------------|
| Grid size | 10×10 = 100 cells | 20×20 = 400 cells |
| Type A agents | 40 | 160 |
| Type B agents | 40 | 160 |
| Empty cells | 20 | 80 |
| Tract division | 3-4-3 | 6-8-6 |

**Census tract sizes for 20×20 grid (9 tracts total):**
- 4 corner tracts: 6×6 = 36 cells each
- 4 edge tracts: 6×8 = 48 cells each
- 1 center tract: 8×8 = 64 cells

The tract logic (`DissimilarityIndex.py`) uses a 3×3 block partition where edge bands are `size // 3` wide (6 for 20×20) and the center takes the remainder (8).

**Task:**
1. Update the slide diagram to show 20×20 grid with 6-8-6 tract divisions
2. Update agent counts: 160 + 160 agents, 80 empty cells
3. Ensure the DI formula explanation still matches

**Files:**
- `pres-overleaf.link/pres/sakoda-schelling-llm-social-context.tex` — the slide to update
- `DissimilarityIndex.py` — tract definition reference

#### Task 3: Add value function explanation slide(s) ✓ DONE

Created slides explaining value functions with visual examples:

1. **"Value Functions: From LLM to Simulation"** — conceptual explanation
2. **"Value Functions: Olmo 2 32B"** — context-sensitive example (6 scenarios)
3. **"Value Functions: Phi 4 14B"** — context-insensitive example (6 scenarios)
4. **"Value Function Comparison"** — side-by-side Olmo vs Phi

**Value function plot script:** `analysis_tools/ccs2026_presentation/generate_value_function_plots.py`

Generates 2D value function plots showing P(MOVE) vs fraction of similar neighbors:
- Per-model plots: 2×3 grid of all 6 scenarios
- Comparison plot: Olmo vs Phi side-by-side

```bash
python analysis_tools/ccs2026_presentation/generate_value_function_plots.py --yes
```

Output: `pics/<model>_value_functions.png`, `pics/value_function_comparison.png`

**Data source:** `<run>/value_functions/vf_*__<scenario>__R3_dual_count.json`

#### Task 4: Create individual model result slides

The presentation has 6 old model result slides that must be **removed and replaced** with new slides for the 9 current models.

**Old slides to remove (lines 431-615 approximately):**
- Mistral (Mixtral-8x22b-instruct) — 100 runs, 7 scenarios
- Gemma (Gemma 3-27B) — 57 runs, 7 scenarios
- Qwen (Qwen 2.5-coder) — 100 runs, 7 scenarios
- Phi (Phi 4) — 22 runs, 7 scenarios
- Hermes (Hermes 3) — 22 runs, 7 scenarios
- Llama (Llama 3.3) — 81 runs, 7 scenarios
- Granite (commented out)

These old slides reference different model versions, fewer runs, and outdated images. They need to be completely replaced with new slides for the 9 models with 10,000 runs each.

**Current slide structure (to replicate for each model):**
```latex
%--- Model Results ---%
\begin{frame}[t]{Model Name \hfill \small 10,000 runs, 6 scenarios}
\begin{columns}[T]
    \begin{column}{0.5\textwidth}
        \centering
        \textbf{Segregation by Context}
        \includegraphics[width=\textwidth]{pics/model_segregation_metrics_comparison_dissimilarity_index.png}
    \end{column}
    \begin{column}{0.5\textwidth}
        \centering
        \textbf{Convergence Patterns}
        \includegraphics[width=\textwidth]{pics/model_convergence_patterns_dissimilarity_index.png}
    \end{column}
\end{columns}
\medskip
\textbf{Interpretation:} [Key findings for this model]
\end{frame}
```

**Models to create slides for (9 total, in goodness order):**
1. Olmo 2 32B (2/3 matches)
2. Deepseek V4 Flash (1/3 matches)
3. Granite 4.2 30B (1/3 matches)
4. Llama 3.3 70B (1/3 matches)
5. Mistral Small 4 119B (1/3 matches)
6. Phi 4 14B (1/3 matches)
7. Qwen3.6 27B (1/3 matches)
8. Gemma 4 31B (0/3 matches)
9. Hermes 4.3 36B (0/3 matches)

**Image deployment script:** `analysis_tools/ccs2026_presentation/deploy_model_images.py`

Copies images from each model's `<run>/plots/` folder to `pres-overleaf.link/pres/pics/`:

| Source (in `<run>/plots/`) | Destination (in `pics/`) |
|---------------------------|-------------------------|
| `segregation_metrics_comparison_dissimilarity_index.png` | `<model>_segregation_by_context.png` |
| `convergence_patterns_dissimilarity_index.png` | `<model>_convergence_patterns.png` |

Where `<model>` is: `olmo`, `deepseek`, `granite`, `llama`, `mistral`, `phi`, `qwen`, `gemma`, `hermes`

**Usage:**
```bash
# Deploy images interactively (on ECON-0FM96LD-L)
python analysis_tools/ccs2026_presentation/deploy_model_images.py

# Auto-deploy without asking
python analysis_tools/ccs2026_presentation/deploy_model_images.py --yes

# List what would be copied, no action
python analysis_tools/ccs2026_presentation/deploy_model_images.py --no
```

**Task steps:**
1. Run `deploy_model_images.py --yes` to copy images to pics/
2. Remove the 6 old model slides from the presentation
3. Create 9 new slides using the template below
4. Fill in model-specific interpretations

**Sorted segregation plots:** `analysis_tools/ccs2026_presentation/generate_sorted_segregation_plots.py`

Generates alternative segregation plots where scenarios are **sorted by mean DI** (lowest to highest) for each model, helping readers understand the ordering visually. Uses shorter labels with **bold** for the 3 empirical comparison scenarios:
- Color(GvY), Color(RvB), **Economic**, Ethnic, **Racial**, **Political**

```bash
python analysis_tools/ccs2026_presentation/generate_sorted_segregation_plots.py --yes
```

Output: `pics/<model>_segregation_sorted.png`

**Clean convergence plots:** `analysis_tools/ccs2026_presentation/generate_convergence_plots.py`

Generates simplified convergence plots with:
- Shorter labels (same as above, with bold)
- No "active runs" strip (cleaner for slides)
- Highest DI scenarios drawn last (visible on top when lines overlap)

**Requires:** `<run>/analysis/dissimilarity_index/dissimilarity_by_step_all.csv.gz` (not in git due to size)

```bash
python analysis_tools/ccs2026_presentation/generate_convergence_plots.py --yes
```

Output: `pics/<model>_convergence_clean.png`

**Cross-model comparison images (already generated):**
```
experiments_with_llama_cpp/cross_model/cross_model_bump_*.png
experiments_with_llama_cpp/cross_model/cross_model_level_*.png
```
These show all models together and may be useful for a summary slide.

#### Task 5: Create LLM overview slide ✓ DONE

Created slide "Large Language Models Tested" with a table listing all 9 models:
- Model name, parameter count, brief description
- Placed in Results section, before Key Results table

## 🔬 The Bias Paradox Revealed

Our research uncovers a fundamental paradox in AI systems:

### The Paradox
- **No Explicit Bias Programming**: LLMs have no coded rules for discrimination
- **Yet Reproduce Human Prejudices**: They segregate based on race, politics, and ethnicity
- **Context-Dependent Biases**: Same LLM shows 12.3× different segregation levels based on framing
- **Emergent from Training Data**: Biases absorbed from human text, not programmed rules

### Why This Matters
1. **For AI Safety**: LLMs perpetuate hidden biases that vary by context
2. **For Social Science**: These biases make LLMs superior for modeling realistic human behavior  
3. **For Policy**: Understanding bias patterns enables better intervention design
4. **For Society**: Reveals how AI systems can amplify societal prejudices

### 🕒 NEW: Temporal Dynamics of Bias
Our analysis reveals that biases don't just vary in magnitude but in how they evolve:
- **Political contexts crystallize rapidly** (within 20 steps) - reflecting polarization dynamics
- **Economic contexts never stabilize** - showing perpetual residential fluidity
- **Racial/ethnic patterns develop slowly** (50-80 steps) - mirroring historical segregation

### 🏆 Key Research Findings

#### Agent Architecture Comparison (Baseline Red/Blue)
- **⚡ LLM agents converge 2.2× faster** than mechanical agents (84 vs 187 steps)
- **🏘️ Memory reduces extreme segregation** by 53.8% ("ghetto" formation)
- **📊 Similar final segregation levels** (~55% vs 58%) but different dynamics
- **🎯 100% convergence rate** for LLM agents vs 50% for mechanical

#### The Bias Paradox in Action
- **🔴 Political contexts show EXTREME segregation**: Ghetto rate 61.6 with rapid lock-in (1.95× early volatility)
- **💰 Economic contexts show MINIMAL segregation**: Ghetto rate 5.0 but never stabilizes (continuous churn)
- **🏘️ Racial/Ethnic contexts mirror real-world patterns**: ~40 ghetto rate with gradual historical development
- **🎭 Same LLM, different biases**: 12.3× difference based solely on social framing
- **🚨 No explicit bias rules**: These patterns emerge from implicit associations in training data

#### NEW: Temporal Dynamics Discovered
- **⚡ Political**: Rapid crystallization in first 20 steps (1.95× early volatility) - reflects polarization dynamics
- **🔄 Economic**: Perpetual fluidity (0.91× early/late volatility) - ongoing mobility
- **📈 Racial**: Slow burn over 50-80 steps (1.47× early volatility) - matches historical segregation patterns
- **🎯 Intervention Windows**: Different contexts require different timing strategies
- **📊 Stability Rankings**: Ethnic (most stable) > Baseline > Political > Race > Income (least stable)

## 📄 Scientific Papers

### Paper 1: Agent Architecture Comparison
**"Human-like Decision Making in Agent-Based Models: A Comparative Study of Large Language Model Agents versus Traditional Utility Maximization in the Schelling Segregation Model"**

- **Focus**: Comparing mechanical vs standard LLM vs memory-enhanced LLM agents
- **Key Finding**: LLM agents converge 2.2× faster with memory reducing extreme segregation
- **Status**: Original version prepared for submission
- **File**: [`schelling_llm_paper.qmd`](schelling_llm_paper.qmd)

### Paper 2: Social Context Effects (NEW)
**"Social Context Matters: How Large Language Model Agents Reproduce Real-World Segregation Patterns in the Schelling Model"**

- **Focus**: How different social framings (political, racial, economic) affect segregation
- **Key Finding**: Political contexts produce 12.3× more segregation than economic contexts
- **NEW Finding**: Temporal dynamics reveal context-specific evolution patterns
- **Status**: Analysis complete with dynamics, paper draft available
- **File**: [`schelling_llm_paper_updated.qmd`](schelling_llm_paper_updated.qmd)

### Paper 3: The Bias Paradox Study (FEATURED)
**"The Bias Paradox: How Large Language Models Reveal Human Prejudices While Advancing Agent-Based Social Science"**

- **Focus**: How LLMs reproduce human biases without explicit programming
- **Key Finding**: LLMs' greatest flaw (absorbing biases) is also their greatest strength for social science
- **NEW Analysis**: Temporal dynamics show biases evolve differently - political lock-in vs economic fluidity
- **Implications**: Both a warning for AI deployment and opportunity for research
- **Status**: Comprehensive analysis with AI ethics focus and policy recommendations
- **File**: [`schelling_llm_paper_comprehensive.qmd`](schelling_llm_paper_comprehensive.qmd)

- **Authors**: Andreas Pape, Carl Lipo, et al.
- **Institution**: Binghamton University
- **Render Instructions**: See [`paper_README.md`](paper_README.md)

## ✨ New Features (2024)

### 🔄 Easy LLM Model Switching
- **8 predefined LLM configurations** (Mixtral, Qwen, GPT-4, Claude, etc.)
- **Simple preset commands**: `--preset mixtral`, `--preset qwen`, `--preset gpt4`
- **Automatic validation** of API keys and configurations
- **Consistent arguments** across all scripts

### ⏱️ Real-Time Progress Monitoring
- **Live progress dashboard** showing "Run X of Y, Step Z of 1000"
- **Auto-refreshing monitoring** every 10 seconds
- **Progress files** updated every 10 simulation steps
- **No more guessing** if experiments are stuck or progressing

### 📊 Enhanced Dashboard System
- **Auto-detection** of active experiments
- **Quick launch** for progress monitoring
- **Multiple dashboard options** (analysis vs monitoring)
- **Smart experiment selection** based on current activity

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/clipo/PancsVriend.git
cd PancsVriend
pip install -r requirements.txt
```

### Easy LLM Model Switching
```bash
# List available LLM models
python switch_llm.py --list

# Test connectivity with different models
python switch_llm.py --preset mixtral --test
python switch_llm.py --preset qwen --test
```

### Basic Usage
```bash
# Test LLM connectivity (uses Mixtral by default)
python check_llm.py

# Run comprehensive study with real-time progress monitoring
python comprehensive_comparison_study.py --quick-test

# Launch interactive dashboard with progress monitoring
python launch_dashboard_menu.py

# Multi-scenario runner
python run_all_contexts.py --runs 10 --processes 5
```

## 📋 Available LLM Models

| Preset | Model | Provider | Status |
|--------|-------|----------|---------|
| `mixtral` | Mixtral 8x22B | Binghamton Uni | ✅ **Default** - Ready to use |
| `qwen` | Qwen 2.5 Coder 32B | Binghamton Uni | ✅ Ready to use |
| `gpt4` | GPT-4 | OpenAI | Requires API key |
| `gpt4o` | GPT-4o | OpenAI | Requires API key |
| `claude-sonnet` | Claude 3 Sonnet | Anthropic | Requires API key |
| `local-llama` | Llama2 | Local Ollama | Requires local setup |

## 🎯 Usage Examples

### Quick Testing
```bash
# Quick test with default model (Mixtral)
python comprehensive_comparison_study.py --quick-test

# Test different models
python switch_llm.py --preset mixtral --test
```

### Production Experiments
```bash
# Full comprehensive study
python comprehensive_comparison_study.py --preset mixtral

# Specific scenario testing
python switch_llm.py --preset qwen --llm race_white_black 30

# Custom configuration
python run_all_contexts.py --llm-model "gpt-4" --llm-url "https://api.openai.com/v1/chat/completions" --llm-api-key "your-key"
```

### Real-Time Monitoring
```bash
# Launch progress dashboard
python launch_dashboard_menu.py

# Direct progress monitoring
streamlit run dashboard_with_progress.py
```

## 🖥️ Running with a Local GGUF Model (llama.cpp)

Run the full simulation pipeline against a locally-hosted quantized model — no external API required.

### Overview

You serve a GGUF file via `llama-cpp-python`'s built-in OpenAI-compatible HTTP server, then point the simulation at `localhost:8080`. The pipeline skips the token-probability stage and runs simulation + analysis directly. Two scale profiles are available:

| Profile | Runs | Steps | Scenarios |
|---------|------|-------|-----------|
| `smoke_test` | 5 | 200 | baseline only |
| `production` | 100 | 1000 | all scenarios |

Always run `smoke_test` first to confirm end-to-end connectivity before committing to the long production run.

### Step 1 — Install llama-cpp-python server

```bash
pip install "llama-cpp-python[server]"
```

For GPU acceleration, install the CUDA build instead — see the
[llama-cpp-python installation docs](https://github.com/abetlen/llama-cpp-python#installation).
CPU-only works (slower) and is fine for `smoke_test`.

### Step 2 — Configure and start the server

Edit **one line** in `configs/llama_cpp_server.yaml` — set `model:` to the absolute path of your GGUF file:

```yaml
models:
  - model: "/absolute/path/to/your-model.gguf"
```

Then start the server (leave it running in a separate terminal or under `screen`/`tmux`):

```bash
python -m llama_cpp.server --config_file configs/llama_cpp_server.yaml
```

Ready when the terminal prints `Uvicorn running on http://0.0.0.0:8080`.

> **GPU OOM?** Lower `n_gpu_layers` in `configs/llama_cpp_server.yaml` from `-1` to a positive number (e.g. `28`) until the model loads.

### Step 3 — Label the run

In `configs/llama_cpp_simulation_run.yaml`, set `llm_model:` to a short label for the GGUF you loaded (e.g. `gemma-3-4b-it-q4`). This names the output folders — nothing else needs editing.

### Step 4 — Run the pipeline

```bash
# Validate setup first (fast — ~5 runs × 200 steps)
python run_llm_probability_simulation_analysis.py \
  --config-yaml configs/llama_cpp_simulation_run.yaml \
  --config-profile smoke_test

# Full production run (slow — run under screen/tmux)
screen -S schelling
python run_llm_probability_simulation_analysis.py \
  --config-yaml configs/llama_cpp_simulation_run.yaml \
  --config-profile production
```

### Output

Results land in a timestamped directory under `experiments_with_llama_cpp/`. Each folder is one pipeline run for one model, named `run_<YYYYMMDD_HHMMSS>_<model>-vf-<family>`:

| Suffix | What it is | Status |
| --- | --- | --- |
| `-vf-lp` | 10,000 runs per scenario from the model's **exact** value function (token log-probabilities). | **The result.** Nine models, all rank-stability `SETTLED`. |
| `-vf-s` | 100 runs per scenario from the sequential **sanity** resample (n=100 draws per cell). | A check on the `-vf-lp` tables, not a result. |
| `cross_model/` | Bump and level charts and paired tests across every model's newest `-vf-lp` run. | Regenerated automatically by the pipeline's last stage. |

#### The key data file

**`<run>/analysis/run_summary_by_run_all_scenarios.csv`** — all six scenarios in one table (60,000 rows for `-vf-lp`), with a `scenario_key` column. This is what ANOVA, normality, metrics-comparison, and cross-model steps read.

| Column | Meaning |
| --- | --- |
| `run_id` | Seed of the run; per-move detail is regenerable from it under the `rng_scheme` in `config.json`. |
| `scenario` | `baseline`, `race_white_black`, `ethnic_asian_hispanic`, `income_high_low`, `political_liberal_conservative`, `green_yellow`. |
| `converged`, `convergence_step` | Whether the run hit 5 consecutive zero-move steps, and the first step of that streak. |
| `dissimilarity_index`, `clusters`, `switch_rate`, `distance`, `mix_deviation`, `share`, `ghetto_rate` | The seven metrics on the final grid. DI is the headline metric. |
| `initial_<metric>` | The same seven on frame 0 (random allocation): the run's own paired draw from the chance distribution. |
| `stop_reason` | `converged` / `max_steps` / `incomplete`. |

#### One run folder structure

```
run_<ts>_<model>-vf-lp/
├── run_config_effective.yaml     # yaml with the chosen profile resolved
├── value_functions/              # decision tables the run simulated FROM (frozen)
│   └── vf_<label>-lp__<scenario>__R3_dual_count.json
├── experiments/                  # one folder per scenario
│   └── llm_<scenario>_<ts>/
│       ├── config.json
│       ├── run_summary.csv       # <- per-scenario data file
│       ├── metrics_history.csv.gz
│       ├── states/               # states_packed.npz (not in git)
│       └── move_logs/            # step_moves_packed.csv.gz (not in git)
├── analysis/
│   ├── run_summary_by_run_all_scenarios.csv  # <- ALL scenarios combined
│   ├── anova_results_by_metric.{csv,md}
│   ├── segregation_scenario_rankings_<model>.{csv,md}
│   └── rank_stability/           # is the DI ordering settled?
└── plots/
    ├── segregation_metrics_comparison_dissimilarity_index.png
    ├── convergence_patterns_dissimilarity_index.png
    └── metric_panels/
```

#### cross_model/ folder

| File | Content |
| --- | --- |
| `cross_model_bump_<metric>.png` | Scenario ranking per model, gap bars binned on Cohen's d. |
| `cross_model_level_<metric>.png` | Scenario means per model with the chance level. |
| `cross_model_pairwise_tests.csv` | Paired t-tests between adjacent scenarios, per model and metric. |
| `cross_model_rankings.csv` | The ranking table behind the bump charts. |

Monitor a running job:

```bash
wc -l experiments_with_llama_cpp/run_*/experiments/*/metrics_history.csv
wc -l experiments_with_llama_cpp/run_*/experiments/*/run_summary.csv
nvidia-smi -l 5    # GPU utilization
```

### Throughput note

The llama.cpp server is single-stream (one request at a time). `processes` is pinned to `1` in both profiles — raising it only queues requests. For faster production runs, start several server instances on different ports behind a round-robin proxy, point `llm_url` at the proxy, and raise `processes` accordingly.

### Full guide

See [`LLAMA_CPP_SIMULATION_RUN_GUIDE.md`](LLAMA_CPP_SIMULATION_RUN_GUIDE.md) for troubleshooting and additional notes.

---

## 📊 Key Features

### Agent Types
- **Mechanical Agents**: Traditional Schelling model with utility functions
- **Standard LLM Agents**: Context-aware decisions using current neighborhood state
- **Memory LLM Agents**: Human-like agents with persistent memory and history

### Social Context Scenarios
- **baseline**: Red vs blue teams (control group)
- **race_white_black**: White middle class vs Black families
- **ethnic_asian_hispanic**: Asian American vs Hispanic/Latino families
- **economic_high_working**: High-income vs working-class households
- **political_liberal_conservative**: Liberal vs conservative households

### Segregation Metrics
- **Clustering Index**: Spatial concentration measurement
- **Switch Rate**: Frequency of agent relocations
- **Distance Metrics**: Average separation between groups
- **Mix Deviation**: Departure from random mixing
- **Share Metrics**: Proportional representation analysis
- **Ghetto Rate**: Extreme segregation detection

## 🛠 Core Components

### Experiment Scripts
- `comprehensive_comparison_study.py` - **Main entry point** for three-way comparisons
- `run_all_contexts.py` - Multi-scenario runner (resume, manifests, run-summary roll-up)
- `baseline_runner.py` - Runs mechanical agent simulations
- `llm_runner.py` - Runs LLM agent simulations with social contexts

### LLM Management
- `switch_llm.py` - Easy model switching and testing
- `llm_presets.py` - Predefined LLM configurations
- `update_default_llm.py` - Change default model in config.py
- `check_llm.py` - Connectivity testing

### Real-Time Monitoring
- `dashboard_with_progress.py` - Live progress monitoring dashboard
- `launch_dashboard_menu.py` - Smart dashboard launcher
- `dashboard.py` - Comprehensive analysis dashboard

### Analysis Tools
- `statistical_analysis.py` - ANOVA, effect sizes, multivariate analysis
- `plateau_detection.py` - Convergence detection and segregation speed calculation

## 🔧 Configuration

### Default Configuration
The system uses **Mixtral 8x22B** by default via Binghamton University's endpoint:
```python
# config.py
OLLAMA_MODEL = "mixtral:8x22b-instruct"
OLLAMA_URL = "https://chat.binghamton.edu/api/chat/completions"  
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")  # set in .env (copy .env.example)
```

### Easy Model Switching
```bash
# Change default model
python update_default_llm.py --set qwen
python update_default_llm.py --set mixtral

# Show current default
python update_default_llm.py --show
```

### Command Line Overrides
All scripts support consistent LLM configuration:
```bash
# Use presets (recommended)
--preset mixtral
--preset qwen
--preset gpt4

# Custom configuration
--llm-model "model-name"
--llm-url "https://api.provider.com/v1/chat/completions"
--llm-api-key "your-api-key"
```

## 📊 Analysis & Visualization Tools

### Statistical Analysis
```bash
# Generate comprehensive statistical report
python statistical_analysis.py

# Detailed pairwise comparisons
python pairwise_comparison_analysis.py

# Convergence analysis with speed metrics
python convergence_analysis.py
```

### Comprehensive Visualization
```bash
# Generate publication-quality PDF report with all analyses
python comprehensive_visualization_report.py

# Creates: comprehensive_comparison_report.pdf with:
# - Executive summary
# - Convergence analysis 
# - Time series evolution
# - Final state comparisons
# - Statistical tables
# - Pairwise comparisons
```

### Academic Paper
```bash
# Render the scientific paper (requires Quarto + R)
quarto render schelling_llm_paper.qmd --to pdf

# See paper_README.md for setup instructions
```

## 📈 Experiment Workflows

### Comprehensive Comparison
```bash
# Quick validation
python comprehensive_comparison_study.py --quick-test

# Full three-way study (mechanical vs standard LLM vs memory LLM)
python comprehensive_comparison_study.py --preset mixtral

# Monitor progress in real-time
python launch_dashboard_menu.py  # Choose progress dashboard
```

### Social Context Studies
```bash
# Compare specific scenarios
python switch_llm.py --preset mixtral --llm race_white_black 30
python switch_llm.py --preset qwen --llm economic_high_working 30

# Multi-scenario comparison
python run_all_contexts.py --scenarios baseline race_white_black economic_high_working
```

### Model Comparisons
```bash
# Test different models on same scenario
python switch_llm.py --preset mixtral --llm race_white_black 30
python switch_llm.py --preset qwen --llm race_white_black 30
python switch_llm.py --preset gpt4 --llm race_white_black 30  # (requires OpenAI key)
```

## 📊 Output Structure

Experiments generate structured outputs with real-time progress files:
```
experiments/
├── baseline_[timestamp]/              # Mechanical agent results
├── llm_[scenario]_[timestamp]/        # LLM scenario results
└── comprehensive_study_[timestamp]/   # Multi-agent comparisons
    ├── llm_results/
    │   ├── progress_realtime.json     # ← Real-time progress monitoring
    │   ├── experiments/exp_0001/
    │   └── logs/
    ├── comprehensive_analysis/
    └── three_way_comparison_report.md

reports/
├── comprehensive_report_[timestamp].pdf
├── statistical_analysis_[timestamp].txt
└── experiment_summary_[timestamp].json
```

## 🔍 Progress Monitoring

### Real-Time Dashboard Features
- **Live progress counters**: "Run 15 of 30, Step 450 of 1000"
- **Progress bars** for both run and step completion
- **Status indicators** (running/completed/failed)
- **Auto-refresh** every 10 seconds
- **Multi-experiment monitoring** for parallel runs

### Dashboard Access
```bash
# Smart launcher (detects active experiments)
python launch_dashboard_menu.py

# Direct progress dashboard
streamlit run dashboard_with_progress.py

# Analysis dashboard
streamlit run dashboard.py
```

## 🚨 AI Ethics and Bias Detection

### The Dual Nature of LLM Biases

Our research reveals that LLMs are not neutral tools but carriers of human cultural biases:

#### As a Problem
- **Hidden Biases**: LLMs perpetuate prejudices they were never explicitly taught
- **Context-Dependent**: Same model shows different biases based on framing
- **Unpredictable**: Biases may emerge in unexpected ways in applications
- **Amplification Risk**: Could reinforce societal prejudices at scale

#### As an Opportunity  
- **Bias Detection**: Use our framework to measure implicit prejudices in AI systems
- **Social Mirror**: LLMs reveal hidden biases in society through their outputs
- **Research Tool**: Study prejudice without survey response bias
- **Policy Testing**: Pre-test interventions across different bias scenarios

### Recommendations for AI Deployment

1. **Context-Aware Testing**: Test LLMs across multiple social framings before deployment
2. **Bias Monitoring**: Implement continuous monitoring for emergent biases
3. **Ensemble Approaches**: Use multiple LLMs to identify consistent vs variable biases
4. **Transparent Reporting**: Document bias profiles for different contexts

## 🎯 Research Applications

### Urban Planning
- Study segregation patterns under different policy scenarios
- Test integration policies virtually before implementation
- Understand how biases affect housing decisions

### Social Science  
- Use LLMs as mirrors to reveal societal biases
- Model realistic human behavior without explicit bias programming
- Study emergence of prejudice and discrimination

### AI Safety
- Develop comprehensive bias testing frameworks
- Create context-sensitive bias detection tools
- Design bias-aware AI governance structures

## 📈 Recent Analysis Tools Added

### Temporal Dynamics Analysis
- **analyze_rate_of_change.py** - Calculates how quickly segregation patterns evolve
- **analyze_stability_patterns.py** - Measures consistency and volatility across contexts  
- **analyze_convergence_patterns.py** - Identifies when patterns stabilize (or don't)
- **dynamics_analysis_summary.md** - Key findings from temporal analysis

### Key Dynamics Findings
1. **Political contexts show rapid crystallization** - 1.95× more volatile early then lock-in
2. **Economic contexts never stabilize** - Equal volatility throughout (0.91× ratio)
3. **Racial/ethnic show historical patterns** - Gradual development over 50-80 steps
4. **Different intervention windows** - Political needs immediate action, economic needs continuous management

## 🔍 Key Findings: The Bias Paradox Demonstrated

### Without Any Explicit Programming for Discrimination:

#### LLMs Reproduce Real-World Biases
- **Political**: 61.6 ghetto formation rate (extreme segregation)
- **Racial**: ~40 ghetto formation rate (matches empirical data)
- **Economic**: 5.0 ghetto formation rate (minimal segregation)
- **12.3× Variation**: Based solely on how groups are framed

### Emergent vs Programmed Behavior
Our research reveals the fundamental difference:

- **Mechanical agents**: Follow explicit rules (threshold = 0.5)
- **LLM agents**: No bias rules, yet produce human-like prejudices
- **Emergence**: Biases arise from training data, not programming
- **Variability**: Different contexts activate different implicit biases

### Why This Makes LLMs Superior for Social Science

#### Political Segregation (Most Extreme)
- **Ghetto formation rate**: 61.6 ± 9.3 (highest across all contexts)
- **Segregation share**: 0.928 ± 0.042 (near-complete segregation)
- **Switch rate**: 0.076 ± 0.036 (agents rarely move once settled)
- **Interpretation**: Reflects contemporary political polarization

#### Economic Integration (Most Integrated)
- **Ghetto formation rate**: 5.0 ± 3.1 (12.3× lower than political)
- **Number of clusters**: 25.0 ± 3.1 (most fragmented/mixed)
- **Convergence speed**: ~7 steps (10× faster than other contexts)
- **Interpretation**: Economic diversity more tolerable than other differences

#### Racial/Ethnic Patterns (Real-World Alignment)
- **Race (White/Black)**: Ghetto rate 40.8 ± 9.6, share 0.823 ± 0.060
- **Ethnic (Asian/Hispanic)**: Ghetto rate 38.9 ± 11.2, share 0.821 ± 0.076
- **Interpretation**: Matches empirical segregation indices from urban studies

#### Statistical Significance
- **Methodology**: Independent samples t-test (Welch's), Holm-corrected, requiring both p < 0.01 AND |Cohen's d| ≥ 0.2
- **3-way comparison** (Economic, Political, Racial): 6/9 models show at least one significant difference; 3 models (Granite, Mistral, Phi) are context-insensitive with no significant differences
- **Context-sensitive models**: Olmo (2/3 empirical matches), Deepseek/Llama/Qwen (1/3 each), Gemma/Hermes (0/3 but 1 sig diff)
- Full pairwise test results: `analysis_tools/ccs2026_presentation/pairwise_tests_all.csv`

## 📚 Documentation

- **[LLM Switching Guide](README_LLM_SWITCHING.md)** - Complete guide to model switching
- **[CLAUDE.md](CLAUDE.md)** - Project instructions and command reference
- **Code Documentation** - Inline documentation throughout codebase

## 🔧 Advanced Features

### Supported LLM Providers
- **Local models**: Ollama, LM Studio, vLLM
- **OpenAI**: GPT-4, GPT-4o, GPT-3.5-turbo
- **Anthropic**: Claude-3-sonnet, Claude-3-haiku (via proxy)
- **Cloud providers**: Azure OpenAI, AWS Bedrock (via proxy)

### Performance Optimization
- **Parallel processing** with circuit breakers and automatic fallback
- **Error handling** with graceful degradation when LLM services fail
- **Scalability** for 100+ simulation runs with configurable parameters
- **Cost estimation** and resource planning tools

## 🤝 Contributing

We welcome contributions! Please:
- Report bugs or suggest features via GitHub issues
- Submit pull requests for improvements
- Share your research findings using this framework
- Test with different LLM models and report results

## 📁 Repository Structure

### Core Simulation Files
- `Agent.py` - Traditional utility-maximizing agents
- `LLMAgent.py` - LLM-powered agents with human-like decisions
- `Metrics.py` - Six segregation metrics (clusters, distance, share, etc.)
- `SchellingSim.py` - Interactive GUI simulation
- `config.py` - Central configuration for all parameters

### Experiment Runners
- `baseline_runner.py` - Mechanical agent experiments
- `llm_runner.py` - LLM agent experiments with social scenarios
- `comprehensive_comparison_study.py` - Full 3-way comparison
- `run_pure_comparison.py` - Pure agent type comparison
- `experiment_explorer.py` - Design space exploration

### Analysis & Visualization
- `statistical_analysis.py` - ANOVA, effect sizes, statistical tests
- `pairwise_comparison_analysis.py` - Detailed pairwise comparisons
- `convergence_analysis.py` - Convergence speed and rate analysis
- `comprehensive_visualization_report.py` - Complete PDF report generator
- **NEW**: `analyze_experiment_results.py` - Extract and compare final metrics across scenarios
- **NEW**: `analyze_convergence_patterns.py` - Time series analysis of segregation evolution
- **NEW**: `visualize_experiment_comparison.py` - Generate comparison plots and heatmaps
- **NEW**: `analyze_rate_of_change.py` - Temporal dynamics and phase transition analysis
- **NEW**: `analyze_stability_patterns.py` - Stability and volatility measurements

### LLM Configuration
- `switch_llm.py` - Interactive model switching
- `update_default_llm.py` - Update default LLM configuration
- `llm_presets.py` - Predefined LLM configurations
- `check_llm.py` - Connectivity and performance testing

### Dashboard & Monitoring
- `launch_dashboard_menu.py` - Dashboard launcher with options
- `dashboard_with_progress.py` - Real-time progress monitoring
- `cleanup_experiments.py` - Experiment cleanup utility

### Scientific Paper
- `schelling_llm_paper.qmd` - Quarto scientific paper
- `references.bib` - Bibliography
- `paper_README.md` - Paper rendering instructions
- `journal_style_guide.md` - Journal submission guide
- `verify_paper_data.R` - Data verification script

## 📄 Citation

If you use this framework in your research, please cite:
```bibtex
@software{pancs_vriend_llm,
  title={PancsVriend: LLM-Enhanced Schelling Segregation Model},
  author={[Research Team]},
  year={2024},
  url={https://github.com/clipo/PancsVriend},
  note={Research framework comparing mechanical and LLM agents in segregation dynamics}
}
```

## 📧 Contact

For questions about the research or technical issues:
- Create an issue on GitHub
- Contact the research team via [institutional contact]

## 🔮 The Bigger Picture: What This Means

### For AI Development
Our findings challenge the notion of "neutral" AI. LLMs trained on human text inevitably absorb human biases, creating systems that mirror our prejudices in complex, context-dependent ways. This isn't a bug to be fixed but a fundamental characteristic that must be understood and managed.

### For Social Science
The bias paradox offers unprecedented opportunities. Rather than programming our assumptions about human behavior, we can use LLMs to discover emergent patterns we might not have thought to look for. They serve as computational mirrors of society.

### For Society
As LLMs become integrated into decision-making systems (housing, hiring, lending), understanding their implicit biases becomes crucial. Our framework provides a method to detect and measure these biases before they cause harm.

### The Paradox Embraced
We propose embracing rather than eliminating the bias paradox:
- Use it as a tool to understand ourselves
- Leverage it for more realistic social modeling
- Monitor it carefully in applications
- Learn from it to build better societies

## 📜 License

[License information to be added]

---

*"The question is not whether LLMs have biases, but how we can use this mirror wisely." - This framework reveals the bias paradox at the heart of modern AI, providing tools to understand, measure, and leverage it for both research and responsible AI deployment.*