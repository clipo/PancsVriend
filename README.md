# Schelling Segregation Simulator (Enhanced Edition)

This project implements an extended version of the Schelling Segregation Model based on Pancs & Vriend (2007). The simulation demonstrates how individual preferences for neighborhood composition—even preferences *for integration*—can unintentionally lead to segregation.

## 🆕 Comprehensive Experiment Framework

This enhanced version includes a complete experimental framework for comparing mechanical agents with LLM-based agents under various social contexts, with sophisticated statistical analysis and visualization capabilities.

## 🔍 What It Does

This Python-based simulation models a 2D grid-based society of agents from two groups (e.g., Group A and Group B). Each agent evaluates its neighborhood and decides whether to stay or move based on how satisfied it is with its current surroundings. Satisfaction is calculated via a utility function, and agents perform **best-response dynamics** to relocate.

### Core Features

- **Spatial Agent-Based Simulation**
- **6 Segregation Metrics Tracked:**
  - Cluster count
  - Switch rate
  - Distance to unlike neighbors
  - Mix deviation
  - Share measure
  - Ghetto rate
- **Graphical User Interface** using `pygame` and `pygame_gui`
- **CSV logging** of metrics per timestep
- **Live plotting** of metric trends (`matplotlib`)
- **LLM Decision Mode**: Switches from mechanistic decision logic to LLM-based reasoning using Ollama

---

## 💡 Expected Behavior

This simulator demonstrates that:
- Even mild or integration-favoring preferences can result in rapid segregation.
- Segregation persists despite agents preferring mixed neighborhoods.
- LLM agents may exhibit different movement patterns compared to traditional best-response agents depending on their interpretive logic.

You can switch between:
- **Mechanistic Mode** — classic Schelling-style myopic best response.
- **LLM Mode** — context-aware decision making using large language models like `qwen2.5-coder:32B`.

---

## 🧪 Comparing Mechanistic vs. LLM Agents

To evaluate how LLMs differ from traditional agents:

1. **Set `USE_LLM = False`** in `config.py` and run the simulation. Observe CSV metrics and live plots.
2. **Set `USE_LLM = True`**, then choose a model from the dropdown in the GUI.
3. Run identical scenarios (same population and initial conditions) and compare:
   - How quickly segregation occurs.
   - Whether more or fewer agents end up in ghettos.
   - The variation in number of clusters.
   - Whether LLM agents behave in a more nuanced or chaotic way.
4. Export and analyze logged data (`segregation_metrics.csv`) for both modes.

---

## ⚙️ How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Your Environment

Edit `config.py`:

```python
USE_LLM = True  # or False
OLLAMA_MODEL = "qwen2.5-coder:32B"
OLLAMA_URL = "https://chat.binghamton.edu/api/chat/completions"
OLLAMA_API_KEY = "your-api-key-here"
```

### 3. Start the Simulation

```bash
python SchellingSim.py
```

---

## 🧭 Interface Controls

- **LLM Toggle**: Switch between classic and LLM-driven agents
- **Model Dropdown**: Select an active model deployed on your local or institutional Ollama server
- **Live Graphs**: 6 panels show evolving segregation metrics
- **CSV Logging**: `segregation_metrics.csv` accumulates all time-step data

---

## 🧠 LLM Agent Behavior

The LLM agents receive a 3x3 local neighborhood context like:

```
[
 ["S", "E", "O"],
 ["X", "S", "E"],
 ["O", "S", "E"]
]
```

Where:
- `S` = Same group
- `O` = Opposite group
- `E` = Empty cell
- `X` = Out of bounds

They're prompted to return coordinates of the best location to move to. You can use this to test:
- Reasoning under partial satisfaction
- Interpretations of “integration” or “diversity”
- Exploration vs. exploitation in movement

---

## 📊 Output Files

- `segregation_metrics.csv` — logs step-by-step values for 6 metrics
- Live graphical window — shows the evolving grid and metric plots
- Console output — for error/debug logs if needed

---

## 🛠 Future Additions

- Snapshot export of grid states
- Configurable utility curves
- Batch run comparison tools
- Cluster heatmaps or agent movement traces

---


---

## ✅ Recent Fixes and Performance Improvements

The current version of the simulation includes critical updates to make LLM-based movement both correct and efficient:

### 🔁 Correct LLM Coordinate Interpretation
LLM agents return 3×3 neighborhood-relative coordinates (centered at position (1,1)). These are now correctly translated to absolute grid positions before move validation. This has fixed the prior issue where agents repeatedly attempted to move to globally invalid or occupied cells.

### 🧠 Enhanced LLM Prompts
The agent prompt enforces strict formatting, asking for a tuple `(r, c)` or `None` — and nothing else. This eliminates parsing ambiguity and ensures smooth operation.

### 📉 Faster and Parallel LLM Responses
The system uses a background worker model with `queue.Queue` and a thread pool to parallelize LLM calls. This enables smoother GUI responsiveness even when agent decisions are LLM-driven.

### 🎯 GUI and Metric Updates
- The simulation loop now applies moves **before** drawing the grid.
- Metrics and visualizations now reflect real changes, and GUI updates are tied to actual agent behavior.
- Convergence logic prevents premature termination by ensuring the agent queue is fully processed.

If your LLM agents seem too cautious or always return `None`, consider tuning the prompt to reward movement or suggest alternative goals (e.g., diversity, centrality, clustering).

---

## 📚 References

- Pancs, R., & Vriend, N. J. (2007). *Schelling's spatial proximity model of segregation revisited.* Journal of Public Economics, 91(1), 1-24.
- Schelling, T. C. (1971). *Dynamic models of segregation.* Journal of Mathematical Sociology, 1(2), 143-186.



---

## ⚙️ Configurations and Batch Runs

### 🔧 Runtime Configurations

These options allow you to control performance and output:

- **Live Graphs Toggle**: Click "Toggle Graphs" to enable or disable live metric plotting.
  - Live graphs are OFF by default (faster simulation).
  - Turning them ON shows evolving metric plots in real-time.
- **Stop & Graph Button**: Ends the simulation and generates a **PDF summary** of all metric graphs from the current run (`final_metrics_summary.pdf`).
- **LLM Toggle**: Switch between mechanistic and LLM-based agents at any time.
- **LLM Model Dropdown**: Choose the LLM model used when `USE_LLM = True`.

---

### 📦 Batch Mode Analysis

Use the included `batch_run.py` to run multiple simulations and aggregate results.

#### How It Works:

- Runs the simulation multiple times (default = 5)
- Collects metric values from `segregation_metrics.csv` after each run
- Computes **mean** and **standard deviation** per timestep
- Produces line plots with confidence intervals for each metric
- Saves results as `batch_<metric>.pdf` (e.g., `batch_clusters.pdf`)

#### To Run:

```bash
python batch_run.py
```

> Note: Currently assumes you manually close each simulation run.
> Full `--headless` mode is planned for future automation.

---



---

## 🚀 Running Comprehensive Experiments

### Quick Start - Full Experiment Suite

Run the complete experiment comparing baseline mechanical agents with LLM agents across different social contexts:

```bash
python run_experiments.py
```

This will:
1. Run 100 baseline simulations with mechanical agents
2. Run 10 simulations for each LLM context scenario
3. Analyze convergence patterns and plateau detection
4. Generate comprehensive visualizations
5. Perform statistical analysis comparing all scenarios
6. Create a PDF report with all results

### Custom Experiment Configuration

```bash
# Run with custom parameters
python run_experiments.py --baseline-runs 50 --llm-runs 20 --max-steps 500

# Run specific LLM scenarios only
python run_experiments.py --scenarios baseline race_white_black income_high_low

# Quick test mode (5 baseline, 2 LLM runs)
python run_experiments.py --quick-test
```

### Available LLM Context Scenarios

1. **baseline**: Simple red vs blue agents
2. **race_white_black**: White middle class vs predominantly black neighborhoods
3. **race_asian_hispanic**: Asian vs Hispanic neighborhoods
4. **income_high_low**: High-income vs low-income households
5. **political_liberal_conservative**: Liberal vs conservative households

### Running Individual Components

```bash
# Run baseline experiments only
python baseline_runner.py --runs 100

# Run LLM experiments with specific scenario
python llm_runner.py --scenario race_white_black --runs 10

# Analyze existing results
python visualization.py --baseline-dir experiments/baseline_[timestamp] --llm-dirs experiments/llm_*
```

---

## 📊 Analysis Capabilities

### Plateau Detection
- Automatically detects when metrics stabilize
- Calculates convergence rates and half-life
- Compares speed of segregation across scenarios

### Statistical Analysis
- Descriptive statistics for all metrics
- Normality tests (Shapiro-Wilk)
- ANOVA/Kruskal-Wallis for multi-group comparisons
- Post-hoc tests (Tukey HSD, Dunn)
- Effect size calculations (Cohen's d)
- Multivariate analysis (PCA)

### Visualization Features
- Evolution of metrics over time with confidence intervals
- Convergence time distributions
- Scenario comparison bar charts
- Comprehensive PDF reports
- Real-time plotting capabilities

---

## 🧰 Tools and Enhancements

### 🟦 Simulation Progress Bar
The simulation now includes a visual **progress bar** that fills as the simulation proceeds through its defined number of steps (`max_steps`). This is useful for monitoring run progress at a glance.

---

### 📊 Real-Time CSV Visualizer
Run this script in a separate terminal to monitor the evolving metrics:

```bash
python realtime_csv_plot.py
```

This visualizer watches the `segregation_metrics.csv` file and updates live metric plots every second.

---

### 📈 Batch Summary Dashboard

If you've run multiple simulations (e.g., `segregation_metrics_1.csv`, `segregation_metrics_2.csv`, ...), you can visualize their collective results:

```bash
python batch_summary_dashboard.py
```

This tool computes the **mean** and **standard deviation** over time across all simulations and produces line plots with shaded confidence regions for:

- Cluster count
- Switch rate
- Distance to unlike agents
- Mix deviation
- Share of same-type neighbors
- Ghetto rate

---



---

## 🖼 Interface Overview

The simulation features a GUI-based layout with the following controls:

- 🔘 **Start Button**: Begins the simulation.
- ⏹ **Stop & Graph**: Ends the simulation and generates PDF plots of segregation metrics.
- 🔄 **LLM Toggle**: Enables or disables large language model agent logic.
- 📥 **Model Dropdown**: Choose among LLM models if LLM is active.
- 📊 **Progress Bar**: Shows percentage progress through the run.

### Layout Screenshot (Placeholder)
Replace the image below with a screenshot of your simulation GUI:

![Simulation GUI Layout](images/gui_layout_placeholder.png)
*Fig. 1: Main window with grid and control panel.*

---

## 📈 Output Visualizations

Once the simulation is stopped, it creates the following:

- `segregation_metrics.csv`: Time series of segregation metrics
- `final_metrics_summary.pdf`: PDF of plots showing evolution of:
  - Clusters
  - Switch Rate
  - Distance to Unlike
  - Mix Deviation
  - Share of Same-Type Neighbors
  - Ghetto Rate

### Sample Metric Graph (Placeholder)

![Metric Evolution Plot](images/metrics_plot_placeholder.png)
*Fig. 2: Example of plotted segregation metrics over simulation time.*

---

## 🧪 Running in Batch Mode & LLM Experiments

### 🔁 Batch Runs

To compare performance across multiple runs (e.g., with and without LLMs), execute:

```bash
for i in {1..10}; do python SchellingSim.py; done
```

Each run will log its convergence step in `convergence_summary.csv`, including:

- Step (timestep when convergence occurred)
- Whether LLM was used (`USE_LLM`)
- Model name (`OLLAMA_MODEL`)
- Total number of agents

You can analyze these using:

```python
import pandas as pd
df = pd.read_csv("convergence_summary.csv")
df.groupby("Model")["Step"].describe()
```

---

### 🤖 LLM-Driven Behavior

To activate LLM decision-making:

1. Set `USE_LLM = True` in `config.py`.
2. Select a model from the dropdown in the GUI.
3. Ensure your Ollama server is running and the model is available.

---

### 🧠 Customizing LLM Queries

Modify `LLMAgent.py` to explore new agent preferences:

```python
prompt = f"What would an agent who values {context_theme} prefer as a neighborhood? Current neighborhood: {context}"
```

Replace `context_theme` with:

- `"racial composition"`
- `"ethnic background"`
- `"musical taste"`
- `"language spoken"`

This lets you simulate preferences for cultural affinity, communication compatibility, or lifestyle similarity.

---

## 🛠 Tips for Enhancing Visuals

To get good screenshots:
- Use ⌘ + Shift + 4 (Mac) or Snipping Tool (Windows)
- Place images in an `images/` folder
- Update the README paths to reflect your screenshot filenames

---

