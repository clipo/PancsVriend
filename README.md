# Schelling Segregation Simulator (Enhanced Edition)

This project implements an extended version of the Schelling Segregation Model based on Pancs & Vriend (2007). The simulation demonstrates how individual preferences for neighborhood composition—even preferences *for integration*—can unintentionally lead to segregation.

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
pip install pygame pygame_gui matplotlib numpy
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

