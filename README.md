# Schelling Segregation Model: LLM vs Mechanical Agents

**A comprehensive experimental framework comparing traditional utility-maximizing agents with LLM-based agents exhibiting authentic human residential preferences.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

This project extends the classic Schelling Segregation Model to compare how **mechanical utility-maximizing agents** versus **LLM agents acting as authentic residents** make housing decisions. The framework enables systematic study of segregation patterns across different social contexts.

### 🔬 Research Questions

- Do LLM agents exhibit realistic residential segregation patterns?
- How do social contexts (race, income, politics) affect LLM decision-making?
- What's the difference between purely rational and human-like housing choices?
- Can we measure the "speed of segregation" across different scenarios?

## ✨ Key Features

### 🤖 **Dual Agent Systems**
- **Mechanical Agents**: Traditional utility-maximizing best-response dynamics
- **LLM Agents**: Act as authentic residents considering cultural, economic, and social factors

### 🌍 **Social Context Scenarios**
1. **Baseline**: Red vs blue teams (control)
2. **Racial**: White middle class vs Black families
3. **Ethnic**: Asian American vs Hispanic/Latino families  
4. **Economic**: High-income vs working-class households
5. **Political**: Liberal vs conservative households

### 📊 **Comprehensive Analytics**
- **6 Segregation Metrics**: Clusters, switch rate, distance, mix deviation, share, ghetto rate
- **Plateau Detection**: Automatic identification of convergence points
- **Statistical Testing**: ANOVA, effect sizes, multivariate analysis
- **Visualization**: Evolution plots, comparison charts, PDF reports

### 🔧 **Robust Infrastructure**
- **Parallel Processing**: Efficient LLM query handling with circuit breakers
- **Error Handling**: Graceful degradation when LLM services fail
- **Scalability**: 100+ simulation runs with configurable parameters

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/clipo/PancsVriend.git
cd PancsVriend
pip install -r requirements.txt
```

### 2. Configuration

Edit `config.py` with your LLM settings:

```python
# LLM Configuration
OLLAMA_MODEL = "qwen2.5-coder:32B"
OLLAMA_URL = "https://your-llm-server.com/api/chat/completions"
OLLAMA_API_KEY = "your-api-key-here"

# Simulation Parameters
GRID_SIZE = 20
NUM_TYPE_A = 150
NUM_TYPE_B = 150
```

### 3. Verify LLM Connection

```bash
# Test basic connectivity
python check_llm.py

# Test parallel processing robustness (optional)
python test_llm_parallel.py
```

### 4. Run Complete Experiment Suite

```bash
# Full experiment: 100 baseline + 10 runs per LLM scenario
python run_experiments.py

# Quick test: 5 baseline + 2 runs per scenario
python run_experiments.py --quick-test

# Custom configuration
python run_experiments.py --baseline-runs 50 --llm-runs 20 --scenarios baseline race_white_black
```

## 📋 Experiment Workflow

### Automatic Process:
1. **Baseline Simulations** → 100 runs of mechanical agents
2. **LLM Connectivity Check** → Verify LLM availability  
3. **LLM Simulations** → 10 runs per social context scenario
4. **Convergence Analysis** → Detect plateaus and calculate rates
5. **Statistical Testing** → ANOVA, effect sizes, significance tests
6. **Visualization** → Generate comprehensive PDF reports
7. **Summary Report** → JSON with all experiment metadata

### Output Structure:
```
experiments/
├── baseline_[timestamp]/          # Mechanical agent results
├── llm_baseline_[timestamp]/      # LLM baseline results  
├── llm_race_white_black_[timestamp]/  # Racial context results
└── ...

reports/
├── comprehensive_report_[timestamp].pdf  # Visual analysis
├── statistical_analysis_[timestamp].txt  # Statistical tests
└── experiment_summary_[timestamp].json   # Master summary
```

## 🧠 How LLM Agents Work

### Authentic Resident Personas
LLM agents are prompted to act as **real people** making housing decisions:

> *"You are a white middle class family considering whether to move to a different house in your neighborhood. As a real person with your own background, experiences, and family considerations, think about where you would genuinely prefer to live..."*

### Decision Factors Considered:
- **Cultural connections** and community comfort
- **Family needs** and children's friendships  
- **Economic priorities** and property values
- **Social dynamics** and neighborhood composition
- **Safety perceptions** and lifestyle preferences

### Response Format:
- Coordinates `(row, col)` to move to an empty house
- `None` to stay in current location
- **No explanations** - just the decision (for clean data)

## 📊 Analysis Capabilities

### Convergence Detection
- **Plateau identification**: When metrics stabilize
- **Convergence rates**: Speed of segregation
- **Half-life calculations**: Time to reach 50% of final value

### Statistical Methods
- **Descriptive statistics**: Mean, std, quartiles for all metrics
- **Normality testing**: Shapiro-Wilk tests
- **Group comparisons**: ANOVA/Kruskal-Wallis with post-hoc tests
- **Effect sizes**: Cohen's d for practical significance
- **Multivariate analysis**: PCA for pattern recognition

### Visualization Features
- **Time series plots** with confidence intervals
- **Convergence distributions** across scenarios
- **Scenario comparisons** with error bars
- **Heat maps** and correlation matrices

## 🛠️ Individual Components

### Run Components Separately:

```bash
# Baseline mechanical agents only
python baseline_runner.py --runs 100

# Specific LLM scenario
python llm_runner.py --scenario race_white_black --runs 10

# Analysis of existing results
python statistical_analysis.py
python visualization.py --baseline-dir experiments/baseline_xxx --llm-dirs experiments/llm_*
```

### Interactive GUI Simulation:

```bash
# Original interactive simulation
python SchellingSim.py
```

## 🔧 Advanced Configuration

### Parallel Processing Tuning:
- Modify `batch_size` in `llm_runner.py` (default: 5 agents)
- Adjust `max_llm_failures` for circuit breaker sensitivity (default: 20)
- Change timeouts in `get_llm_decision()` method

### Experiment Customization:
- Add new social contexts in `CONTEXT_SCENARIOS`
- Modify convergence thresholds (`no_move_threshold`)
- Adjust utility functions in `Agent.py`

### Statistical Analysis:
- Customize significance levels (`alpha = 0.05`)
- Add new metrics to track
- Modify plateau detection sensitivity

## 📈 Expected Results

### Typical Findings:
- **Mechanical agents**: Rapid, predictable segregation patterns
- **LLM baseline**: Similar to mechanical but with more variability
- **Social contexts**: Different scenarios show varying segregation rates
- **Convergence speeds**: Real-world contexts often segregate faster/slower than baseline

### Performance Metrics:
- **Baseline simulations**: ~30 seconds per run
- **LLM simulations**: ~3-5 minutes per run (depends on LLM speed)
- **Full experiment suite**: 2-4 hours total

## 🚨 Troubleshooting

### LLM Connection Issues:
```bash
# Check connectivity
python check_llm.py

# Common fixes:
# 1. Verify Ollama is running: ollama serve
# 2. Check model is loaded: ollama pull qwen2.5-coder:32B  
# 3. Validate URL and API key in config.py
```

### Performance Issues:
```bash
# Test parallel processing
python test_llm_parallel.py

# If issues detected:
# 1. Reduce batch_size in llm_runner.py
# 2. Increase timeouts
# 3. Use smaller/faster LLM model
```

### Memory/Hanging Issues:
- The system includes circuit breakers and timeouts
- LLM failures automatically fall back to mechanical agents
- All threads have bounded queues and forced cleanup

## 📚 Scientific Background

### Schelling Segregation Model
Based on Thomas Schelling's pioneering work demonstrating how individual preferences can lead to collective segregation patterns, even when individuals prefer integration.

### Extensions in This Framework:
- **Pancs & Vriend (2007)**: Utility-based best response dynamics
- **LLM Integration**: Authentic human-like decision making
- **Multi-context Analysis**: Real-world social scenarios
- **Convergence Analytics**: Mathematical analysis of segregation speed

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📖 Citation

If you use this framework in your research, please cite:

```bibtex
@software{schelling_llm_framework,
  title={Schelling Segregation Model: LLM vs Mechanical Agents},
  author={[Your Name]},
  year={2024},
  url={https://github.com/clipo/PancsVriend}
}
```

## 🔗 References

- Pancs, R., & Vriend, N. J. (2007). *Schelling's spatial proximity model of segregation revisited.* Journal of Public Economics, 91(1), 1-24.
- Schelling, T. C. (1971). *Dynamic models of segregation.* Journal of Mathematical Sociology, 1(2), 143-186.

---

**Ready to explore how artificial intelligence makes housing decisions compared to traditional economic models? Start your experiments today!**