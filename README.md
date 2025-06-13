# PancsVriend: LLM-Enhanced Schelling Segregation Model

A comprehensive research framework comparing traditional utility-maximizing agents with LLM-based agents in the classic Schelling Segregation Model. This project studies how different social contexts (race, income, politics) affect residential segregation patterns and explores the behavioral differences between mechanical and AI-powered agents.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🔬 Research Overview

This framework enables systematic comparison between:

1. **Traditional Mechanical Agents**: Classic utility-maximizing agents using mathematical best-response functions
2. **Standard LLM Agents**: AI agents that make contextual housing decisions based on social scenarios  
3. **Memory LLM Agents**: Human-like agents with persistent memory and decision history
4. **Multiple Social Contexts**: Race, ethnicity, economics, and political scenarios

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

# Run experiments with different models
python switch_llm.py --preset mixtral --quick-test
python switch_llm.py --preset gpt4 --run-experiments  # (requires OpenAI API key)
```

### Basic Usage
```bash
# Test LLM connectivity (uses Mixtral by default)
python check_llm.py

# Run comprehensive study with real-time progress monitoring
python comprehensive_comparison_study.py --quick-test

# Launch interactive dashboard with progress monitoring
python launch_dashboard_menu.py

# Full experiment suite
python run_experiments.py
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
python switch_llm.py --preset qwen --quick-test
python switch_llm.py --preset mixtral --test
```

### Production Experiments
```bash
# Full comprehensive study
python comprehensive_comparison_study.py --preset mixtral

# Specific scenario testing
python switch_llm.py --preset qwen --llm race_white_black 30

# Custom configuration
python run_experiments.py --llm-model "gpt-4" --llm-url "https://api.openai.com/v1/chat/completions" --llm-api-key "your-key"
```

### Real-Time Monitoring
```bash
# Launch progress dashboard
python launch_dashboard_menu.py

# Direct progress monitoring
streamlit run dashboard_with_progress.py
```

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
- `run_experiments.py` - Master orchestrator for complete experiment suites
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
- `visualization.py` - Comprehensive PDF reports with time series and comparisons
- `plateau_detection.py` - Convergence detection and segregation speed calculation

## 🔧 Configuration

### Default Configuration
The system uses **Mixtral 8x22B** by default via Binghamton University's endpoint:
```python
# config.py
OLLAMA_MODEL = "mixtral:8x22b-instruct"
OLLAMA_URL = "https://chat.binghamton.edu/api/chat/completions"  
OLLAMA_API_KEY = "sk-571df6eec7f5495faef553ab5cb2c67a"
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
python run_experiments.py --scenarios baseline race_white_black economic_high_working
```

### Model Comparisons
```bash
# Test different models on same scenario
python switch_llm.py --preset mixtral --quick-test
python switch_llm.py --preset qwen --quick-test
python switch_llm.py --preset gpt4 --quick-test  # (requires OpenAI key)
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

## 🎯 Research Applications

### Urban Planning
- Study segregation patterns under different policy scenarios
- Model the impact of social attitudes on neighborhood formation
- Compare algorithmic vs human-like decision making in housing

### Social Science
- Investigate how context affects segregation outcomes
- Analyze the role of memory and history in residential decisions
- Explore differences between explicit and implicit bias effects

### AI Research
- Compare LLM behavior with traditional agent-based models
- Study emergent behaviors in multi-agent LLM systems
- Benchmark contextual decision-making capabilities

## 🔍 Key Findings

Our research reveals significant differences between agent types:

- **Mechanical agents** produce predictable, rapid segregation
- **Standard LLM agents** show more variable, context-sensitive patterns
- **Memory LLM agents** exhibit slower, more realistic segregation dynamics
- **Social context** dramatically affects outcomes, with racial scenarios showing strongest effects
- **Model choice matters**: Different LLMs show varying segregation patterns

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

## 📜 License

[License information to be added]

---

*This framework provides a rigorous, user-friendly foundation for studying segregation dynamics through the lens of both traditional and AI-powered agent-based modeling, with comprehensive progress monitoring and easy model switching capabilities.*