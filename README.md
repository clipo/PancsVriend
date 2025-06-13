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

#### Default Configuration (config.py)
Edit `config.py` with your default LLM settings:

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

#### Command-Line LLM Override
All experiment scripts support command-line LLM configuration that overrides `config.py`:

```bash
# Use OpenAI GPT-4
python run_experiments.py --llm-model "gpt-4" --llm-url "https://api.openai.com/v1/chat/completions" --llm-api-key "your-openai-key"

# Use Anthropic Claude (via proxy)
python run_experiments.py --llm-model "claude-3-sonnet" --llm-url "https://api.anthropic.com/v1/messages" --llm-api-key "your-anthropic-key"

# Use different local model
python run_experiments.py --llm-model "llama2:13b" --llm-url "http://localhost:11434/api/chat/completions"
```

### 3. Verify LLM Connection

```bash
# Test basic connectivity (uses config.py)
python check_llm.py

# Test with custom LLM configuration
python check_llm.py --llm-model "gpt-4" --llm-url "https://api.openai.com/v1/chat/completions" --llm-api-key "your-key"

# Test parallel processing robustness (optional)
python test_llm_parallel.py

# Test parallel processing with custom LLM
python test_llm_parallel.py --llm-model "claude-3-sonnet" --llm-url "https://api.anthropic.com/v1/messages"
```

### 4. Run Complete Experiment Suite

```bash
# Full experiment: 100 baseline + 10 runs per LLM scenario (uses config.py)
python run_experiments.py

# Full experiment with custom LLM
python run_experiments.py --llm-model "gpt-4o" --llm-url "https://api.openai.com/v1/chat/completions" --llm-api-key "your-key"

# Quick test: 5 baseline + 2 runs per scenario
python run_experiments.py --quick-test

# Custom configuration with different LLM
python run_experiments.py --baseline-runs 50 --llm-runs 20 --scenarios baseline race_white_black --llm-model "claude-3-sonnet"
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

## 🤖 Supported LLM Providers

The framework works with any **OpenAI-compatible API**, making it flexible for different LLM providers:

### 🌐 **Cloud Providers**
- **OpenAI**: GPT-4, GPT-4o, GPT-3.5-turbo
- **Anthropic**: Claude-3-sonnet, Claude-3-haiku (via proxy)
- **Azure OpenAI**: Enterprise GPT models
- **AWS Bedrock**: Claude, Llama (via proxy)

### 🏠 **Local Models**
- **Ollama**: Any model (llama2, mistral, qwen, etc.)
- **LM Studio**: Local model serving
- **vLLM**: High-performance inference server
- **Text Generation WebUI**: Gradio-based local serving

### 📋 **Command-Line Options**
All experiment scripts support these LLM configuration flags:
- `--llm-model`: Model name/identifier
- `--llm-url`: API endpoint URL
- `--llm-api-key`: Authentication key (if required)

### 💡 **Usage Examples**
```bash
# OpenAI GPT-4
--llm-model "gpt-4" --llm-url "https://api.openai.com/v1/chat/completions" --llm-api-key "sk-..."

# Local Ollama model
--llm-model "llama2:13b" --llm-url "http://localhost:11434/api/chat/completions"

# Azure OpenAI
--llm-model "gpt-4" --llm-url "https://your-resource.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2023-12-01-preview" --llm-api-key "your-azure-key"
```

## 🧠 How LLM Agents Work

### 🆕 Enhanced Agents with Memory (NEW!)

The framework now supports **memory-enhanced agents** that act more like real humans by maintaining:

#### **Personal Identity & Characteristics**
Each agent has persistent traits:
- **Demographics**: Family size, children's ages, profession, income level
- **Personality**: Tolerance level, decision style, stability preference
- **Priorities**: Schools, safety, community, affordability
- **Background**: Years in area, homeowner status, extended family nearby

#### **Memory Systems**
Agents remember and learn from experiences:
- **Move History**: Past relocations with reasons and outcomes
- **Neighborhood Experiences**: Memorable events and their impact
- **Neighbor Relationships**: Quality of interactions over time  
- **Satisfaction Tracking**: How happy they've been in each location
- **Time Awareness**: How long they've lived in current location

#### **Human-like Decision Making**
Memory agents consider:
- "I've moved 3 times already and want stability for my children"
- "My neighbors have been friendly, why risk moving?"
- "I've lived here 8 years and built strong community ties"
- "Previous moves didn't improve my situation much"

#### **Usage**
```bash
# Enable memory in config.py
ENABLE_AGENT_MEMORY = True

# Or use the enhanced runner directly
from llm_runner_with_memory import LLMSimulationWithMemory
sim = LLMSimulationWithMemory(run_id=1, enable_memory=True)

# Demo the difference
python demo_memory_agents.py
```

#### **Memory vs Standard Comparison**
| Feature | Standard Agents | Memory Agents |
|---------|----------------|---------------|
| **Decision basis** | Current neighbors only | History + neighbors + identity |
| **Stability** | Reactive to changes | Consider moving costs/benefits |
| **Realism** | Game-theoretic | Human psychological factors |
| **Relationships** | None | Build neighbor familiarity |
| **Context** | 3x3 grid snapshot | Life history and experiences |

### Traditional LLM Agents

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

## 🔬 Research Guide: Comparing LLMs and Social Contexts

### 📊 Comparing Different LLMs

When comparing how different LLMs handle residential decisions, follow this structured approach:

#### 1. **Baseline Comparison Setup**
```bash
# First, always run mechanical baseline for reference
python baseline_runner.py --runs 100

# Then test each LLM with identical parameters
# OpenAI GPT-4
python run_experiments.py --baseline-runs 0 --llm-runs 20 --llm-model "gpt-4" --llm-url "https://api.openai.com/v1/chat/completions" --llm-api-key "sk-..."

# Anthropic Claude
python run_experiments.py --baseline-runs 0 --llm-runs 20 --llm-model "claude-3-sonnet-20240229" --llm-url "https://api.anthropic.com/v1/messages" --llm-api-key "sk-ant-..."

# Local Llama2
python run_experiments.py --baseline-runs 0 --llm-runs 20 --llm-model "llama2:13b" --llm-url "http://localhost:11434/api/chat/completions"
```

#### 2. **Consistent Testing Protocol**
- Use **same number of runs** (recommend 20-50 per LLM)
- Keep **grid size and agent counts** constant
- Test **all social contexts** with each LLM
- Record **response times** for cost/performance analysis

#### 3. **Performance Metrics to Track**
- **Convergence speed**: How quickly segregation emerges
- **Final segregation levels**: Steady-state metrics
- **Decision consistency**: Variance across runs
- **Response time**: API latency impacts
- **Failure rates**: Robustness of each LLM

### 🌍 Modifying Social Contexts

To add new social contexts or modify existing ones:

#### 1. **Edit Social Context Scenarios**
Open `llm_runner.py` and find `CONTEXT_SCENARIOS`:

```python
CONTEXT_SCENARIOS = {
    'your_new_context': {
        'type_a': 'Group A description',
        'type_b': 'Group B description', 
        'prompt_template': """You are a {agent_type} considering whether to move...
        
        [Your context-specific prompt that shapes decision-making]
        
        Respond with ONLY coordinates or None."""
    }
}
```

#### 2. **Design Effective Context Prompts**
Key elements for realistic contexts:
- **Authentic identity**: "You are a [specific demographic]"
- **Real motivations**: Family, culture, economics, safety
- **Decision factors**: What this group actually considers
- **No bias injection**: Let LLM express natural preferences

#### 3. **Example: Adding Economic Contexts**
```python
'economic_tech_service': {
    'type_a': 'tech industry professional',
    'type_b': 'service industry worker',
    'prompt_template': """You are a {agent_type} considering housing options.
    
    Your neighborhood ({context}) reflects different economic realities.
    As someone in your economic situation, consider:
    - Commute to work locations
    - Cost of living pressures  
    - Access to amenities you need
    - Community support systems
    
    Where would you genuinely prefer to live?"""
}
```

### 📈 Systematic Comparison Workflow

#### 1. **Planning Phase**
- Define research questions
- Select 3-5 LLMs to compare
- Choose relevant social contexts
- Set consistent parameters

#### 2. **Execution Phase**
```bash
# Create experiment plan
mkdir experiments_gpt4 experiments_claude experiments_llama

# Run each LLM systematically
for context in baseline race_white_black economic_high_low; do
    python llm_runner.py --scenario $context --runs 30 --llm-model "gpt-4" ...
done
```

#### 3. **Analysis Phase**
```bash
# Generate comparative analysis
python statistical_analysis.py

# Create visualization comparing all LLMs
python visualization.py --baseline-dir experiments/baseline_* \
    --llm-dirs experiments/llm_gpt4_* experiments/llm_claude_* experiments/llm_llama_*
```

### 🎯 Best Practices for LLM Comparison

1. **Control Variables**
   - Same random seeds for reproducibility
   - Identical grid initialization
   - Consistent temperature settings (0.3 recommended)

2. **Statistical Rigor**
   - Minimum 20-30 runs per condition
   - Report confidence intervals
   - Use appropriate statistical tests (in `statistical_analysis.py`)

3. **Document Everything**
   - LLM version/date (models update!)
   - Exact prompts used
   - Any failures or anomalies
   - Total API costs

### 💡 Research Questions to Explore

1. **LLM Behavioral Differences**
   - Do larger models show more nuanced segregation patterns?
   - How do open vs. closed source models differ?
   - What biases emerge across different LLMs?

2. **Context Sensitivity**
   - Which contexts produce fastest segregation?
   - How do economic vs. racial contexts differ?
   - Do LLMs reflect real-world segregation data?

3. **Prompt Engineering Effects**
   - How sensitive are results to prompt wording?
   - Can prompts reduce or increase segregation?
   - What happens with ambiguous identities?

4. **Memory and Human-like Behavior** 🆕
   - Do agents with memory show more realistic stability?
   - How does relationship-building affect segregation speed?
   - Do family considerations reduce reactive moving?
   - Which memories matter most for housing decisions?

### 🧠 Comparing Memory vs Standard Agents

To study the effect of memory on segregation patterns:

```bash
# Run standard agents
python run_experiments.py --llm-runs 20 --scenarios baseline

# Run memory-enhanced agents  
ENABLE_AGENT_MEMORY=True python run_experiments.py --llm-runs 20 --scenarios baseline

# Demo the differences
python demo_memory_agents.py
```

**Expected differences with memory**:
- **Slower segregation**: Agents consider moving costs
- **Higher stability**: Less reactive behavior
- **More realistic patterns**: Family/relationship factors
- **Individual stories**: Each agent has unique journey

### 📊 Publishing Results

When sharing findings:
1. Report **all** model configurations
2. Include **convergence plots** for each LLM/context
3. Show **statistical comparisons** with effect sizes
4. Discuss **cost/performance tradeoffs**
5. Share **reproducible code** with exact versions

## 🔬 Design Space Exploration (NEW!)

For **comprehensive research**, the framework includes a powerful design space exploration system that systematically tests different combinations of:

- **LLM Models**: GPT-4, Claude, local models, etc.
- **Agent Types**: Standard vs memory-enhanced
- **Social Contexts**: All scenarios (racial, economic, political)
- **Grid Sizes**: Small to extra-large populations
- **Multiple Runs**: Statistical significance testing

### 🚀 **Quick Start: Full Design Space**

```bash
# 1. Configure your experiment matrix
cp experiment_configs.yaml my_experiment.yaml
# Edit my_experiment.yaml with your LLM credentials

# 2. Run complete exploration (plan + execute + analyze)
python run_design_space_exploration.py --all --config my_experiment.yaml

# 3. Results automatically saved with comparative analysis
```

### 📋 **Step-by-Step Workflow**

#### **1. Plan Experiments**
```bash
# Generate experimental design matrix
python run_design_space_exploration.py --plan \
    --llms qwen_local gpt4 claude_sonnet \
    --agents standard memory \
    --scenarios baseline race_white_black economic_high_working \
    --grids small medium large

# Output: experiment_plan_TIMESTAMP.json with all combinations
```

#### **2. Estimate Resources**
The planner automatically estimates:
- **Runtime**: 2-48 hours (depending on scope)
- **API Costs**: $0-$2000+ (depending on LLMs used)
- **Storage**: 100MB-10GB of results

#### **3. Run Experiments**
```bash
# Run all planned experiments
python run_design_space_exploration.py --run

# Or run in batches for large experiments
python run_design_space_exploration.py --run --max-experiments 50

# Monitor progress with generated logs
tail -f design_space_exploration/progress_*.json
```

#### **4. Analyze Results**
```bash
# Generate comprehensive comparative analysis
python run_design_space_exploration.py --analyze

# Creates visualizations, statistical tests, and reports
```

### 📊 **Generated Analysis**

The system automatically creates:

#### **Visualizations**
- **Convergence Analysis**: Speed and patterns across configurations
- **Segregation Metrics**: All 6 metrics compared systematically  
- **Memory vs Standard**: Direct comparison of agent types
- **PCA Analysis**: Principal components of experimental variance

#### **Statistical Tests**
- **Significance Testing**: ANOVA, Kruskal-Wallis across conditions
- **Effect Sizes**: Practical significance of differences
- **Confidence Intervals**: Uncertainty quantification

#### **Reports**
- **Comprehensive Report**: `comprehensive_report.md` with key findings
- **Raw Statistics**: `summary_statistics.json` with detailed breakdowns
- **Test Results**: `statistical_tests.json` with p-values and effect sizes

### 🎯 **Example Research Questions**

The design space explorer helps answer:

1. **LLM Comparison**: "Does GPT-4 show different segregation patterns than Claude?"
2. **Memory Effects**: "Do memory agents segregate more slowly than standard agents?"
3. **Context Sensitivity**: "Which social contexts produce fastest segregation?"
4. **Scale Effects**: "How does population size affect convergence speed?"
5. **Interaction Effects**: "Do memory benefits vary by LLM model?"

### 📁 **Output Structure**

```
design_space_exploration/
├── experiment_plan_TIMESTAMP.json     # Full experimental design
├── experiments/                       # Individual experiment results
│   ├── exp_0001/                     # First experiment
│   │   ├── experiment_config.json    # Configuration
│   │   └── results.json              # Raw results
│   └── exp_NNNN/                     # Additional experiments
├── analysis/                         # Comparative analysis
│   ├── comprehensive_report.md       # Main findings report
│   ├── convergence_analysis.png      # Convergence visualizations
│   ├── segregation_metrics_comparison.png  # Metrics comparison
│   ├── memory_vs_standard_analysis.png     # Agent type comparison
│   ├── pca_analysis.png              # Principal component analysis
│   ├── summary_statistics.json       # Detailed statistics
│   ├── statistical_tests.json        # Significance tests
│   └── pca_loadings.csv              # PCA component loadings
└── logs/                             # Execution logs
    └── progress_TIMESTAMP.json       # Real-time progress
```

### 💡 **Configuration Tips**

#### **For Development/Testing**
```yaml
# experiment_configs.yaml
quick_test:
  enabled: true           # Enable quick mode
  runs_per_config: 3      # Fewer runs
  max_steps: 100          # Shorter simulations
  grids: ["small"]        # Smaller populations
  scenarios: ["baseline"] # Single scenario
```

#### **For Production Research**
```yaml
experiment_parameters:
  runs_per_config: 50     # More runs for significance
  max_steps: 1000         # Full convergence
  
grid_configurations:
  small: {grid_size: 10, type_a: 25, type_b: 25}    # Quick
  medium: {grid_size: 15, type_a: 75, type_b: 75}   # Moderate  
  large: {grid_size: 20, type_a: 150, type_b: 150}  # Realistic
```

### 🚨 **Resource Planning**

Before running large explorations:

1. **Estimate costs**: Check generated `estimates_TIMESTAMP.json`
2. **Plan time**: Large studies take 12-48 hours
3. **Monitor progress**: Use progress logs for batch management
4. **Backup results**: Experiments save continuously

### 📈 **Advanced Usage**

#### **Parallel Execution** (for large machines)
```bash
# Split large experiments across multiple processes
python run_design_space_exploration.py --run --start-idx 0 --max-experiments 25 &
python run_design_space_exploration.py --run --start-idx 25 --max-experiments 25 &
python run_design_space_exploration.py --run --start-idx 50 --max-experiments 25 &
```

#### **Custom Analysis**
```python
# Use analyzer directly for custom analysis
from design_space_analyzer import DesignSpaceAnalyzer
analyzer = DesignSpaceAnalyzer("my_experiment_results")
analyzer.create_comprehensive_report()
```

This design space exploration system transforms your simulation from individual experiments into **systematic scientific investigation** with **statistical rigor** and **comparative analysis**! 🎯

## 🏆 **Comprehensive Comparison Study (RECOMMENDED)**

For **complete research**, we provide a **ready-to-run comprehensive study** that compares:

### **🔬 The Complete Agent Comparison Matrix**
1. **📐 Mechanical Baseline** - Traditional utility-maximizing Schelling agents
2. **🤖 Standard LLM Agents** - LLM decisions based on current neighborhood only  
3. **🧠 Memory-Enhanced LLM Agents** - Human-like agents with personal history and relationships

### **🌍 Across All Social Contexts**
- **Baseline** (Red vs Blue control)
- **Racial** (White vs Black families)
- **Ethnic** (Asian vs Hispanic families)  
- **Economic** (High-income vs Working-class)
- **Political** (Liberal vs Conservative)

This gives you a **3×5×2 factorial design** = **30 experiments** with **900 total simulations** providing **publication-ready comparative analysis**!

### 🚀 **Quick Start: Complete Study**

```bash
# 1. Set up your LLM credentials in the config
cp baseline_vs_llm_study.yaml my_study.yaml
# Edit my_study.yaml with your LLM API details

# 2. Quick test first (recommended)
python comprehensive_comparison_study.py --quick-test --config my_study.yaml
# Takes ~2 hours, validates everything works

# 3. Run full study (overnight/weekend recommended)
python comprehensive_comparison_study.py --config my_study.yaml
# Takes ~40 hours, generates complete dataset

# 4. Results automatically analyzed and saved
```

### 📊 **Study Design Overview**

#### **Research Questions Answered**
1. **Mechanical vs LLM**: How do LLM agents differ from traditional Schelling models?
2. **Standard vs Memory LLM**: Does memory make agents more human-like?
3. **Social Context Effects**: Which contexts affect segregation patterns most?
4. **Interaction Effects**: Do agent types respond differently to social contexts?
5. **Scale Effects**: How does population size influence each agent type?

#### **Experimental Matrix**
| Agent Type | Social Contexts | Grid Sizes | Runs Each | Total Runs |
|------------|----------------|------------|-----------|------------|
| 📐 Mechanical | 5 scenarios | 2 sizes | 30 | 300 |
| 🤖 Standard LLM | 5 scenarios | 2 sizes | 30 | 300 |
| 🧠 Memory LLM | 5 scenarios | 2 sizes | 30 | 300 |
| **TOTAL** | **5** | **2** | **30** | **900** |

#### **Expected Findings**
- **📐 Mechanical**: Fast, predictable segregation regardless of social context
- **🤖 Standard LLM**: Context-sensitive, more variable than mechanical
- **🧠 Memory LLM**: Slower, more stable, human-like residential patterns

### 📈 **Generated Analysis & Outputs**

#### **Comprehensive Comparison Analysis**
The study automatically generates:

##### **🔬 Statistical Analysis**
- **3-way ANOVA** comparing all agent types across contexts
- **Post-hoc tests** for pairwise comparisons  
- **Effect size calculations** (Cohen's d) for practical significance
- **Interaction effect analysis** between agent type and social context
- **Convergence speed analysis** with survival curves

##### **📊 Rich Visualizations**
```
comprehensive_study_TIMESTAMP/
├── mechanical_baseline_results/           # Traditional Schelling results
├── llm_results/                          # LLM experiment results  
├── comprehensive_analysis/               # 3-way comparison analysis
│   ├── three_way_comparison_report.md   # Main findings report
│   ├── agent_type_comparison.png        # Side-by-side comparisons
│   ├── social_context_effects.png       # Context sensitivity analysis
│   ├── convergence_speed_analysis.png   # Speed comparisons
│   ├── interaction_effects.png          # Agent×Context interactions  
│   ├── memory_agent_stories.png         # Individual journey analysis
│   ├── segregation_heatmaps.png         # All combinations visualized
│   └── statistical_summary.json         # Detailed test results
└── logs/                                # Execution logs
```

##### **📋 Individual Agent Stories** (Memory agents only)
- **Personal identities**: Family size, profession, priorities
- **Move histories**: Why each agent moved and when
- **Relationship development**: How neighbor relationships evolved
- **Satisfaction trajectories**: Happiness levels over time

#### **Key Output Files**

##### **Main Research Report**
- `three_way_comparison_report.md` - Publication-ready findings summary
- `statistical_summary.json` - All statistical test results with p-values

##### **Visualizations for Papers**
- `agent_type_comparison.png` - Main comparison figure
- `social_context_effects.png` - Context sensitivity results  
- `convergence_speed_analysis.png` - Speed and stability analysis
- `interaction_effects.png` - Statistical interaction plots

##### **Raw Data for Further Analysis**
- `complete_results_dataset.csv` - All simulation outcomes  
- `memory_agent_journeys.json` - Individual agent stories
- `convergence_metrics.csv` - Detailed convergence analysis

### 🔍 **Analyzing Your Results**

#### **Reading the Main Report**
```bash
# Open the comprehensive findings
open comprehensive_study_*/comprehensive_analysis/three_way_comparison_report.md
```

The report includes:
- **Executive Summary**: Key findings and statistical significance
- **Agent Type Comparison**: How mechanical, standard LLM, and memory LLM differ
- **Social Context Analysis**: Which contexts produce different patterns
- **Interaction Effects**: Unique combinations and emergent behaviors
- **Individual Stories**: Notable memory agent journeys

#### **Statistical Significance Guide**
- **p < 0.001**: Highly significant difference (⭐⭐⭐)
- **p < 0.01**: Very significant difference (⭐⭐)  
- **p < 0.05**: Significant difference (⭐)
- **Effect size d > 0.8**: Large practical difference
- **Effect size d > 0.5**: Medium practical difference
- **Effect size d > 0.2**: Small practical difference

#### **Interpreting Visualizations**

##### **Agent Type Comparison Plot**
- **Y-axis**: Segregation metrics (clusters, switch rate, etc.)
- **X-axis**: Social contexts
- **Colors**: Agent types (Mechanical, Standard LLM, Memory LLM)
- **Error bars**: 95% confidence intervals

##### **Convergence Speed Analysis**
- **Survival curves**: Show percentage unconverged over time
- **Median convergence**: 50% convergence point
- **Confidence bands**: Statistical uncertainty

##### **Memory Agent Stories**
- **Individual trajectories**: Each line = one agent's satisfaction over time
- **Move markers**: Points where agents relocated
- **Color coding**: Different family types or characteristics

### 🎯 **Research Applications**

#### **For Academic Papers**
This study design supports research questions like:
- "Do large language models exhibit realistic residential segregation patterns?"
- "How does agent memory affect convergence speed and stability?"
- "Are certain social contexts more segregating across different agent architectures?"

#### **For Policy Analysis**
- **Housing Policy**: Compare interventions across agent types
- **Urban Planning**: Understand how different populations might respond
- **Social Integration**: Test strategies for reducing segregation

#### **For Model Validation**
- **Benchmark Comparison**: Standard reference for LLM segregation models
- **Human Realism**: Memory agents as proxy for real human behavior
- **Context Sensitivity**: Understanding when social factors matter

### 💡 **Customizing the Study**

#### **Modify Social Contexts**
Edit `baseline_vs_llm_study.yaml`:
```yaml
scenarios:
  - baseline
  - custom_context_1  # Add your contexts
  - custom_context_2
  # Remove contexts you don't need
```

#### **Adjust Statistical Power**
```yaml
experiment_parameters:
  runs_per_config: 50    # More runs = stronger statistics
  max_steps: 1500        # Longer runs = full convergence
```

#### **Grid Size Sensitivity**
```yaml
grid_configurations:
  small: {grid_size: 10, type_a: 25, type_b: 25}
  medium: {grid_size: 15, type_a: 75, type_b: 75}  
  large: {grid_size: 20, type_a: 150, type_b: 150}
  xlarge: {grid_size: 25, type_a: 200, type_b: 200}
```

### ⚠️ **Resource Planning**

#### **Time Estimates**
- **Quick Test**: 2-3 hours (validates approach)
- **Full Study**: 35-45 hours (complete analysis)
- **Analysis Generation**: 1-2 hours (automatic)

#### **Computing Requirements**
- **CPU**: Multi-core recommended for parallel execution
- **Memory**: 8GB+ RAM for large grid sizes
- **Storage**: 2-5GB for complete study results
- **Network**: Stable connection for LLM API calls

#### **Cost Estimates** (using cloud LLMs)
| LLM Provider | Cost per Run | 900 Runs | 
|--------------|-------------|----------|
| Local Models | $0 | $0 |
| GPT-3.5-turbo | $0.50 | $450 |
| GPT-4 | $15.00 | $13,500 |
| Claude-3-Haiku | $0.125 | $112.50 |

### 🔄 **Running Studies in Parallel**

For faster execution on multi-core machines:
```bash
# Split study across multiple processes
python comprehensive_comparison_study.py --config my_study.yaml &
# Start additional parallel runs with different config subsets
```

### 📚 **Citation Guidelines**

When publishing results from this framework:
```bibtex
@software{schelling_llm_comprehensive,
  title={Comprehensive LLM vs Mechanical Agent Segregation Study},
  author={[Your Name]},
  year={2024},
  note={Framework: Schelling Segregation Model with LLM Agents},
  url={https://github.com/clipo/PancsVriend}
}
```

This comprehensive study design provides **the most rigorous comparison** of mechanical vs LLM segregation models available, with **statistical significance testing** and **publication-ready outputs**! 🎓

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

## ⏱️ Timing and Cost Considerations

### Simulation Duration Estimates

#### **Baseline Mechanical Agents**
- **Per run**: ~30 seconds
- **100 runs**: ~50 minutes
- **Cost**: Free (no LLM calls)

#### **LLM Simulations**
Time depends on LLM response speed, grid size, and memory settings:

| Configuration | Agents | Steps to Converge | LLM Calls | Time per Run | 
|--------------|--------|-------------------|-----------|--------------|
| Default (20x20) | 300 | ~30-50 | ~10,000-15,000 | 2-5 minutes |
| Default + Memory | 300 | ~40-60 | ~12,000-18,000 | 3-7 minutes |
| Small (10x10) | 50 | ~20-30 | ~1,000-1,500 | 1-2 minutes |
| Small + Memory | 50 | ~25-35 | ~1,250-1,750 | 1.5-3 minutes |
| Tiny (5x5) | 10 | ~10-20 | ~100-200 | 30-60 seconds |

#### **Full Experiment Suite**
- **Quick test**: 10-20 minutes
- **Standard (100 baseline + 50 LLM)**: 2-4 hours  
- **Comprehensive (100 baseline + 200 LLM)**: 8-12 hours

### 💰 API Cost Estimates

Approximate costs per run (default 20x20 grid):

#### Standard Agents
| LLM Provider | Cost per 1K tokens | Est. tokens per run | Cost per run | 30 runs |
|--------------|-------------------|---------------------|--------------|---------|
| GPT-3.5-turbo | $0.001 | ~500K | $0.50 | $15 |
| GPT-4 | $0.03 | ~500K | $15.00 | $450 |
| Claude-3-Haiku | $0.00025 | ~500K | $0.125 | $3.75 |
| Claude-3-Sonnet | $0.003 | ~500K | $1.50 | $45 |
| Local Models | Free | - | $0 | $0 |

#### Memory-Enhanced Agents (+20-40% tokens)
| LLM Provider | Cost per 1K tokens | Est. tokens per run | Cost per run | 30 runs |
|--------------|-------------------|---------------------|--------------|---------|
| GPT-3.5-turbo | $0.001 | ~650K | $0.65 | $19.50 |
| GPT-4 | $0.03 | ~650K | $19.50 | $585 |
| Claude-3-Haiku | $0.00025 | ~650K | $0.16 | $4.88 |
| Claude-3-Sonnet | $0.003 | ~650K | $1.95 | $58.50 |
| Local Models | Free | - | $0 | $0 |

*Memory agents use more tokens due to personal history and relationship context in prompts*

### 🚀 Performance Optimization Tips

1. **For Development/Testing**
   ```bash
   # Use smaller grids
   python llm_runner.py --scenario baseline --runs 2
   # Grid size 10x10 = 50 agents = 5x faster
   ```

2. **For Batch Experiments**
   ```bash
   # Run overnight
   nohup python run_experiments.py > experiment.log 2>&1 &
   
   # Monitor progress
   tail -f experiment.log
   ```

3. **For Cost Reduction**
   - Use cheaper models (GPT-3.5, Claude-Haiku) for initial tests
   - Reduce grid size for exploratory analysis
   - Run fewer simulations with higher step counts
   - Use local models (Ollama) when possible

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

### 🏆 **Comprehensive Study Troubleshooting**

#### **Study Fails to Start**
```bash
# Check configuration file
python show_study_matrix.py  # Validates your setup

# Test LLM connectivity first
python check_llm.py --llm-model "your-model" --llm-url "your-url" --llm-api-key "your-key"

# Run quick test mode first
python comprehensive_comparison_study.py --quick-test
```

#### **Study Stops Mid-Execution**
```bash
# Check progress
tail -f comprehensive_study_*/logs/progress_*.json

# Resume from last checkpoint (experiments save continuously)
python comprehensive_comparison_study.py --config my_study.yaml
# Will skip completed experiments automatically
```

#### **Memory Issues with Large Studies**
```bash
# Reduce grid sizes temporarily
# Edit baseline_vs_llm_study.yaml:
grid_configurations:
  medium: {grid_size: 15, type_a: 75, type_b: 75}  # Remove 'large'

# Or run in smaller batches
python run_design_space_exploration.py --run --max-experiments 10
```

#### **Analysis Generation Fails**
```bash
# Run analysis separately after experiments complete
python design_space_analyzer.py --input-dir comprehensive_study_TIMESTAMP/llm_results

# Check for missing dependencies
pip install -r requirements.txt
```

#### **Unexpected Results**
- **All agent types look similar**: Check if LLM is actually being used (should see LLM call counts > 0)
- **Memory agents identical to standard**: Verify `ENABLE_AGENT_MEMORY = True` in config.py
- **No convergence**: Increase `max_steps` in configuration
- **High variance**: Increase `runs_per_config` for better statistics

### LLM-Specific Issues:

#### **Slow Performance**
```bash
# Check LLM response time
python check_llm.py --llm-model "your-model"

# If slow (>1s per response), consider:
# 1. Using a faster model
# 2. Reducing grid size
# 3. Implementing caching for repeated contexts
```

#### **Different LLMs Give Different Results**
This is expected! Document the differences:
- **Response format variations**: Some LLMs may format coordinates differently
- **Decision patterns**: Models have different "personalities"
- **Consistency**: Some models are more deterministic than others

#### **API Rate Limits**
```python
# Add delays between runs in your script
import time
for run in range(n_runs):
    run_simulation()
    time.sleep(5)  # 5 second delay between runs
```

#### **Comparing Incompatible APIs**
Some LLMs require API adapters:
- **Anthropic**: Use their native API or a proxy
- **Google PaLM/Gemini**: May need API wrapper
- **Local models**: Ensure OpenAI-compatible endpoint

### General Troubleshooting:
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