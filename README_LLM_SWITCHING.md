# 🔄 Easy LLM Model Switching

This guide shows you how to easily switch between different LLM models for your Schelling segregation experiments.

## 🚀 Quick Start

### List Available Models
```bash
python switch_llm.py --list
```

### Test Model Connectivity
```bash
# Test current default (Mixtral)
python switch_llm.py --preset mixtral --test

# Test other models
python switch_llm.py --preset qwen --test
python switch_llm.py --preset gpt4 --test  # (requires OpenAI API key)
```

### Run Experiments with Different Models
```bash
# Quick test with Mixtral
python switch_llm.py --preset mixtral --quick-test

# Full experiments with Qwen
python switch_llm.py --preset qwen --run-experiments

# LLM experiments with specific scenario
python switch_llm.py --preset mixtral --llm baseline 30
```

## 📋 Available LLM Presets

| Preset | Model | Provider | Notes |
|--------|-------|----------|--------|
| `mixtral` | Mixtral 8x22B | Binghamton Uni | **Default** - High performance, ready to use |
| `qwen` | Qwen 2.5 Coder 32B | Binghamton Uni | Code-optimized, ready to use |
| `gpt4` | GPT-4 | OpenAI | Requires OpenAI API key |
| `gpt4o` | GPT-4o | OpenAI | Latest OpenAI model, requires API key |
| `gpt35` | GPT-3.5 Turbo | OpenAI | Fast and cost-effective, requires API key |
| `claude-sonnet` | Claude 3 Sonnet | Anthropic | Requires Anthropic API key |
| `claude-haiku` | Claude 3 Haiku | Anthropic | Fast model, requires API key |
| `local-llama` | Llama2 | Local Ollama | Requires local Ollama installation |

## 🛠 Configuration Management

### Show Current Default
```bash
python update_default_llm.py --show
```

### Change Default Model
```bash
# Set Mixtral as default
python update_default_llm.py --set mixtral

# Set Qwen as default
python update_default_llm.py --set qwen

# Set GPT-4 as default (need to add API key first)
python update_default_llm.py --set gpt4
```

### Adding Your Own API Keys

Edit `llm_presets.py` to add your API keys:

```python
# For OpenAI models
"gpt4": {
    "api_key": "sk-your-actual-openai-key-here",
    # ... other settings
},

# For Anthropic models  
"claude-sonnet": {
    "api_key": "sk-ant-your-actual-anthropic-key-here",
    # ... other settings
},
```

## 🎯 Common Use Cases

### Compare Models Quickly
```bash
# Test connectivity of all ready-to-use models
python switch_llm.py --preset mixtral --test
python switch_llm.py --preset qwen --test

# Run quick experiments to compare
python switch_llm.py --preset mixtral --quick-test
python switch_llm.py --preset qwen --quick-test
```

### Production Experiments
```bash
# Full experiment suite with your preferred model
python switch_llm.py --preset mixtral --run-experiments

# Specific LLM scenario testing
python switch_llm.py --preset qwen --llm race_white_black 30
python switch_llm.py --preset mixtral --llm economic_high_working 30
```

### Troubleshooting Connectivity
```bash
# Test specific model if experiments are failing
python switch_llm.py --preset your-model --test

# Check current default configuration
python update_default_llm.py --show
```

## 🔧 Advanced Usage

### Pass Additional Arguments
```bash
# Run experiments with custom parameters
python switch_llm.py --preset mixtral --run-experiments --extra-args --baseline-runs 50 --llm-runs 20

# Quick test with specific scenarios
python switch_llm.py --preset qwen --quick-test --extra-args --scenarios baseline race_white_black
```

### Using with Other Scripts
You can still use the original command-line approach:

```bash
# Original way (still works)
python run_experiments.py --llm-model "gpt-4" --llm-url "https://api.openai.com/v1/chat/completions" --llm-api-key "your-key"

# New easy way
python switch_llm.py --preset gpt4 --run-experiments
```

## ✅ Current Status

- **Default Model**: Mixtral 8x22B (high performance, ready to use)
- **Ready Models**: Mixtral, Qwen (both via Binghamton University)
- **Available Models**: GPT-4, Claude, local models (need API keys/setup)
- **Fallback**: All scripts still accept manual `--llm-model`, `--llm-url`, `--llm-api-key` arguments

The new switching tools make it much easier to test different models and find the best one for your research needs!