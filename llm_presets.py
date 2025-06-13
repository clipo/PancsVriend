#!/usr/bin/env python3
"""
LLM Presets and Easy Switching
Provides predefined LLM configurations and easy switching between models
"""

LLM_PRESETS = {
    "mixtral": {
        "name": "Mixtral 8x22B (Default)",
        "model": "mixtral:8x22b-instruct",
        "url": "https://chat.binghamton.edu/api/chat/completions",
        "api_key": "sk-571df6eec7f5495faef553ab5cb2c67a",
        "description": "High-performance open model via Binghamton University"
    },
    "qwen": {
        "name": "Qwen 2.5 Coder 32B",
        "model": "qwen2.5-coder:32B",
        "url": "https://chat.binghamton.edu/api/chat/completions",
        "api_key": "sk-571df6eec7f5495faef553ab5cb2c67a",
        "description": "Code-optimized model, good for structured responses"
    },
    "gpt4": {
        "name": "GPT-4",
        "model": "gpt-4",
        "url": "https://api.openai.com/v1/chat/completions",
        "api_key": "your-openai-key-here",
        "description": "OpenAI's flagship model (requires OpenAI API key)"
    },
    "gpt4o": {
        "name": "GPT-4o",
        "model": "gpt-4o",
        "url": "https://api.openai.com/v1/chat/completions",
        "api_key": "your-openai-key-here",
        "description": "OpenAI's latest optimized model (requires OpenAI API key)"
    },
    "gpt35": {
        "name": "GPT-3.5 Turbo",
        "model": "gpt-3.5-turbo",
        "url": "https://api.openai.com/v1/chat/completions",
        "api_key": "your-openai-key-here",
        "description": "Fast and cost-effective OpenAI model (requires OpenAI API key)"
    },
    "claude-sonnet": {
        "name": "Claude 3 Sonnet",
        "model": "claude-3-sonnet-20240229",
        "url": "https://api.anthropic.com/v1/messages",
        "api_key": "your-anthropic-key-here",
        "description": "Anthropic's balanced model (requires Anthropic API key)"
    },
    "claude-haiku": {
        "name": "Claude 3 Haiku",
        "model": "claude-3-haiku-20240307",
        "url": "https://api.anthropic.com/v1/messages",
        "api_key": "your-anthropic-key-here",
        "description": "Anthropic's fast and efficient model (requires Anthropic API key)"
    },
    "local-llama": {
        "name": "Local Llama (Ollama)",
        "model": "llama2",
        "url": "http://localhost:11434/api/chat/completions",
        "api_key": "not-required",
        "description": "Local Llama model via Ollama (requires local Ollama setup)"
    }
}

def get_preset_names():
    """Get list of available preset names"""
    return list(LLM_PRESETS.keys())

def get_preset_info(preset_name):
    """Get information about a specific preset"""
    return LLM_PRESETS.get(preset_name, None)

def list_presets():
    """Print all available presets with descriptions"""
    print("\n📋 Available LLM Presets:")
    print("=" * 50)
    for key, config in LLM_PRESETS.items():
        print(f"🔹 {key:15} - {config['name']}")
        print(f"   {config['description']}")
        print(f"   Model: {config['model']}")
        print()

def get_preset_args(preset_name):
    """Get command line arguments for a preset"""
    config = LLM_PRESETS.get(preset_name)
    if not config:
        return None
    
    return {
        'llm_model': config['model'],
        'llm_url': config['url'],
        'llm_api_key': config['api_key']
    }

def validate_preset(preset_name):
    """Check if a preset is valid and has required API keys"""
    config = LLM_PRESETS.get(preset_name)
    if not config:
        return False, f"Unknown preset: {preset_name}"
    
    # Check for placeholder API keys
    if "your-" in config['api_key'] and config['api_key'] != "not-required":
        return False, f"Preset '{preset_name}' requires a valid API key. Please update the API key in llm_presets.py"
    
    return True, "Preset is valid"

if __name__ == "__main__":
    list_presets()