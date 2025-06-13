#!/usr/bin/env python3
"""
Update Default LLM Configuration
Updates config.py with a new default LLM model
"""

import argparse
import sys
from llm_presets import LLM_PRESETS, list_presets, validate_preset
import re

def update_config_file(preset_name):
    """Update config.py with new default LLM settings"""
    valid, message = validate_preset(preset_name)
    if not valid:
        print(f"❌ Error: {message}")
        return False
    
    config = LLM_PRESETS[preset_name]
    
    # Read current config.py
    try:
        with open('config.py', 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print("❌ Error: config.py not found")
        return False
    
    # Update the values
    content = re.sub(
        r'OLLAMA_MODEL = "[^"]*"',
        f'OLLAMA_MODEL = "{config["model"]}"',
        content
    )
    content = re.sub(
        r'OLLAMA_URL = "[^"]*"',
        f'OLLAMA_URL = "{config["url"]}"',
        content
    )
    content = re.sub(
        r'OLLAMA_API_KEY = "[^"]*"',
        f'OLLAMA_API_KEY = "{config["api_key"]}"',
        content
    )
    
    # Write back to file
    try:
        with open('config.py', 'w') as f:
            f.write(content)
        
        print(f"✅ Successfully updated config.py with {config['name']}")
        print(f"📋 Model: {config['model']}")
        print(f"🌐 URL: {config['url']}")
        print(f"🔑 API Key: {config['api_key'][:10]}..." if len(config['api_key']) > 10 else config['api_key'])
        print("\n💡 This will be the new default for all experiments unless overridden with command-line arguments")
        return True
        
    except Exception as e:
        print(f"❌ Error writing to config.py: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Update Default LLM Configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                    # List all available presets
  %(prog)s --set mixtral             # Set Mixtral as default
  %(prog)s --set gpt4                # Set GPT-4 as default
  %(prog)s --show                    # Show current default settings
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--list', action='store_true',
                      help='List all available LLM presets')
    group.add_argument('--set', type=str, metavar='PRESET',
                      help='Set a new default LLM preset')
    group.add_argument('--show', action='store_true',
                      help='Show current default configuration')
    
    args = parser.parse_args()
    
    if args.list:
        list_presets()
        return
    
    if args.show:
        try:
            with open('config.py', 'r') as f:
                content = f.read()
            
            # Extract current values
            model_match = re.search(r'OLLAMA_MODEL = "([^"]*)"', content)
            url_match = re.search(r'OLLAMA_URL = "([^"]*)"', content)
            key_match = re.search(r'OLLAMA_API_KEY = "([^"]*)"', content)
            
            print("🔍 Current Default LLM Configuration:")
            print("=" * 40)
            print(f"Model: {model_match.group(1) if model_match else 'Not found'}")
            print(f"URL: {url_match.group(1) if url_match else 'Not found'}")
            print(f"API Key: {key_match.group(1)[:10] + '...' if key_match and len(key_match.group(1)) > 10 else key_match.group(1) if key_match else 'Not found'}")
            
        except FileNotFoundError:
            print("❌ Error: config.py not found")
        return
    
    if args.set:
        success = update_config_file(args.set)
        if not success:
            print("\nUse --list to see available presets")
            sys.exit(1)

if __name__ == "__main__":
    main()