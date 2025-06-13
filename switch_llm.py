#!/usr/bin/env python3
"""
Easy LLM Model Switching
Simple script to switch between different LLM configurations
"""

import argparse
import sys
from llm_presets import LLM_PRESETS, list_presets, validate_preset, get_preset_args
import subprocess

def run_with_preset(preset_name, script, extra_args=None):
    """Run a script with a specific LLM preset"""
    valid, message = validate_preset(preset_name)
    if not valid:
        print(f"❌ Error: {message}")
        return False
    
    preset_args = get_preset_args(preset_name)
    config = LLM_PRESETS[preset_name]
    
    print(f"🚀 Running {script} with {config['name']}")
    print(f"📋 Model: {config['model']}")
    print(f"🌐 URL: {config['url']}")
    print(f"🔑 API Key: {config['api_key'][:10]}..." if len(config['api_key']) > 10 else config['api_key'])
    print()
    
    # Build command
    cmd = [
        'python', script,
        '--llm-model', preset_args['llm_model'],
        '--llm-url', preset_args['llm_url'],
        '--llm-api-key', preset_args['llm_api_key']
    ]
    
    # Add extra arguments
    if extra_args:
        cmd.extend(extra_args)
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Easy LLM Model Switching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                           # List all available presets
  %(prog)s --preset mixtral --test          # Test Mixtral connectivity
  %(prog)s --preset gpt4 --run-experiments  # Run experiments with GPT-4
  %(prog)s --preset qwen --check            # Check Qwen connectivity
  %(prog)s --preset mixtral --quick-test    # Quick experiment test with Mixtral
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--list', action='store_true',
                      help='List all available LLM presets')
    group.add_argument('--preset', type=str, metavar='NAME',
                      help='LLM preset to use (see --list for options)')
    
    parser.add_argument('--test', action='store_true',
                       help='Test LLM connectivity (runs check_llm.py)')
    parser.add_argument('--run-experiments', action='store_true',
                       help='Run full experiments (runs run_experiments.py)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick experiment test (runs run_experiments.py --quick-test)')
    parser.add_argument('--check', action='store_true',
                       help='Check LLM connectivity (same as --test)')
    parser.add_argument('--baseline', type=int, metavar='RUNS',
                       help='Run baseline experiments with specified number of runs')
    parser.add_argument('--llm', nargs=2, metavar=('SCENARIO', 'RUNS'),
                       help='Run LLM experiments with scenario and number of runs')
    
    # Additional arguments to pass through
    parser.add_argument('--extra-args', nargs='*', default=[],
                       help='Additional arguments to pass to the script')
    
    args = parser.parse_args()
    
    if args.list:
        list_presets()
        return
    
    if not args.preset:
        parser.error("--preset is required when not using --list")
    
    # Validate preset
    valid, message = validate_preset(args.preset)
    if not valid:
        print(f"❌ Error: {message}")
        print("\nUse --list to see available presets")
        sys.exit(1)
    
    # Determine which script to run
    success = False
    
    if args.test or args.check:
        success = run_with_preset(args.preset, 'check_llm.py', args.extra_args)
    
    elif args.run_experiments:
        success = run_with_preset(args.preset, 'run_experiments.py', args.extra_args)
    
    elif args.quick_test:
        extra = ['--quick-test'] + args.extra_args
        success = run_with_preset(args.preset, 'run_experiments.py', extra)
    
    elif args.baseline:
        extra = ['--runs', str(args.baseline)] + args.extra_args
        success = run_with_preset(args.preset, 'baseline_runner.py', extra)
    
    elif args.llm:
        scenario, runs = args.llm
        extra = ['--scenario', scenario, '--runs', runs] + args.extra_args
        success = run_with_preset(args.preset, 'llm_runner.py', extra)
    
    else:
        print("❓ No action specified. Use one of: --test, --run-experiments, --quick-test, --baseline, --llm")
        print("   Or use --help for more options")
        sys.exit(1)
    
    if success:
        print("✅ Command completed successfully")
    else:
        print("❌ Command failed")
        sys.exit(1)

if __name__ == "__main__":
    main()