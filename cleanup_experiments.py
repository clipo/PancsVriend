#!/usr/bin/env python3
"""
Experiment Cleanup Utility
Helps clean up old, incomplete, or test experiments to keep the workspace organized.
"""

import os
import shutil
import glob
import json
from pathlib import Path
from datetime import datetime, timedelta
import argparse

def analyze_experiment_status(exp_path, deletion_mode='safe'):
    """Analyze experiment status and determine if it's complete/incomplete/failed"""
    exp_path = Path(exp_path)
    
    info = {
        'path': str(exp_path),
        'name': exp_path.name,
        'type': 'unknown',
        'status': 'unknown',
        'size_mb': 0,
        'last_modified': None,
        'files_count': 0,
        'can_delete': False,
        'can_delete_aggressive': False,
        'can_delete_force': False,
        'reason': ''
    }
    
    if not exp_path.exists():
        return info
    
    # Calculate directory size and file count
    total_size = 0
    files_count = 0
    for root, dirs, files in os.walk(exp_path):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)
                files_count += 1
    
    info['size_mb'] = total_size / (1024 * 1024)
    info['files_count'] = files_count
    info['last_modified'] = datetime.fromtimestamp(exp_path.stat().st_mtime)
    
    # Determine experiment type (order matters - check llm_ before baseline_)
    if 'comprehensive_study' in exp_path.name:
        info['type'] = 'comprehensive_study'
        info.update(analyze_comprehensive_study(exp_path))
    elif 'llm_' in exp_path.name:
        info['type'] = 'llm'
        info.update(analyze_llm_experiment(exp_path))
    elif 'baseline_' in exp_path.name:
        info['type'] = 'baseline'
        info.update(analyze_baseline_experiment(exp_path))
    elif 'design_space_' in exp_path.name:
        info['type'] = 'design_space'
        info.update(analyze_design_space(exp_path))
    
    # Determine deletion eligibility based on different modes
    # Use the most recent timestamp available (progress file or directory modification)
    effective_timestamp = info['last_modified']
    
    # For LLM experiments, try to use the progress file timestamp if it's more recent
    if info['type'] == 'llm':
        progress_file = exp_path / 'progress_realtime.json'
        if progress_file.exists():
            try:
                import json
                with open(progress_file) as f:
                    progress_data = json.load(f)
                timestamp_str = progress_data.get('timestamp', '')
                if timestamp_str:
                    progress_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    if progress_time.tzinfo:
                        progress_time = progress_time.replace(tzinfo=None)
                    # Use progress timestamp for age calculation
                    effective_timestamp = progress_time
            except:
                pass
    
    age_days = (datetime.now() - effective_timestamp).days
    age_hours = (datetime.now() - effective_timestamp).total_seconds() / 3600
    
    # Safe mode (conservative)
    if info['status'] == 'failed' and age_days > 1:
        info['can_delete'] = True
        info['reason'] = f'Failed experiment older than 1 day'
    elif info['status'] == 'incomplete' and age_days > 3:
        info['can_delete'] = True
        info['reason'] = f'Incomplete experiment older than 3 days'
    elif info['status'] == 'empty':
        info['can_delete'] = True
        info['reason'] = f'Empty experiment directory'
    elif info['size_mb'] < 0.1 and age_days > 1:
        info['can_delete'] = True
        info['reason'] = f'Very small experiment (<0.1MB) older than 1 day'
    
    # Aggressive mode (less conservative)
    if info['status'] == 'failed' and age_hours > 1:
        info['can_delete_aggressive'] = True
    elif info['status'] == 'incomplete' and age_hours > 0.5:  # Very aggressive: 30 minutes for incomplete
        info['can_delete_aggressive'] = True
    elif info['status'] == 'empty':
        info['can_delete_aggressive'] = True
    elif info['size_mb'] < 1.0 and age_hours > 2:
        info['can_delete_aggressive'] = True
    
    # Force mode (delete anything that's not currently running)
    if info['status'] != 'running':
        info['can_delete_force'] = True
    
    return info

def analyze_comprehensive_study(exp_path):
    """Analyze comprehensive study status"""
    info = {'status': 'incomplete', 'progress': 0, 'total_experiments': 0}
    
    # Check for progress files
    progress_files = list(exp_path.glob('**/progress_*.json'))
    if progress_files:
        try:
            latest_progress = max(progress_files, key=lambda x: x.stat().st_mtime)
            with open(latest_progress) as f:
                progress_data = json.load(f)
            
            completed = progress_data.get('completed', 0)
            total = progress_data.get('total_planned', 0)
            info['progress'] = completed
            info['total_experiments'] = total
            
            if completed >= total and total > 0:
                info['status'] = 'completed'
            elif completed > 0:
                info['status'] = 'running'
        except:
            pass
    
    # Check if directory is essentially empty
    if info['total_experiments'] == 0 and len(list(exp_path.glob('**/*'))) < 5:
        info['status'] = 'empty'
    
    return info

def analyze_baseline_experiment(exp_path):
    """Analyze baseline experiment status"""
    info = {'status': 'incomplete'}
    
    # Check for results file
    results_file = exp_path / 'results.json'
    if results_file.exists():
        try:
            with open(results_file) as f:
                results = json.load(f)
            if results.get('runs_completed', 0) > 0:
                info['status'] = 'completed'
        except:
            info['status'] = 'failed'
    
    # Check if has minimal files
    if len(list(exp_path.glob('*'))) < 2:
        info['status'] = 'empty'
    
    return info

def analyze_llm_experiment(exp_path):
    """Analyze LLM experiment status"""
    info = {'status': 'incomplete'}
    
    # Check for progress file
    progress_file = exp_path / 'progress_realtime.json'
    if progress_file.exists():
        try:
            with open(progress_file) as f:
                progress = json.load(f)
            
            status = progress.get('status', 'unknown')
            current_run = progress.get('current_run', 0)
            total_runs = progress.get('total_runs', 0)
            timestamp_str = progress.get('timestamp', '')
            
            # Check if timestamp is recent (within last 5 minutes) to determine if truly running
            is_recently_active = False
            if timestamp_str:
                try:
                    progress_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    # Remove timezone info for comparison if present
                    if progress_time.tzinfo:
                        progress_time = progress_time.replace(tzinfo=None)
                    time_diff = (datetime.now() - progress_time).total_seconds()
                    is_recently_active = time_diff < 300  # 5 minutes
                except:
                    pass
            
            if status == 'completed' or (current_run >= total_runs and total_runs > 0):
                info['status'] = 'completed'
            elif status == 'running' and is_recently_active:
                info['status'] = 'running'
            elif status == 'running' and not is_recently_active:
                info['status'] = 'incomplete'  # Stale "running" status
            elif status == 'failed':
                info['status'] = 'failed'
        except:
            info['status'] = 'failed'
    
    # Check if has minimal files
    if len(list(exp_path.glob('*'))) < 2:
        info['status'] = 'empty'
    
    return info

def analyze_design_space(exp_path):
    """Analyze design space experiment status"""
    info = {'status': 'incomplete'}
    
    # Similar logic to comprehensive study
    # This is a placeholder - adjust based on actual design space structure
    if len(list(exp_path.glob('*'))) < 2:
        info['status'] = 'empty'
    
    return info

def find_all_experiments(deletion_mode='safe'):
    """Find all experiment directories"""
    experiments = []
    
    patterns = [
        "comprehensive_study_*",
        "experiments/baseline_*",
        "experiments/llm_*",
        "design_space_*"
    ]
    
    for pattern in patterns:
        for exp_path in glob.glob(pattern):
            if os.path.isdir(exp_path):
                experiments.append(analyze_experiment_status(exp_path, deletion_mode))
    
    return experiments

def display_experiments(experiments, show_all=False):
    """Display experiment summary"""
    print("\n" + "="*80)
    print("🧹 EXPERIMENT CLEANUP ANALYZER")
    print("="*80)
    
    if not experiments:
        print("No experiments found.")
        return
    
    # Sort by last modified (newest first)
    experiments.sort(key=lambda x: x['last_modified'] or datetime(1900, 1, 1), reverse=True)
    
    # Group by status
    by_status = {}
    total_size = 0
    
    for exp in experiments:
        status = exp['status']
        if status not in by_status:
            by_status[status] = []
        by_status[status].append(exp)
        total_size += exp['size_mb']
    
    print(f"\n📊 SUMMARY:")
    print(f"Total experiments: {len(experiments)}")
    print(f"Total disk usage: {total_size:.1f} MB")
    print()
    
    status_emojis = {
        'completed': '✅',
        'running': '🔄',
        'incomplete': '⚠️',
        'failed': '❌',
        'empty': '📭',
        'unknown': '❓'
    }
    
    for status, exps in by_status.items():
        emoji = status_emojis.get(status, '❓')
        deletable = sum(1 for e in exps if e['can_delete'])
        print(f"{emoji} {status.upper()}: {len(exps)} experiments ({deletable} can be deleted)")
    
    if show_all:
        print(f"\n📋 DETAILED LIST:")
        for i, exp in enumerate(experiments, 1):
            age_days = (datetime.now() - exp['last_modified']).days
            status_emoji = status_emojis.get(exp['status'], '❓')
            delete_emoji = "🗑️" if exp['can_delete'] else "🔒"
            
            print(f"{i:2d}. {delete_emoji} {status_emoji} {exp['name']}")
            print(f"    Type: {exp['type']} | Size: {exp['size_mb']:.1f}MB | Age: {age_days} days")
            print(f"    Status: {exp['status']} | Files: {exp['files_count']}")
            if exp['can_delete']:
                print(f"    💡 Can delete: {exp['reason']}")
            print()

def interactive_cleanup(experiments, mode='safe'):
    """Interactive cleanup process"""
    if mode == 'safe':
        deletable = [exp for exp in experiments if exp['can_delete']]
        mode_name = "SAFE MODE"
    elif mode == 'aggressive':
        deletable = [exp for exp in experiments if exp['can_delete_aggressive']]
        mode_name = "AGGRESSIVE MODE"
    elif mode == 'force':
        deletable = [exp for exp in experiments if exp['can_delete_force']]
        mode_name = "FORCE MODE (⚠️ DANGER)"
    
    if not deletable:
        print(f"\n✨ No experiments marked for deletion in {mode_name}.")
        return
    
    print(f"\n🗑️ DELETION CANDIDATES ({len(deletable)} experiments):")
    total_size = sum(exp['size_mb'] for exp in deletable)
    print(f"Total size to be freed: {total_size:.1f} MB")
    print("-" * 60)
    
    for i, exp in enumerate(deletable, 1):
        age_days = (datetime.now() - exp['last_modified']).days
        print(f"{i:2d}. {exp['name']}")
        print(f"    Reason: {exp['reason']}")
        print(f"    Size: {exp['size_mb']:.1f}MB | Age: {age_days} days")
        print()
    
    # Confirmation
    response = input("Do you want to delete these experiments? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        print("\n🗑️ Deleting experiments...")
        deleted_count = 0
        freed_space = 0
        
        for exp in deletable:
            try:
                shutil.rmtree(exp['path'])
                print(f"   ✅ Deleted: {exp['name']}")
                deleted_count += 1
                freed_space += exp['size_mb']
            except Exception as e:
                print(f"   ❌ Failed to delete {exp['name']}: {e}")
        
        print(f"\n✨ Cleanup complete!")
        print(f"   Deleted: {deleted_count} experiments")
        print(f"   Freed space: {freed_space:.1f} MB")
    else:
        print("\n❌ Cleanup cancelled.")

def main():
    parser = argparse.ArgumentParser(description="Clean up old and incomplete experiments")
    parser.add_argument('--list', action='store_true', help='List all experiments with details')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted without actually deleting')
    parser.add_argument('--auto', action='store_true', help='Automatically delete candidates without confirmation')
    
    # Deletion mode options
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--safe', action='store_true', default=True, 
                           help='Safe mode: Conservative deletion (default)')
    mode_group.add_argument('--aggressive', action='store_true', 
                           help='Aggressive mode: Delete incomplete experiments older than 12 hours')
    mode_group.add_argument('--force', action='store_true', 
                           help='Force mode: Delete all non-running experiments (DANGEROUS)')
    
    args = parser.parse_args()
    
    # Determine deletion mode
    if args.force:
        mode = 'force'
        print("⚠️  WARNING: FORCE MODE ENABLED - Will delete ALL non-running experiments!")
        confirm = input("Type 'YES' to continue with force mode: ")
        if confirm != 'YES':
            print("❌ Force mode cancelled.")
            return
    elif args.aggressive:
        mode = 'aggressive'
        print("⚡ AGGRESSIVE MODE - Will delete experiments older than 12 hours")
    else:
        mode = 'safe'
    
    experiments = find_all_experiments(mode)
    
    if args.list:
        display_experiments(experiments, show_all=True)
        show_deletion_summary(experiments, mode)
    elif args.dry_run:
        display_experiments(experiments, show_all=False)
        show_deletion_summary(experiments, mode)
    elif args.auto:
        display_experiments(experiments, show_all=False)
        auto_cleanup(experiments, mode)
    else:
        display_experiments(experiments, show_all=False)
        interactive_cleanup(experiments, mode)

def show_deletion_summary(experiments, mode):
    """Show what would be deleted in the current mode"""
    if mode == 'safe':
        deletable = [exp for exp in experiments if exp['can_delete']]
        mode_name = "SAFE MODE"
    elif mode == 'aggressive':
        deletable = [exp for exp in experiments if exp['can_delete_aggressive']]
        mode_name = "AGGRESSIVE MODE"
    elif mode == 'force':
        deletable = [exp for exp in experiments if exp['can_delete_force']]
        mode_name = "FORCE MODE"
    
    if deletable:
        total_size = sum(exp['size_mb'] for exp in deletable)
        print(f"\n🔍 {mode_name} - Would delete {len(deletable)} experiments ({total_size:.1f}MB):")
        for exp in deletable:
            age_hours = (datetime.now() - exp['last_modified']).total_seconds() / 3600
            print(f"   🗑️ {exp['name']} ({exp['status']}, {age_hours:.1f}h old)")
    else:
        print(f"\n✨ {mode_name} - No experiments marked for deletion.")

def auto_cleanup(experiments, mode):
    """Automatically delete experiments in the specified mode"""
    if mode == 'safe':
        deletable = [exp for exp in experiments if exp['can_delete']]
    elif mode == 'aggressive':
        deletable = [exp for exp in experiments if exp['can_delete_aggressive']]
    elif mode == 'force':
        deletable = [exp for exp in experiments if exp['can_delete_force']]
    
    if deletable:
        total_size = sum(exp['size_mb'] for exp in deletable)
        print(f"\n🤖 AUTO CLEANUP - Deleting {len(deletable)} experiments ({total_size:.1f}MB)...")
        deleted_count = 0
        freed_space = 0
        
        for exp in deletable:
            try:
                shutil.rmtree(exp['path'])
                print(f"   ✅ Deleted: {exp['name']}")
                deleted_count += 1
                freed_space += exp['size_mb']
            except Exception as e:
                print(f"   ❌ Failed: {exp['name']} - {e}")
        
        print(f"\n✨ Auto cleanup complete! Deleted {deleted_count} experiments, freed {freed_space:.1f}MB")
    else:
        print(f"\n✨ No experiments to delete in current mode.")

if __name__ == "__main__":
    main()