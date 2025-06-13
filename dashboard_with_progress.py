#!/usr/bin/env python3
"""
Enhanced Dashboard with Real-Time Progress Monitoring
Shows run-by-run and step-by-step progress for LLM experiments
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import glob
from pathlib import Path
from datetime import datetime
import time
import subprocess
import sys

# Page configuration
st.set_page_config(
    page_title="Schelling Dashboard - Live Progress",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .progress-card {
        background-color: #e1f5fe;
        padding: 15px;
        border-radius: 8px;
        margin: 5px 0;
        border-left: 4px solid #1976d2;
    }
    .status-running { color: #1976d2; font-weight: bold; }
    .status-completed { color: #388e3c; font-weight: bold; }
    .status-failed { color: #d32f2f; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def load_progress_data(experiment_dir):
    """Load real-time progress data from experiment directory"""
    progress_file = f"{experiment_dir}/progress_realtime.json"
    if Path(progress_file).exists():
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading progress: {e}")
            return None
    return None

def load_experiment_set_progress(experiment_dir):
    """Load experiment set progress data"""
    # Look for progress files in the experiment directory
    progress_files = glob.glob(f"{experiment_dir}/progress_*.json")
    if progress_files:
        # Get the most recent progress file
        latest_progress = max(progress_files, key=lambda x: Path(x).stat().st_mtime)
        try:
            with open(latest_progress, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading experiment set progress: {e}")
    return None

def load_current_experiment_status(experiment_dir):
    """Load current experiment status information"""
    # Look for status files in logs directory
    logs_dir = Path(experiment_dir) / "logs"
    if logs_dir.exists():
        status_files = list(logs_dir.glob("*_status.json"))
        if status_files:
            # Find the most recently modified status file
            latest_status = max(status_files, key=lambda x: x.stat().st_mtime)
            try:
                with open(latest_status, 'r') as f:
                    return json.load(f)
            except Exception as e:
                st.error(f"Error loading experiment status: {e}")
    return None

def display_enhanced_progress_card(progress_data, experiment_name, experiment_dir):
    """Display an enhanced progress card with experiment set information"""
    if not progress_data:
        return
    
    # Load additional data
    set_progress = load_experiment_set_progress(experiment_dir)
    current_exp_status = load_current_experiment_status(experiment_dir)
    
    # Calculate overall progress
    run_progress = progress_data.get('run_progress_percent', 0)
    step_progress = progress_data.get('step_progress_percent', 0)
    
    # Create header with experiment set information
    if set_progress:
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #1976d2;">
            <h3>🔬 {experiment_name}</h3>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4 style="color: #1976d2; margin: 0;">Experiment Set Progress</h4>
                    <p style="margin: 5px 0; font-size: 18px;"><strong>{set_progress.get('completed', 0)} of {set_progress.get('total_planned', 0)} experiments completed</strong></p>
                    <p style="margin: 0; color: #666;">Overall Progress: {set_progress.get('progress_percent', 0):.1f}%</p>
                </div>
                <div style="width: 200px;">
                    <div style="background-color: #e0e0e0; border-radius: 10px; height: 20px; overflow: hidden;">
                        <div style="background-color: #1976d2; height: 100%; width: {set_progress.get('progress_percent', 0)}%;"></div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Show current experiment details
    if current_exp_status:
        st.markdown(f"""
        <div style="background-color: #fff3e0; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #ff9800;">
            <h4 style="color: #ff9800; margin-top: 0;">📋 Current Experiment: {current_exp_status.get('experiment_id', 'Unknown')}</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <div>
                    <p><strong>Description:</strong> {current_exp_status.get('description', 'Unknown')}</p>
                    <p><strong>Agent Type:</strong> {current_exp_status.get('agent_type', 'Unknown')}</p>
                    <p><strong>Scenario:</strong> {current_exp_status.get('scenario', 'Unknown')}</p>
                </div>
                <div>
                    <p><strong>Grid Size:</strong> {current_exp_status.get('grid_size', 'Unknown')}x{current_exp_status.get('grid_size', 'Unknown')}</p>
                    <p><strong>Agents:</strong> {current_exp_status.get('num_type_a', 0)} + {current_exp_status.get('num_type_b', 0)} = {current_exp_status.get('num_type_a', 0) + current_exp_status.get('num_type_b', 0)}</p>
                    <p><strong>Total Runs:</strong> {current_exp_status.get('n_runs', 'Unknown')}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Create columns for the detailed progress
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="progress-card">
            <h4>🏃‍♂️ Current Run Progress</h4>
            <p><strong>Run:</strong> {progress_data.get('current_run', 0)} of {progress_data.get('total_runs', 0)}</p>
            <p><strong>Step:</strong> {progress_data.get('current_step', 0)} of {progress_data.get('max_steps', 0)}</p>
            <p><strong>Status:</strong> <span class="status-{progress_data.get('status', 'unknown')}">{progress_data.get('status', 'Unknown').upper()}</span></p>
            <p><strong>Last Update:</strong> {progress_data.get('timestamp', 'Unknown')[:19].replace('T', ' ')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Run progress
        st.metric("Run Progress", f"{run_progress:.1f}%")
        st.progress(run_progress / 100)
    
    with col3:
        # Step progress (for current run)
        st.metric("Step Progress", f"{step_progress:.1f}%")
        st.progress(step_progress / 100)

def find_active_experiments():
    """Find all active experiment directories (only those with recent activity)"""
    experiments = []
    
    def is_recently_active(progress_file_path):
        """Check if experiment has been active within last 5 minutes"""
        try:
            with open(progress_file_path, 'r') as f:
                progress_data = json.load(f)
            
            timestamp_str = progress_data.get('timestamp', '')
            if timestamp_str:
                progress_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                if progress_time.tzinfo:
                    progress_time = progress_time.replace(tzinfo=None)
                time_diff = (datetime.now() - progress_time).total_seconds()
                return time_diff < 300  # 5 minutes
        except:
            pass
        return False
    
    # Look for comprehensive study directories
    for study_dir in glob.glob("comprehensive_study_*/"):
        progress_file = f"{study_dir}/progress_realtime.json"
        if Path(progress_file).exists() and is_recently_active(progress_file):
            experiments.append({
                'name': study_dir.rstrip('/'),
                'path': study_dir.rstrip('/'),
                'type': 'comprehensive_study'
            })
    
    # Look for individual experiment directories
    for exp_dir in glob.glob("**/progress_realtime.json", recursive=True):
        if is_recently_active(exp_dir):
            exp_path = exp_dir.replace('/progress_realtime.json', '')
            if not exp_path.startswith('comprehensive_study_'):
                experiments.append({
                    'name': exp_path,
                    'path': exp_path,
                    'type': 'individual'
                })
    
    return experiments

def main():
    st.title("🏘️ Schelling Segregation Dashboard - Live Progress")
    st.markdown("Real-time monitoring of LLM agent experiments")
    
    # Create tabs for different functionality
    tab1, tab2 = st.tabs(["📊 Live Progress", "🧹 Cleanup"])
    
    with tab1:
        show_progress_tab()
    
    with tab2:
        show_cleanup_tab()

def show_progress_tab():
    """Show the live progress monitoring tab"""
    # Auto-refresh controls
    col1, col2 = st.columns([3, 1])
    with col1:
        auto_refresh = st.checkbox("Auto-refresh every 10 seconds", value=True)
    with col2:
        if st.button("🔄 Refresh Now"):
            st.rerun()
    
    # Find active experiments
    experiments = find_active_experiments()
    
    if not experiments:
        st.warning("No active experiments found. Start an experiment to see progress here.")
        st.info("Looking for experiments with progress_realtime.json files...")
        return
    
    st.markdown(f"**Found {len(experiments)} active experiment(s)**")
    
    # Display progress for each experiment
    for exp in experiments:
        progress_data = load_progress_data(exp['path'])
        if progress_data:
            display_enhanced_progress_card(progress_data, exp['name'], exp['path'])
            st.markdown("---")
    
    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(10)
        st.rerun()
    
    # Show instructions
    with st.expander("ℹ️ About This Dashboard"):
        st.markdown("""
        This dashboard shows comprehensive real-time progress for LLM experiments:
        
        **Experiment Set Level:**
        - **Total Progress**: Shows how many experiments in the set are completed (e.g., "20 of 40 experiments")
        - **Overall Percentage**: Shows the completion percentage for the entire experiment set
        
        **Current Experiment Level:**
        - **Experiment ID**: Which specific experiment is running (e.g., exp_0015)
        - **Configuration**: Agent type, scenario, grid size, and number of agents
        - **Description**: Human-readable description of the experiment
        
        **Individual Run Level:**
        - **Run Progress**: Shows which run (out of total) is currently executing within the current experiment
        - **Step Progress**: Shows which step (out of 1000) within the current run
        - **Status**: Current run status (running, completed, failed)
        
        **Technical Details:**
        - **Auto-refresh**: Updates every 10 seconds when enabled
        - **Data Source**: Combines experiment set progress, experiment status, and real-time run progress
        - **Update Frequency**: Progress data is updated every 10 simulation steps
        """)

def show_cleanup_tab():
    """Show the experiment cleanup tab"""
    st.markdown("### 🧹 Experiment Cleanup Utility")
    st.markdown("Clean up old, incomplete, or failed experiments to keep your workspace organized.")
    
    # Quick info section
    with st.expander("ℹ️ About Cleanup"):
        st.markdown("""
        **What gets marked for deletion:**
        - Failed experiments older than 1 day
        - Incomplete experiments older than 3 days  
        - Empty experiment directories
        - Very small experiments (<0.1MB) older than 1 day
        
        **Safety features:**
        - Only suggests safe deletions based on age and status
        - Never deletes running or recently completed experiments
        - Shows exactly what will be deleted before confirmation
        - Provides dry-run option to preview changes
        """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📋 List All Experiments", use_container_width=True):
            run_cleanup_command("--list")
    
    with col2:
        if st.button("🔍 Dry Run", use_container_width=True):
            run_cleanup_command("--dry-run")
    
    with col3:
        if st.button("🧹 Interactive Cleanup", use_container_width=True):
            st.info("Interactive cleanup must be run from terminal: `python cleanup_experiments.py`")
    
    with col4:
        if st.button("⚡ Auto Cleanup", use_container_width=True):
            if st.checkbox("I understand this will automatically delete safe candidates"):
                run_cleanup_command("--auto")
            else:
                st.warning("Please check the box to confirm auto cleanup")
    
    # Show current cleanup status
    st.markdown("---")
    if st.button("🔄 Refresh Cleanup Status"):
        run_cleanup_command("--dry-run")

def run_cleanup_command(args):
    """Run the cleanup command and display results"""
    try:
        result = subprocess.run(
            [sys.executable, "cleanup_experiments.py", args],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            st.code(result.stdout, language="text")
        else:
            st.error(f"Cleanup command failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        st.error("Cleanup command timed out")
    except Exception as e:
        st.error(f"Error running cleanup: {e}")

if __name__ == "__main__":
    main()