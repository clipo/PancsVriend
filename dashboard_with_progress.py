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

def display_progress_card(progress_data, experiment_name):
    """Display a progress card for an experiment"""
    if not progress_data:
        return
    
    # Calculate overall progress
    run_progress = progress_data.get('run_progress_percent', 0)
    step_progress = progress_data.get('step_progress_percent', 0)
    
    # Create columns for the card
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="progress-card">
            <h4>{experiment_name}</h4>
            <p><strong>Scenario:</strong> {progress_data.get('scenario', 'Unknown')}</p>
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
    """Find all active experiment directories"""
    experiments = []
    
    # Look for comprehensive study directories
    for study_dir in glob.glob("comprehensive_study_*/"):
        if Path(f"{study_dir}/progress_realtime.json").exists():
            experiments.append({
                'name': study_dir.rstrip('/'),
                'path': study_dir.rstrip('/'),
                'type': 'comprehensive_study'
            })
    
    # Look for individual experiment directories
    for exp_dir in glob.glob("*/progress_realtime.json"):
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
            display_progress_card(progress_data, exp['name'])
            st.markdown("---")
    
    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(10)
        st.rerun()
    
    # Show instructions
    with st.expander("ℹ️ About This Dashboard"):
        st.markdown("""
        This dashboard shows real-time progress for LLM experiments:
        
        - **Run Progress**: Shows which run (out of total) is currently executing
        - **Step Progress**: Shows which step (out of 1000) within the current run
        - **Status**: Current experiment status (running, completed, failed)
        - **Auto-refresh**: Updates every 10 seconds when enabled
        
        The progress data is updated every 10 simulation steps to balance 
        responsiveness with performance.
        """)

if __name__ == "__main__":
    main()