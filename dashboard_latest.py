#!/usr/bin/env python3
"""
Launch dashboard for the latest running experiment
"""

import streamlit as st
import glob
import os
from pathlib import Path

# Import the main dashboard
from dashboard import *

# Override the sidebar to auto-detect latest experiment
def setup_sidebar_with_auto_detect():
    """Enhanced sidebar that finds latest experiment"""
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Find latest experiment directories
        comp_dirs = glob.glob("comprehensive_study_*/llm_results")
        design_dirs = glob.glob("design_space_exploration")
        all_dirs = sorted(comp_dirs + design_dirs, key=lambda x: Path(x).stat().st_mtime if Path(x).exists() else 0, reverse=True)
        
        if all_dirs:
            # Default to most recent
            default_dir = all_dirs[0]
            st.success(f"📍 Auto-detected: {default_dir}")
        else:
            default_dir = "design_space_exploration"
            st.warning("No experiments found. Using default directory.")
        
        # Directory selection with auto-detected default
        experiment_dir = st.selectbox(
            "Experiment Directory",
            options=all_dirs if all_dirs else [default_dir],
            index=0,
            help="Select experiment results directory"
        )
        
        # Show directory info
        if Path(experiment_dir).exists():
            progress_files = glob.glob(f"{experiment_dir}/progress_*.json")
            if progress_files:
                latest_progress = max(progress_files, key=os.path.getmtime)
                mtime = Path(latest_progress).stat().st_mtime
                from datetime import datetime
                last_update = datetime.fromtimestamp(mtime).strftime("%H:%M:%S")
                st.caption(f"Last update: {last_update}")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        refresh_interval = st.slider("Refresh interval (seconds)", 5, 60, 10)
        
        return experiment_dir, auto_refresh, refresh_interval

# Monkey-patch the main dashboard
if __name__ == "__main__":
    st.set_page_config(
        page_title="Schelling Experiments - Latest",
        page_icon="🏘️",
        layout="wide"
    )
    
    st.title("🏘️ Schelling Segregation Model Dashboard")
    st.markdown("**Monitoring latest experiment automatically**")
    
    # Get configuration from enhanced sidebar
    experiment_dir, auto_refresh, refresh_interval = setup_sidebar_with_auto_detect()
    
    # Load data using the selected directory
    df = load_experiment_data(experiment_dir)
    progress_data = load_progress_data(experiment_dir)
    metrics_df = load_segregation_metrics(experiment_dir)
    
    # Create the standard dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "⏱️ Real-time Progress", "📈 Analysis", "🔬 Experiment Details"])
    
    with tab1:
        st.header("Experiment Overview")
        
        if progress_data:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Experiments", progress_data.get('total_planned', 0))
            
            with col2:
                st.metric("Completed", progress_data.get('completed', 0),
                         delta=f"{progress_data.get('successful', 0)} successful")
            
            with col3:
                st.metric("Failed", progress_data.get('failed', 0), delta_color="inverse")
            
            with col4:
                percent = progress_data.get('progress_percent', 0)
                st.metric("Progress", f"{percent:.1f}%")
        
        # Progress gauge
        st.subheader("Overall Progress")
        progress_fig = create_progress_gauge(progress_data)
        if progress_fig:
            st.plotly_chart(progress_fig, use_container_width=True)
    
    # Auto-refresh logic
    if auto_refresh and progress_data and progress_data.get('completed', 0) < progress_data.get('total_planned', 1):
        import time
        time.sleep(refresh_interval)
        st.experimental_rerun()