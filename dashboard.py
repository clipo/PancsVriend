#!/usr/bin/env python3
"""
Interactive Dashboard for Schelling Segregation Experiments
Real-time monitoring and analysis of mechanical vs LLM agents
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import glob
from pathlib import Path
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Schelling Segregation Dashboard",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .status-running { color: #1f77b4; }
    .status-completed { color: #2ca02c; }
    .status-failed { color: #d62728; }
</style>
""", unsafe_allow_html=True)

def load_experiment_data(experiment_dir):
    """Load experiment results from directory"""
    results = []
    
    # Find all experiment directories
    exp_dirs = glob.glob(f"{experiment_dir}/experiments/exp_*")
    
    for exp_dir in sorted(exp_dirs):
        try:
            # Load experiment config
            config_file = Path(exp_dir) / "experiment_config.json"
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)
                
                # Load results if available
                results_file = Path(exp_dir) / "results.json"
                if results_file.exists():
                    with open(results_file) as f:
                        result_data = json.load(f)
                    config.update(result_data)
                
                results.append(config)
        except:
            continue
    
    return pd.DataFrame(results)

def load_progress_data(experiment_dir):
    """Load latest progress data"""
    progress_files = glob.glob(f"{experiment_dir}/progress_*.json")
    
    if not progress_files:
        return None
    
    latest_file = max(progress_files, key=lambda x: Path(x).stat().st_mtime)
    
    try:
        with open(latest_file) as f:
            return json.load(f)
    except:
        return None

def load_segregation_metrics(results_dir):
    """Load segregation metrics from simulation results"""
    metrics_data = []
    
    # Find all .npz files
    npz_files = glob.glob(f"{results_dir}/**/*.npz", recursive=True)
    
    for npz_file in npz_files:
        try:
            data = np.load(npz_file)
            
            # Extract metrics over time
            if 'segregation_clusters' in data:
                steps = len(data['segregation_clusters'])
                for step in range(steps):
                    metrics_data.append({
                        'file': Path(npz_file).stem,
                        'step': step,
                        'clusters': data['segregation_clusters'][step] if 'segregation_clusters' in data else 0,
                        'switch_rate': data['segregation_switchrate'][step] if 'segregation_switchrate' in data else 0,
                        'distance': data['segregation_distance'][step] if 'segregation_distance' in data else 0,
                        'mix_deviation': data['segregation_mix'][step] if 'segregation_mix' in data else 0,
                        'share': data['segregation_share'][step] if 'segregation_share' in data else 0,
                        'ghetto_rate': data['segregation_ghetto'][step] if 'segregation_ghetto' in data else 0
                    })
        except:
            continue
    
    return pd.DataFrame(metrics_data)

def create_progress_gauge(progress_data):
    """Create a progress gauge chart"""
    if not progress_data:
        return None
    
    completed = progress_data.get('completed', 0)
    total = progress_data.get('total_planned', 1)
    percent = (completed / total) * 100 if total > 0 else 0
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=percent,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall Progress"},
        delta={'reference': 100},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_experiment_timeline(df):
    """Create timeline visualization of experiments"""
    if df.empty:
        return None
    
    # Prepare data for timeline
    timeline_data = []
    for _, row in df.iterrows():
        if 'start_time' in row and pd.notna(row['start_time']):
            timeline_data.append({
                'Task': f"{row.get('agent_type', 'unknown')} - {row.get('scenario', 'unknown')}",
                'Start': pd.to_datetime(row['start_time']),
                'Finish': pd.to_datetime(row.get('end_time', datetime.now())),
                'Status': row.get('status', 'pending'),
                'Agent Type': row.get('agent_type', 'unknown')
            })
    
    if not timeline_data:
        return None
    
    timeline_df = pd.DataFrame(timeline_data)
    
    # Create Gantt chart
    fig = px.timeline(
        timeline_df,
        x_start="Start",
        x_end="Finish",
        y="Task",
        color="Agent Type",
        title="Experiment Timeline"
    )
    
    fig.update_yaxes(categoryorder="total ascending")
    fig.update_layout(height=400)
    
    return fig

def create_convergence_comparison(metrics_df, selected_experiments):
    """Create convergence comparison chart"""
    if metrics_df.empty or not selected_experiments:
        return None
    
    fig = go.Figure()
    
    for exp in selected_experiments:
        exp_data = metrics_df[metrics_df['file'].str.contains(exp)]
        if not exp_data.empty:
            fig.add_trace(go.Scatter(
                x=exp_data['step'],
                y=exp_data['clusters'],
                mode='lines',
                name=exp,
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title="Segregation Convergence Comparison",
        xaxis_title="Simulation Steps",
        yaxis_title="Number of Clusters",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_agent_type_comparison(df):
    """Create comparison of metrics by agent type"""
    if df.empty:
        return None
    
    # Aggregate metrics by agent type
    metrics = ['convergence_speed', 'final_clusters', 'final_distance']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        return None
    
    fig = make_subplots(
        rows=1, cols=len(available_metrics),
        subplot_titles=available_metrics
    )
    
    for i, metric in enumerate(available_metrics):
        for agent_type in df['agent_type'].unique():
            agent_data = df[df['agent_type'] == agent_type][metric].dropna()
            
            fig.add_trace(
                go.Box(y=agent_data, name=agent_type, showlegend=(i == 0)),
                row=1, col=i+1
            )
    
    fig.update_layout(height=400, title="Agent Type Performance Comparison")
    return fig

def main():
    st.title("🏘️ Schelling Segregation Model Dashboard")
    st.markdown("**Real-time monitoring and analysis of mechanical vs LLM agent experiments**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Directory selection
        experiment_dir = st.text_input(
            "Experiment Directory",
            value="design_space_exploration",
            help="Path to experiment results directory"
        )
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        refresh_interval = st.slider("Refresh interval (seconds)", 5, 60, 10)
        
        # Experiment filters
        st.header("🔍 Filters")
        
        # Load experiment data
        df = load_experiment_data(experiment_dir)
        
        if not df.empty:
            agent_types = st.multiselect(
                "Agent Types",
                options=df['agent_type'].unique() if 'agent_type' in df.columns else [],
                default=[]
            )
            
            scenarios = st.multiselect(
                "Scenarios",
                options=df['scenario'].unique() if 'scenario' in df.columns else [],
                default=[]
            )
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "⏱️ Real-time Progress", "📈 Analysis", "🔬 Experiment Details"])
    
    with tab1:
        st.header("Experiment Overview")
        
        # Load progress data
        progress_data = load_progress_data(experiment_dir)
        
        if progress_data:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Experiments",
                    progress_data.get('total_planned', 0)
                )
            
            with col2:
                st.metric(
                    "Completed",
                    progress_data.get('completed', 0),
                    delta=f"{progress_data.get('successful', 0)} successful"
                )
            
            with col3:
                st.metric(
                    "Failed",
                    progress_data.get('failed', 0),
                    delta_color="inverse"
                )
            
            with col4:
                percent = progress_data.get('progress_percent', 0)
                st.metric(
                    "Progress",
                    f"{percent:.1f}%",
                    delta=f"{progress_data.get('completed', 0) - progress_data.get('failed', 0)} net successful"
                )
        
        # Progress gauge
        st.subheader("Overall Progress")
        progress_fig = create_progress_gauge(progress_data)
        if progress_fig:
            st.plotly_chart(progress_fig, use_container_width=True)
        
        # Experiment timeline
        if not df.empty:
            st.subheader("Experiment Timeline")
            timeline_fig = create_experiment_timeline(df)
            if timeline_fig:
                st.plotly_chart(timeline_fig, use_container_width=True)
    
    with tab2:
        st.header("⏱️ Real-time Progress Monitor")
        
        # Create placeholder for live updates
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        if auto_refresh:
            while True:
                # Load latest progress
                progress_data = load_progress_data(experiment_dir)
                
                if progress_data:
                    # Update progress display
                    with progress_placeholder.container():
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            completed = progress_data.get('completed', 0)
                            total = progress_data.get('total_planned', 1)
                            progress = completed / total if total > 0 else 0
                            
                            st.progress(progress)
                            st.caption(f"Progress: {completed}/{total} experiments ({progress*100:.1f}%)")
                        
                        with col2:
                            st.metric("Success Rate", 
                                     f"{(progress_data.get('successful', 0) / max(completed, 1) * 100):.1f}%")
                    
                    # Show current experiment status
                    with status_placeholder.container():
                        st.subheader("Current Status")
                        
                        # Find most recent experiment
                        exp_dirs = sorted(glob.glob(f"{experiment_dir}/experiments/exp_*"))
                        if exp_dirs:
                            latest_exp = exp_dirs[-1]
                            try:
                                with open(Path(latest_exp) / "experiment_config.json") as f:
                                    current_exp = json.load(f)
                                
                                st.info(f"🔄 Running: {current_exp.get('description', 'Unknown experiment')}")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.caption(f"Agent Type: {current_exp.get('agent_type', 'N/A')}")
                                with col2:
                                    st.caption(f"Scenario: {current_exp.get('scenario', 'N/A')}")
                                with col3:
                                    st.caption(f"Grid: {current_exp.get('grid_config', 'N/A')}")
                            except:
                                pass
                
                # Wait before refresh
                time.sleep(refresh_interval)
                
                # Check if user navigated away
                if not auto_refresh:
                    break
        else:
            st.info("Enable auto-refresh to see live updates")
    
    with tab3:
        st.header("📈 Comparative Analysis")
        
        # Load metrics data
        metrics_df = load_segregation_metrics(experiment_dir)
        
        if not metrics_df.empty:
            # Convergence comparison
            st.subheader("Convergence Patterns")
            
            available_experiments = metrics_df['file'].unique()
            selected_exps = st.multiselect(
                "Select experiments to compare",
                options=available_experiments,
                default=available_experiments[:3] if len(available_experiments) >= 3 else available_experiments
            )
            
            convergence_fig = create_convergence_comparison(metrics_df, selected_exps)
            if convergence_fig:
                st.plotly_chart(convergence_fig, use_container_width=True)
            
            # Agent type comparison
            if not df.empty and 'agent_type' in df.columns:
                st.subheader("Agent Type Performance")
                comparison_fig = create_agent_type_comparison(df)
                if comparison_fig:
                    st.plotly_chart(comparison_fig, use_container_width=True)
            
            # Metric heatmap
            st.subheader("Segregation Metrics Heatmap")
            
            # Aggregate final metrics
            final_metrics = metrics_df.groupby('file').tail(1)
            metric_cols = ['clusters', 'switch_rate', 'distance', 'mix_deviation', 'share', 'ghetto_rate']
            available_metric_cols = [col for col in metric_cols if col in final_metrics.columns]
            
            if available_metric_cols:
                heatmap_data = final_metrics.set_index('file')[available_metric_cols].T
                
                fig = px.imshow(
                    heatmap_data,
                    labels=dict(x="Experiment", y="Metric", color="Value"),
                    title="Final Segregation Metrics by Experiment",
                    color_continuous_scale="RdBu_r"
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No metrics data available yet. Experiments may still be running.")
    
    with tab4:
        st.header("🔬 Experiment Details")
        
        if not df.empty:
            # Filter dataframe
            filtered_df = df.copy()
            if agent_types:
                filtered_df = filtered_df[filtered_df['agent_type'].isin(agent_types)]
            if scenarios:
                filtered_df = filtered_df[filtered_df['scenario'].isin(scenarios)]
            
            # Display experiments table
            st.subheader(f"Experiments ({len(filtered_df)} total)")
            
            # Format status with colors
            def format_status(status):
                color_map = {
                    'completed': 'green',
                    'running': 'blue',
                    'failed': 'red',
                    'pending': 'gray'
                }
                color = color_map.get(status, 'black')
                return f'<span style="color: {color};">{status}</span>'
            
            # Display key columns
            display_cols = ['experiment_id', 'agent_type', 'scenario', 'status', 'runtime_seconds']
            available_cols = [col for col in display_cols if col in filtered_df.columns]
            
            if available_cols:
                display_df = filtered_df[available_cols].copy()
                
                # Format runtime
                if 'runtime_seconds' in display_df.columns:
                    display_df['runtime_seconds'] = display_df['runtime_seconds'].apply(
                        lambda x: f"{x:.1f}s" if pd.notna(x) else "N/A"
                    )
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
            
            # Experiment details expander
            st.subheader("Detailed Results")
            
            for _, exp in filtered_df.iterrows():
                with st.expander(f"{exp.get('experiment_id', 'Unknown')} - {exp.get('description', 'No description')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Configuration:**")
                        st.json({
                            'agent_type': exp.get('agent_type'),
                            'scenario': exp.get('scenario'),
                            'grid_config': exp.get('grid_config'),
                            'llm_model': exp.get('llm_model')
                        })
                    
                    with col2:
                        st.write("**Results:**")
                        if 'status' in exp and exp['status'] == 'completed':
                            st.success(f"✅ Completed in {exp.get('runtime_seconds', 'N/A')} seconds")
                            
                            # Show final metrics if available
                            metric_keys = ['final_clusters', 'convergence_speed', 'final_distance']
                            metrics = {k: exp.get(k) for k in metric_keys if k in exp and pd.notna(exp[k])}
                            if metrics:
                                st.json(metrics)
                        elif 'status' in exp and exp['status'] == 'failed':
                            st.error(f"❌ Failed: {exp.get('error', 'Unknown error')}")
                        else:
                            st.info(f"⏳ {exp.get('status', 'Pending')}")
        else:
            st.info("No experiments found. Make sure the experiment directory is correct.")

if __name__ == "__main__":
    main()