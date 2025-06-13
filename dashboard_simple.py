#!/usr/bin/env python3
"""
Simple Dashboard for Schelling Segregation Experiments
Minimal dependencies version for easy setup
"""

import os
import sys
import json
import glob
from pathlib import Path
from datetime import datetime
import time

# Try to import optional dependencies
try:
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    FULL_FEATURES = True
except ImportError:
    FULL_FEATURES = False
    print("⚠️  Some features unavailable. Install with: pip install streamlit pandas plotly")

def simple_progress_display():
    """Simple text-based progress display for minimal dependencies"""
    print("\n" + "="*60)
    print("SCHELLING SEGREGATION EXPERIMENT MONITOR")
    print("="*60)
    
    while True:
        # Find latest progress file
        progress_files = glob.glob("design_space_exploration/progress_*.json")
        
        if progress_files:
            latest_file = max(progress_files, key=lambda x: Path(x).stat().st_mtime)
            
            try:
                with open(latest_file) as f:
                    progress = json.load(f)
                
                completed = progress.get('completed', 0)
                total = progress.get('total_planned', 0)
                successful = progress.get('successful', 0)
                failed = progress.get('failed', 0)
                percent = (completed / total * 100) if total > 0 else 0
                
                # Clear screen (works on most terminals)
                os.system('cls' if os.name == 'nt' else 'clear')
                
                print("\n" + "="*60)
                print("SCHELLING SEGREGATION EXPERIMENT MONITOR")
                print("="*60)
                print(f"\nTime: {datetime.now().strftime('%H:%M:%S')}")
                print(f"\nProgress: {completed}/{total} experiments ({percent:.1f}%)")
                print(f"✅ Successful: {successful}")
                print(f"❌ Failed: {failed}")
                
                # Progress bar
                bar_length = 40
                filled = int((percent / 100) * bar_length)
                bar = "█" * filled + "░" * (bar_length - filled)
                print(f"\n[{bar}] {percent:.1f}%")
                
                # Recent experiments
                print("\nRecent Activity:")
                exp_files = sorted(glob.glob("design_space_exploration/experiments/exp_*/experiment_config.json"))[-5:]
                for exp_file in exp_files:
                    try:
                        with open(exp_file) as f:
                            exp = json.load(f)
                        print(f"  • {exp.get('description', 'Unknown')}")
                    except:
                        pass
                
                if completed >= total:
                    print("\n🎉 EXPERIMENTS COMPLETE!")
                    break
                    
            except Exception as e:
                print(f"Error reading progress: {e}")
        else:
            print("\nWaiting for experiments to start...")
        
        time.sleep(5)

def create_simple_dashboard():
    """Create Streamlit dashboard with basic features"""
    st.set_page_config(page_title="Schelling Experiments", layout="wide")
    
    st.title("🏘️ Schelling Segregation Experiments")
    st.markdown("**Real-time monitoring of mechanical vs LLM agent simulations**")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Find all available experiment directories
        comp_dirs = glob.glob("comprehensive_study_*/llm_results")
        design_dirs = glob.glob("design_space_exploration")
        exp_dirs = glob.glob("experiments/llm_*")
        
        # Combine and sort by modification time
        all_dirs = comp_dirs + design_dirs + exp_dirs
        all_dirs = [d for d in all_dirs if Path(d).exists()]
        all_dirs.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
        
        # Add default if no directories found
        if not all_dirs:
            all_dirs = ["design_space_exploration"]
            st.warning("No experiment directories found")
        else:
            st.success(f"Found {len(all_dirs)} experiment(s)")
        
        # Directory selection dropdown
        experiment_dir = st.selectbox(
            "Select Experiment",
            options=all_dirs,
            index=0,
            help="Choose from available experiment directories"
        )
        
        auto_refresh = st.checkbox("Auto-refresh", True)
        
        if st.button("🔄 Refresh Now"):
            st.rerun()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📊 Progress", "📈 Results", "📋 Experiments"])
    
    with tab1:
        st.header("Current Progress")
        
        # Find progress file
        progress_files = glob.glob(f"{experiment_dir}/progress_*.json")
        
        if progress_files:
            latest_file = max(progress_files, key=lambda x: Path(x).stat().st_mtime)
            
            with open(latest_file) as f:
                progress = json.load(f)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Total", progress.get('total_planned', 0))
            col2.metric("Completed", progress.get('completed', 0))
            col3.metric("Successful", progress.get('successful', 0))
            col4.metric("Failed", progress.get('failed', 0))
            
            # Progress bar
            percent = progress.get('progress_percent', 0)
            st.progress(percent / 100)
            st.caption(f"{percent:.1f}% complete")
            
            # Current status
            if progress.get('completed', 0) < progress.get('total_planned', 1):
                st.info("🔄 Experiments running...")
            else:
                st.success("✅ All experiments complete!")
        else:
            st.warning("No progress data found. Experiments may not have started.")
    
    with tab2:
        st.header("Results Overview")
        
        # Load experiment results
        results = []
        exp_dirs = glob.glob(f"{experiment_dir}/experiments/exp_*")
        
        for exp_dir in exp_dirs[:20]:  # Limit to prevent overload
            try:
                config_file = Path(exp_dir) / "experiment_config.json"
                if config_file.exists():
                    with open(config_file) as f:
                        config = json.load(f)
                    
                    # Check for results
                    results_file = Path(exp_dir) / "results.json"
                    if results_file.exists():
                        config['status'] = 'completed'
                    else:
                        config['status'] = 'running'
                    
                    results.append(config)
            except:
                continue
        
        if results:
            df = pd.DataFrame(results)
            
            # Summary stats
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("By Agent Type")
                if 'agent_type' in df.columns:
                    agent_counts = df['agent_type'].value_counts()
                    st.bar_chart(agent_counts)
            
            with col2:
                st.subheader("By Scenario")
                if 'scenario' in df.columns:
                    scenario_counts = df['scenario'].value_counts()
                    st.bar_chart(scenario_counts)
            
            # Status breakdown
            if 'status' in df.columns:
                st.subheader("Experiment Status")
                status_counts = df['status'].value_counts()
                fig = px.pie(values=status_counts.values, names=status_counts.index, 
                            title="Experiment Status Distribution")
                st.plotly_chart(fig)
        else:
            st.info("No results available yet.")
    
    with tab3:
        st.header("Experiment List")
        
        # Load experiments
        experiments = []
        config_files = glob.glob(f"{experiment_dir}/experiments/exp_*/experiment_config.json")
        
        for config_file in sorted(config_files)[:50]:  # Limit display
            try:
                with open(config_file) as f:
                    exp = json.load(f)
                exp['exp_id'] = Path(config_file).parent.name
                experiments.append(exp)
            except:
                continue
        
        if experiments:
            # Create dataframe
            exp_df = pd.DataFrame(experiments)
            
            # Display selected columns
            display_cols = ['exp_id', 'agent_type', 'scenario', 'grid_config']
            available_cols = [col for col in display_cols if col in exp_df.columns]
            
            if available_cols:
                st.dataframe(exp_df[available_cols], use_container_width=True)
            else:
                st.dataframe(exp_df)
        else:
            st.info("No experiments found.")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(5)
        st.rerun()

def main():
    """Main entry point"""
    import sys
    
    if '--simple' in sys.argv or not FULL_FEATURES:
        # Run simple text-based monitor
        simple_progress_display()
    else:
        # Run Streamlit dashboard
        print("Starting Streamlit dashboard...")
        print("If the dashboard doesn't open automatically, go to: http://localhost:8501")
        create_simple_dashboard()

if __name__ == "__main__":
    # Check if running directly or through streamlit
    if FULL_FEATURES and __name__ == "__main__" and "streamlit" not in sys.modules:
        # Launch streamlit
        import subprocess
        import sys
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
    else:
        main()