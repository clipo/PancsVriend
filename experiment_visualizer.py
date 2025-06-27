#!/usr/bin/env python3
"""
Experiment Visualizer for Schelling Model Simulations

This tool loads and visualizes experiments from saved simulation data, allowing you to:
- Browse and select experiments from the experiments/ folder using GUI
- Select which run to visualize within each experiment (defaults to first run)
- View step-by-step evolution of the simulation grid with synchronized metrics
- Control playback speed, pause/resume, and jump to specific steps
- Display real-time metrics and convergence information
- Color-coded visualization with legend (red=Type A, blue=Type B, black=empty)

Command line options:
- --select-experiment (-s): GUI to choose experiment
- --gui (-g): GUI to choose run within experiment  
- --full-gui (-f): GUI to choose both experiment and run
- --list-experiments (-l): List all available experiments
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider, CheckButtons
from matplotlib.patches import Patch
import json
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

class ExperimentVisualizer:
    def __init__(self, experiment_dir=None, run_id=0):
        """
        Initialize the experiment visualizer.
        
        Args:
            experiment_dir: Path to experiment directory. If None, loads latest.
            run_id: Which run to visualize (default: 0)
        """
        self.experiment_dir = experiment_dir or self._find_latest_experiment()
        self.run_id = run_id
        self.current_step = 0
        self.is_playing = False
        self.speed = 1.0  # Steps per second
        
        # Data containers
        self.states = None
        self.metrics_df = None
        self.config = None
        self.convergence_data = None
        self.max_steps = 0  # Maximum step number from metrics
        self.step_to_state_mapping = {}  # Maps step number to state index
        
        # Animation and UI
        self.fig = None
        self.ax_grid = None
        self.ax_metrics = None
        self.animation_obj = None
        self.grid_im = None
        
        # Load data
        self._load_experiment_data()
        self._setup_visualization()
        
    def _find_latest_experiment(self):
        """Find the most recent experiment directory from the experiments folder."""
        experiments_dir = "experiments"
        if not os.path.exists(experiments_dir):
            raise ValueError("No 'experiments' folder found. Please ensure experiments are saved in the experiments/ directory.")
        
        experiment_dirs = glob.glob(os.path.join(experiments_dir, "*"))
        experiment_dirs = [d for d in experiment_dirs if os.path.isdir(d)]
        
        if not experiment_dirs:
            raise ValueError("No experiment directories found in experiments/ folder.")
        
        # Find the latest experiment by creation time
        latest = max(experiment_dirs, key=lambda x: os.path.getctime(x))
        print(f"📂 Loading latest experiment from experiments folder: {latest}")
        return latest
        
    def _load_experiment_data(self):
        """Load all relevant data for the experiment."""
        print(f"🔄 Loading experiment data from {self.experiment_dir}...")
        
        # Check if this is a direct experiment directory (experiments folder format)
        # or a comprehensive_study format with nested baseline directories
        if os.path.exists(os.path.join(self.experiment_dir, "states")):
            # Direct format (experiments folder)
            baseline_dir = self.experiment_dir
            print(f"📁 Using direct experiment directory: {baseline_dir}")
        else:
            # Comprehensive study format - look for baseline directories
            baseline_dirs = glob.glob(os.path.join(self.experiment_dir, "mechanical_baseline", "*"))
            if not baseline_dirs:
                raise ValueError(f"No baseline directories found in {self.experiment_dir}")
            baseline_dir = baseline_dirs[0]
            print(f"📁 Using nested baseline directory: {baseline_dir}")
        
        # Load configuration
        config_path = os.path.join(baseline_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Default config values if file not found
            self.config = {
                "grid_size": 20,
                "num_type_a": 160,
                "num_type_b": 160,
                "n_runs": 100,
                "max_steps": 1000
            }
            
        # Load states for the specified run
        states_path = os.path.join(baseline_dir, "states", f"states_run_{self.run_id}.npz")
        if not os.path.exists(states_path):
            available_runs = self._get_available_runs(baseline_dir)
            if available_runs:
                original_run_id = self.run_id
                self.run_id = available_runs[0]
                states_path = os.path.join(baseline_dir, "states", f"states_run_{self.run_id}.npz")
                print(f"⚠️ Run {original_run_id} not found, using run {self.run_id} instead")
            else:
                raise ValueError(f"No state files found in {baseline_dir}/states/")
                
        print(f"📊 Loading states for run {self.run_id}...")
        states_data = np.load(states_path)
        self.states = states_data['states']
        print(f"✅ Loaded {len(self.states)} states")
        
        # Load metrics if available
        metrics_path = os.path.join(baseline_dir, "metrics_history.csv")
        if os.path.exists(metrics_path):
            self.metrics_df = pd.read_csv(metrics_path)
            # Filter for current run
            self.metrics_df = self.metrics_df[self.metrics_df['run_id'] == self.run_id]
            print(f"📈 Loaded metrics: {len(self.metrics_df)} entries")
            
            # Create mapping from step number to state index
            if len(self.metrics_df) > 0:
                self.max_steps = self.metrics_df['step'].max()
                # Estimate state indices for each step (assuming roughly even distribution)
                total_states = len(self.states)
                for i, step in enumerate(self.metrics_df['step']):
                    # Map each step to approximately the right state index
                    state_idx = min(int((step + 1) * total_states / (self.max_steps + 1)), total_states - 1)
                    self.step_to_state_mapping[step] = state_idx
                print(f"🔗 Created step-to-state mapping for {self.max_steps + 1} steps")
        else:
            print("⚠️ No metrics file found")
            self.max_steps = len(self.states) - 1 if self.states is not None else 0
            
        # Load convergence data if available
        conv_path = os.path.join(baseline_dir, "convergence_summary.csv")
        if os.path.exists(conv_path):
            conv_df = pd.read_csv(conv_path)
            run_conv_data = conv_df[conv_df['run_id'] == self.run_id]
            if not run_conv_data.empty:
                self.convergence_data = run_conv_data.iloc[0].to_dict()
                print(f"🎯 Convergence: {'Yes' if self.convergence_data.get('converged', False) else 'No'}")
        
    def _get_available_runs(self, baseline_dir):
        """Get list of available run IDs."""
        states_dir = os.path.join(baseline_dir, "states")
        if not os.path.exists(states_dir):
            return []
            
        files = glob.glob(os.path.join(states_dir, "states_run_*.npz"))
        run_ids = []
        for f in files:
            try:
                run_id = int(os.path.basename(f).split('_')[-1].split('.')[0])
                run_ids.append(run_id)
            except ValueError:
                continue
        return sorted(run_ids)
        
    def _setup_visualization(self):
        """Set up the matplotlib visualization with controls."""
        # Create figure with subplots, leaving space at bottom for controls
        self.fig = plt.figure(figsize=(16, 10))
        
        # Adjust subplot layout to leave space for controls at bottom
        plt.subplots_adjust(bottom=0.2)
        
        # Grid visualization (left side)
        self.ax_grid = plt.subplot2grid((3, 3), (0, 0), rowspan=3, colspan=2)
        self.ax_grid.set_title(f"Schelling Model - Run {self.run_id} (Step {self.current_step})")
        self.ax_grid.set_xticks([])
        self.ax_grid.set_yticks([])
        
        # Metrics visualization (top right)
        self.ax_metrics = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
        self.ax_metrics.set_title("Metrics Over Time")
        
        # Info panel (bottom right)
        self.ax_info = plt.subplot2grid((3, 3), (2, 2))
        self.ax_info.axis('off')
        
        # Initialize grid display
        self._update_grid_display()
        self._update_metrics_display()
        self._update_info_panel()
        
        # Add control widgets
        self._add_controls()
        
        # Set up animation
        self.animation_obj = animation.FuncAnimation(
            self.fig, self._animate, interval=1000/self.speed, 
            blit=False, repeat=True
        )
        
    def _update_grid_display(self):
        """Update the grid visualization."""
        if self.states is None:
            return
            
        # Map current step to appropriate state index
        if self.step_to_state_mapping and self.current_step in self.step_to_state_mapping:
            state_index = self.step_to_state_mapping[self.current_step]
        else:
            # Fallback: use current_step as state index (for backward compatibility)
            state_index = min(self.current_step, len(self.states) - 1)
            
        if state_index >= len(self.states):
            return
            
        current_state = self.states[state_index]
        
        # Create RGB color-coded display
        # -1 = empty (black), 0 = type A (red), 1 = type B (blue)
        height, width = current_state.shape
        display_grid = np.zeros((height, width, 3), dtype=float)  # RGB array
        
        # Set colors: black for empty, red for type A, blue for type B
        display_grid[current_state == -1] = [0.0, 0.0, 0.0]  # Empty = black
        display_grid[current_state == 0] = [1.0, 0.0, 0.0]   # Type A = red
        display_grid[current_state == 1] = [0.0, 0.0, 1.0]   # Type B = blue
        
        if self.grid_im is None and self.ax_grid is not None:
            self.grid_im = self.ax_grid.imshow(
                display_grid, interpolation='nearest', aspect='equal'
            )
            
            # Add white grid lines to show cell boundaries
            grid_size = current_state.shape[0]
            # Set minor ticks at cell boundaries
            self.ax_grid.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
            self.ax_grid.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
            # Show grid lines
            self.ax_grid.grid(which='minor', color='white', linestyle='-', linewidth=1)
            self.ax_grid.tick_params(which='minor', size=0)  # Hide tick marks
            
            # Add legend for cell colors
            legend_elements = [
                Patch(facecolor='red', edgecolor='white', label='Type A Agents'),
                Patch(facecolor='blue', edgecolor='white', label='Type B Agents'),
                Patch(facecolor='black', edgecolor='white', label='Empty Cells')
            ]
            self.ax_grid.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(-0.25, 0.5))
            
        elif self.grid_im is not None:
            self.grid_im.set_array(display_grid)
            
        if self.ax_grid is not None:
            self.ax_grid.set_title(f"Schelling Model - Run {self.run_id} (Step {self.current_step}/{self.max_steps})")
        
    def _update_metrics_display(self):
        """Update the metrics visualization."""
        if self.ax_metrics is None:
            return
            
        if self.metrics_df is None or len(self.metrics_df) == 0:
            self.ax_metrics.text(0.5, 0.5, "No metrics data available", 
                               ha='center', va='center', transform=self.ax_metrics.transAxes)
            return
            
        self.ax_metrics.clear()
        
        # Plot key metrics - check which ones are available
        available_metrics = []
        metric_colors = []
        
        metrics_to_try = [
            ('clusters', 'blue'),
            ('switch_rate', 'green'), 
            ('segregation_index', 'red'),
            ('distance', 'orange'),
            ('mix_deviation', 'purple'),
            ('ghetto_rate', 'brown')
        ]
        
        for metric, color in metrics_to_try:
            if metric in self.metrics_df.columns:
                available_metrics.append(metric)
                metric_colors.append(color)
        
        # Plot available metrics
        for metric, color in zip(available_metrics[:3], metric_colors[:3]):  # Limit to 3 for readability
            steps = self.metrics_df['step'].values
            values = self.metrics_df[metric].values
            self.ax_metrics.plot(steps, values, label=metric, color=color, alpha=0.7)
                
        # Mark current step
        if self.current_step < len(self.metrics_df):
            self.ax_metrics.axvline(x=self.current_step, color='black', linestyle='--', alpha=0.5)
            
        self.ax_metrics.set_xlabel('Step')
        self.ax_metrics.set_ylabel('Value')
        if available_metrics:
            self.ax_metrics.legend()
        self.ax_metrics.grid(True, alpha=0.3)
        
    def _update_info_panel(self):
        """Update the information panel."""
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        # Gather information to display
        info_text = []
        info_text.append(f"Experiment: {os.path.basename(self.experiment_dir)}")
        info_text.append(f"Run ID: {self.run_id}")
        info_text.append(f"Current Step: {self.current_step}/{self.max_steps}")
        
        if self.config:
            info_text.append(f"Grid Size: {self.config.get('grid_size', 'N/A')}")
            info_text.append(f"Type A: {self.config.get('num_type_a', 'N/A')}")
            info_text.append(f"Type B: {self.config.get('num_type_b', 'N/A')}")
            
        if self.convergence_data:
            conv_status = "Yes" if self.convergence_data.get('converged', False) else "No"
            info_text.append(f"Converged: {conv_status}")
            if self.convergence_data.get('convergence_step'):
                info_text.append(f"Convergence Step: {self.convergence_data['convergence_step']}")
                
        # Current metrics
        if self.metrics_df is not None and self.current_step < len(self.metrics_df):
            current_metrics = self.metrics_df.iloc[self.current_step]
            info_text.append("--- Current Metrics ---")
            
            # Show available metrics
            for col in ['clusters', 'switch_rate', 'segregation_index', 'distance', 'mix_deviation']:
                if col in current_metrics:
                    info_text.append(f"{col}: {current_metrics[col]:.3f}")
                    
        # Display info text
        info_str = '\n'.join(info_text)
        self.ax_info.text(0.05, 0.95, info_str, transform=self.ax_info.transAxes,
                         fontsize=9, verticalalignment='top', fontfamily='monospace')
        
    def _add_controls(self):
        """Add control widgets to the figure."""
        # Play/Pause button
        ax_play = plt.axes([0.1, 0.02, 0.1, 0.04])
        self.btn_play = Button(ax_play, 'Play')
        self.btn_play.on_clicked(self._toggle_play)
        
        # Step controls
        ax_prev = plt.axes([0.22, 0.02, 0.1, 0.04])
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_prev.on_clicked(self._step_backward)
        
        ax_next = plt.axes([0.34, 0.02, 0.1, 0.04])
        self.btn_next = Button(ax_next, 'Next')
        self.btn_next.on_clicked(self._step_forward)
        
        # Speed control
        ax_speed = plt.axes([0.5, 0.02, 0.2, 0.04])
        self.slider_speed = Slider(ax_speed, 'Speed', 0.1, 5.0, valinit=self.speed)
        self.slider_speed.on_changed(self._change_speed)
        
        # Step slider
        ax_step = plt.axes([0.1, 0.08, 0.6, 0.04])
        self.slider_step = Slider(ax_step, 'Step', 0, self.max_steps, 
                                 valinit=self.current_step, valfmt='%d')
        self.slider_step.on_changed(self._change_step)
        
    def _toggle_play(self, event):
        """Toggle play/pause."""
        self.is_playing = not self.is_playing
        self.btn_play.label.set_text('Pause' if self.is_playing else 'Play')
        
    def _step_forward(self, event):
        """Step forward one frame."""
        if self.current_step < self.max_steps:
            self.current_step += 1
            self.slider_step.set_val(self.current_step)
            self._update_displays()
            
    def _step_backward(self, event):
        """Step backward one frame."""
        if self.current_step > 0:
            self.current_step -= 1
            self.slider_step.set_val(self.current_step)
            self._update_displays()
            
    def _change_speed(self, val):
        """Change animation speed."""
        self.speed = val
        self.animation_obj.event_source.interval = 1000 / self.speed
        
    def _change_step(self, val):
        """Change current step via slider."""
        self.current_step = int(val)
        self._update_displays()
        
    def _animate(self, frame):
        """Animation function called by matplotlib."""
        if self.is_playing:
            if self.current_step < self.max_steps:
                self.current_step += 1
                self.slider_step.set_val(self.current_step)
                self._update_displays()
            else:
                self.is_playing = False
                self.btn_play.label.set_text('Play')
        return []
        
    def _update_displays(self):
        """Update all display components."""
        self._update_grid_display()
        self._update_metrics_display()
        self._update_info_panel()
        
    def show(self):
        """Show the visualization."""
        plt.show()
        
    def save_frame(self, filename=None):
        """Save current frame as image."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"frame_run_{self.run_id}_step_{self.current_step}_{timestamp}.png"
        
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 Saved frame: {filename}")

def get_available_experiments():
    """Get list of available experiments from the experiments folder."""
    experiments = []
    if os.path.exists("experiments"):
        exp_dirs = glob.glob(os.path.join("experiments", "*"))
        experiments.extend([d for d in exp_dirs if os.path.isdir(d)])
    
    # Sort by creation time (newest first)
    experiments.sort(key=lambda x: os.path.getctime(x), reverse=True)
    return experiments

def create_experiment_selector_gui():
    """Create a GUI to select which experiment to visualize."""
    root = tk.Tk()
    root.title("Select Experiment to Visualize")
    root.geometry("500x400")
    
    # Get available experiments
    available_experiments = get_available_experiments()
    
    if not available_experiments:
        messagebox.showerror("Error", "No experiments found in experiments/ folder.")
        root.destroy()
        return None
    
    selected_experiment = tk.StringVar(value=available_experiments[0])
    
    # Create UI
    ttk.Label(root, text="Available Experiments", font=("Arial", 12, "bold")).pack(pady=10)
    ttk.Label(root, text=f"Found {len(available_experiments)} experiments in experiments/ folder").pack()
    
    frame = ttk.Frame(root)
    frame.pack(pady=20, padx=20, fill='both', expand=True)
    
    ttk.Label(frame, text="Select Experiment:").pack()
    
    # Listbox for experiment selection with scrollbar
    list_frame = ttk.Frame(frame)
    list_frame.pack(fill='both', expand=True, pady=10)
    
    scrollbar = ttk.Scrollbar(list_frame)
    scrollbar.pack(side='right', fill='y')
    
    listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set)
    listbox.pack(side='left', fill='both', expand=True)
    scrollbar.config(command=listbox.yview)
    
    # Populate listbox with experiment names and dates
    for exp_dir in available_experiments:
        exp_name = os.path.basename(exp_dir)
        try:
            # Get creation time for display
            creation_time = datetime.fromtimestamp(os.path.getctime(exp_dir))
            time_str = creation_time.strftime("%Y-%m-%d %H:%M:%S")
            display_text = f"{exp_name} ({time_str})"
        except:
            display_text = exp_name
        listbox.insert(tk.END, display_text)
    
    listbox.selection_set(0)  # Select first (newest) item
    
    def on_select(event=None):
        selection = listbox.curselection()
        if selection:
            selected_experiment.set(available_experiments[selection[0]])
    
    listbox.bind('<<ListboxSelect>>', on_select)
    
    # Info label to show selected experiment
    info_label = ttk.Label(frame, text="", wraplength=400)
    info_label.pack(pady=5)
    
    def update_info():
        exp_path = selected_experiment.get()
        info_text = f"Selected: {os.path.basename(exp_path)}"
        
        # Try to get some basic info about the experiment
        try:
            # Check for config file
            config_path = os.path.join(exp_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    info_text += f"\nGrid: {config.get('grid_size', 'N/A')}, "
                    info_text += f"Runs: {config.get('n_runs', 'N/A')}"
            
            # Count available run files
            states_dir = os.path.join(exp_path, "states")
            if os.path.exists(states_dir):
                run_files = glob.glob(os.path.join(states_dir, "states_run_*.npz"))
                info_text += f"\nAvailable runs: {len(run_files)}"
        except:
            pass
        
        info_label.config(text=info_text)
    
    # Update info when selection changes
    def on_selection_change(event=None):
        on_select(event)
        update_info()
    
    listbox.bind('<<ListboxSelect>>', on_selection_change)
    update_info()  # Initial update
    
    # Buttons
    button_frame = ttk.Frame(frame)
    button_frame.pack(pady=10)
    
    result = {'experiment_dir': None, 'cancelled': True}
    
    def on_ok():
        result['experiment_dir'] = selected_experiment.get()
        result['cancelled'] = False
        root.destroy()
    
    def on_cancel():
        root.destroy()
    
    ttk.Button(button_frame, text="Select", command=on_ok).pack(side='left', padx=5)
    ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side='left', padx=5)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()
    
    return None if result['cancelled'] else result['experiment_dir']

def create_run_selector_gui(experiment_dir):
    """Create a GUI to select which run to visualize."""
    root = tk.Tk()
    root.title("Select Run to Visualize")
    root.geometry("400x300")
    
    # Check if this is a direct experiment directory or comprehensive_study format
    if os.path.exists(os.path.join(experiment_dir, "states")):
        # Direct format (experiments folder)
        baseline_dir = experiment_dir
    else:
        # Comprehensive study format - look for baseline directories
        baseline_dirs = glob.glob(os.path.join(experiment_dir, "mechanical_baseline", "*"))
        if not baseline_dirs:
            messagebox.showerror("Error", f"No baseline directories found in {experiment_dir}")
            root.destroy()
            return None
        baseline_dir = baseline_dirs[0]
    
    # Get available runs
    states_dir = os.path.join(baseline_dir, "states")
    available_runs = []
    if os.path.exists(states_dir):
        files = glob.glob(os.path.join(states_dir, "states_run_*.npz"))
        for f in files:
            try:
                run_id = int(os.path.basename(f).split('_')[-1].split('.')[0])
                available_runs.append(run_id)
            except ValueError:
                continue
    available_runs.sort()
    
    if not available_runs:
        messagebox.showerror("Error", f"No run files found in {states_dir}")
        root.destroy()
        return None
    
    selected_run = tk.IntVar(value=available_runs[0])
    
    # Create UI
    ttk.Label(root, text=f"Experiment: {os.path.basename(experiment_dir)}").pack(pady=10)
    ttk.Label(root, text=f"Available runs: {len(available_runs)}").pack()
    
    frame = ttk.Frame(root)
    frame.pack(pady=20, padx=20, fill='both', expand=True)
    
    ttk.Label(frame, text="Select Run ID:").pack()
    
    # Listbox for run selection
    listbox = tk.Listbox(frame, height=8)
    listbox.pack(fill='both', expand=True, pady=10)
    
    for run_id in available_runs:
        listbox.insert(tk.END, f"Run {run_id}")
    
    listbox.selection_set(0)  # Select first item
    
    def on_select(event=None):
        selection = listbox.curselection()
        if selection:
            selected_run.set(available_runs[selection[0]])
    
    listbox.bind('<<ListboxSelect>>', on_select)
    
    # Buttons
    button_frame = ttk.Frame(frame)
    button_frame.pack(pady=10)
    
    result = {'run_id': None, 'cancelled': True}
    
    def on_ok():
        result['run_id'] = selected_run.get()
        result['cancelled'] = False
        root.destroy()
    
    def on_cancel():
        root.destroy()
    
    ttk.Button(button_frame, text="Visualize", command=on_ok).pack(side='left', padx=5)
    ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side='left', padx=5)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()
    
    return None if result['cancelled'] else result['run_id']

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Visualize Schelling Model experiments")
    parser.add_argument("--experiment", "-e", type=str, help="Experiment directory (default: latest)")
    parser.add_argument("--run", "-r", type=int, help="Run ID to visualize")
    parser.add_argument("--list-experiments", "-l", action="store_true", 
                       help="List available experiments")
    parser.add_argument("--gui", "-g", action="store_true", 
                       help="Use GUI to select run")
    parser.add_argument("--select-experiment", "-s", action="store_true",
                       help="Use GUI to select experiment")
    parser.add_argument("--full-gui", "-f", action="store_true",
                       help="Use GUI to select both experiment and run")
    
    args = parser.parse_args()
    
    if args.list_experiments:
        # Only check experiments folder
        experiments = get_available_experiments()
        
        if experiments:
            print("Available experiments in experiments/ folder:")
            for exp in sorted(experiments):
                exp_name = os.path.basename(exp)
                try:
                    creation_time = datetime.fromtimestamp(os.path.getctime(exp))
                    time_str = creation_time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"  {exp_name} ({time_str})")
                except:
                    print(f"  {exp_name}")
        else:
            print("No experiments found in experiments/ folder.")
        return
    
    try:
        # Determine experiment directory
        experiment_dir = args.experiment
        
        # If no specific arguments are provided, use full GUI by default
        use_full_gui = (args.select_experiment or args.full_gui or 
                       (experiment_dir is None and args.run is None and not args.gui))
        
        if use_full_gui:
            # Use GUI to select experiment
            experiment_dir = create_experiment_selector_gui()
            if experiment_dir is None:
                print("❌ No experiment selected.")
                return
            print(f"📂 Selected experiment: {os.path.basename(experiment_dir)}")
        
        if experiment_dir is None:
            # Find latest experiment from experiments folder
            if not os.path.exists("experiments"):
                print("❌ No experiments folder found. Please ensure experiments are saved in the experiments/ directory.")
                return
            
            available_experiments = get_available_experiments()
            
            if not available_experiments:
                print("❌ No experiments found in experiments/ folder. Run an experiment first.")
                return
                
            experiment_dir = available_experiments[0]  # Get newest
            print(f"📂 Using latest experiment: {os.path.basename(experiment_dir)}")
        
        # Determine run ID
        run_id = args.run
        if args.gui or use_full_gui or run_id is None:
            run_id = create_run_selector_gui(experiment_dir)
            if run_id is None:
                print("❌ No run selected.")
                return
        
        if run_id is None:
            run_id = 0  # Default to first run
            
        print(f"🎬 Starting visualization for run {run_id}...")
        
        # Create and show visualizer
        visualizer = ExperimentVisualizer(experiment_dir, run_id)
        visualizer.show()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()