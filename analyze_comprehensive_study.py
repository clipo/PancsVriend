#!/usr/bin/env python3
"""
Analyze Comprehensive Study Results
Analyzes data from comprehensive_study_* directories containing both mechanical and LLM results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
import sys

def load_mechanical_data(study_dir):
    """Load mechanical baseline data from study directory"""
    mechanical_dir = study_dir / "mechanical_baseline"
    mechanical_data = {}
    
    if not mechanical_dir.exists():
        print(f"⚠️  No mechanical baseline data found in {mechanical_dir}")
        return mechanical_data
    
    for baseline_dir in mechanical_dir.glob("baseline_*"):
        grid_name = baseline_dir.name.replace("baseline_", "")
        
        # Load the data files
        try:
            metrics_file = baseline_dir / "metrics_history.csv"
            convergence_file = baseline_dir / "convergence_summary.csv"
            config_file = baseline_dir / "config.json"
            
            if all(f.exists() for f in [metrics_file, convergence_file, config_file]):
                metrics_df = pd.read_csv(metrics_file)
                convergence_df = pd.read_csv(convergence_file)
                
                with open(config_file) as f:
                    config_data = json.load(f)
                
                # Add metadata
                metrics_df['agent_type'] = 'mechanical'
                metrics_df['grid_config'] = grid_name
                metrics_df['scenario'] = 'baseline'  # Mechanical is always baseline scenario
                
                convergence_df['agent_type'] = 'mechanical'
                convergence_df['grid_config'] = grid_name
                convergence_df['scenario'] = 'baseline'
                
                mechanical_data[grid_name] = {
                    'metrics': metrics_df,
                    'convergence': convergence_df,
                    'config': config_data
                }
                
                print(f"✅ Loaded mechanical data: {grid_name} ({len(metrics_df)} steps, {len(convergence_df)} runs)")
                
        except Exception as e:
            print(f"❌ Error loading mechanical data from {baseline_dir}: {e}")
    
    return mechanical_data

def load_llm_data(study_dir):
    """Load LLM experiment data from study directory"""
    llm_dir = study_dir / "llm_results"
    llm_data = {}
    
    if not llm_dir.exists():
        print(f"⚠️  No LLM results found in {llm_dir}")
        return llm_data
    
    experiments_dir = llm_dir / "experiments"
    if not experiments_dir.exists():
        print(f"⚠️  No LLM experiments directory found in {experiments_dir}")
        return llm_data
    
    # Load individual experiment results
    all_metrics = []
    all_convergence = []
    
    for exp_dir in experiments_dir.glob("exp_*"):
        try:
            config_file = exp_dir / "config.json"
            metrics_file = exp_dir / "metrics_history.csv" 
            convergence_file = exp_dir / "convergence_summary.csv"
            
            if all(f.exists() for f in [config_file, metrics_file, convergence_file]):
                with open(config_file) as f:
                    config_data = json.load(f)
                
                metrics_df = pd.read_csv(metrics_file)
                convergence_df = pd.read_csv(convergence_file)
                
                # Extract metadata from config
                agent_type = config_data.get('agent_type', 'unknown')
                scenario = config_data.get('scenario', 'unknown')
                grid_size = config_data.get('grid_size', 'unknown')
                
                # Add metadata to dataframes
                metrics_df['agent_type'] = agent_type
                metrics_df['scenario'] = scenario
                metrics_df['grid_size'] = grid_size
                metrics_df['experiment'] = exp_dir.name
                
                convergence_df['agent_type'] = agent_type
                convergence_df['scenario'] = scenario
                convergence_df['grid_size'] = grid_size
                convergence_df['experiment'] = exp_dir.name
                
                all_metrics.append(metrics_df)
                all_convergence.append(convergence_df)
                
        except Exception as e:
            print(f"❌ Error loading LLM data from {exp_dir}: {e}")
    
    if all_metrics:
        llm_data['metrics'] = pd.concat(all_metrics, ignore_index=True)
        llm_data['convergence'] = pd.concat(all_convergence, ignore_index=True)
        
        print(f"✅ Loaded LLM data: {len(all_metrics)} experiments")
        print(f"   Agent types: {llm_data['metrics']['agent_type'].unique()}")
        print(f"   Scenarios: {llm_data['metrics']['scenario'].unique()}")
    
    return llm_data

def create_comparative_analysis(mechanical_data, llm_data, output_dir):
    """Create comparative analysis between mechanical and LLM agents"""
    
    print("\n📊 Creating comparative analysis...")
    
    # Combine all data
    all_metrics = []
    all_convergence = []
    
    # Add mechanical data
    for grid_name, data in mechanical_data.items():
        all_metrics.append(data['metrics'])
        all_convergence.append(data['convergence'])
    
    # Add LLM data
    if 'metrics' in llm_data:
        all_metrics.append(llm_data['metrics'])
        all_convergence.append(llm_data['convergence'])
    
    if not all_metrics:
        print("❌ No data available for analysis")
        return
    
    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    combined_convergence = pd.concat(all_convergence, ignore_index=True)
    
    # Save combined data
    combined_metrics.to_csv(output_dir / "combined_metrics.csv", index=False)
    combined_convergence.to_csv(output_dir / "combined_convergence.csv", index=False)
    
    print(f"📁 Saved combined data to {output_dir}/")
    
    # Create summary statistics
    summary_stats = []
    
    for agent_type in combined_convergence['agent_type'].unique():
        agent_data = combined_convergence[combined_convergence['agent_type'] == agent_type]
        
        if not agent_data.empty:
            stats = {
                'agent_type': agent_type,
                'n_runs': len(agent_data),
                'convergence_rate': agent_data['converged'].mean() if 'converged' in agent_data.columns else 0,
                'avg_convergence_step': agent_data['convergence_step'].mean() if 'convergence_step' in agent_data.columns else 0,
                'avg_final_clusters': agent_data['final_clusters'].mean() if 'final_clusters' in agent_data.columns else 0,
            }
            summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(output_dir / "summary_statistics.csv", index=False)
    
    print("📊 Summary Statistics:")
    print(summary_df.to_string(index=False))
    
    # Create basic visualizations if possible
    try:
        import matplotlib.pyplot as plt
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Comprehensive Study Results: Mechanical vs LLM Agents', fontsize=16)
        
        # Convergence rates
        if 'converged' in combined_convergence.columns:
            convergence_by_type = combined_convergence.groupby('agent_type')['converged'].mean()
            axes[0, 0].bar(convergence_by_type.index, convergence_by_type.values)
            axes[0, 0].set_title('Convergence Rate by Agent Type')
            axes[0, 0].set_ylabel('Convergence Rate')
            
        # Convergence steps
        if 'convergence_step' in combined_convergence.columns:
            converged_data = combined_convergence[combined_convergence['converged'] == True]
            if not converged_data.empty:
                converged_data.boxplot(column='convergence_step', by='agent_type', ax=axes[0, 1])
                axes[0, 1].set_title('Convergence Speed by Agent Type')
                axes[0, 1].set_ylabel('Steps to Convergence')
        
        # Final metrics comparison
        final_metrics = combined_metrics.groupby(['agent_type', 'run_id']).last().reset_index()
        if 'clusters' in final_metrics.columns:
            final_metrics.boxplot(column='clusters', by='agent_type', ax=axes[1, 0])
            axes[1, 0].set_title('Final Clustering by Agent Type')
            axes[1, 0].set_ylabel('Number of Clusters')
            
        if 'segregation_index' in final_metrics.columns:
            final_metrics.boxplot(column='segregation_index', by='agent_type', ax=axes[1, 1])
            axes[1, 1].set_title('Final Segregation by Agent Type')
            axes[1, 1].set_ylabel('Segregation Index')
        
        plt.tight_layout()
        plt.savefig(output_dir / "comparative_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Saved visualization: {output_dir}/comparative_analysis.png")
        
    except ImportError:
        print("⚠️  Matplotlib not available, skipping visualizations")
    except Exception as e:
        print(f"⚠️  Could not create visualizations: {e}")

def analyze_study(study_path):
    """Analyze a comprehensive study directory"""
    
    study_dir = Path(study_path)
    if not study_dir.exists():
        print(f"❌ Study directory not found: {study_dir}")
        return False
    
    print(f"🔍 Analyzing comprehensive study: {study_dir.name}")
    print("=" * 60)
    
    # Load data
    mechanical_data = load_mechanical_data(study_dir)
    llm_data = load_llm_data(study_dir)
    
    if not mechanical_data and not llm_data:
        print("❌ No data found in study directory")
        return False
    
    # Create analysis directory
    analysis_dir = study_dir / "comprehensive_analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    # Perform analysis
    create_comparative_analysis(mechanical_data, llm_data, analysis_dir)
    
    print(f"\n✅ Analysis complete! Results saved in: {analysis_dir}")
    print(f"📊 Key files:")
    print(f"   - Combined data: combined_metrics.csv, combined_convergence.csv")
    print(f"   - Summary: summary_statistics.csv")
    print(f"   - Visualization: comparative_analysis.png")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Analyze comprehensive study results (mechanical + LLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s comprehensive_study_20250613_160112  # Analyze specific study
  %(prog)s --latest                             # Analyze most recent study
        """
    )
    
    parser.add_argument("study_path", nargs="?", 
                       help="Path to comprehensive study directory")
    parser.add_argument("--latest", action="store_true",
                       help="Analyze the most recent comprehensive study")
    
    args = parser.parse_args()
    
    if args.latest:
        # Find most recent comprehensive study
        study_dirs = list(Path(".").glob("comprehensive_study_*"))
        if not study_dirs:
            print("❌ No comprehensive study directories found")
            return 1
        
        latest_study = max(study_dirs, key=lambda x: x.name)
        study_path = latest_study
        
    elif args.study_path:
        study_path = args.study_path
        
    else:
        print("❌ Please specify a study path or use --latest")
        return 1
    
    success = analyze_study(study_path)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())