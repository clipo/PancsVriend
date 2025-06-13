#!/usr/bin/env python3
"""
Design Space Analysis Module
Generates comprehensive comparative analysis across experimental configurations
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class DesignSpaceAnalyzer:
    """Comprehensive analysis of experimental design space results"""
    
    def __init__(self, base_output_dir="design_space_exploration"):
        self.base_output_dir = Path(base_output_dir)
        self.analysis_dir = self.base_output_dir / "analysis"
        self.analysis_dir.mkdir(exist_ok=True)
        
        # Load all experiment results
        self.experiment_data = self.load_all_experiments()
        self.metrics_df = self.create_metrics_dataframe()
        
    def load_all_experiments(self) -> List[Dict[str, Any]]:
        """Load all completed experiment results"""
        experiments_dir = self.base_output_dir / "experiments"
        all_experiments = []
        
        for exp_dir in experiments_dir.glob("exp_*"):
            config_file = exp_dir / "experiment_config.json"
            results_file = exp_dir / "results.json"
            
            if config_file.exists() and results_file.exists():
                try:
                    with open(config_file) as f:
                        config = json.load(f)
                    with open(results_file) as f:
                        results = json.load(f)
                    
                    if config.get("status") == "completed":
                        all_experiments.append({
                            "config": config,
                            "results": results
                        })
                except Exception as e:
                    print(f"Error loading {exp_dir}: {e}")
                    
        print(f"📊 Loaded {len(all_experiments)} completed experiments")
        return all_experiments
    
    def create_metrics_dataframe(self) -> pd.DataFrame:
        """Create comprehensive metrics dataframe for analysis"""
        rows = []
        
        for exp_data in self.experiment_data:
            config = exp_data["config"]
            results = exp_data["results"]
            
            # Extract metrics from each run
            for run_idx, run_result in enumerate(results):
                
                # Basic experiment identifiers
                row = {
                    "experiment_id": config["experiment_id"],
                    "experiment_name": config["name"],
                    "run_id": run_idx,
                    
                    # Configuration variables
                    "llm_model": config["llm_model"],
                    "agent_type": config["agent_type"],
                    "scenario": config["scenario"],
                    "grid_size": config["grid_size"],
                    "num_agents": config["num_type_a"] + config["num_type_b"],
                    
                    # Run outcomes
                    "converged": run_result.get("converged", False),
                    "convergence_step": run_result.get("convergence_step", config["max_steps"]),
                    "total_steps": run_result.get("steps", config["max_steps"]),
                    "llm_call_count": run_result.get("llm_call_count", 0),
                    "llm_failure_count": run_result.get("llm_failure_count", 0),
                }
                
                # Final segregation metrics (from last step)
                if "metrics_history" in run_result and run_result["metrics_history"]:
                    final_metrics = run_result["metrics_history"][-1]
                    row.update({
                        "final_clusters": final_metrics.get("clusters", 0),
                        "final_switch_rate": final_metrics.get("switch_rate", 0),
                        "final_distance": final_metrics.get("distance", 0),
                        "final_mix_deviation": final_metrics.get("mix_deviation", 0),
                        "final_share": final_metrics.get("share", 0),
                        "final_ghetto_rate": final_metrics.get("ghetto_rate", 0)
                    })
                    
                # Convergence speed (steps to 90% of final value)
                if "metrics_history" in run_result and len(run_result["metrics_history"]) > 5:
                    row["convergence_speed"] = self.calculate_convergence_speed(
                        run_result["metrics_history"]
                    )
                    
                # Memory-specific metrics
                if "agent_memories" in run_result:
                    memories = run_result["agent_memories"]
                    if memories:
                        row.update({
                            "avg_moves_per_agent": np.mean([a.get("total_moves", 0) for a in memories]),
                            "avg_final_satisfaction": np.mean([a.get("final_satisfaction", 5) for a in memories if a.get("final_satisfaction") is not None]),
                            "avg_time_in_final_location": np.mean([a.get("time_in_final_location", 0) for a in memories])
                        })
                        
                rows.append(row)
                
        df = pd.DataFrame(rows)
        print(f"📈 Created metrics dataframe: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def calculate_convergence_speed(self, metrics_history: List[Dict]) -> float:
        """Calculate convergence speed (steps to reach 90% of final segregation)"""
        if len(metrics_history) < 5:
            return len(metrics_history)
            
        # Use clusters as primary segregation metric
        cluster_values = [m.get("clusters", 0) for m in metrics_history]
        final_value = cluster_values[-1]
        target_value = final_value * 0.9
        
        for step, value in enumerate(cluster_values):
            if value >= target_value:
                return step
                
        return len(cluster_values)
    
    def generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive summary statistics"""
        summary = {}
        
        # Overall statistics
        summary["total_experiments"] = len(self.metrics_df["experiment_id"].unique())
        summary["total_runs"] = len(self.metrics_df)
        summary["convergence_rate"] = self.metrics_df["converged"].mean()
        
        # By configuration
        for grouping in ["llm_model", "agent_type", "scenario", "grid_size"]:
            summary[f"by_{grouping}"] = {}
            
            grouped = self.metrics_df.groupby(grouping).agg({
                "converged": ["count", "mean"],
                "convergence_step": ["mean", "std"],
                "final_clusters": ["mean", "std"],
                "final_switch_rate": ["mean", "std"]
            }).round(3)
            
            summary[f"by_{grouping}"] = grouped.to_dict()
            
        # Save summary
        summary_file = self.analysis_dir / "summary_statistics.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        print(f"📋 Summary statistics saved: {summary_file}")
        return summary
    
    def create_convergence_analysis(self):
        """Analyze convergence patterns across configurations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Convergence Analysis Across Configurations", fontsize=16)
        
        # 1. Convergence rate by configuration
        convergence_by_config = self.metrics_df.groupby(["llm_model", "agent_type", "scenario"])["converged"].mean().reset_index()
        sns.barplot(data=convergence_by_config, x="scenario", y="converged", 
                   hue="agent_type", ax=axes[0,0])
        axes[0,0].set_title("Convergence Rate by Scenario and Agent Type")
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Convergence speed distribution
        converged_data = self.metrics_df[self.metrics_df["converged"]]
        sns.boxplot(data=converged_data, x="agent_type", y="convergence_step", 
                   hue="scenario", ax=axes[0,1])
        axes[0,1].set_title("Convergence Speed Distribution")
        
        # 3. Final segregation levels
        sns.scatterplot(data=self.metrics_df, x="convergence_step", y="final_clusters",
                       hue="agent_type", style="scenario", ax=axes[1,0])
        axes[1,0].set_title("Convergence Speed vs Final Segregation")
        
        # 4. Grid size effects
        if len(self.metrics_df["grid_size"].unique()) > 1:
            sns.boxplot(data=self.metrics_df, x="grid_size", y="final_clusters",
                       hue="agent_type", ax=axes[1,1])
            axes[1,1].set_title("Grid Size Effects on Final Segregation")
        
        plt.tight_layout()
        convergence_file = self.analysis_dir / "convergence_analysis.png"
        plt.savefig(convergence_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Convergence analysis saved: {convergence_file}")
    
    def create_segregation_metrics_comparison(self):
        """Compare all segregation metrics across configurations"""
        
        segregation_metrics = ["final_clusters", "final_switch_rate", "final_distance", 
                             "final_mix_deviation", "final_share", "final_ghetto_rate"]
        
        # Filter metrics that exist in the data
        available_metrics = [m for m in segregation_metrics if m in self.metrics_df.columns]
        
        if not available_metrics:
            print("⚠️ No segregation metrics found in data")
            return
            
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        fig.suptitle("Segregation Metrics Comparison", fontsize=16)
        
        for i, metric in enumerate(available_metrics):
            if i < len(axes):
                # Box plot comparing agent types across scenarios
                sns.boxplot(data=self.metrics_df, x="scenario", y=metric, 
                           hue="agent_type", ax=axes[i])
                axes[i].set_title(f"{metric.replace('final_', '').replace('_', ' ').title()}")
                axes[i].tick_params(axis='x', rotation=45)
                
                # Remove legend from all but first plot
                if i > 0:
                    axes[i].get_legend().remove()
        
        # Hide unused subplots
        for i in range(len(available_metrics), len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        metrics_file = self.analysis_dir / "segregation_metrics_comparison.png"
        plt.savefig(metrics_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Segregation metrics comparison saved: {metrics_file}")
    
    def create_memory_vs_standard_analysis(self):
        """Specific analysis comparing memory vs standard agents"""
        
        if "memory" not in self.metrics_df["agent_type"].values or "standard" not in self.metrics_df["agent_type"].values:
            print("⚠️ Both memory and standard agents needed for comparison")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Memory vs Standard Agents Analysis", fontsize=16)
        
        # 1. Convergence speed comparison
        sns.violinplot(data=self.metrics_df, x="agent_type", y="convergence_step", ax=axes[0,0])
        axes[0,0].set_title("Convergence Speed Distribution")
        
        # 2. Final segregation comparison
        sns.violinplot(data=self.metrics_df, x="agent_type", y="final_clusters", ax=axes[0,1])
        axes[0,1].set_title("Final Segregation Level")
        
        # 3. Memory-specific metrics (if available)
        if "avg_moves_per_agent" in self.metrics_df.columns:
            memory_data = self.metrics_df[self.metrics_df["agent_type"] == "memory"]
            sns.scatterplot(data=memory_data, x="avg_moves_per_agent", y="final_clusters",
                           hue="scenario", ax=axes[1,0])
            axes[1,0].set_title("Agent Mobility vs Final Segregation (Memory Agents)")
        
        # 4. Scenario-specific comparison
        scenario_comparison = self.metrics_df.groupby(["scenario", "agent_type"])["final_clusters"].mean().reset_index()
        pivot_data = scenario_comparison.pivot(index="scenario", columns="agent_type", values="final_clusters")
        sns.heatmap(pivot_data, annot=True, cmap="viridis", ax=axes[1,1])
        axes[1,1].set_title("Final Segregation by Scenario and Agent Type")
        
        plt.tight_layout()
        memory_file = self.analysis_dir / "memory_vs_standard_analysis.png"
        plt.savefig(memory_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Memory vs standard analysis saved: {memory_file}")
    
    def create_statistical_significance_tests(self):
        """Perform statistical significance tests"""
        
        results = {}
        
        # Test 1: Memory vs Standard convergence speed
        if "memory" in self.metrics_df["agent_type"].values and "standard" in self.metrics_df["agent_type"].values:
            memory_convergence = self.metrics_df[self.metrics_df["agent_type"] == "memory"]["convergence_step"]
            standard_convergence = self.metrics_df[self.metrics_df["agent_type"] == "standard"]["convergence_step"]
            
            stat, p_value = stats.mannwhitneyu(memory_convergence, standard_convergence, alternative='two-sided')
            results["memory_vs_standard_convergence"] = {
                "test": "Mann-Whitney U",
                "statistic": float(stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "memory_median": float(memory_convergence.median()),
                "standard_median": float(standard_convergence.median())
            }
        
        # Test 2: ANOVA across scenarios
        scenarios = self.metrics_df["scenario"].unique()
        if len(scenarios) > 2:
            scenario_groups = [self.metrics_df[self.metrics_df["scenario"] == s]["final_clusters"] for s in scenarios]
            stat, p_value = stats.kruskal(*scenario_groups)
            results["scenario_differences"] = {
                "test": "Kruskal-Wallis",
                "statistic": float(stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "scenarios_tested": list(scenarios)
            }
        
        # Test 3: Grid size effects
        grid_sizes = self.metrics_df["grid_size"].unique()
        if len(grid_sizes) > 1:
            grid_groups = [self.metrics_df[self.metrics_df["grid_size"] == g]["final_clusters"] for g in grid_sizes]
            stat, p_value = stats.kruskal(*grid_groups)
            results["grid_size_effects"] = {
                "test": "Kruskal-Wallis",
                "statistic": float(stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "grid_sizes_tested": list(map(int, grid_sizes))
            }
        
        # Save results
        stats_file = self.analysis_dir / "statistical_tests.json"
        with open(stats_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"📊 Statistical tests saved: {stats_file}")
        return results
    
    def create_pca_analysis(self):
        """Principal Component Analysis of experimental configurations"""
        
        # Prepare data for PCA
        numeric_cols = ["grid_size", "num_agents", "convergence_step", "total_steps", 
                       "final_clusters", "final_switch_rate", "final_distance"]
        
        # Filter columns that exist in the data
        available_cols = [col for col in numeric_cols if col in self.metrics_df.columns]
        
        if len(available_cols) < 3:
            print("⚠️ Not enough numeric variables for PCA")
            return
            
        # Prepare data
        pca_data = self.metrics_df[available_cols].dropna()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pca_data)
        
        # Perform PCA
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        
        # Create visualizations
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Explained variance
        explained_var = pca.explained_variance_ratio_
        axes[0].plot(range(1, len(explained_var) + 1), np.cumsum(explained_var), 'bo-')
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Cumulative Explained Variance')
        axes[0].set_title('PCA Explained Variance')
        axes[0].grid(True)
        
        # 2. PCA scatter plot
        pca_df = pd.DataFrame({
            'PC1': pca_result[:, 0],
            'PC2': pca_result[:, 1],
            'agent_type': self.metrics_df.loc[pca_data.index, 'agent_type'],
            'scenario': self.metrics_df.loc[pca_data.index, 'scenario']
        })
        
        sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='agent_type', 
                       style='scenario', ax=axes[1])
        axes[1].set_title('PCA: First Two Components')
        
        plt.tight_layout()
        pca_file = self.analysis_dir / "pca_analysis.png"
        plt.savefig(pca_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save component loadings
        loadings = pd.DataFrame(
            pca.components_[:3].T,  # First 3 components
            columns=['PC1', 'PC2', 'PC3'],
            index=available_cols
        )
        
        loadings_file = self.analysis_dir / "pca_loadings.csv"
        loadings.to_csv(loadings_file)
        
        print(f"📊 PCA analysis saved: {pca_file}")
        print(f"📊 PCA loadings saved: {loadings_file}")
    
    def create_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        
        print(f"\n📊 GENERATING COMPREHENSIVE ANALYSIS")
        print(f"====================================")
        
        # Generate all analyses
        summary = self.generate_summary_statistics()
        self.create_convergence_analysis()
        self.create_segregation_metrics_comparison()
        self.create_memory_vs_standard_analysis()
        stats_results = self.create_statistical_significance_tests()
        self.create_pca_analysis()
        
        # Create summary report
        report_file = self.analysis_dir / "comprehensive_report.md"
        with open(report_file, 'w') as f:
            f.write("# Experimental Design Space Analysis Report\n\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")
            
            f.write("## Summary Statistics\n")
            f.write(f"- Total Experiments: {summary['total_experiments']}\n")
            f.write(f"- Total Runs: {summary['total_runs']}\n")
            f.write(f"- Overall Convergence Rate: {summary['convergence_rate']:.3f}\n\n")
            
            f.write("## Key Findings\n")
            
            if stats_results.get("memory_vs_standard_convergence"):
                result = stats_results["memory_vs_standard_convergence"]
                f.write(f"### Memory vs Standard Agents\n")
                f.write(f"- Memory agents median convergence: {result['memory_median']:.1f} steps\n")
                f.write(f"- Standard agents median convergence: {result['standard_median']:.1f} steps\n")
                f.write(f"- Statistically significant difference: {result['significant']}\n")
                f.write(f"- p-value: {result['p_value']:.6f}\n\n")
                
            f.write("## Generated Visualizations\n")
            f.write("- convergence_analysis.png: Convergence patterns\n")
            f.write("- segregation_metrics_comparison.png: All segregation metrics\n")
            f.write("- memory_vs_standard_analysis.png: Memory vs standard comparison\n")
            f.write("- pca_analysis.png: Principal component analysis\n\n")
            
            f.write("## Data Files\n")
            f.write("- summary_statistics.json: Detailed statistics\n")
            f.write("- statistical_tests.json: Significance test results\n")
            f.write("- pca_loadings.csv: PCA component loadings\n")
        
        print(f"📄 Comprehensive report saved: {report_file}")
        print(f"📁 All analysis files in: {self.analysis_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Design Space Analysis")
    parser.add_argument("--input-dir", default="design_space_exploration",
                       help="Input directory with experiment results")
    
    args = parser.parse_args()
    
    analyzer = DesignSpaceAnalyzer(args.input_dir)
    analyzer.create_comprehensive_report()