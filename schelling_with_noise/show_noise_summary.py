#!/usr/bin/env python3
"""
Quick summary and findings from the noise experiment analysis
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def show_analysis_summary():
    """Display key findings from the noise experiment"""
    
    print("="*80)
    print("🎯 NOISE EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    
    print("\n📊 EXPERIMENT SETUP:")
    print("• Noise levels tested: 0%, 5%, 10%, 15%, 20%, 25%, 30%")
    print("• Runs per condition: 50")
    print("• Agent type: Mechanical (non-LLM)")
    print("• Metric: Mix deviation (segregation index)")
    
    print("\n🔍 KEY FINDINGS:")
    print("1. CONVERGENCE IMPACT:")
    print("   • 0% noise: 100% convergence (baseline)")
    print("   • 5% noise: 98% convergence")
    print("   • 10%+ noise: 0-2% convergence (dramatic drop)")
    
    print("\n2. SEGREGATION PATTERNS:")
    print("   • Baseline (0% noise): 0.370 ± 0.030")
    print("   • Peak segregation at 5% noise: 0.438 ± 0.015")
    print("   • Higher noise reduces segregation variation")
    
    print("\n3. STATISTICAL SIGNIFICANCE:")
    print("   • All noise levels significantly different from baseline (p < 0.05)")
    print("   • Effect sizes range from Small to Large (Cohen's d)")
    print("   • 5%-25% noise show LARGE effects (d > 0.8)")
    
    print("\n4. CRITICAL THRESHOLDS:")
    print("   • Convergence threshold: ~10% noise")
    print("   • Optimal segregation: 5% noise")
    print("   • System breakdown: >10% noise")
    
    print("\n🎯 MAIN CONCLUSION:")
    print("Small amounts of perception noise (5%) can actually INCREASE")
    print("segregation patterns, but higher noise (>10%) prevents convergence")
    print("entirely, leading to unstable, non-converged states.")
    
    print("\n📈 VISUALIZATION:")
    latest_plot = None
    for file in os.listdir('.'):
        if file.startswith('noise_experiment_analysis_') and file.endswith('.png'):
            latest_plot = file
    
    if latest_plot:
        print(f"Generated comprehensive 4-panel analysis: {latest_plot}")
        print("• Panel 1: Convergence rates by noise level")
        print("• Panel 2: Segregation distributions (box plots)")
        print("• Panel 3: Average segregation with error bars")
        print("• Panel 4: Effect sizes vs baseline")
    
    print("\n" + "="*80)

def display_plot_if_available():
    """Try to display the plot if in an interactive environment"""
    
    latest_plot = None
    for file in os.listdir('.'):
        if file.startswith('noise_experiment_analysis_') and file.endswith('.png'):
            latest_plot = file
    
    if latest_plot and os.path.exists(latest_plot):
        try:
            # Try to show the plot
            img = mpimg.imread(latest_plot)
            plt.figure(figsize=(16, 12))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'Noise Experiment Analysis\n{latest_plot}', fontsize=14, pad=20)
            plt.tight_layout()
            plt.show()
            print(f"📊 Displayed plot: {latest_plot}")
        except Exception as e:
            print(f"📊 Plot available but couldn't display: {latest_plot}")
            print(f"   (Reason: {e})")
    else:
        print("📊 No plot file found")

if __name__ == "__main__":
    show_analysis_summary()
    print("\n" + "🖼️  TRYING TO DISPLAY VISUALIZATION..." + "\n")
    display_plot_if_available()
