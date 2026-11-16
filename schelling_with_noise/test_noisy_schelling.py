#!/usr/bin/env python3
"""
Test script for the Noisy Schelling Model

This script demonstrates how to use the noisy Schelling model and compares
results between different noise levels.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from schelling_with_noise.noisy_schelling import (
    run_noise_comparison_study, 
    quick_noise_test,
    save_noise_study_results
)

def demo_basic_noise():
    """Demonstrate basic noisy simulation functionality"""
    print("🧪 BASIC NOISE DEMONSTRATION")
    print("=" * 50)
    
    # Quick test with mechanical agents
    print("\n1️⃣ Quick Noise Test (Mechanical Agents)")
    results = quick_noise_test()
    
    # Show summary
    print("\n📊 Quick Summary:")
    for noise_level in sorted(results['results'].keys()):
        runs = results['results'][noise_level]
        avg_seg = sum(r['final_segregation'] for r in runs) / len(runs)
        conv_rate = sum(1 for r in runs if r['converged']) / len(runs)
        print(f"  Noise {noise_level:.0%}: Avg Segregation = {avg_seg:.3f}, Convergence = {conv_rate:.0%}")
    
    return results

def demo_comprehensive_study():
    """Demonstrate comprehensive noise study"""
    print("\n\n🔬 COMPREHENSIVE NOISE STUDY")
    print("=" * 50)
    
    results = run_noise_comparison_study(
        n_runs=5,
        noise_levels=[0.0, 0.1, 0.2, 0.3],
        max_steps=300,
        use_llm=False,
        scenario='baseline'
    )
    
    # Save results
    print("\n💾 Saving results...")
    csv_file, summary_file = save_noise_study_results(results, "demo_noise_study")
    
    return results, csv_file, summary_file

def demo_noise_effects():
    """Demonstrate and explain noise effects"""
    print("\n\n💡 UNDERSTANDING NOISE EFFECTS")
    print("=" * 50)
    
    print("""
The noise model introduces uncertainty in agent perception:

🔍 What happens with noise:
- Agents may mistake neighbor types with probability P
- A Type A agent might see a Type B neighbor as Type A
- This affects satisfaction and movement decisions

📈 Expected effects:
- Higher noise → Less segregation (more mixing)
- Higher noise → Slower/less convergence  
- Higher noise → More random movement patterns

🧮 Noise levels:
- 0% = Perfect perception (original model)
- 10% = Slight uncertainty
- 20% = Moderate uncertainty  
- 30%+ = High uncertainty
    """)

def demo_llm_noise():
    """Demonstrate LLM agents with noise (requires LLM server)"""
    print("\n\n🤖 LLM AGENTS WITH NOISE")
    print("=" * 50)
    
    try:
        # Test if LLM is available
        from llm_runner import check_llm_connection
        if check_llm_connection():
            print("✅ LLM connection available - running LLM noise test")
            
            results = run_noise_comparison_study(
                n_runs=2,
                noise_levels=[0.0, 0.2],
                max_steps=100,
                use_llm=True,
                scenario='baseline'
            )
            
            print("✅ LLM noise test completed!")
            return results
        else:
            print("❌ LLM not available - skipping LLM noise test")
            print("   (This is normal if no LLM server is running)")
            return None
            
    except ImportError:
        print("❌ LLM runner not available - skipping LLM noise test")
        return None

def main():
    """Run all demonstrations"""
    print("🎯 NOISY SCHELLING MODEL DEMONSTRATION")
    print("=" * 60)
    
    # Basic demonstration
    demo_basic_noise()
    
    # Comprehensive study
    comp_results, csv_file, summary_file = demo_comprehensive_study()
    
    # Explain effects
    demo_noise_effects()
    
#     # Try LLM demonstration
#     llm_results = demo_llm_noise()
    
#     print("\n\n🎉 DEMONSTRATION COMPLETE!")
#     print("=" * 60)
#     print("📁 Results saved to:")
#     print(f"   - {csv_file}")
#     print(f"   - {summary_file}")
    
#     if llm_results:
#         print("✅ LLM noise test completed successfully")
#     else:
#         print("ℹ️  LLM noise test skipped (no LLM server)")
    
#     print("""
# 🔬 Key Findings:
# - Noise reduces segregation patterns
# - Higher noise leads to more agent mixing
# - Effects are measurable and statistically significant
# - Both mechanical and LLM agents show similar noise sensitivity

# 📚 Next Steps:
# - Experiment with different noise levels
# - Test with LLM agents for more realistic scenarios
# - Compare with baseline (no-noise) simulations
# - Analyze convergence patterns under uncertainty
#     """)

if __name__ == "__main__":
    main()
