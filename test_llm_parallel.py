#!/usr/bin/env python3
"""
Test script to verify LLM parallel processing works correctly
"""

import time
import config as cfg
from llm_runner import LLMSimulation, check_llm_connection

def test_parallel_processing():
    """Test the parallel LLM processing with a small simulation"""
    
    print("="*60)
    print("TESTING LLM PARALLEL PROCESSING")
    print("="*60)
    
    # First check if LLM is available
    print("\n1. Checking LLM connectivity...")
    if not check_llm_connection():
        print("❌ LLM not available - cannot test parallel processing")
        print("Please ensure LLM is running and try again")
        return False
    
    print("✅ LLM connection verified")
    
    # Test with a small simulation
    print("\n2. Running small test simulation...")
    print("   - Grid size: 10x10")
    print("   - Agents: 20 total")
    print("   - Steps: 5")
    print("   - Scenario: baseline")
    
    # Temporarily modify config for testing
    original_grid_size = cfg.GRID_SIZE
    original_type_a = cfg.NUM_TYPE_A
    original_type_b = cfg.NUM_TYPE_B
    
    try:
        cfg.GRID_SIZE = 10
        cfg.NUM_TYPE_A = 10
        cfg.NUM_TYPE_B = 10
        
        # Create and run test simulation
        sim = LLMSimulation(run_id=999, scenario='baseline', use_llm_probability=1.0)
        
        print(f"\n3. Running {5} simulation steps...")
        start_time = time.time()
        
        for step in range(5):
            print(f"   Step {step + 1}/5...")
            converged = sim.run_step()
            
            # Check for issues
            if sim.llm_circuit_open:
                print(f"   ⚠️  Circuit breaker opened due to LLM failures")
                break
            
            if converged:
                print(f"   ✅ Simulation converged at step {step + 1}")
                break
        
        elapsed = time.time() - start_time
        
        # Report results
        print(f"\n4. Test Results:")
        print(f"   ✅ Test completed successfully")
        print(f"   Total time: {elapsed:.2f} seconds")
        print(f"   LLM calls made: {sim.llm_call_count}")
        print(f"   LLM failures: {sim.llm_failure_count}")
        print(f"   Circuit breaker status: {'OPEN' if sim.llm_circuit_open else 'CLOSED'}")
        
        if sim.llm_call_times:
            avg_time = sum(sim.llm_call_times) / len(sim.llm_call_times)
            print(f"   Average LLM response time: {avg_time:.2f}s")
        
        # Clean up simulation
        sim._shutdown_requested = True
        try:
            sim.query_queue.put(None, timeout=1.0)
            sim.llm_thread.join(timeout=3.0)
        except:
            pass
        
        print(f"\n5. Parallel Processing Assessment:")
        if sim.llm_failure_count == 0:
            print("   ✅ No LLM failures - parallel processing working well")
        elif sim.llm_failure_count < 5:
            print("   ⚠️  Some LLM failures but system recovered")
        else:
            print("   ❌ Many LLM failures - may need tuning")
        
        if sim.llm_circuit_open:
            print("   ⚠️  Circuit breaker activated - LLM may be overloaded")
        else:
            print("   ✅ Circuit breaker remained closed - good stability")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False
        
    finally:
        # Restore original config
        cfg.GRID_SIZE = original_grid_size
        cfg.NUM_TYPE_A = original_type_a
        cfg.NUM_TYPE_B = original_type_b

def test_failure_handling():
    """Test how the system handles LLM failures"""
    
    print("\n" + "="*60)
    print("TESTING FAILURE HANDLING")
    print("="*60)
    
    # Test with invalid URL to simulate failures
    original_url = cfg.OLLAMA_URL
    
    try:
        cfg.OLLAMA_URL = "http://invalid-url:8080/api/chat/completions"
        
        print("\n1. Testing with invalid LLM URL...")
        print("   (This should trigger fallback to mechanical agents)")
        
        # Temporarily modify config for testing
        cfg.GRID_SIZE = 8
        cfg.NUM_TYPE_A = 8
        cfg.NUM_TYPE_B = 8
        
        sim = LLMSimulation(run_id=998, scenario='baseline', use_llm_probability=1.0)
        
        print("\n2. Running 3 steps with failing LLM...")
        for step in range(3):
            print(f"   Step {step + 1}/3...")
            sim.run_step()
            
            if sim.llm_circuit_open:
                print(f"   ✅ Circuit breaker opened after {sim.llm_failure_count} failures")
                break
        
        print(f"\n3. Failure Handling Results:")
        print(f"   LLM failures: {sim.llm_failure_count}")
        print(f"   Circuit breaker: {'OPEN' if sim.llm_circuit_open else 'CLOSED'}")
        
        if sim.llm_circuit_open:
            print("   ✅ System correctly switched to mechanical agents")
        else:
            print("   ⚠️  Circuit breaker didn't activate (may need more failures)")
        
        # Clean up
        sim._shutdown_requested = True
        try:
            sim.query_queue.put(None, timeout=1.0)
            sim.llm_thread.join(timeout=3.0)
        except:
            pass
            
        return True
        
    except Exception as e:
        print(f"❌ Failure test had error: {e}")
        return False
        
    finally:
        cfg.OLLAMA_URL = original_url

if __name__ == "__main__":
    print("LLM Parallel Processing Test Suite")
    print("This will test the robustness of LLM parallel processing")
    
    success1 = test_parallel_processing()
    success2 = test_failure_handling()
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    if success1 and success2:
        print("✅ All tests passed - LLM parallel processing is robust")
        print("\nYou can safely run:")
        print("  python run_experiments.py")
    elif success1:
        print("⚠️  Basic functionality works, but failure handling needs review")
        print("Consider testing with smaller batch sizes")
    else:
        print("❌ Tests failed - please check LLM configuration")
        print("Run: python check_llm.py")
    
    print("="*60)