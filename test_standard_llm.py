#!/usr/bin/env python3
"""
Test if standard LLM runner is working properly
"""

import time
import subprocess
import tempfile
import os

def test_standard_llm_runner():
    """Test the standard LLM runner with small configuration"""
    
    print("🧪 TESTING STANDARD LLM RUNNER")
    print("=" * 50)
    
    # Save original config
    import config as cfg
    orig_grid = cfg.GRID_SIZE
    orig_a = cfg.NUM_TYPE_A
    orig_b = cfg.NUM_TYPE_B
    
    try:
        # Set small test configuration
        cfg.GRID_SIZE = 5
        cfg.NUM_TYPE_A = 3
        cfg.NUM_TYPE_B = 3
        
        print(f"📊 Test Configuration:")
        print(f"   Grid: {cfg.GRID_SIZE}x{cfg.GRID_SIZE}")
        print(f"   Agents: {cfg.NUM_TYPE_A + cfg.NUM_TYPE_B}")
        print(f"   Scenario: baseline")
        print(f"   Runs: 1")
        
        cmd = [
            "python", "llm_runner.py",
            "--scenario", "baseline",
            "--runs", "1",
            "--max-steps", "50"  # Limit steps for quick test
        ]
        
        print(f"\n🚀 Running command: {' '.join(cmd)}")
        print(f"⏱️  Starting timer...")
        
        start_time = time.time()
        
        # Run with timeout
        try:
            result = subprocess.run(
                cmd,
                timeout=60,  # 60 second timeout
                capture_output=True,
                text=True
            )
            
            runtime = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ SUCCESS! Completed in {runtime:.1f} seconds")
                
                # Parse output for key metrics
                output_lines = result.stdout.split('\n')
                for line in output_lines[-10:]:  # Last 10 lines
                    if 'LLM calls' in line or 'convergence' in line or 'Completed' in line:
                        print(f"   📊 {line.strip()}")
                
                if runtime > 30:
                    print(f"   ⚠️  Runtime ({runtime:.1f}s) is longer than expected for this small test")
                
                return True
                
            else:
                print(f"❌ FAILED with exit code {result.returncode}")
                print(f"   Runtime: {runtime:.1f} seconds")
                print(f"   STDOUT: {result.stdout[-500:]}")  # Last 500 chars
                print(f"   STDERR: {result.stderr[-500:]}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ TIMEOUT after 60 seconds!")
            print(f"   This confirms the hanging issue")
            return False
            
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        return False
        
    finally:
        # Restore original config
        cfg.GRID_SIZE = orig_grid
        cfg.NUM_TYPE_A = orig_a
        cfg.NUM_TYPE_B = orig_b

def main():
    print("🔍 DIAGNOSING LLM RUNNER HANGING ISSUE")
    print("=" * 60)
    
    # Check if there are running processes we should kill first
    print("🔍 Checking for running processes...")
    result = subprocess.run(
        ["ps", "aux"],
        capture_output=True,
        text=True
    )
    
    llm_processes = [line for line in result.stdout.split('\n') 
                    if 'llm_runner.py' in line and 'grep' not in line]
    
    if llm_processes:
        print(f"⚠️  Found {len(llm_processes)} running llm_runner processes:")
        for proc in llm_processes:
            parts = proc.split()
            if len(parts) > 1:
                pid = parts[1]
                print(f"   PID {pid}: {' '.join(parts[10:])}")
        
        print(f"\n💡 You may want to kill these processes first:")
        for proc in llm_processes:
            parts = proc.split()
            if len(parts) > 1:
                pid = parts[1]
                print(f"   kill {pid}")
        
        print(f"\n🤔 Should I continue with the test anyway? (existing processes might interfere)")
        
    # Run the test
    success = test_standard_llm_runner()
    
    if success:
        print(f"\n🎉 Standard LLM runner is working correctly!")
        print(f"   The hanging issue might be specific to certain configurations")
        print(f"   or due to process conflicts.")
    else:
        print(f"\n🚨 Standard LLM runner has issues!")
        print(f"   This explains why experiments are hanging.")
        
        print(f"\n💡 Possible causes:")
        print(f"   1. LLM response times are much slower than expected")
        print(f"   2. Threading issues in llm_runner.py")
        print(f"   3. Network connectivity problems")
        print(f"   4. Memory/resource exhaustion")

if __name__ == "__main__":
    main()