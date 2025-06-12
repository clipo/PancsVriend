#!/usr/bin/env python3
"""
Standalone LLM connectivity checker for the Schelling segregation simulation
"""

import requests
import time
import argparse
import config as cfg

def comprehensive_llm_check(llm_model=None, llm_url=None, llm_api_key=None):
    """
    Comprehensive LLM connectivity and performance check
    """
    model = llm_model or cfg.OLLAMA_MODEL
    url = llm_url or cfg.OLLAMA_URL
    api_key = llm_api_key or cfg.OLLAMA_API_KEY
    
    print("="*60)
    print("LLM CONNECTIVITY CHECK")
    print("="*60)
    
    print(f"\nConfiguration:")
    print(f"  URL: {url}")
    print(f"  Model: {model}")
    print(f"  API Key: {'*' * (len(api_key) - 8) + api_key[-8:] if len(api_key) > 8 else 'Set'}")
    if llm_model or llm_url or llm_api_key:
        print(f"  (Using command-line override)")
    
    # Test 1: Basic connectivity
    print(f"\n1. Testing basic connectivity...")
    try:
        test_payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Respond with only the word 'TEST' and nothing else."}],
            "stream": False,
            "temperature": 0
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        start_time = time.time()
        response = requests.post(url, headers=headers, json=test_payload, timeout=15)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            if "choices" in data and data["choices"]:
                content = data["choices"][0]["message"]["content"].strip()
                print(f"   ✅ SUCCESS - Response time: {elapsed:.2f}s")
                print(f"   Response: '{content}'")
            else:
                print(f"   ❌ FAILED - Invalid response structure")
                print(f"   Response: {data}")
                return False
        else:
            print(f"   ❌ FAILED - HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"   ❌ FAILED - Timeout after 15 seconds")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"   ❌ FAILED - Connection error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ FAILED - Error: {type(e).__name__}: {e}")
        return False
    
    # Test 2: Response parsing test
    print(f"\n2. Testing response parsing...")
    try:
        parse_test_payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Respond with exactly: (1, 2)"}],
            "stream": False,
            "temperature": 0
        }
        
        response = requests.post(url, headers=headers, json=parse_test_payload, timeout=15)
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            
            # Test if we can parse coordinates
            import re
            match = re.search(r"\((\d+),\s*(\d+)\)", content)
            if match:
                print(f"   ✅ SUCCESS - Parsing works")
                print(f"   Response: '{content}'")
                print(f"   Parsed: ({match.group(1)}, {match.group(2)})")
            else:
                print(f"   ⚠️  WARNING - Parsing may be unreliable")
                print(f"   Response: '{content}'")
        else:
            print(f"   ❌ FAILED - HTTP {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ FAILED - Error: {type(e).__name__}: {e}")
    
    # Test 3: Load test (multiple quick requests)
    print(f"\n3. Testing load handling (5 quick requests)...")
    success_count = 0
    total_time = 0
    
    for i in range(5):
        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=test_payload, timeout=10)
            elapsed = time.time() - start_time
            total_time += elapsed
            
            if response.status_code == 200:
                success_count += 1
                print(f"   Request {i+1}: ✅ ({elapsed:.2f}s)")
            else:
                print(f"   Request {i+1}: ❌ HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   Request {i+1}: ❌ {type(e).__name__}")
    
    print(f"   Success rate: {success_count}/5 ({success_count/5*100:.0f}%)")
    if success_count > 0:
        print(f"   Average response time: {total_time/success_count:.2f}s")
    
    # Test 4: Context-aware test
    print(f"\n4. Testing context-aware decision making...")
    try:
        context_payload = {
            "model": model,
            "messages": [{"role": "user", "content": """You are a red team resident living in a neighborhood, considering whether to move to a different house.

You are looking at your immediate 3x3 neighborhood:
S O E
O S E  
E E E

Where:
- 'S' = neighbors who are also red team residents like you
- 'O' = neighbors from the blue team resident community  
- 'E' = empty houses you could move to

As a real person, think about where you'd genuinely want to live, then respond with ONLY:
- The coordinates (row, col) of the empty house you'd move to, OR
- None (if you prefer to stay where you are)

Your choice:"""}],
            "stream": False,
            "temperature": 0.3
        }
        
        response = requests.post(url, headers=headers, json=context_payload, timeout=15)
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            print(f"   ✅ SUCCESS - Context response")
            print(f"   Response: '{content}'")
        else:
            print(f"   ❌ FAILED - HTTP {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ FAILED - Error: {type(e).__name__}: {e}")
    
    print(f"\n" + "="*60)
    if success_count >= 4:
        print("OVERALL: ✅ LLM is ready for experiments")
        print("\nYou can proceed with:")
        print("  python run_experiments.py")
    elif success_count >= 2:
        print("OVERALL: ⚠️  LLM works but may be slow/unreliable")
        print("\nConsider:")
        print("  - Using fewer parallel requests")
        print("  - Increasing timeouts")
        print("  - Using a faster model")
    else:
        print("OVERALL: ❌ LLM is not ready")
        print("\nPlease check:")
        print("  1. Is Ollama running? (ollama serve)")
        print(f"  2. Is the model pulled? (ollama pull {model})")
        print("  3. Is the URL correct in config.py?")
        print("  4. Is the API key valid?")
    
    print("="*60)
    return success_count >= 4

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check LLM connectivity and performance")
    parser.add_argument('--llm-model', type=str, help='LLM model to use (overrides config.py)')
    parser.add_argument('--llm-url', type=str, help='LLM API URL (overrides config.py)')
    parser.add_argument('--llm-api-key', type=str, help='LLM API key (overrides config.py)')
    
    args = parser.parse_args()
    comprehensive_llm_check(args.llm_model, args.llm_url, args.llm_api_key)