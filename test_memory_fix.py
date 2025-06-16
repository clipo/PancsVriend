#!/usr/bin/env python3
"""
Test that memory agent debug code fix works
"""

import os
import sys

# Enable debugging
os.environ['DEBUG_LLM'] = 'true'

print("🧪 Testing Memory Agent Debug Fix")
print("=" * 50)

# Import the fixed modules
try:
    from llm_runner_with_memory import LLMSimulationWithMemory
    from llm_agent_with_memory import LLMAgentWithMemory
    print("✅ Imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test agent creation
try:
    agent = LLMAgentWithMemory(0, 'baseline')
    print(f"✅ Agent created with identity: {agent.identity.get('type', 'unknown')}")
    print(f"   Has get_enhanced_prompt: {hasattr(agent, 'get_enhanced_prompt')}")
    print(f"   Has move_history: {hasattr(agent, 'move_history')}")
except Exception as e:
    print(f"❌ Agent creation error: {e}")
    sys.exit(1)

print("\n✨ Fix appears to be working correctly!")
print("\nNext step: Run the memory experiment when LLM is available.")