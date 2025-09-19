#!/usr/bin/env python3
"""
Quick integration test for Anthropic provider
"""
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from agenesis.providers import create_llm_provider, AnthropicProvider


async def test_anthropic_integration():
    """Test Anthropic provider with real API calls"""
    print("🔧 Testing Anthropic Integration...")
    
    # Create provider
    provider = create_llm_provider()
    print(f"✅ Provider created: {type(provider).__name__}")
    
    if not isinstance(provider, AnthropicProvider):
        print(f"❌ Expected AnthropicProvider, got {type(provider).__name__}")
        print("❌ Check your ANTHROPIC_API_KEY in .env file")
        return False
    
    try:
        # Test 1: Basic completion
        print("\n📝 Testing completion...")
        result = await provider.complete("What is 2+2? Respond with just the number.")
        print(f"✅ Completion result: '{result}'")
        
        # Test 2: Scoring
        print("\n📊 Testing scoring...")
        score = await provider.score("This is very important information that should be remembered")
        print(f"✅ Scoring result: {score}")
        
        # Test 3: Classification
        print("\n🏷️ Testing classification...")
        classification = await provider.classify(
            "What is the weather like today?", 
            ["question", "request", "statement", "conversation"]
        )
        print(f"✅ Classification result: '{classification}'")
        
        print("\n🎉 All Anthropic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Anthropic test failed: {e}")
        return False


async def test_cognition_with_anthropic():
    """Test cognition module with Anthropic provider"""
    print("\n🧠 Testing Cognition with Anthropic...")
    
    try:
        from agenesis.cognition import BasicCognition
        from agenesis.perception import TextPerception
        from agenesis.memory import ImmediateMemory, WorkingMemory
        
        # Set up components
        cognition = BasicCognition()
        perception = TextPerception()
        immediate_memory = ImmediateMemory()
        working_memory = WorkingMemory()
        
        print(f"✅ Cognition uses LLM: {cognition.use_llm}")
        print(f"✅ LLM provider: {type(cognition.llm_provider).__name__}")
        
        # Process a test input
        test_input = "Can you help me learn Python programming?"
        perception_result = perception.process(test_input)
        immediate_memory.store(perception_result)
        
        print(f"\n📝 Processing: '{test_input}'")
        cognition_result = await cognition.process(immediate_memory, working_memory)
        
        print(f"✅ Intent: {cognition_result.intent}")
        print(f"✅ Context type: {cognition_result.context_type}")
        print(f"✅ Persistence score: {cognition_result.persistence_score}")
        print(f"✅ Summary: {cognition_result.summary}")
        print(f"✅ Confidence: {cognition_result.confidence}")
        print(f"✅ Reasoning: {cognition_result.reasoning}")
        
        print("\n🎉 Cognition with Anthropic works perfectly!")
        return True
        
    except Exception as e:
        print(f"❌ Cognition test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all integration tests"""
    print("🚀 Starting Anthropic Integration Tests\n")
    
    # Test 1: Direct Anthropic provider
    anthropic_success = await test_anthropic_integration()
    
    # Test 2: Cognition with Anthropic
    cognition_success = await test_cognition_with_anthropic()
    
    print(f"\n📊 Results:")
    print(f"   Anthropic Provider: {'✅ PASS' if anthropic_success else '❌ FAIL'}")
    print(f"   Cognition Integration: {'✅ PASS' if cognition_success else '❌ FAIL'}")
    
    if anthropic_success and cognition_success:
        print(f"\n🎉 All integration tests PASSED! Your setup is ready.")
    else:
        print(f"\n❌ Some tests failed. Check your configuration.")


if __name__ == "__main__":
    asyncio.run(main())