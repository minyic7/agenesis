#!/usr/bin/env python3
"""
Demo script showing complete pipeline: perception → memory → cognition → action
"""
import asyncio
from dotenv import load_dotenv

from agenesis.core import Agent

# Load environment variables
load_dotenv()


async def demo_complete_pipeline():
    """Demonstrate complete agent pipeline"""
    print("🤖 Complete Agent Pipeline Demo\n")
    
    # Create anonymous agent (no persistent memory)
    agent = Agent(profile=None)
    
    print("✅ Agent initialized:")
    profile_info = agent.get_profile_info()
    print(f"   Anonymous: {profile_info['is_anonymous']}")
    print(f"   Persistent memory: {profile_info['has_persistent_memory']}")
    print(f"   LLM available: {agent.cognition.use_llm}")
    if agent.cognition.use_llm:
        print(f"   LLM provider: {type(agent.cognition.llm_provider).__name__}")
        print(f"   Action LLM: {agent.action.use_llm}")
    print()
    
    # Test different types of interactions
    test_inputs = [
        "Hello! How are you today?",
        "What is machine learning?", 
        "Can you help me learn Python programming?",
        "I am working on an AI project",
        "How do neural networks work?"
    ]
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"🔄 Interaction #{i}")
        print(f"👤 User: {test_input}")
        
        # Process through complete pipeline
        response = await agent.process_input(test_input)
        
        print(f"🤖 Agent: {response}")
        
        # Show session context building
        context = agent.get_session_context(2)
        print(f"📝 Session context: {len(context)} items")
        
        print("-" * 80)
    
    print("\n📊 Final Session Summary:")
    profile_info = agent.get_profile_info()
    print(f"   Total interactions: {profile_info['session_size']}")
    print(f"   Current focus: {profile_info['current_focus']}")
    
    print("\n🎉 Complete pipeline demo finished!")


async def demo_context_awareness():
    """Demonstrate context awareness across interactions"""
    print("\n🧠 Context Awareness Demo\n")
    
    agent = Agent(profile=None)
    
    # Build context through related interactions
    interactions = [
        "I'm learning about Python programming",
        "What are variables in Python?",
        "How do I create functions?",
        "Can you explain loops?"
    ]
    
    for i, interaction in enumerate(interactions, 1):
        print(f"💬 {i}. User: {interaction}")
        response = await agent.process_input(interaction)
        print(f"   Agent: {response[:100]}{'...' if len(response) > 100 else ''}")
        print()
    
    # Show accumulated context
    context = agent.get_session_context(10)
    print(f"📚 Built context from {len(context)} interactions:")
    for i, ctx in enumerate(context, 1):
        print(f"   {i}. {ctx.content}")


if __name__ == "__main__":
    asyncio.run(demo_complete_pipeline())
    asyncio.run(demo_context_awareness())