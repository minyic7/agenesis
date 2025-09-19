#!/usr/bin/env python3
"""
Demo script showing evolution learning capabilities
"""
import asyncio
from dotenv import load_dotenv

from agenesis.core import Agent

# Load environment variables
load_dotenv()


async def demo_evolution_learning():
    """Demonstrate evolution learning with different interaction types"""
    print("🧠 Evolution Learning Demo\n")
    
    # Create named agent (with persistent memory for evolution)
    agent = Agent(profile="evolution_demo")
    
    print("✅ Agent initialized:")
    profile_info = agent.get_profile_info()
    print(f"   Profile: {agent.profile}")
    print(f"   Persistent memory: {profile_info['has_persistent_memory']}")
    print(f"   Evolution LLM: {agent.evolution.use_llm}")
    if agent.evolution.use_llm:
        print(f"   LLM provider: {type(agent.evolution.llm_provider).__name__}")
    print()
    
    # Test cases: mix of learnable and non-learnable interactions
    test_interactions = [
        # Should NOT be learned (casual)
        "Hello! How are you today?",
        "What's 2+2?",
        "Thanks for the help!",
        
        # SHOULD be learned (preferences/patterns)
        "I prefer detailed technical explanations with code examples",
        "I work in healthcare, so please consider HIPAA compliance",
        "I usually need Python solutions rather than JavaScript",
        
        # Should NOT be learned (one-off questions)
        "What's the weather like?",
        "How do I spell 'definitely'?",
        
        # SHOULD be learned (work context)
        "I'm building a machine learning pipeline for medical data analysis"
    ]
    
    learned_count = 0
    
    for i, interaction in enumerate(test_interactions, 1):
        print(f"🔄 Interaction #{i}")
        print(f"👤 User: {interaction}")
        
        # Process interaction with evolution learning
        response = await agent.process_input(interaction)
        
        print(f"🤖 Agent: {response[:100]}{'...' if len(response) > 100 else ''}")
        
        # Check if this triggered learning (output should show if learning detected)
        # The evolution integration will print learning messages
        
        print("-" * 60)
    
    print(f"\n📊 Evolution Learning Demo Complete!")
    print(f"   Processed {len(test_interactions)} interactions")
    print(f"   Check output above for '🧠 Learning detected:' messages")


async def demo_evolution_vs_heuristic():
    """Compare LLM vs heuristic evolution analysis"""
    print("\n🔬 LLM vs Heuristic Evolution Analysis\n")
    
    # Test with both LLM and heuristic modes
    from agenesis.evolution import EvolutionAnalyzer
    from agenesis.perception import TextPerception
    from agenesis.memory import ImmediateMemory, WorkingMemory
    
    perception = TextPerception()
    immediate_memory = ImmediateMemory()
    working_memory = WorkingMemory()
    
    test_input = "I prefer Python over JavaScript for data analysis because it has better libraries"
    perception_result = perception.process(test_input)
    immediate_memory.store(perception_result)
    working_memory.store(perception_result)
    
    print(f"📝 Test Input: '{test_input}'\n")
    
    # Test with LLM (if available)
    llm_analyzer = EvolutionAnalyzer()
    if llm_analyzer.use_llm:
        print("🤖 LLM Analysis:")
        llm_decision = await llm_analyzer.analyze_memory_session(immediate_memory, working_memory)
        print(f"   Should persist: {llm_decision.should_persist}")
        print(f"   Learning type: {llm_decision.learning_type}")
        print(f"   Description: {llm_decision.learning_description}")
        print(f"   Confidence: {llm_decision.confidence}")
        print(f"   Future application: {llm_decision.future_application}")
        if llm_decision.rejection_reason:
            print(f"   Rejection reason: {llm_decision.rejection_reason}")
    else:
        print("⚠️  No LLM available - set ANTHROPIC_API_KEY or OPENAI_API_KEY")
    
    print()
    
    # Test with heuristics
    heuristic_analyzer = EvolutionAnalyzer()
    heuristic_analyzer.use_llm = False  # Force heuristic mode
    
    print("🔧 Heuristic Analysis:")
    heuristic_decision = await heuristic_analyzer.analyze_memory_session(immediate_memory, working_memory)
    print(f"   Should persist: {heuristic_decision.should_persist}")
    print(f"   Learning type: {heuristic_decision.learning_type}")
    print(f"   Description: {heuristic_decision.learning_description}")
    print(f"   Confidence: {heuristic_decision.confidence}")
    print(f"   Future application: {heuristic_decision.future_application}")
    if heuristic_decision.rejection_reason:
        print(f"   Rejection reason: {heuristic_decision.rejection_reason}")


if __name__ == "__main__":
    asyncio.run(demo_evolution_learning())
    asyncio.run(demo_evolution_vs_heuristic())