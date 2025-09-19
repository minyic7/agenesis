#!/usr/bin/env python3
"""
Demo script showing cognition module capabilities
"""
import asyncio
from dotenv import load_dotenv

from agenesis.cognition import BasicCognition, SimplePatternLearning
from agenesis.perception import TextPerception
from agenesis.memory import ImmediateMemory, WorkingMemory

# Load environment variables
load_dotenv()


async def demo_cognition():
    """Demonstrate cognition module capabilities"""
    print("🧠 Cognition Module Demo\n")
    
    # Initialize components
    cognition = BasicCognition()
    learning = SimplePatternLearning()
    perception = TextPerception()
    immediate_memory = ImmediateMemory()
    working_memory = WorkingMemory()
    
    print(f"✅ LLM available: {cognition.use_llm}")
    if cognition.use_llm:
        print(f"✅ LLM provider: {type(cognition.llm_provider).__name__}")
    print()
    
    # Test different types of inputs
    test_inputs = [
        "Hello! How are you?",
        "What is machine learning?", 
        "Can you help me learn Python?",
        "I am interested in AI development",
        "How do neural networks work?"
    ]
    
    cognition_results = []
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"🔄 Processing #{i}: '{test_input}'")
        
        # Process input through perception and cognition
        perception_result = perception.process(test_input)
        immediate_memory.store(perception_result)
        working_memory.store(perception_result)  # Also add to working memory for context
        
        cognition_result = await cognition.process(immediate_memory, working_memory)
        cognition_results.append(cognition_result)
        
        print(f"   Intent: {cognition_result.intent}")
        print(f"   Context: {cognition_result.context_type}")
        print(f"   Persistence: {cognition_result.persistence_score:.2f}")
        print(f"   Confidence: {cognition_result.confidence:.2f}")
        print(f"   Summary: {cognition_result.summary}")
        print()
    
    # Demonstrate learning from cognition results
    print("📚 Learning Analysis:")
    patterns = learning.extract_patterns(cognition_results)
    learning.update_knowledge(patterns)
    
    print(f"   Most common intent: {patterns.get('most_common_intent')}")
    print(f"   Most common context: {patterns.get('most_common_context')}")
    print(f"   Intent distribution: {patterns.get('intent_distribution')}")
    print(f"   Context distribution: {patterns.get('context_distribution')}")
    
    knowledge_summary = learning.get_knowledge_summary()
    print(f"   Total interactions processed: {knowledge_summary['total_interactions']}")
    
    print("\n🎉 Cognition demo completed!")


if __name__ == "__main__":
    asyncio.run(demo_cognition())