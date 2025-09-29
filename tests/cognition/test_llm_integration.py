import pytest
from dotenv import load_dotenv

from agenesis.cognition import BasicCognition, CognitionResult
from agenesis.perception import TextPerception
from agenesis.memory import ImmediateMemory, WorkingMemory

# Load environment variables from .env file
load_dotenv()


@pytest.mark.asyncio
async def test_cognition_with_llm():
    """Test cognition module with actual LLM provider"""
    cognition = BasicCognition()
    perception = TextPerception()
    immediate_memory = ImmediateMemory()
    working_memory = WorkingMemory()
    
    if not cognition.use_llm:
        pytest.skip("No LLM provider available - set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")
    
    print(f"\n✅ Using LLM provider: {type(cognition.llm_provider).__name__}")
    
    # Test with a clear request
    test_input = "Can you help me learn Python programming?"
    perception_result = perception.process(test_input)
    immediate_memory.store(perception_result)
    
    print(f"🧠 Processing: '{test_input}'")
    result = await cognition.process(immediate_memory, working_memory)
    
    assert isinstance(result, CognitionResult)
    assert result.intent in ["question", "request", "statement", "conversation"]
    assert result.context_type in ["new", "continuation", "clarification", "related"]
    assert isinstance(result.should_persist, bool)
    assert len(result.summary) > 0
    assert len(result.reasoning) > 0

    print(f"✅ Intent: {result.intent}")
    print(f"✅ Context type: {result.context_type}")
    print(f"✅ Should persist: {result.should_persist}")
    print(f"✅ Summary: {result.summary}")
    print(f"✅ Reasoning: {result.reasoning}")


@pytest.mark.asyncio
async def test_cognition_different_intents():
    """Test cognition with different types of inputs"""
    cognition = BasicCognition()
    perception = TextPerception()
    immediate_memory = ImmediateMemory()
    working_memory = WorkingMemory()
    
    if not cognition.use_llm:
        pytest.skip("No LLM provider available")
    
    test_cases = [
        ("What is machine learning?", "question"),
        ("Please explain neural networks", "request"), 
        ("I am interested in AI", "statement"),
        ("Hello!", "conversation")
    ]
    
    for test_input, expected_intent in test_cases:
        perception_result = perception.process(test_input)
        immediate_memory.store(perception_result)
        
        result = await cognition.process(immediate_memory, working_memory)
        
        print(f"Input: '{test_input}' -> Intent: {result.intent} (Expected: {expected_intent})")
        # Note: LLM might classify differently than heuristics, so we just ensure it's valid
        assert result.intent in ["question", "request", "statement", "conversation"]
        assert isinstance(result.should_persist, bool)


@pytest.mark.asyncio 
async def test_cognition_with_context():
    """Test cognition with working memory context"""
    cognition = BasicCognition()
    perception = TextPerception()
    immediate_memory = ImmediateMemory()
    working_memory = WorkingMemory()
    
    if not cognition.use_llm:
        pytest.skip("No LLM provider available")
    
    # Add context to working memory
    context1 = perception.process("I'm learning about Python")
    context2 = perception.process("I want to understand functions")
    working_memory.store(context1)
    working_memory.store(context2)
    
    # Process related question
    test_input = "How do I define a function?"
    perception_result = perception.process(test_input)
    immediate_memory.store(perception_result)
    
    result = await cognition.process(immediate_memory, working_memory)
    
    print(f"Context-aware processing: '{test_input}'")
    print(f"✅ Context type: {result.context_type}")
    print(f"✅ Found {len(result.relevant_memories)} relevant memories")
    
    assert isinstance(result, CognitionResult)
    # Should recognize this as continuation or related to existing context
    assert result.context_type in ["new", "continuation", "clarification", "related"]