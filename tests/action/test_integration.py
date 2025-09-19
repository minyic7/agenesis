import pytest
from dotenv import load_dotenv

from agenesis.action import BasicAction
from agenesis.cognition import BasicCognition
from agenesis.perception import TextPerception
from agenesis.memory import ImmediateMemory, WorkingMemory
from agenesis.core import Agent

# Load environment variables from .env file
load_dotenv()


class TestActionIntegration:
    
    def setup_method(self):
        self.perception = TextPerception()
        self.immediate_memory = ImmediateMemory()
        self.working_memory = WorkingMemory()
        self.cognition = BasicCognition()
        self.action = BasicAction()
    
    @pytest.mark.asyncio
    async def test_full_pipeline_manual(self):
        """Test full pipeline manually: perception → memory → cognition → action"""
        test_input = "What is machine learning?"
        
        # 1. Perception
        perception_result = self.perception.process(test_input)
        assert perception_result.content == test_input
        
        # 2. Memory
        self.immediate_memory.store(perception_result)
        self.working_memory.store(perception_result)
        
        # 3. Cognition
        cognition_result = await self.cognition.process(self.immediate_memory, self.working_memory)
        assert cognition_result.intent in ["question", "request", "statement", "conversation"]
        assert 0.0 <= cognition_result.confidence <= 1.0
        
        # 4. Action
        action_result = await self.action.generate_response(cognition_result)
        assert isinstance(action_result.response_text, str)
        assert len(action_result.response_text) > 0
        assert 0.0 <= action_result.confidence <= 1.0
        
        print(f"Input: '{test_input}'")
        print(f"Intent: {cognition_result.intent}")
        print(f"Response: '{action_result.response_text}'")
    
    @pytest.mark.asyncio
    async def test_agent_end_to_end(self):
        """Test complete agent interaction"""
        agent = Agent(profile=None)  # Anonymous agent
        
        test_cases = [
            "Hello! How are you?",
            "What is Python?",
            "Can you help me learn programming?",
            "I am interested in AI development"
        ]
        
        for test_input in test_cases:
            response = await agent.process_input(test_input)
            
            assert isinstance(response, str)
            assert len(response) > 0
            
            print(f"\nInput: '{test_input}'")
            print(f"Response: '{response}'")
            
            # Check that working memory is building context
            context = agent.get_session_context(2)
            assert len(context) > 0
    
    @pytest.mark.asyncio
    async def test_context_awareness(self):
        """Test that agent maintains context across interactions"""
        agent = Agent(profile=None)
        
        # First interaction
        response1 = await agent.process_input("I'm learning Python programming")
        assert isinstance(response1, str)
        
        # Second interaction should be context-aware
        response2 = await agent.process_input("What are functions?")
        assert isinstance(response2, str)
        
        # Check working memory has both interactions
        context = agent.get_session_context(5)
        assert len(context) == 2
        assert any("Python" in ctx.content for ctx in context)
        assert any("functions" in ctx.content for ctx in context)
        
        print(f"First: 'I'm learning Python programming' → '{response1}'")
        print(f"Second: 'What are functions?' → '{response2}'")
    
    @pytest.mark.asyncio
    async def test_different_intents(self):
        """Test agent responses to different intent types"""
        agent = Agent(profile=None)
        
        test_cases = [
            ("What is AI?", "question"),
            ("Please help me", "request"),
            ("I like programming", "statement"),
            ("Hello there!", "conversation")
        ]
        
        for test_input, expected_intent in test_cases:
            response = await agent.process_input(test_input)
            
            assert isinstance(response, str)
            assert len(response) > 0
            
            # Response should be appropriate for the intent
            if expected_intent == "question":
                # Should attempt to answer or ask for clarification
                assert any(word in response.lower() for word in ["understand", "help", "answer", "information"])
            elif expected_intent == "request":
                # Should offer assistance
                assert any(word in response.lower() for word in ["help", "assist", "let me"])
            elif expected_intent == "statement":
                # Should acknowledge or engage (LLM might respond differently than heuristics)
                assert any(word in response.lower() for word in ["thank", "interesting", "appreciate", "great", "good", "nice"])
            elif expected_intent == "conversation":
                # Should engage socially
                assert any(word in response.lower() for word in ["appreciate", "help", "hello", "hi"])
            
            print(f"Intent '{expected_intent}': '{test_input}' → '{response}'")


@pytest.mark.asyncio
async def test_action_with_llm():
    """Test action module with actual LLM provider if available"""
    action = BasicAction()
    
    if not action.use_llm:
        pytest.skip("No LLM provider available - set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")
    
    print(f"\n✅ Using LLM provider: {type(action.llm_provider).__name__}")
    
    # Create a sample cognition result
    from agenesis.cognition import CognitionResult
    cognition_result = CognitionResult(
        intent="question",
        context_type="new",
        persistence_score=0.7,
        summary="User asking about machine learning",
        relevant_memories=[],
        confidence=0.9,
        reasoning="Clear question about technical topic"
    )
    
    action_result = await action.generate_response(cognition_result)
    
    assert isinstance(action_result.response_text, str)
    assert len(action_result.response_text) > 0
    assert 0.0 <= action_result.confidence <= 1.0
    
    print(f"🧠 Cognition: {cognition_result.summary}")
    print(f"🤖 Action: '{action_result.response_text}'")
    print(f"📊 Confidence: {action_result.confidence}")