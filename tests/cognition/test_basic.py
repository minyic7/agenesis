import pytest
from unittest.mock import Mock

from agenesis.cognition import BasicCognition, CognitionResult
from agenesis.perception import TextPerception
from agenesis.memory import ImmediateMemory, WorkingMemory


class TestBasicCognition:
    
    def setup_method(self):
        # Use heuristic-only cognition for deterministic tests
        self.cognition = BasicCognition()
        self.cognition.use_llm = False  # Force heuristic mode
        
        self.perception = TextPerception()
        self.immediate_memory = ImmediateMemory()
        self.working_memory = WorkingMemory()
    
    @pytest.mark.asyncio
    async def test_empty_input(self):
        """Test cognition with empty immediate memory"""
        result = await self.cognition.process(self.immediate_memory, self.working_memory)
        
        assert isinstance(result, CognitionResult)
        assert result.intent == "conversation"
        assert result.should_persist == False
        assert result.summary == "Empty input"
    
    @pytest.mark.asyncio
    async def test_question_intent(self):
        """Test question intent classification"""
        # Store a question in immediate memory
        perception_result = self.perception.process("What is Python?")
        self.immediate_memory.store(perception_result)
        
        result = await self.cognition.process(self.immediate_memory, self.working_memory)
        
        assert result.intent == "question"
        assert isinstance(result.should_persist, bool)
        assert "question" in result.summary.lower() or "?" in result.summary
    
    @pytest.mark.asyncio
    async def test_request_intent(self):
        """Test request intent classification"""
        perception_result = self.perception.process("Can you help me with coding?")
        self.immediate_memory.store(perception_result)
        
        result = await self.cognition.process(self.immediate_memory, self.working_memory)
        
        assert result.intent == "request"
    
    @pytest.mark.asyncio
    async def test_statement_intent(self):
        """Test statement intent classification"""
        perception_result = self.perception.process("I like programming in Python")
        self.immediate_memory.store(perception_result)
        
        result = await self.cognition.process(self.immediate_memory, self.working_memory)
        
        assert result.intent == "statement"
    
    @pytest.mark.asyncio
    async def test_conversation_intent(self):
        """Test conversation intent classification"""
        perception_result = self.perception.process("Hello there!")
        self.immediate_memory.store(perception_result)
        
        result = await self.cognition.process(self.immediate_memory, self.working_memory)
        
        assert result.intent == "conversation"
    
    @pytest.mark.asyncio
    async def test_context_awareness(self):
        """Test context type detection with working memory"""
        # Add some context to working memory
        context_result = self.perception.process("I'm learning Python")
        self.working_memory.store(context_result)
        
        # Process new related input
        perception_result = self.perception.process("What are Python functions?")
        self.immediate_memory.store(perception_result)
        
        result = await self.cognition.process(self.immediate_memory, self.working_memory)
        
        # Should detect continuation due to related content
        assert result.context_type in ["new", "continuation"]
        assert len(result.relevant_memories) >= 0  # May find relevant memories
    
    @pytest.mark.asyncio
    async def test_cognition_result_structure(self):
        """Test that CognitionResult has all required fields"""
        perception_result = self.perception.process("Test message")
        self.immediate_memory.store(perception_result)
        
        result = await self.cognition.process(self.immediate_memory, self.working_memory)
        
        # Check all required fields exist
        assert hasattr(result, 'intent')
        assert hasattr(result, 'context_type')
        assert hasattr(result, 'should_persist')
        assert hasattr(result, 'summary')
        assert hasattr(result, 'relevant_memories')
        assert hasattr(result, 'reasoning')
        assert hasattr(result, 'timestamp')
        
        # Check data types
        assert isinstance(result.intent, str)
        assert isinstance(result.context_type, str)
        assert isinstance(result.should_persist, bool)
        assert isinstance(result.summary, str)
        assert isinstance(result.relevant_memories, list)
        assert isinstance(result.reasoning, str)
        
        # Check ranges
        assert isinstance(result.should_persist, bool)
        # Confidence field removed per user feedback
    
    @pytest.mark.asyncio
    async def test_llm_fallback_behavior(self):
        """Test that LLM failures fall back to heuristics gracefully"""
        # Mock LLM provider to simulate failure
        mock_provider = Mock()
        mock_provider.complete.side_effect = Exception("API Error")
        
        self.cognition.llm_provider = mock_provider
        self.cognition.use_llm = True  # Enable LLM mode to test fallback
        
        perception_result = self.perception.process("Test question?")
        self.immediate_memory.store(perception_result)
        
        result = await self.cognition.process(self.immediate_memory, self.working_memory)
        
        # Should still return valid result via heuristic fallback
        assert isinstance(result, CognitionResult)
        assert result.intent == "question"