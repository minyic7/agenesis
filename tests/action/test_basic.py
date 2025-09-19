import pytest

from agenesis.action import BasicAction, ActionResult
from agenesis.cognition import CognitionResult


class TestBasicAction:
    
    def setup_method(self):
        # Use heuristic-only action for deterministic tests
        self.action = BasicAction()
        self.action.use_llm = False  # Force heuristic mode
    
    @pytest.mark.asyncio
    async def test_question_intent_response(self):
        """Test response generation for question intent"""
        cognition_result = CognitionResult(
            intent="question",
            context_type="new",
            persistence_score=0.7,
            summary="User asking about Python",
            relevant_memories=[],
            confidence=0.8,
            reasoning="Question about programming language"
        )
        
        action_result = await self.action.generate_response(cognition_result)
        
        assert isinstance(action_result, ActionResult)
        assert isinstance(action_result.response_text, str)
        assert len(action_result.response_text) > 0
        assert "Python" in action_result.response_text
        assert "asking about" in action_result.response_text
        assert 0.0 <= action_result.confidence <= 1.0
        assert action_result.internal_reasoning is not None
    
    @pytest.mark.asyncio
    async def test_request_intent_response(self):
        """Test response generation for request intent"""
        cognition_result = CognitionResult(
            intent="request",
            context_type="new", 
            persistence_score=0.8,
            summary="User requesting help with coding",
            relevant_memories=[],
            confidence=0.9,
            reasoning="Help request"
        )
        
        action_result = await self.action.generate_response(cognition_result)
        
        assert isinstance(action_result, ActionResult)
        assert "help with" in action_result.response_text
        assert "coding" in action_result.response_text
        assert action_result.confidence == cognition_result.confidence * 0.7  # Heuristic factor
    
    @pytest.mark.asyncio
    async def test_statement_intent_response(self):
        """Test response generation for statement intent"""
        cognition_result = CognitionResult(
            intent="statement",
            context_type="new",
            persistence_score=0.6,
            summary="User sharing information about AI",
            relevant_memories=[],
            confidence=0.8,
            reasoning="Information sharing"
        )
        
        action_result = await self.action.generate_response(cognition_result)
        
        assert isinstance(action_result, ActionResult)
        assert "Thank you" in action_result.response_text
        assert "AI" in action_result.response_text
        assert "interesting" in action_result.response_text
    
    @pytest.mark.asyncio
    async def test_conversation_intent_response(self):
        """Test response generation for conversation intent"""
        cognition_result = CognitionResult(
            intent="conversation",
            context_type="new",
            persistence_score=0.2,
            summary="Casual greeting",
            relevant_memories=[],
            confidence=0.7,
            reasoning="Social interaction"
        )
        
        action_result = await self.action.generate_response(cognition_result)
        
        assert isinstance(action_result, ActionResult)
        assert "appreciate" in action_result.response_text or "help" in action_result.response_text
    
    @pytest.mark.asyncio
    async def test_heuristic_mode(self):
        """Test heuristic mode when no LLM provider available"""
        # Ensure we're in heuristic mode
        self.action.use_llm = False
        self.action.llm_provider = None
        
        cognition_result = CognitionResult(
            intent="question",
            context_type="new",
            persistence_score=0.7,
            summary="Test question",
            relevant_memories=[],
            confidence=0.8,
            reasoning="Test reasoning"
        )
        
        action_result = await self.action.generate_response(cognition_result)
        
        # Should return valid heuristic result
        assert isinstance(action_result, ActionResult)
        assert isinstance(action_result.response_text, str)
        assert len(action_result.response_text) > 0
        assert action_result.confidence == cognition_result.confidence * 0.7  # Heuristic confidence
    
    @pytest.mark.asyncio
    async def test_action_result_structure(self):
        """Test that ActionResult has all required fields"""
        cognition_result = CognitionResult(
            intent="question",
            context_type="new",
            persistence_score=0.7,
            summary="Test",
            relevant_memories=[],
            confidence=0.8,
            reasoning="Test"
        )
        
        action_result = await self.action.generate_response(cognition_result)
        
        # Check all required fields exist
        assert hasattr(action_result, 'response_text')
        assert hasattr(action_result, 'confidence')
        assert hasattr(action_result, 'timestamp')
        assert hasattr(action_result, 'internal_reasoning')
        
        # Check data types
        assert isinstance(action_result.response_text, str)
        assert isinstance(action_result.confidence, float)
        assert action_result.timestamp is not None
        
        # Check ranges
        assert 0.0 <= action_result.confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_confidence_calculation(self):
        """Test confidence adjustment for heuristic mode"""
        cognition_result = CognitionResult(
            intent="question",
            context_type="new",
            persistence_score=0.7,
            summary="Test",
            relevant_memories=[],
            confidence=1.0,  # Maximum confidence
            reasoning="Test"
        )
        
        # Test heuristic confidence factor
        self.action.use_llm = False
        action_result = await self.action.generate_response(cognition_result)
        assert action_result.confidence == 0.7  # 1.0 * 0.7
        
        # Test with different input confidence
        cognition_result.confidence = 0.8
        action_result = await self.action.generate_response(cognition_result)
        assert abs(action_result.confidence - 0.56) < 0.01  # 0.8 * 0.7 (with float precision tolerance)