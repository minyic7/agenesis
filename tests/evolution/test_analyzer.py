import pytest

from agenesis.evolution import EvolutionAnalyzer, EvolutionDecision, EvolvedKnowledge
from agenesis.perception import TextPerception
from agenesis.memory import ImmediateMemory, WorkingMemory


class TestEvolutionAnalyzer:
    
    def setup_method(self):
        # Use heuristic-only evolution for deterministic tests
        self.evolution = EvolutionAnalyzer()
        self.evolution.use_llm = False  # Force heuristic mode
        
        self.perception = TextPerception()
        self.immediate_memory = ImmediateMemory()
        self.working_memory = WorkingMemory()
    
    @pytest.mark.asyncio
    async def test_empty_session_analysis(self):
        """Test evolution analysis with empty memories"""
        decision = await self.evolution.analyze_memory_session(
            self.immediate_memory, self.working_memory
        )
        
        assert isinstance(decision, EvolutionDecision)
        assert decision.should_persist is False
        assert decision.rejection_reason is not None
        assert "No meaningful session content" in decision.rejection_reason
    
    @pytest.mark.asyncio
    async def test_preference_detection_heuristic(self):
        """Test that evolution analysis without LLM returns rejection (heuristic mode)"""
        # Add preference statement to memory
        perception_result = self.perception.process("I prefer Python over JavaScript for data analysis")
        self.immediate_memory.store(perception_result)
        self.working_memory.store(perception_result)

        decision = await self.evolution.analyze_memory_session(
            self.immediate_memory, self.working_memory
        )

        assert isinstance(decision, EvolutionDecision)
        # Without LLM, evolution analysis should return rejection
        assert decision.should_persist is False
        assert "No LLM provider available" in decision.rejection_reason
    
    @pytest.mark.asyncio
    async def test_casual_conversation_rejection(self):
        """Test that casual conversation is rejected by heuristics"""
        # Add casual conversation to memory
        perception_result1 = self.perception.process("Hello! How are you?")
        perception_result2 = self.perception.process("I'm doing well, thanks for asking!")
        
        self.immediate_memory.store(perception_result1)
        self.working_memory.store(perception_result1)
        self.working_memory.store(perception_result2)
        
        decision = await self.evolution.analyze_memory_session(
            self.immediate_memory, self.working_memory
        )
        
        assert isinstance(decision, EvolutionDecision)
        assert decision.should_persist is False
        assert decision.rejection_reason is not None
    
    def test_evolved_knowledge_metadata_creation(self):
        """Test creation of evolved knowledge metadata"""
        decision = EvolutionDecision(
            should_persist=True,
            learning_type="preference",
            learning_description="User prefers Python for data analysis",
            future_application="Use Python examples when discussing data analysis"
        )
        
        metadata = self.evolution.create_evolved_knowledge_metadata(decision)
        
        assert isinstance(metadata, EvolvedKnowledge)
        assert metadata.knowledge_summary == decision.learning_description
        assert metadata.learning_context == decision.learning_type
        assert metadata.future_relevance == decision.future_application
        assert metadata.evolved_at is not None  # Should have timestamp
    
    def test_evolved_knowledge_metadata_invalid_decision(self):
        """Test that invalid decisions raise errors"""
        decision = EvolutionDecision(should_persist=False)
        
        with pytest.raises(ValueError):
            self.evolution.create_evolved_knowledge_metadata(decision)
    
    def test_trigger_analysis_conditions(self):
        """Test evolution analysis trigger conditions"""
        # Session end should always trigger
        assert self.evolution.should_trigger_analysis("session_end", {}) is True
        
        # High confidence trigger has been disabled (confidence scores removed)
        assert self.evolution.should_trigger_analysis("high_confidence", {"confidence": 0.9}) is False
        assert self.evolution.should_trigger_analysis("high_confidence", {"confidence": 0.5}) is False
        
        # User learning indicators should trigger when present
        assert self.evolution.should_trigger_analysis("user_learning", {"contains_learning_indicators": True}) is True
        assert self.evolution.should_trigger_analysis("user_learning", {"contains_learning_indicators": False}) is False
        
        # Unknown trigger types should not trigger
        assert self.evolution.should_trigger_analysis("unknown_trigger", {}) is False
    
    def test_session_summary_extraction(self):
        """Test session content extraction from memories"""
        # Add content to memories
        perception_result1 = self.perception.process("I work in healthcare")
        perception_result2 = self.perception.process("I need HIPAA compliant solutions")
        
        self.immediate_memory.store(perception_result1)
        self.working_memory.store(perception_result1)
        self.working_memory.store(perception_result2)
        
        summary = self.evolution._extract_session_summary(
            self.immediate_memory, self.working_memory
        )
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "healthcare" in summary
        assert "HIPAA" in summary
        assert "Current user input:" in summary  # Should include current focus
        assert "User input" in summary     # Should include working memory


@pytest.mark.asyncio
async def test_evolution_with_llm():
    """Test evolution analyzer with actual LLM provider if available"""
    evolution = EvolutionAnalyzer()
    
    if not evolution.use_llm:
        pytest.skip("No LLM provider available - set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")
    
    print(f"\n✅ Using LLM provider: {type(evolution.llm_provider).__name__}")
    
    perception = TextPerception()
    immediate_memory = ImmediateMemory()
    working_memory = WorkingMemory()
    
    # Test with preference statement (should be learned)
    preference_input = "I prefer detailed technical explanations with code examples when learning programming"
    perception_result = perception.process(preference_input)
    immediate_memory.store(perception_result)
    working_memory.store(perception_result)
    
    decision = await evolution.analyze_memory_session(immediate_memory, working_memory)
    
    print(f"🧠 Input: '{preference_input}'")
    print(f"📊 Should persist: {decision.should_persist}")
    print(f"📝 Learning: {decision.learning_description}")
    print(f"🎯 Application: {decision.future_application}")
    if decision.rejection_reason:
        print(f"❌ Rejection: {decision.rejection_reason}")
    
    assert isinstance(decision, EvolutionDecision)
    assert isinstance(decision.should_persist, bool)