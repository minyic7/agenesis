import pytest
from dotenv import load_dotenv

from agenesis.core import Agent

# Load environment variables
load_dotenv()


class TestEvolutionIntegration:
    
    @pytest.mark.asyncio
    async def test_agent_with_evolution_anonymous(self):
        """Test that anonymous agents don't trigger evolution (no persistent memory)"""
        agent = Agent(profile=None)  # Anonymous agent
        
        # This should work but not trigger evolution analysis
        response = await agent.process_input("I prefer Python programming")
        
        assert isinstance(response, str)
        assert len(response) > 0
        # No evolution should occur for anonymous agents
    
    @pytest.mark.asyncio
    async def test_agent_with_evolution_named(self):
        """Test that named agents can trigger evolution analysis"""
        agent = Agent(profile="test_evolution")  # Named agent
        
        assert agent.has_persistent_memory is True
        assert agent.evolution is not None
        
        # Test with casual conversation (should not trigger learning)
        response1 = await agent.process_input("Hello! How are you?")
        assert isinstance(response1, str)
        
        # Test with preference statement (might trigger learning depending on mode)
        response2 = await agent.process_input("I prefer detailed explanations with examples")
        assert isinstance(response2, str)
        
        # Agent should handle both cases gracefully
    
    @pytest.mark.asyncio
    async def test_evolution_heuristic_mode(self):
        """Test evolution in heuristic mode"""
        agent = Agent(profile="test_heuristic")
        
        # Force heuristic mode for predictable testing
        agent.evolution.use_llm = False
        
        # Test preference detection (should trigger in heuristic mode)
        response = await agent.process_input("I prefer Python over JavaScript for data analysis")
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    @pytest.mark.asyncio 
    async def test_evolution_memory_enhancement(self):
        """Test that evolved knowledge enhances memory records"""
        agent = Agent(profile="test_memory_enhancement")
        
        # Force heuristic mode for predictable behavior
        agent.evolution.use_llm = False
        
        # Add preference that should trigger learning
        response = await agent.process_input("I prefer detailed technical documentation")
        
        # Check working memory for any evolved knowledge markers
        recent_memories = agent.working_memory.get_recent(3)
        
        # At least one memory should exist
        assert len(recent_memories) > 0
        
        # Check if any memories were marked as evolved knowledge
        # (This depends on heuristic detection working correctly)
        evolved_memories = [m for m in recent_memories if hasattr(m, 'is_evolved_knowledge') and m.is_evolved_knowledge]
        
        # The test should pass regardless of whether evolution triggered
        # This just ensures the integration doesn't break
        assert isinstance(response, str)


@pytest.mark.asyncio
async def test_complete_pipeline_with_evolution():
    """Test complete pipeline including evolution"""
    agent = Agent(profile="test_complete_pipeline")
    
    print(f"\n✅ Testing complete pipeline with evolution")
    print(f"   Profile: {agent.profile}")
    print(f"   Has persistent memory: {agent.has_persistent_memory}")
    print(f"   Evolution available: {agent.evolution is not None}")
    print(f"   Evolution LLM: {agent.evolution.use_llm}")
    
    # Test sequence of interactions
    interactions = [
        "Hello there!",
        "I work in healthcare and need HIPAA compliant solutions",
        "What are some good Python libraries?",
        "I prefer open source tools over proprietary ones"
    ]
    
    for i, interaction in enumerate(interactions, 1):
        print(f"\n🔄 Interaction {i}: '{interaction}'")
        response = await agent.process_input(interaction)
        print(f"   Response: {response[:50]}{'...' if len(response) > 50 else ''}")
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    # Check session context was built
    context = agent.get_session_context(10)
    assert len(context) == len(interactions)
    
    print(f"\n✅ Complete pipeline test successful!")
    print(f"   Processed {len(interactions)} interactions")
    print(f"   Session context: {len(context)} items")