import pytest
from agenesis.core import Agent
from agenesis.persona import PersonaContext


class TestPersonaIntegration:
    
    def test_agent_without_persona(self):
        """Test agent works normally without persona"""
        agent = Agent(profile=None)  # Anonymous agent
        
        # No persona should be set
        assert agent.persona is None
        
        # Profile info should show no persona
        profile_info = agent.get_profile_info()
        assert profile_info['persona']['active'] is False
        assert profile_info['persona']['name'] is None
        assert profile_info['persona']['description'] is None
    
    def test_agent_with_builtin_persona(self):
        """Test agent with built-in persona"""
        agent = Agent(profile="test_user", persona="technical_mentor")
        
        # Persona should be loaded
        assert agent.persona is not None
        assert agent.persona.get_name() == "technical_mentor"
        assert "problem-solving mentor" in agent.persona.get_description()
        
        # Profile info should show persona details
        profile_info = agent.get_profile_info()
        assert profile_info['persona']['active'] is True
        assert profile_info['persona']['name'] == "technical_mentor"
        assert "problem-solving mentor" in profile_info['persona']['description']
    
    def test_agent_with_persona_config_dict(self):
        """Test agent with persona from dictionary config"""
        persona_config = {
            "name": "test_integration_persona",
            "description": "Persona for integration testing",
            "context_template": {
                "focus_areas": ["integration", "testing"],
                "detail_level": "comprehensive"
            }
        }
        
        agent = Agent(profile="test_user", persona=persona_config)
        
        assert agent.persona is not None
        assert agent.persona.get_name() == "test_integration_persona"
        assert agent.persona.get_description() == "Persona for integration testing"
    
    def test_agent_persona_runtime_switching(self):
        """Test switching persona at runtime"""
        agent = Agent(profile="test_user", persona="technical_mentor")
        
        # Initial persona
        assert agent.persona.get_name() == "technical_mentor"
        
        # Switch to different persona
        agent.set_persona("customer_support")
        assert agent.persona.get_name() == "customer_support"
        assert "customer service" in agent.persona.get_description()
        
        # Switch to None (remove persona)
        agent.set_persona(None)
        assert agent.persona is None
    
    def test_agent_invalid_persona_handling(self):
        """Test agent handles invalid persona gracefully"""
        # Invalid persona name should not crash, just print warning
        agent = Agent(profile="test_user", persona="nonexistent_persona")
        
        # Should fallback to no persona
        assert agent.persona is None
    
    @pytest.mark.asyncio
    async def test_persona_context_in_pipeline(self):
        """Test that persona context flows through the processing pipeline"""
        agent = Agent(profile="test_persona_pipeline", persona="technical_mentor")
        
        # Mock the modules to capture persona context
        original_perception_process = agent.perception.process
        original_cognition_process = agent.cognition.process
        original_action_generate = agent.action.generate_response
        
        captured_contexts = []
        
        def mock_perception_process(text, context=None):
            captured_contexts.append(('perception', context))
            return original_perception_process(text)
        
        async def mock_cognition_process(immediate_memory, working_memory, context=None):
            captured_contexts.append(('cognition', context))
            return await original_cognition_process(immediate_memory, working_memory)
        
        async def mock_action_generate(cognition_result, context=None):
            captured_contexts.append(('action', context))
            return await original_action_generate(cognition_result)
        
        # Apply mocks
        agent.perception.process = mock_perception_process
        agent.cognition.process = mock_cognition_process
        agent.action.generate_response = mock_action_generate
        
        # Process input that should trigger persona context
        await agent.process_input("I have a performance bug in my code")
        
        # Verify persona context was passed to all modules
        assert len(captured_contexts) == 3
        
        perception_context = captured_contexts[0][1]
        cognition_context = captured_contexts[1][1]
        action_context = captured_contexts[2][1]
        
        # All should receive the same persona context
        assert isinstance(perception_context, PersonaContext)
        assert isinstance(cognition_context, PersonaContext)
        assert isinstance(action_context, PersonaContext)
        
        # Context should have technical mentor characteristics
        assert "error_analysis" in perception_context.focus_areas
        assert "performance" in perception_context.priority_signals
        assert perception_context.detail_level == "comprehensive"  # Should be upgraded due to "bug"
    
    @pytest.mark.asyncio
    async def test_persona_context_none_when_no_persona(self):
        """Test that context is None when no persona is set"""
        agent = Agent(profile="test_no_persona")
        
        # Mock perception to capture context
        original_process = agent.perception.process
        captured_context = None
        
        def mock_process(text, context=None):
            nonlocal captured_context
            captured_context = context
            return original_process(text)
        
        agent.perception.process = mock_process
        
        await agent.process_input("Hello world")
        
        # Context should be None when no persona
        assert captured_context is None
    
    @pytest.mark.asyncio
    async def test_persona_priority_signal_detection(self):
        """Test that persona priority signals affect context generation"""
        agent = Agent(profile="test_priority", persona="technical_mentor")
        
        # Mock perception to capture contexts
        original_process = agent.perception.process
        captured_contexts = []
        
        def mock_process(text, context=None):
            captured_contexts.append(context)
            return original_process(text)
        
        agent.perception.process = mock_process
        
        # Test normal input
        await agent.process_input("Hello there")
        normal_context = captured_contexts[-1]
        
        # Test priority signal input
        await agent.process_input("I have an error in my code")
        priority_context = captured_contexts[-1]
        
        # Priority signal should increase detail level
        assert normal_context.detail_level == "comprehensive"  # From YAML config
        assert priority_context.detail_level == "comprehensive"  # Already at max
        
        # Both should have same focus areas from persona
        assert normal_context.focus_areas == priority_context.focus_areas
        assert "error_analysis" in priority_context.focus_areas
    
    def test_persona_config_from_file_path(self):
        """Test loading persona from file path"""
        import tempfile
        import yaml
        from pathlib import Path
        
        persona_data = {
            "name": "file_test_persona",
            "description": "Persona loaded from file",
            "context_template": {
                "focus_areas": ["file_testing"],
                "detail_level": "normal"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(persona_data, f)
            temp_path = f.name
        
        try:
            agent = Agent(profile="test_file_persona", persona_config=temp_path)
            
            assert agent.persona is not None
            assert agent.persona.get_name() == "file_test_persona"
            assert agent.persona.get_description() == "Persona loaded from file"
        finally:
            Path(temp_path).unlink()


@pytest.mark.asyncio
async def test_complete_persona_pipeline():
    """Test complete pipeline with persona end-to-end"""
    agent = Agent(profile="test_complete_persona", persona="customer_support")
    
    print(f"\n✅ Testing complete persona pipeline")
    print(f"   Profile: {agent.profile}")
    print(f"   Has persona: {agent.persona is not None}")
    print(f"   Persona name: {agent.persona.get_name() if agent.persona else None}")
    
    # Test sequence of interactions that should be influenced by customer support persona
    interactions = [
        "Hello there!",
        "I'm having trouble with my account",
        "The system is not working properly",
        "Thanks for your help!"
    ]
    
    for i, interaction in enumerate(interactions, 1):
        print(f"\n🔄 Interaction {i}: '{interaction}'")
        response = await agent.process_input(interaction)
        print(f"   Response: {response[:100]}{'...' if len(response) > 100 else ''}")
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    # Check profile info includes persona
    profile_info = agent.get_profile_info()
    assert profile_info['persona']['active'] is True
    assert profile_info['persona']['name'] == "customer_support"
    
    print(f"\n✅ Complete persona pipeline test successful!")
    print(f"   Persona: {profile_info['persona']['name']}")
    print(f"   Description: {profile_info['persona']['description']}")
    print(f"   Processed {len(interactions)} interactions")