import pytest
from agenesis.persona import PersonaContext, BasePersona, DefaultPersona


class TestPersonaContext:
    
    def test_persona_context_creation(self):
        """Test PersonaContext creation with default values"""
        context = PersonaContext()
        
        assert context.focus_areas == []
        assert context.priority_signals == []
        assert context.relevance_boosts == {}
        assert context.context_filters == []
        assert context.reasoning_approach is None
        assert context.decision_criteria == []
        assert context.response_structure is None
        assert context.include_examples is True
        assert context.detail_level == "normal"
        assert context.system_additions == []
    
    def test_persona_context_with_data(self):
        """Test PersonaContext with actual data"""
        context = PersonaContext(
            focus_areas=["technical", "debugging"],
            priority_signals=["error", "bug"],
            relevance_boosts={"solutions": 1.5},
            reasoning_approach="systematic",
            detail_level="comprehensive",
            system_additions=["Be thorough"]
        )
        
        assert context.focus_areas == ["technical", "debugging"]
        assert context.priority_signals == ["error", "bug"]
        assert context.relevance_boosts == {"solutions": 1.5}
        assert context.reasoning_approach == "systematic"
        assert context.detail_level == "comprehensive"
        assert context.system_additions == ["Be thorough"]
    
    def test_has_content_empty(self):
        """Test has_content with empty context"""
        context = PersonaContext()
        assert context.has_content() is False
    
    def test_has_content_with_focus_areas(self):
        """Test has_content with focus areas"""
        context = PersonaContext(focus_areas=["technical"])
        assert context.has_content() is True
    
    def test_has_content_with_reasoning_approach(self):
        """Test has_content with reasoning approach"""
        context = PersonaContext(reasoning_approach="systematic")
        assert context.has_content() is True
    
    def test_has_content_with_system_additions(self):
        """Test has_content with system additions"""
        context = PersonaContext(system_additions=["Be helpful"])
        assert context.has_content() is True


class TestDefaultPersona:
    
    def test_default_persona_creation(self):
        """Test DefaultPersona creation with basic config"""
        config = {
            "name": "test_persona",
            "description": "A test persona",
            "context_template": {}
        }
        
        persona = DefaultPersona(config)
        assert persona.get_name() == "test_persona"
        assert persona.get_description() == "A test persona"
    
    def test_create_context_empty_template(self):
        """Test context creation with empty template"""
        config = {
            "name": "test_persona",
            "description": "A test persona",
            "context_template": {}
        }
        
        persona = DefaultPersona(config)
        context = persona.create_context("Hello world")
        
        assert isinstance(context, PersonaContext)
        assert context.focus_areas == []
        assert context.priority_signals == []
        assert context.detail_level == "normal"
    
    def test_create_context_with_template(self):
        """Test context creation with full template"""
        config = {
            "name": "technical_mentor",
            "description": "Technical mentor persona",
            "context_template": {
                "focus_areas": ["debugging", "optimization"],
                "priority_signals": ["error", "slow"],
                "relevance_boosts": {"solutions": 1.5},
                "reasoning_approach": "systematic",
                "detail_level": "normal",
                "system_additions": ["Explain step by step"]
            }
        }
        
        persona = DefaultPersona(config)
        context = persona.create_context("Hello world")
        
        assert context.focus_areas == ["debugging", "optimization"]
        assert context.priority_signals == ["error", "slow"]
        assert context.relevance_boosts == {"solutions": 1.5}
        assert context.reasoning_approach == "systematic"
        assert context.detail_level == "normal"
        assert context.system_additions == ["Explain step by step"]
    
    def test_priority_signal_detection(self):
        """Test that priority signals adjust detail level"""
        config = {
            "name": "test_persona",
            "description": "Test persona",
            "context_template": {
                "priority_signals": ["error", "bug"],
                "detail_level": "normal"
            }
        }
        
        persona = DefaultPersona(config)
        
        # Normal input - detail level stays normal
        context1 = persona.create_context("Hello there")
        assert context1.detail_level == "normal"
        
        # Input with priority signal - detail level increases
        context2 = persona.create_context("I have an error in my code")
        assert context2.detail_level == "comprehensive"
    
    def test_priority_signal_minimal_to_normal(self):
        """Test priority signal upgrading minimal to normal"""
        config = {
            "name": "test_persona", 
            "description": "Test persona",
            "context_template": {
                "priority_signals": ["error"],
                "detail_level": "minimal"
            }
        }
        
        persona = DefaultPersona(config)
        context = persona.create_context("I have an error")
        assert context.detail_level == "normal"
    
    def test_missing_name_description(self):
        """Test persona with missing name/description"""
        config = {}
        
        persona = DefaultPersona(config)
        assert persona.get_name() == "unknown"
        assert persona.get_description() == ""