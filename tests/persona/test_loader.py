import pytest
import tempfile
import yaml
from pathlib import Path

from agenesis.persona import PersonaLoader, load_persona, list_builtin_personas, get_builtin_persona_config
from agenesis.persona import DefaultPersona


class TestPersonaLoader:
    
    def test_loader_creation(self):
        """Test PersonaLoader instantiation"""
        loader = PersonaLoader()
        assert loader is not None
        assert hasattr(loader, '_cache')
        assert hasattr(loader, '_config_path')
    
    def test_load_from_yaml_file(self):
        """Test loading persona from YAML file"""
        # Create temporary YAML file
        persona_data = {
            "name": "test_persona",
            "description": "Test persona for unit testing",
            "context_template": {
                "focus_areas": ["testing", "validation"],
                "detail_level": "comprehensive"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(persona_data, f)
            temp_path = f.name
        
        try:
            loader = PersonaLoader()
            config = loader.load_from_yaml(temp_path)
            
            assert config["name"] == "test_persona"
            assert config["description"] == "Test persona for unit testing"
            assert config["context_template"]["focus_areas"] == ["testing", "validation"]
            assert config["context_template"]["detail_level"] == "comprehensive"
        finally:
            Path(temp_path).unlink()
    
    def test_load_from_yaml_nonexistent_file(self):
        """Test loading from non-existent file raises error"""
        loader = PersonaLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_from_yaml("nonexistent_file.yaml")
    
    def test_load_from_yaml_empty_file(self):
        """Test loading from empty YAML file raises error"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")  # Empty file
            temp_path = f.name
        
        try:
            loader = PersonaLoader()
            with pytest.raises(ValueError, match="Empty or invalid YAML"):
                loader.load_from_yaml(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_load_from_yaml_missing_name(self):
        """Test loading YAML without required name field"""
        persona_data = {
            "description": "Missing name field"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(persona_data, f)
            temp_path = f.name
        
        try:
            loader = PersonaLoader()
            with pytest.raises(ValueError, match="missing required 'name' field"):
                loader.load_from_yaml(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_load_from_yaml_missing_description(self):
        """Test loading YAML without required description field"""
        persona_data = {
            "name": "test_persona"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(persona_data, f)
            temp_path = f.name
        
        try:
            loader = PersonaLoader()
            with pytest.raises(ValueError, match="missing required 'description' field"):
                loader.load_from_yaml(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_caching(self):
        """Test that loaded configurations are cached"""
        persona_data = {
            "name": "cached_persona",
            "description": "Test caching behavior"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(persona_data, f)
            temp_path = f.name
        
        try:
            loader = PersonaLoader()
            
            # First load
            config1 = loader.load_from_yaml(temp_path)
            assert len(loader._cache) == 1
            
            # Second load - should use cache
            config2 = loader.load_from_yaml(temp_path)
            assert config1 is config2  # Same object reference
            assert len(loader._cache) == 1
        finally:
            Path(temp_path).unlink()
    
    def test_clear_cache(self):
        """Test cache clearing functionality"""
        persona_data = {
            "name": "cache_test",
            "description": "Test cache clearing"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(persona_data, f)
            temp_path = f.name
        
        try:
            loader = PersonaLoader()
            loader.load_from_yaml(temp_path)
            assert len(loader._cache) == 1
            
            loader.clear_cache()
            assert len(loader._cache) == 0
        finally:
            Path(temp_path).unlink()
    
    def test_create_persona_from_dict(self):
        """Test creating persona from dictionary"""
        config_dict = {
            "name": "dict_persona",
            "description": "Persona from dictionary",
            "context_template": {
                "focus_areas": ["dict_test"]
            }
        }
        
        loader = PersonaLoader()
        persona = loader.create_persona(config_dict)
        
        assert isinstance(persona, DefaultPersona)
        assert persona.get_name() == "dict_persona"
        assert persona.get_description() == "Persona from dictionary"
    
    def test_create_persona_from_file_path(self):
        """Test creating persona from file path"""
        persona_data = {
            "name": "file_persona",
            "description": "Persona from file"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(persona_data, f)
            temp_path = f.name
        
        try:
            loader = PersonaLoader()
            persona = loader.create_persona(temp_path)
            
            assert isinstance(persona, DefaultPersona)
            assert persona.get_name() == "file_persona"
        finally:
            Path(temp_path).unlink()


class TestBuiltinPersonas:
    
    def test_list_builtin_personas(self):
        """Test listing built-in personas"""
        personas = list_builtin_personas()
        
        assert isinstance(personas, list)
        # Should have at least the personas we created
        expected_personas = ["technical_mentor", "customer_support", "casual_assistant", "professional"]
        for expected in expected_personas:
            assert expected in personas
    
    def test_get_builtin_persona_config_technical_mentor(self):
        """Test loading technical mentor persona"""
        config = get_builtin_persona_config("technical_mentor")
        
        assert config["name"] == "technical_mentor"
        assert config["description"] == "Systematic problem-solving mentor for developers"
        assert "context_template" in config
        assert "focus_areas" in config["context_template"]
        assert "error_analysis" in config["context_template"]["focus_areas"]
    
    def test_get_builtin_persona_config_customer_support(self):
        """Test loading customer support persona"""
        config = get_builtin_persona_config("customer_support")
        
        assert config["name"] == "customer_support"
        assert config["description"] == "Helpful, patient customer service representative"
        assert "context_template" in config
    
    def test_get_builtin_persona_config_nonexistent(self):
        """Test loading non-existent built-in persona"""
        with pytest.raises(ValueError, match="Unknown builtin persona"):
            get_builtin_persona_config("nonexistent_persona")


class TestConvenienceFunctions:
    
    def test_load_persona_from_dict(self):
        """Test load_persona convenience function with dict"""
        config = {
            "name": "convenience_test",
            "description": "Test convenience function"
        }
        
        persona = load_persona(config)
        assert isinstance(persona, DefaultPersona)
        assert persona.get_name() == "convenience_test"
    
    def test_load_persona_builtin_name(self):
        """Test load_persona convenience function with builtin name"""
        persona = load_persona("technical_mentor")
        assert isinstance(persona, DefaultPersona)
        assert persona.get_name() == "technical_mentor"
    
    def test_load_persona_file_path(self):
        """Test load_persona convenience function with file path"""
        persona_data = {
            "name": "file_convenience_test",
            "description": "Test file loading"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(persona_data, f)
            temp_path = f.name
        
        try:
            persona = load_persona(temp_path)
            assert isinstance(persona, DefaultPersona)
            assert persona.get_name() == "file_convenience_test"
        finally:
            Path(temp_path).unlink()