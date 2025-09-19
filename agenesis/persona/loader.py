import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union

from .base import BasePersona, DefaultPersona


class PersonaLoader:
    """Loads persona configurations from YAML files"""
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        # Point to project config directory
        self._config_path = Path(__file__).parent.parent.parent / "config" / "persona"
    
    def load_from_yaml(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load persona configuration from YAML file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Persona file not found: {file_path}")
        
        # Check cache first
        cache_key = str(file_path.absolute())
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if not config:
            raise ValueError(f"Empty or invalid YAML file: {file_path}")
        
        # Validate required fields
        if 'name' not in config:
            raise ValueError(f"Persona YAML missing required 'name' field: {file_path}")
        if 'description' not in config:
            raise ValueError(f"Persona YAML missing required 'description' field: {file_path}")
        
        self._cache[cache_key] = config
        return config
    
    def load_builtin_persona(self, name: str) -> Dict[str, Any]:
        """Load a built-in persona by name"""
        persona_file = self._config_path / f"{name}.yaml"
        
        if not persona_file.exists():
            available = self.list_builtin_personas()
            raise ValueError(f"Unknown builtin persona '{name}'. Available: {available}")
        
        return self.load_from_yaml(persona_file)
    
    def list_builtin_personas(self) -> list[str]:
        """List all available built-in persona names"""
        if not self._config_path.exists():
            return []
        
        personas = []
        for yaml_file in self._config_path.glob("*.yaml"):
            personas.append(yaml_file.stem)
        
        return sorted(personas)
    
    def create_persona(self, source: Union[str, Path, Dict[str, Any]]) -> BasePersona:
        """Create a persona instance from various sources"""
        if isinstance(source, dict):
            # Direct configuration
            config = source
        elif isinstance(source, str):
            if "/" in source or "\\" in source or source.endswith('.yaml'):
                # File path
                config = self.load_from_yaml(source)
            else:
                # Built-in persona name
                config = self.load_builtin_persona(source)
        else:
            # Path object
            config = self.load_from_yaml(source)
        
        return DefaultPersona(config)
    
    def clear_cache(self):
        """Clear the persona configuration cache"""
        self._cache.clear()
    
    def get_cached_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all currently cached persona configurations"""
        return self._cache.copy()


# Global loader instance
_loader = PersonaLoader()


def load_persona(source: Union[str, Path, Dict[str, Any]]) -> BasePersona:
    """Convenience function to load persona from various sources"""
    return _loader.create_persona(source)


def list_builtin_personas() -> list[str]:
    """Convenience function to list built-in personas"""
    return _loader.list_builtin_personas()


def get_builtin_persona_config(name: str) -> Dict[str, Any]:
    """Convenience function to get built-in persona config"""
    return _loader.load_builtin_persona(name)