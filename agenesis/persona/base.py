from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class PersonaContext:
    """Context object that flows through the processing pipeline"""
    
    # Attention guidance
    focus_areas: List[str] = field(default_factory=list)
    priority_signals: List[str] = field(default_factory=list)
    
    # Memory preferences  
    relevance_boosts: Dict[str, float] = field(default_factory=dict)
    context_filters: List[str] = field(default_factory=list)
    
    # Thinking framework
    reasoning_approach: Optional[str] = None
    decision_criteria: List[str] = field(default_factory=list)
    
    # Content preferences
    response_structure: Optional[str] = None
    include_examples: bool = True
    detail_level: str = "normal"
    
    # System prompt additions
    system_additions: List[str] = field(default_factory=list)
    
    def has_content(self) -> bool:
        """Check if context has any meaningful content"""
        return (
            bool(self.focus_areas) or
            bool(self.priority_signals) or
            bool(self.relevance_boosts) or
            bool(self.context_filters) or
            self.reasoning_approach is not None or
            bool(self.decision_criteria) or
            self.response_structure is not None or
            bool(self.system_additions)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to serializable dictionary"""
        return {
            "focus_areas": self.focus_areas,
            "priority_signals": self.priority_signals,
            "relevance_boosts": self.relevance_boosts,
            "context_filters": self.context_filters,
            "reasoning_approach": self.reasoning_approach,
            "decision_criteria": self.decision_criteria,
            "response_structure": self.response_structure,
            "include_examples": self.include_examples,
            "detail_level": self.detail_level,
            "system_additions": self.system_additions
        }


class BasePersona(ABC):
    """Abstract base class for agent personas"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('name', 'unknown')
        self.description = config.get('description', '')
    
    @abstractmethod
    def create_context(self, input_text: str) -> PersonaContext:
        """Create persona context based on input and configuration"""
        pass
    
    def get_name(self) -> str:
        """Get persona name"""
        return self.name
    
    def get_description(self) -> str:
        """Get persona description"""
        return self.description

    def get_learning_preferences(self) -> Optional[Dict[str, Any]]:
        """Get learning preferences configuration"""
        return self.config.get('learning_preferences')


class DefaultPersona(BasePersona):
    """Default persona implementation using YAML configuration"""
    
    def create_context(self, input_text: str) -> PersonaContext:
        """Create persona context from YAML configuration"""
        context_template = self.config.get('context_template', {})
        
        context = PersonaContext()
        
        # Extract static configuration
        context.focus_areas = context_template.get('focus_areas', [])
        context.priority_signals = context_template.get('priority_signals', [])
        context.relevance_boosts = context_template.get('relevance_boosts', {})
        context.context_filters = context_template.get('context_filters', [])
        context.reasoning_approach = context_template.get('reasoning_approach')
        context.decision_criteria = context_template.get('decision_criteria', [])
        context.response_structure = context_template.get('response_structure')
        context.include_examples = context_template.get('include_examples', True)
        context.detail_level = context_template.get('detail_level', 'normal')
        context.system_additions = context_template.get('system_additions', [])
        
        # Apply dynamic adjustments based on input
        input_lower = input_text.lower()
        
        # Check for priority signals to adjust detail level
        if any(signal in input_lower for signal in context.priority_signals):
            # High priority input - increase detail level
            if context.detail_level == 'normal':
                context.detail_level = 'comprehensive'
            elif context.detail_level == 'minimal':
                context.detail_level = 'normal'
        
        return context