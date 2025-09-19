from abc import ABC, abstractmethod
from typing import List, Dict, Any
from datetime import datetime

from .base import CognitionResult


class BaseLearning(ABC):
    """Abstract base class for learning capabilities"""
    
    @abstractmethod
    def extract_patterns(self, cognition_results: List[CognitionResult]) -> Dict[str, Any]:
        """Extract patterns from cognition results"""
        pass
    
    @abstractmethod
    def update_knowledge(self, pattern: Dict[str, Any]) -> None:
        """Update knowledge base with new patterns"""
        pass


class SimplePatternLearning(BaseLearning):
    """Basic pattern learning implementation"""
    
    def __init__(self):
        self.patterns = {}
        self.interaction_count = 0
    
    def extract_patterns(self, cognition_results: List[CognitionResult]) -> Dict[str, Any]:
        """Extract simple patterns from cognition results"""
        if not cognition_results:
            return {}
        
        # Count intent patterns
        intent_counts = {}
        context_counts = {}
        
        for result in cognition_results:
            intent_counts[result.intent] = intent_counts.get(result.intent, 0) + 1
            context_counts[result.context_type] = context_counts.get(result.context_type, 0) + 1
        
        return {
            "most_common_intent": max(intent_counts, key=intent_counts.get) if intent_counts else None,
            "most_common_context": max(context_counts, key=context_counts.get) if context_counts else None,
            "intent_distribution": intent_counts,
            "context_distribution": context_counts,
            "sample_size": len(cognition_results),
            "extracted_at": datetime.now()
        }
    
    def update_knowledge(self, pattern: Dict[str, Any]) -> None:
        """Update knowledge base with new patterns"""
        self.interaction_count += pattern.get("sample_size", 0)
        
        # Simple knowledge update - merge patterns
        for key, value in pattern.items():
            if key.endswith("_distribution") and isinstance(value, dict):
                if key not in self.patterns:
                    self.patterns[key] = {}
                for sub_key, count in value.items():
                    self.patterns[key][sub_key] = self.patterns[key].get(sub_key, 0) + count
            else:
                self.patterns[key] = value
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary of learned knowledge"""
        return {
            "total_interactions": self.interaction_count,
            "patterns": self.patterns,
            "knowledge_updated_at": datetime.now()
        }