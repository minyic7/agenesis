from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any
from datetime import datetime, timezone


@dataclass
class EvolutionDecision:
    """Result of evolution analysis - whether session contains valuable learning"""
    should_persist: bool  # Whether this session contains valuable learning (default: False)
    learning_type: Optional[str] = None  # pattern, preference, knowledge, skill
    learning_description: Optional[str] = None  # What was learned
    future_application: Optional[str] = None  # How this could help in future interactions
    rejection_reason: Optional[str] = None  # Why learning was rejected (if should_persist=False)


@dataclass
class EvolvedKnowledge:
    """Metadata for memories marked as evolved knowledge"""
    knowledge_summary: str  # What valuable knowledge was identified
    learning_context: str  # The context in which this was learned
    future_relevance: str  # When this knowledge would be useful
    evolved_at: Optional[datetime] = None  # When this knowledge was identified
    
    def __post_init__(self):
        if self.evolved_at is None:
            self.evolved_at = datetime.now(timezone.utc)


class BaseEvolution(ABC):
    """Abstract base class for evolution analysis"""
    
    @abstractmethod
    async def analyze_for_learning(self, data_source: Any) -> EvolutionDecision:
        """Analyze data source for learning opportunities"""
        pass
    
    @abstractmethod
    def create_evolved_knowledge_metadata(self, decision: EvolutionDecision) -> EvolvedKnowledge:
        """Create metadata for evolved knowledge based on evolution decision"""
        pass