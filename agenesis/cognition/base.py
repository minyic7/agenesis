from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from ..memory.base import MemoryRecord


@dataclass
class CognitionResult:
    """Result of cognitive processing"""
    intent: str  # question, request, statement, conversation
    context_type: str  # new, continuation, clarification, related
    persistence_score: float  # 0.0-1.0
    summary: str  # brief description of what user wants
    relevant_memories: List[str]  # IDs of relevant memory records
    confidence: float  # 0.0-1.0 confidence in the analysis
    reasoning: str  # explanation of the analysis
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        
        # Validate ranges
        self.persistence_score = max(0.0, min(1.0, self.persistence_score))
        self.confidence = max(0.0, min(1.0, self.confidence))


class BaseCognition(ABC):
    """Abstract base class for cognitive processing"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    @abstractmethod
    async def process(self, immediate_memory, working_memory) -> CognitionResult:
        """Process immediate focus and working context to understand user intent"""
        pass
    
    def _get_user_input(self, immediate_memory) -> str:
        """Extract user input from immediate memory"""
        current_record = immediate_memory.get_current()
        if current_record:
            return current_record.perception_result.content
        return ""
    
    def _summarize_working_memory(self, working_memory, max_items: int = 3) -> str:
        """Create brief summary of recent conversation for context"""
        recent = working_memory.get_recent(max_items)
        if not recent:
            return "No recent context"
        
        summaries = []
        for record in recent:
            content = record.perception_result.content[:50]
            if len(record.perception_result.content) > 50:
                content += "..."
            summaries.append(content)
        
        return "; ".join(summaries)
    
    def _find_relevant_memories(self, working_memory, user_input: str) -> List[str]:
        """Simple keyword-based relevance detection (placeholder)"""
        # TODO: Implement more sophisticated relevance detection
        recent = working_memory.get_recent(5)
        relevant_ids = []
        
        input_words = set(user_input.lower().split())
        
        for record in recent:
            content_words = set(record.perception_result.content.lower().split())
            # Simple overlap detection
            if len(input_words.intersection(content_words)) > 0:
                relevant_ids.append(record.id)
        
        return relevant_ids