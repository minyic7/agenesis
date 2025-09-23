from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone
import uuid

from ..perception.base import PerceptionResult


@dataclass
class MemoryRecord:
    id: str
    perception_result: PerceptionResult
    stored_at: datetime
    context: Dict[str, Any]
    metadata: Dict[str, Any]

    # Evolution support
    is_evolved_knowledge: bool = False
    evolution_metadata: Optional[Dict[str, Any]] = None
    reliability_multiplier: float = 1.0  # Higher for evolved knowledge

    # Semantic search support
    embedding: Optional[List[float]] = None  # OpenAI embedding for semantic search

    # Full interaction context (when learning occurs)
    agent_response: Optional[str] = None  # Agent's response to the user input
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.stored_at:
            self.stored_at = datetime.now(timezone.utc)


class BaseMemory(ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    @abstractmethod
    def store(self, perception_result: PerceptionResult, context: Optional[Dict[str, Any]] = None) -> str:
        """Store a perception result and return its memory ID"""
        pass

    def store_record(self, memory_record: 'MemoryRecord') -> str:
        """Store a complete memory record and return its memory ID"""
        # Default implementation - can be overridden by subclasses for efficiency
        return self.store(memory_record.perception_result, memory_record.context)
    
    @abstractmethod
    def retrieve(self, memory_id: str) -> Optional[MemoryRecord]:
        """Retrieve a memory record by ID"""
        pass
    
    @abstractmethod
    def get_recent(self, count: int = 10) -> List[MemoryRecord]:
        """Get the most recent memory records"""
        pass
    
    def _create_context(self, additional_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create context metadata for storage"""
        context = {
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        if additional_context:
            context.update(additional_context)
        return context
    
    def _create_metadata(self) -> Dict[str, Any]:
        """Create storage metadata"""
        return {
            "memory_type": self.__class__.__name__,
            "instance_id": str(id(self))
        }