from typing import Dict, List, Optional, Any
from .base import BaseMemory, MemoryRecord, PerceptionResult


class ImmediateMemory(BaseMemory):
    """Single-slot memory for current focus - what the agent is thinking about RIGHT NOW"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._current_record: Optional[MemoryRecord] = None
    
    def store(self, perception_result: PerceptionResult, context: Optional[Dict[str, Any]] = None) -> str:
        """Store current perception - replaces previous focus"""
        record = MemoryRecord(
            id="",  # Generated in __post_init__
            perception_result=perception_result,
            stored_at=None,  # Set in __post_init__
            context=self._create_context(context),
            metadata=self._create_metadata()
        )
        
        self._current_record = record
        return record.id
    
    def retrieve(self, memory_id: str) -> Optional[MemoryRecord]:
        """Retrieve by ID - only current record available"""
        if self._current_record and self._current_record.id == memory_id:
            return self._current_record
        return None
    
    def get_recent(self, count: int = 10) -> List[MemoryRecord]:
        """Get recent records - only current record exists"""
        return [self._current_record] if self._current_record else []
    
    def get_current(self) -> Optional[MemoryRecord]:
        """Get the current focus"""
        return self._current_record
    
    def clear(self):
        """Clear current focus - ready for next input"""
        self._current_record = None
    
    def has_focus(self) -> bool:
        """Check if agent has current focus"""
        return self._current_record is not None