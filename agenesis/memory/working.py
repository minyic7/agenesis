from typing import Dict, List, Optional, Any
from .base import BaseMemory, MemoryRecord, PerceptionResult


class WorkingMemory(BaseMemory):
    """In-memory storage for current session's perception results"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_capacity = self.config.get('max_capacity', 100)
        self._records: List[MemoryRecord] = []
        self._id_index: Dict[str, MemoryRecord] = {}
    
    def store(self, perception_result: PerceptionResult, context: Optional[Dict[str, Any]] = None) -> str:
        """Store a perception result in working memory"""
        record = MemoryRecord(
            id="",  # Will be generated in __post_init__
            perception_result=perception_result,
            stored_at=None,  # Will be set in __post_init__
            context=self._create_context(context),
            metadata=self._create_metadata()
        )
        
        # Add to storage
        self._records.append(record)
        self._id_index[record.id] = record
        
        # Maintain capacity limit
        if len(self._records) > self.max_capacity:
            removed = self._records.pop(0)  # Remove oldest
            del self._id_index[removed.id]
        
        return record.id
    
    def retrieve(self, memory_id: str) -> Optional[MemoryRecord]:
        """Retrieve a memory record by ID"""
        return self._id_index.get(memory_id)
    
    def get_recent(self, count: int = 10) -> List[MemoryRecord]:
        """Get the most recent memory records in reverse chronological order (newest first)"""
        recent = self._records[-count:] if self._records else []
        return list(reversed(recent))
    
    def get_all(self) -> List[MemoryRecord]:
        """Get all memory records in chronological order"""
        return self._records.copy()
    
    def clear(self):
        """Clear all memory records"""
        self._records.clear()
        self._id_index.clear()
    
    def size(self) -> int:
        """Get current memory size"""
        return len(self._records)