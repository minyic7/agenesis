from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timezone

from ..cognition.base import CognitionResult


@dataclass
class ActionResult:
    """Result of action processing - simple text response"""
    response_text: str              # The actual text response to user
    confidence: float              # 0.0-1.0 confidence in response  
    timestamp: Optional[datetime] = None
    
    # Optional metadata for debugging/logging (not user-facing)
    internal_reasoning: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        
        # Validate confidence range
        self.confidence = max(0.0, min(1.0, self.confidence))


class BaseAction(ABC):
    """Abstract base class for action processing"""
    
    @abstractmethod
    async def generate_response(self, cognition_result: CognitionResult) -> ActionResult:
        """Generate simple text response based on cognition analysis"""
        pass