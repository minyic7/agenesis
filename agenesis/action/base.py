from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timezone

from ..cognition.base import CognitionResult


@dataclass
class ActionResult:
    """Result of action processing - simple text response"""
    response_text: str              # The actual text response to user
    timestamp: Optional[datetime] = None

    # Optional metadata for debugging/logging (not user-facing)
    internal_reasoning: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class BaseAction(ABC):
    """Abstract base class for action processing"""
    
    @abstractmethod
    async def generate_response(
        self,
        cognition_result: CognitionResult,
        context: Optional = None
    ) -> ActionResult:
        """Generate simple text response based on cognition analysis and memory context"""
        pass