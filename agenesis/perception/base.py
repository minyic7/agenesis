from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, List
from datetime import datetime, timezone
from enum import Enum


class InputType(Enum):
    TEXT = "text"
    STRUCTURED = "structured"
    MULTIMODAL = "multimodal"


@dataclass
class PerceptionResult:
    content: str
    input_type: InputType
    metadata: Dict[str, Any]
    features: Dict[str, Any]
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class BasePerception(ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    @abstractmethod
    def validate_input(self, input_data: Any) -> bool:
        pass
    
    @abstractmethod
    def process(self, input_data: Any, context: Any = None) -> PerceptionResult:
        pass
    
    def _create_metadata(self, input_data: Any) -> Dict[str, Any]:
        return {
            "length": len(str(input_data)) if input_data else 0,
            "source": "user_input",
            "instance_id": str(id(self))
        }
    
    @staticmethod
    def _extract_basic_features(content: str) -> Dict[str, Any]:
        return {
            "word_count": len(content.split()) if content else 0,
            "char_count": len(content) if content else 0
        }