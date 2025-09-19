from typing import Any
from .base import BasePerception, PerceptionResult


class MultimodalPerception(BasePerception):
    def validate_input(self, input_data: Any) -> bool:
        raise NotImplementedError("Multimodal perception not yet implemented")
    
    def process(self, input_data: Any) -> PerceptionResult:
        raise NotImplementedError("Multimodal perception not yet implemented")