import re
from typing import Any, Dict, Optional
from .base import BasePerception, PerceptionResult, InputType


class TextPerception(BasePerception):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_length = self.config.get('max_length', 10000)
        self.min_length = self.config.get('min_length', 1)
        self.strip_whitespace = self.config.get('strip_whitespace', True)
        self.normalize_unicode = self.config.get('normalize_unicode', True)
    
    def validate_input(self, input_data: Any) -> bool:
        if not isinstance(input_data, str):
            return False
        
        if len(input_data) < self.min_length:
            return False
            
        if len(input_data) > self.max_length:
            return False
            
        return True
    
    def process(self, input_data: Any, context: Any = None) -> PerceptionResult:
        if not self.validate_input(input_data):
            raise ValueError(f"Invalid input: {type(input_data)} with length {len(str(input_data))}")
        
        content = self._preprocess_text(input_data)
        metadata = self._create_text_metadata(input_data, content)
        features = self._extract_text_features(content)
        
        return PerceptionResult(
            content=content,
            input_type=InputType.TEXT,
            metadata=metadata,
            features=features,
            timestamp=None  # Will be set by __post_init__
        )
    
    def _preprocess_text(self, text: str) -> str:
        if self.strip_whitespace:
            text = text.strip()
        
        if self.normalize_unicode:
            text = self._normalize_unicode(text)
        
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        import unicodedata
        return unicodedata.normalize('NFKC', text)
    
    def _create_text_metadata(self, original: str, processed: str) -> Dict[str, Any]:
        base_metadata = self._create_metadata(original)
        
        text_metadata = {
            "original_length": len(original),
            "processed_length": len(processed),
            "preprocessing_applied": {
                "strip_whitespace": self.strip_whitespace,
                "normalize_unicode": self.normalize_unicode
            }
        }
        
        return {**base_metadata, **text_metadata}
    
    def _extract_text_features(self, content: str) -> Dict[str, Any]:
        base_features = self._extract_basic_features(content)
        
        text_features = {
            "line_count": len(content.split('\n')),
            "paragraph_count": len([p for p in content.split('\n\n') if p.strip()]),
            "has_urls": bool(re.search(r'https?://', content)),
            "has_emails": bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)),
            "has_code_blocks": bool(re.search(r'```|`[^`]+`', content)),
            "language_hints": self._detect_language_hints(content)
        }
        
        return {**base_features, **text_features}
    
    def _detect_language_hints(self, content: str) -> Dict[str, bool]:
        return {
            "has_punctuation": bool(re.search(r'[.!?]', content)),
            "has_code_syntax": bool(re.search(r'[{}()\[\];]', content)),
            "has_markdown": bool(re.search(r'[#*_`]', content)),
            "likely_code": self._is_likely_code(content)
        }
    
    def _is_likely_code(self, content: str) -> bool:
        code_indicators = [
            r'\bdef\s+\w+\(',  # Python function
            r'\bfunction\s+\w+\(',  # JavaScript function
            r'\bclass\s+\w+',  # Class definition
            r'\bimport\s+\w+',  # Import statement
            r'\bfrom\s+\w+\s+import',  # Python import
            r'^\s*[{}]\s*$',  # Standalone braces
        ]
        
        score = sum(1 for pattern in code_indicators if re.search(pattern, content, re.MULTILINE))
        return score >= 2