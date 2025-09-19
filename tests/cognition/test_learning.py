import pytest
from datetime import datetime

from agenesis.cognition import CognitionResult, SimplePatternLearning


class TestSimplePatternLearning:
    
    def setup_method(self):
        self.learning = SimplePatternLearning()
    
    def test_extract_patterns_empty(self):
        """Test pattern extraction with no data"""
        patterns = self.learning.extract_patterns([])
        assert patterns == {}
    
    def test_extract_patterns_single_result(self):
        """Test pattern extraction with single result"""
        result = CognitionResult(
            intent="question",
            context_type="new",
            persistence_score=0.7,
            summary="Test question",
            relevant_memories=[],
            confidence=0.8,
            reasoning="Test reasoning"
        )
        
        patterns = self.learning.extract_patterns([result])
        
        assert patterns["most_common_intent"] == "question"
        assert patterns["most_common_context"] == "new"
        assert patterns["intent_distribution"] == {"question": 1}
        assert patterns["context_distribution"] == {"new": 1}
        assert patterns["sample_size"] == 1
        assert isinstance(patterns["extracted_at"], datetime)
    
    def test_extract_patterns_multiple_results(self):
        """Test pattern extraction with multiple results"""
        results = [
            CognitionResult("question", "new", 0.7, "Test 1", [], 0.8, "Reasoning 1"),
            CognitionResult("request", "continuation", 0.8, "Test 2", [], 0.9, "Reasoning 2"),
            CognitionResult("question", "new", 0.6, "Test 3", [], 0.7, "Reasoning 3"),
        ]
        
        patterns = self.learning.extract_patterns(results)
        
        assert patterns["most_common_intent"] == "question"  # 2 questions vs 1 request
        assert patterns["most_common_context"] == "new"     # 2 new vs 1 continuation
        assert patterns["intent_distribution"] == {"question": 2, "request": 1}
        assert patterns["context_distribution"] == {"new": 2, "continuation": 1}
        assert patterns["sample_size"] == 3
    
    def test_update_knowledge(self):
        """Test knowledge base updates"""
        pattern = {
            "most_common_intent": "question",
            "intent_distribution": {"question": 3, "request": 1},
            "context_distribution": {"new": 2, "continuation": 2},
            "sample_size": 4
        }
        
        self.learning.update_knowledge(pattern)
        
        assert self.learning.interaction_count == 4
        assert "intent_distribution" in self.learning.patterns
        assert self.learning.patterns["intent_distribution"] == {"question": 3, "request": 1}
    
    def test_multiple_knowledge_updates(self):
        """Test accumulating knowledge from multiple updates"""
        pattern1 = {
            "intent_distribution": {"question": 2, "request": 1},
            "sample_size": 3
        }
        pattern2 = {
            "intent_distribution": {"question": 1, "statement": 2},
            "sample_size": 3
        }
        
        self.learning.update_knowledge(pattern1)
        self.learning.update_knowledge(pattern2)
        
        assert self.learning.interaction_count == 6
        # Should merge distributions
        expected_intent_dist = {"question": 3, "request": 1, "statement": 2}
        assert self.learning.patterns["intent_distribution"] == expected_intent_dist
    
    def test_get_knowledge_summary(self):
        """Test knowledge summary generation"""
        pattern = {
            "intent_distribution": {"question": 5, "request": 3},
            "sample_size": 8
        }
        self.learning.update_knowledge(pattern)
        
        summary = self.learning.get_knowledge_summary()
        
        assert summary["total_interactions"] == 8
        assert "patterns" in summary
        assert "knowledge_updated_at" in summary
        assert isinstance(summary["knowledge_updated_at"], datetime)