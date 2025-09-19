from typing import Optional, Dict, Any
import json

from .base import BaseEvolution, EvolutionDecision, EvolvedKnowledge
from ..providers import create_llm_provider


class EvolutionAnalyzer(BaseEvolution):
    """Main evolution analyzer supporting multiple data sources"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.llm_provider = create_llm_provider()
        self.use_llm = self.llm_provider is not None
    
    async def analyze_memory_session(self, immediate_memory, working_memory) -> EvolutionDecision:
        """Analyze memory session for valuable learning opportunities"""
        
        # Extract session content
        session_content = self._extract_session_summary(immediate_memory, working_memory)
        
        if not session_content:
            return EvolutionDecision(
                should_persist=False,
                rejection_reason="No meaningful session content to analyze"
            )
        
        if self.use_llm:
            try:
                return await self._analyze_with_llm(session_content)
            except Exception as e:
                print(f"LLM evolution analysis failed, falling back to heuristics: {e}")
                return self._analyze_with_heuristics(session_content)
        else:
            return self._analyze_with_heuristics(session_content)
    
    async def analyze_for_learning(self, data_source: Any) -> EvolutionDecision:
        """Generic learning analysis interface"""
        # For now, assume data_source is (immediate_memory, working_memory) tuple
        if isinstance(data_source, tuple) and len(data_source) == 2:
            return await self.analyze_memory_session(data_source[0], data_source[1])
        else:
            return EvolutionDecision(
                should_persist=False,
                rejection_reason="Unsupported data source type"
            )
    
    async def _analyze_with_llm(self, session_content: str) -> EvolutionDecision:
        """Use LLM to analyze session for learning opportunities"""
        
        prompt = f"""Analyze this conversation session for valuable learning:

Session Content:
{session_content}

Critical Questions:
1. Does this session contain genuinely valuable patterns, preferences, or knowledge that would help the agent perform better in future similar situations?
2. Is this information specific and actionable enough to warrant long-term storage?
3. Would storing this information improve future interactions, or would it just add noise?

IMPORTANT: Be selective. Most casual conversations should NOT be marked for evolution learning.

Only identify truly valuable insights like:
- Clear user preferences or requirements
- Successful problem-solving approaches
- Important context about user's work/interests
- Patterns that consistently work well

AVOID learning from:
- Casual small talk or greetings
- One-off questions without broader application
- Generic information easily available elsewhere
- Temporary context that won't be relevant later

Respond with JSON only:
{{
    "should_persist": false,
    "learning_type": null,
    "learning_description": null,
    "confidence": 0.0,
    "future_application": null,
    "rejection_reason": "reason why this should not be learned (if should_persist=false)"
}}

Default to should_persist=false unless there's clear, valuable learning."""

        response = await self.llm_provider.complete(
            prompt=prompt,
            temperature=0.1,  # Low temperature for consistent analysis
            max_tokens=300
        )
        
        try:
            analysis = json.loads(response)
            
            return EvolutionDecision(
                should_persist=analysis.get("should_persist", False),
                learning_type=analysis.get("learning_type"),
                learning_description=analysis.get("learning_description"),
                confidence=float(analysis.get("confidence", 0.0)),
                future_application=analysis.get("future_application"),
                rejection_reason=analysis.get("rejection_reason")
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # JSON parsing failed, default to no learning
            return EvolutionDecision(
                should_persist=False,
                rejection_reason=f"LLM response parsing failed: {e}"
            )
    
    def _analyze_with_heuristics(self, session_content: str) -> EvolutionDecision:
        """Simple heuristic analysis when LLM unavailable"""
        
        # Very conservative heuristics - default to no learning
        content_lower = session_content.lower()
        
        # Only identify very obvious preference statements
        preference_indicators = ['i prefer', 'i like', 'i usually', 'i always', 'i work with', 'my job']
        
        if any(indicator in content_lower for indicator in preference_indicators):
            return EvolutionDecision(
                should_persist=True,
                learning_type="preference",
                learning_description="User preference statement detected",
                confidence=0.6,
                future_application="Apply user preferences in similar contexts"
            )
        
        # Default: no learning for heuristic mode
        return EvolutionDecision(
            should_persist=False,
            rejection_reason="Heuristic analysis found no clear learning value"
        )
    
    def _extract_session_summary(self, immediate_memory, working_memory) -> str:
        """Extract meaningful content from memory for analysis"""
        content_parts = []
        
        # Get current focus
        current_record = immediate_memory.get_current()
        if current_record:
            content_parts.append(f"Current: {current_record.perception_result.content}")
        
        # Get recent working memory
        recent_records = working_memory.get_recent(5)
        for i, record in enumerate(recent_records):
            content_parts.append(f"Memory {i+1}: {record.perception_result.content}")
        
        return "\n".join(content_parts) if content_parts else ""
    
    def create_evolved_knowledge_metadata(self, decision: EvolutionDecision) -> EvolvedKnowledge:
        """Create metadata for evolved knowledge based on evolution decision"""
        if not decision.should_persist:
            raise ValueError("Cannot create evolved knowledge metadata for non-persistent decision")
        
        return EvolvedKnowledge(
            knowledge_summary=decision.learning_description or "Valuable learning identified",
            learning_context=decision.learning_type or "general",
            future_relevance=decision.future_application or "Future similar contexts",
            reliability_boost=1.0 + (decision.confidence * 0.5)  # 1.0-1.5x boost based on confidence
        )
    
    def should_trigger_analysis(self, trigger_type: str, context: Dict[str, Any]) -> bool:
        """Determine if evolution analysis should be triggered"""
        
        # Always trigger for end of session
        if trigger_type == "session_end":
            return True
        
        # Trigger for high-confidence interactions
        if trigger_type == "high_confidence" and context.get("confidence", 0) > 0.8:
            return True
        
        # Trigger for explicit user learning indicators
        if trigger_type == "user_learning" and context.get("contains_learning_indicators", False):
            return True
        
        return False