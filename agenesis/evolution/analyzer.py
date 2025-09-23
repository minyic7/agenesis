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
    
    async def analyze_memory_session(self, immediate_memory, working_memory, persona=None) -> EvolutionDecision:
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
                return await self._analyze_with_llm(session_content, persona)
            except Exception as e:
                print(f"LLM evolution analysis failed: {e}")
                return EvolutionDecision(
                    should_persist=False,
                    rejection_reason=f"LLM analysis failed: {e}"
                )
        else:
            return EvolutionDecision(
                should_persist=False,
                rejection_reason="No LLM provider available for evolution analysis"
            )
    
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
    
    async def _analyze_with_llm(self, session_content: str, persona=None) -> EvolutionDecision:
        """Use LLM to analyze session for learning opportunities"""

        # Build persona-specific prompt
        base_prompt = f"""Analyze this conversation session for valuable learning:

Session Content:
{session_content}

"""

        # Add persona-specific learning instructions if available
        if persona and persona.get_learning_preferences():
            learning_prefs = persona.get_learning_preferences()

            persona_prompt = f"""PERSONA LEARNING CONFIGURATION:
Persona: {persona.get_name()} - {persona.get_description()}

Focus on learning about:
{self._format_list(learning_prefs.get('learn_about', []))}

Ignore these topics:
{self._format_list(learning_prefs.get('ignore_topics', []))}

Learning aggressiveness: {learning_prefs.get('learning_aggressiveness', 'moderate')}

CUSTOM INSTRUCTIONS:
{learning_prefs.get('evolution_instructions', 'Use general learning criteria.')}

"""
            base_prompt += persona_prompt
        else:
            # Default learning criteria when no persona
            base_prompt += """DEFAULT LEARNING CRITERIA:
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

"""

        base_prompt += """
Critical Questions:
1. Does this session contain genuinely valuable patterns, preferences, or knowledge that would help the agent perform better in future similar situations?
2. Is this information specific and actionable enough to warrant long-term storage?
3. Would storing this information improve future interactions, or would it just add noise?

IMPORTANT: Be selective. Most casual conversations should NOT be marked for evolution learning.

Respond with JSON only:
{
    "should_persist": false,
    "learning_type": null,
    "learning_description": null,
    "future_application": null,
    "rejection_reason": "reason why this should not be learned (if should_persist=false)"
}

Default to should_persist=false unless there's clear, valuable learning."""

        response = await self.llm_provider.complete(
            prompt=base_prompt,
            temperature=0.1,  # Low temperature for consistent analysis
            max_tokens=300
        )
        
        try:
            analysis = json.loads(response)
            
            return EvolutionDecision(
                should_persist=analysis.get("should_persist", False),
                learning_type=analysis.get("learning_type"),
                learning_description=analysis.get("learning_description"),
                future_application=analysis.get("future_application"),
                rejection_reason=analysis.get("rejection_reason")
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # JSON parsing failed, default to no learning
            return EvolutionDecision(
                should_persist=False,
                rejection_reason=f"LLM response parsing failed: {e}"
            )
    
    
    def _extract_session_summary(self, immediate_memory, working_memory) -> str:
        """Extract meaningful content from memory for analysis (user input only)"""
        content_parts = []

        # Get current focus - user input only
        current_record = immediate_memory.get_current()
        if current_record:
            content_parts.append(f"Current user input: {current_record.perception_result.content}")

        # Get recent working memory - user inputs only
        recent_records = working_memory.get_recent(5)
        for i, record in enumerate(recent_records):
            content_parts.append(f"User input {i+1}: {record.perception_result.content}")

        return "\n".join(content_parts) if content_parts else ""
    
    def create_evolved_knowledge_metadata(self, decision: EvolutionDecision) -> EvolvedKnowledge:
        """Create metadata for evolved knowledge based on evolution decision"""
        if not decision.should_persist:
            raise ValueError("Cannot create evolved knowledge metadata for non-persistent decision")

        return EvolvedKnowledge(
            knowledge_summary=decision.learning_description or "Valuable learning identified",
            learning_context=decision.learning_type or "general",
            future_relevance=decision.future_application or "Future similar contexts"
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

    def _format_list(self, items: list) -> str:
        """Format a list of items for prompt inclusion"""
        if not items:
            return "- (none specified)"
        return "\n".join(f"- {item}" for item in items)

    def _should_learn_from_interaction(self, user_input: str, agent_response: str, persona=None) -> bool:
        """Check if interaction should be learned from using validation functions"""

        # If persona has validation functions configured, use them
        if persona and persona.get_learning_preferences():
            learning_prefs = persona.get_learning_preferences()
            validation_config = learning_prefs.get('validation')

            if validation_config:
                # Create metadata for validation functions
                metadata = {
                    'user_input_length': len(user_input),
                    'agent_response_length': len(agent_response),
                    'has_question': '?' in user_input,
                    'has_technical_keywords': self._has_technical_keywords(user_input),
                    'confidence_indicators': self._count_confidence_words(user_input),
                    'timestamp': None,  # Could add timestamp if needed
                }

                # Run user-defined validation function if configured
                validation_function = validation_config.get('interaction_function')
                if validation_function:
                    # This is a placeholder for user function loading
                    # In practice, this would load and execute user-provided validation code
                    # For now, return True to allow the implementation to be added later
                    print(f"📋 User validation function configured: {validation_function}")
                    print(f"   Input: '{user_input[:50]}...'")
                    print(f"   Response: '{agent_response[:50]}...'")
                    print(f"   Metadata: {metadata}")
                    # TODO: Implement function loading and execution
                    return True

        # Default: allow learning (will be filtered by LLM analysis)
        return True

    def _has_technical_keywords(self, text: str) -> bool:
        """Check if text contains technical keywords"""
        technical_words = ['framework', 'api', 'code', 'function', 'method', 'class', 'variable',
                          'database', 'server', 'client', 'algorithm', 'debug', 'error', 'bug']
        text_lower = text.lower()
        return any(word in text_lower for word in technical_words)

    def _count_confidence_words(self, text: str) -> int:
        """Count confidence-indicating words"""
        confidence_words = ['always', 'prefer', 'usually', 'typically', 'often', 'regularly']
        text_lower = text.lower()
        return sum(1 for word in confidence_words if word in text_lower)