import json
import asyncio
from typing import Dict, Any, Optional

from .base import BaseCognition, CognitionResult
from ..providers import create_llm_provider, OpenAIProvider, AnthropicProvider


# Basic Cognition Constants
class BasicCognitionConfig:
    """Constants for basic cognition to avoid magic numbers"""
    # LLM Configuration
    LLM_TEMPERATURE = 0.1  # Low temperature for consistent responses
    LLM_MAX_TOKENS = 300   # Token limit for cognition responses

    # Text Processing
    SUMMARY_MAX_LENGTH = 100  # Maximum length for summary text


class BasicCognition(BaseCognition):
    """Basic cognitive processing with LLM enhancement and heuristic fallback"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.llm_provider = create_llm_provider()
        self.use_llm = self.llm_provider is not None
        
    async def process(self, immediate_memory, working_memory, context: Any = None) -> CognitionResult:
        """Main cognitive processing entry point"""
        user_input = self._get_user_input(immediate_memory)
        
        if not user_input:
            return self._create_empty_result()
        
        recent_context = self._summarize_working_memory(working_memory)
        relevant_memories = self._find_relevant_memories(working_memory, user_input)
        
        if self.use_llm:
            try:
                return await self._process_with_llm(user_input, recent_context, relevant_memories)
            except Exception as e:
                print(f"LLM processing failed, falling back to heuristics: {e}")
                return self._process_with_heuristics(user_input, recent_context, relevant_memories)
        else:
            return self._process_with_heuristics(user_input, recent_context, relevant_memories)
    
    async def _process_with_llm(self, user_input: str, recent_context: str, relevant_memories: list) -> CognitionResult:
        """Process using LLM with structured JSON response"""
        prompt = f"""Analyze this user input and provide structured analysis:

User Input: "{user_input}"
Recent Context: {recent_context}

Respond with JSON only:
{{
    "intent": "question|request|statement|conversation",
    "context_type": "new|continuation|clarification|related",
    "should_persist": true|false,
    "summary": "brief description of what user wants",
    "reasoning": "why you classified it this way"
}}

Remember:
- should_persist should be true for important requests, preferences, meaningful information that would help in future interactions
- should_persist should be false for casual conversation, greetings, confirmations, or one-off questions
- intent should capture the primary purpose of the message
- context_type should indicate how this relates to the recent conversation"""

        try:
            response = await self.llm_provider.complete(
                prompt=prompt,
                temperature=BasicCognitionConfig.LLM_TEMPERATURE,
                max_tokens=BasicCognitionConfig.LLM_MAX_TOKENS
            )
            
            # Parse JSON response
            analysis = json.loads(response)
            
            return CognitionResult(
                intent=analysis.get("intent", "conversation"),
                context_type=analysis.get("context_type", "new"),
                should_persist=bool(analysis.get("should_persist", False)),
                summary=analysis.get("summary", user_input[:BasicCognitionConfig.SUMMARY_MAX_LENGTH]),
                relevant_memories=relevant_memories,
                reasoning=analysis.get("reasoning", "LLM analysis")
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # JSON parsing failed, try to extract what we can
            return self._parse_partial_llm_response(response, user_input, recent_context, relevant_memories)
    
    def _parse_partial_llm_response(self, response: str, user_input: str, recent_context: str, relevant_memories: list) -> CognitionResult:
        """Extract information from malformed LLM response"""
        response_lower = response.lower()
        
        # Try to extract intent
        intent = "conversation"
        if "question" in response_lower or "?" in user_input:
            intent = "question"
        elif "request" in response_lower or any(word in user_input.lower() for word in ["please", "can you", "help"]):
            intent = "request"
        elif "statement" in response_lower:
            intent = "statement"
        
        # Try to extract persistence decision
        should_persist = False
        if "important" in response_lower or "persist" in response_lower or "true" in response_lower:
            should_persist = True
        elif "casual" in response_lower or "false" in response_lower or "greeting" in response_lower:
            should_persist = False

        return CognitionResult(
            intent=intent,
            context_type="new",
            should_persist=should_persist,
            summary=user_input[:BasicCognitionConfig.SUMMARY_MAX_LENGTH],
            relevant_memories=relevant_memories,
            reasoning="Partial LLM response parsing"
        )
    
    def _process_with_heuristics(self, user_input: str, recent_context: str, relevant_memories: list) -> CognitionResult:
        """Fallback heuristic processing when LLM is not available"""
        
        # Simple intent classification
        intent = self._classify_intent_heuristic(user_input)
        
        # Simple context detection
        context_type = "continuation" if relevant_memories else "new"

        # Simple heuristic persistence decision
        should_persist = intent in ["request", "statement"]  # Persist requests and statements, not questions/conversation

        return CognitionResult(
            intent=intent,
            context_type=context_type,
            should_persist=should_persist,
            summary=f"Heuristic analysis: {intent}",
            relevant_memories=relevant_memories,
            reasoning="Heuristic pattern matching"
        )
    
    def _classify_intent_heuristic(self, user_input: str) -> str:
        """Simple keyword-based intent classification"""
        user_input_lower = user_input.lower()
        
        # Request indicators (check first - more specific than questions)
        if any(phrase in user_input_lower for phrase in ['can you', 'could you', 'would you', 'please help', 'help me']):
            return 'request'
        elif any(word in user_input_lower for word in ['please', 'help', 'do', 'make', 'create', 'show']):
            return 'request'
        
        # Question indicators
        elif '?' in user_input or any(word in user_input_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            return 'question'
        
        # Statement indicators
        elif any(word in user_input_lower for word in ['i am', 'i have', 'i think', 'my', 'i like', 'i prefer']):
            return 'statement'
        
        # Default to conversation
        return 'conversation'
    
    def _create_empty_result(self) -> CognitionResult:
        """Create result for empty or missing input"""
        return CognitionResult(
            intent="conversation",
            context_type="new",
            should_persist=False,  # Never persist empty input
            summary="Empty input",
            relevant_memories=[],
            reasoning="No input provided"
        )