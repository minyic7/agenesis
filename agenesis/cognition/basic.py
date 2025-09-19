import json
import asyncio
from typing import Dict, Any, Optional

from .base import BaseCognition, CognitionResult
from ..providers import create_llm_provider, OpenAIProvider, AnthropicProvider


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
    "persistence_score": 0.0-1.0,
    "summary": "brief description of what user wants",
    "reasoning": "why you classified it this way"
}}

Remember:
- persistence_score should be higher (0.7-1.0) for important requests, preferences, or meaningful information
- persistence_score should be lower (0.0-0.4) for casual conversation, greetings, confirmations
- intent should capture the primary purpose of the message
- context_type should indicate how this relates to the recent conversation"""

        try:
            response = await self.llm_provider.complete(
                prompt=prompt,
                temperature=0.1,
                max_tokens=300
            )
            
            # Parse JSON response
            analysis = json.loads(response)
            
            return CognitionResult(
                intent=analysis.get("intent", "conversation"),
                context_type=analysis.get("context_type", "new"),
                persistence_score=float(analysis.get("persistence_score", 0.5)),
                summary=analysis.get("summary", user_input[:100]),
                relevant_memories=relevant_memories,
                confidence=0.9,  # High confidence for LLM analysis
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
        
        # Try to extract persistence score
        persistence_score = 0.5
        if "high" in response_lower or "important" in response_lower:
            persistence_score = 0.8
        elif "low" in response_lower or "casual" in response_lower:
            persistence_score = 0.2
        
        return CognitionResult(
            intent=intent,
            context_type="new",
            persistence_score=persistence_score,
            summary=user_input[:100],
            relevant_memories=relevant_memories,
            confidence=0.6,  # Lower confidence for partial parsing
            reasoning="Partial LLM response parsing"
        )
    
    def _process_with_heuristics(self, user_input: str, recent_context: str, relevant_memories: list) -> CognitionResult:
        """Fallback heuristic processing when LLM is not available"""
        
        # Simple intent classification
        intent = self._classify_intent_heuristic(user_input)
        
        # Simple context detection
        context_type = "continuation" if relevant_memories else "new"
        
        # Placeholder persistence scoring (neutral default as agreed)
        persistence_score = 0.5
        
        return CognitionResult(
            intent=intent,
            context_type=context_type,
            persistence_score=persistence_score,
            summary=f"Heuristic analysis: {intent}",
            relevant_memories=relevant_memories,
            confidence=0.7,  # Medium confidence for heuristics
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
            persistence_score=0.1,  # Low persistence for empty input
            summary="Empty input",
            relevant_memories=[],
            confidence=0.1,
            reasoning="No input provided"
        )