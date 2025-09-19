from typing import Optional, Dict, Any

from .base import BaseAction, ActionResult
from ..cognition.base import CognitionResult
from ..providers import create_llm_provider


class BasicAction(BaseAction):
    """Basic action processor with LLM enhancement and heuristic fallback"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.llm_provider = create_llm_provider()
        self.use_llm = self.llm_provider is not None
    
    async def generate_response(self, cognition_result: CognitionResult, context: Any = None) -> ActionResult:
        """Main response generation entry point - returns simple text response"""
        if self.use_llm:
            try:
                return await self._generate_with_llm(cognition_result)
            except Exception as e:
                print(f"LLM response generation failed, falling back to heuristics: {e}")
                return self._generate_with_heuristics(cognition_result)
        else:
            return self._generate_with_heuristics(cognition_result)
    
    async def _generate_with_llm(self, cognition_result: CognitionResult) -> ActionResult:
        """Generate response using LLM with structured prompt"""
        
        prompt = f"""You are a helpful AI assistant. Based on the analysis below, generate an appropriate response.

Cognition Analysis:
- User Intent: {cognition_result.intent}
- Context Type: {cognition_result.context_type}  
- Summary: {cognition_result.summary}
- Reasoning: {cognition_result.reasoning}

Guidelines:
- For "question" intent: Provide informative answers
- For "request" intent: Offer helpful assistance  
- For "statement" intent: Acknowledge and engage appropriately
- For "conversation" intent: Respond naturally and socially

Generate a helpful, relevant response that addresses the user's intent."""

        response = await self.llm_provider.complete(
            prompt=prompt,
            temperature=self.config.get('llm_temperature', 0.7),
            max_tokens=self.config.get('max_response_tokens', 300)
        )
        
        return ActionResult(
            response_text=response.strip(),
            confidence=cognition_result.confidence * self.config.get('confidence_factor', 0.9),
            internal_reasoning=f"LLM response based on {cognition_result.intent} intent"
        )
    
    def _generate_with_heuristics(self, cognition_result: CognitionResult) -> ActionResult:
        """Generate response using heuristic rules when LLM unavailable"""
        
        intent = cognition_result.intent
        summary = cognition_result.summary
        
        # Simple template-based responses
        if intent == "question":
            response = f"I understand you're asking about: {summary}. I'd be happy to help, but I need more specific information to provide a detailed answer."
        elif intent == "request":
            response = f"I see you need help with: {summary}. Let me assist you with that."
        elif intent == "statement":
            response = f"Thank you for sharing that information about: {summary}. That's interesting!"
        else:  # conversation
            response = "I appreciate you reaching out! How can I help you today?"
        
        return ActionResult(
            response_text=response,
            confidence=cognition_result.confidence * 0.7,  # Lower confidence for heuristics
            internal_reasoning=f"Heuristic response for {intent} intent"
        )