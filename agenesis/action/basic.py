from typing import Optional, Dict, Any

from .base import BaseAction, ActionResult
from ..cognition.base import CognitionResult
from ..providers import create_llm_provider


# Basic Action Constants
class BasicActionConfig:
    """Constants for action generation to avoid magic numbers"""
    # LLM Configuration defaults
    DEFAULT_LLM_TEMPERATURE = 0.7  # Slightly higher temperature for creative responses
    DEFAULT_MAX_RESPONSE_TOKENS = 300  # Token limit for response generation


class BasicAction(BaseAction):
    """Basic action processor with LLM enhancement and heuristic fallback"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.llm_provider = create_llm_provider()
        self.use_llm = self.llm_provider is not None
    
    async def generate_response(
        self,
        cognition_result: CognitionResult,
        context: Any = None
    ) -> ActionResult:
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
        """Generate response using LLM with structured prompt and memory context"""

        # Base cognition analysis
        prompt = f"""You are a helpful AI assistant. Based on the analysis and context below, generate an appropriate response.

Cognition Analysis:
- User Intent: {cognition_result.intent}
- Context Type: {cognition_result.context_type}
- Summary: {cognition_result.summary}
- Reasoning: {cognition_result.reasoning}"""

        # Add structured memory context from cognition result
        memory_context = cognition_result.memory_context
        if memory_context and memory_context.get('has_memories'):
            prompt += "\n\nMemory Context:"

            # Current Focus
            if memory_context.get('focus'):
                prompt += f"\n\nCURRENT INPUT:\n- {memory_context['focus'][0]}"

            # Recent conversation (Working Memory)
            if memory_context.get('working'):
                prompt += "\n\nRECENT CONVERSATION:"
                for memory in memory_context['working']:
                    prompt += f"\n- {memory}"

            # Long-term knowledge (Persistent Memory)
            if memory_context.get('persistent'):
                prompt += "\n\nRELEVANT KNOWLEDGE:"
                for memory in memory_context['persistent']:
                    prompt += f"\n- {memory}"

            prompt += "\n\nUse this memory context to provide a more informed and contextually relevant response."

        # Guidelines
        prompt += """

Guidelines:
- For "question" intent: Provide informative answers using available context
- For "request" intent: Offer helpful assistance based on relevant knowledge
- For "statement" intent: Acknowledge and engage using conversation history
- For "conversation" intent: Respond naturally referencing appropriate context

Generate a helpful, relevant response that incorporates the available memory context."""

        response = await self.llm_provider.complete(
            prompt=prompt,
            temperature=self.config.get('llm_temperature', BasicActionConfig.DEFAULT_LLM_TEMPERATURE),
            max_tokens=self.config.get('max_response_tokens', BasicActionConfig.DEFAULT_MAX_RESPONSE_TOKENS)
        )
        
        return ActionResult(
            response_text=response.strip(),
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
            internal_reasoning=f"Heuristic response for {intent} intent"
        )