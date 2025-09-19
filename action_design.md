# Action Module Design

## Purpose
The Action module is the final step in the cognitive pipeline: **Perception → Memory → Cognition → ACTION**

It takes `CognitionResult` and generates a simple text response back to the user, completing the agent's interaction cycle.

## Scope: Start Simple
- **Current Focus**: Text-only responses
- **Future Expansion**: May support images, videos, files, and other response types
- **Keep It Simple**: Minimal viable action module to complete the core pipeline

## Current Architecture Analysis

### What Exists:
- **Cognition Output**: `CognitionResult` with intent, context_type, persistence_score, summary, confidence, reasoning
- **Agent Integration Point**: `agent.py:64` has TODO for cognition and action modules
- **Memory Context**: Immediate, working, and persistent memory available
- **LLM Providers**: Anthropic and OpenAI providers ready for use

### What's Missing:
- Response generation based on cognition analysis
- Integration between cognition results and LLM providers for response generation
- Action result structure for tracking what was done
- Integration with existing agent flow

## Design Principles

### 1. Follow Existing Patterns
- **Reuse LLM Provider Pattern**: Use existing `create_llm_provider()` infrastructure
- **Async/Await**: Consistent with cognition module
- **Graceful Fallback**: Heuristic responses when LLM unavailable
- **Clean Interfaces**: Abstract base class + concrete implementation

### 2. Input/Output Clarity
- **Input**: `CognitionResult` from cognition module
- **Output**: `ActionResult` with response text and metadata
- **Context**: Access to memory systems for additional context

### 3. Separation of Concerns
- **Action Module**: Response generation logic
- **Agent Class**: Orchestration and integration
- **Memory**: Context retrieval (already handled by cognition)

## Core Classes Design (Simplified)

### `ActionResult`
```python
@dataclass
class ActionResult:
    """Result of action processing - simple text response"""
    response_text: str              # The actual text response to user
    confidence: float              # 0.0-1.0 confidence in response  
    timestamp: Optional[datetime] = None
    
    # Optional metadata for debugging/logging (not user-facing)
    internal_reasoning: Optional[str] = None
```

### `BaseAction` (Abstract)
```python
class BaseAction(ABC):
    """Abstract base class for action processing"""
    
    @abstractmethod
    async def generate_response(self, cognition_result: CognitionResult) -> ActionResult:
        """Generate simple text response based on cognition analysis"""
        pass
```

### `BasicAction` (Implementation)
```python
class BasicAction(BaseAction):
    """Basic action processor with LLM enhancement and heuristic fallback"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.llm_provider = create_llm_provider()  # Reuse existing pattern
        self.use_llm = self.llm_provider is not None
    
    async def generate_response(self, cognition_result: CognitionResult) -> ActionResult:
        """Main response generation entry point - returns simple text response"""
        if self.use_llm:
            try:
                return await self._generate_with_llm(cognition_result)
            except Exception as e:
                print(f"LLM response generation failed, falling back to heuristics: {e}")
                return self._generate_with_heuristics(cognition_result)
        else:
            return self._generate_with_heuristics(cognition_result)
```

## Response Generation Strategy

### LLM-Enhanced Generation
```python
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
        temperature=0.7,  # More creative than cognition
        max_tokens=300    # Keep responses concise
    )
    
    return ActionResult(
        response_text=response.strip(),
        confidence=cognition_result.confidence * 0.9,  # Slightly lower than cognition
        internal_reasoning=f"LLM response based on {cognition_result.intent} intent"
    )
```

### Heuristic Fallback
```python
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
```

## Integration with Agent Class

### Updated Agent.process_input()
```python
async def process_input(self, text_input: str) -> str:
    """Main processing pipeline: perception → memory → cognition → action"""
    # 1. Perceive input
    perception_result = self.perception.process(text_input)
    
    # 2. Store in memory systems
    self.immediate_memory.store(perception_result)
    self.working_memory.store(perception_result)
    if self.has_persistent_memory:
        self.persistent_memory.store(perception_result)
    
    # 3. Cognitive processing  
    cognition_result = await self.cognition.process(self.immediate_memory, self.working_memory)
    
    # 4. Action generation (simple text response)
    action_result = await self.action.generate_response(cognition_result)
    
    # 5. Return the response text to user
    return action_result.response_text
```

## Error Handling & Fallbacks

1. **LLM Unavailable**: Fall back to heuristic templates
2. **LLM Errors**: Catch exceptions, use heuristics  
3. **Empty Results**: Generate polite "I don't understand" responses
4. **Confidence Tracking**: Lower confidence for fallback methods

## Testing Strategy

### Unit Tests:
- Response generation for each intent type
- Heuristic fallback behavior
- Error handling scenarios
- Confidence calculation

### Integration Tests:
- End-to-end flow with real LLM providers
- Full pipeline: perception → memory → cognition → action
- Agent orchestration

## Implementation Plan

### Phase 1: Core Action Module
1. Create `ActionResult` dataclass (simplified)
2. Implement `BaseAction` interface  
3. Build `BasicAction` with LLM + heuristic generation
4. Update imports and exports

### Phase 2: Agent Integration  
1. Update `Agent.process_input()` to use action module
2. Make method async to support action generation
3. Add cognition + action modules to agent initialization

### Phase 3: Testing & Validation
1. Unit tests for action generation
2. Integration tests with full pipeline
3. Test with both LLM and heuristic modes
4. Validate response quality and appropriateness

## Configuration Options (Simple)

```python
action_config = {
    'llm_temperature': 0.7,        # Creativity for responses
    'max_response_tokens': 300,    # Keep responses concise
    'confidence_factor': 0.9,      # Confidence adjustment
}
```

## Future Enhancements

- **Multi-modal Responses**: Images, videos, files, and other response types
- **Action Types**: Function calls, tool usage, external API calls
- **Response Storage**: Store ActionResults in memory for learning
- **Personality Integration**: Consistent voice and style from persona module
- **Response Templates**: More sophisticated heuristic responses

## Success Criteria

- [ ] Complete perception → memory → cognition → action pipeline working
- [ ] Simple text responses for each intent type (question, request, statement, conversation)
- [ ] Graceful fallback when LLM unavailable
- [ ] Integration with existing agent architecture
- [ ] Comprehensive test coverage
- [ ] Agent returns helpful text responses to user input