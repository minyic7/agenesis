# LLM Provider Design

## Overview
Extensible LLM provider system for cognitive processing. Start with OpenAI API compatibility, designed for easy expansion to other providers.

## Core Architecture

### Abstract Provider Interface
```python
class BaseLLMProvider(ABC):
    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str
    
    @abstractmethod
    async def score(self, prompt: str, **kwargs) -> float
    
    @abstractmethod  
    async def classify(self, prompt: str, options: List[str], **kwargs) -> str
```

### Provider Configuration
- **Environment-based**: Load API keys from `.env` file
- **Provider selection**: Auto-detect or explicit configuration
- **Fallback strategy**: Graceful degradation if provider unavailable

## Phase 1: OpenAI Provider

### Configuration (.env)
```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini  # Default model for cost efficiency
OPENAI_BASE_URL=https://api.openai.com/v1  # For compatibility with OpenAI-compatible APIs
```

### OpenAI Provider Features
- **Complete**: General text completion for complex tasks
- **Score**: Return numerical scores (0.0-1.0) for persistence rating
- **Classify**: Return one of provided options for intent classification
- **Error handling**: Retry logic with exponential backoff
- **Rate limiting**: Respect OpenAI rate limits

## Usage Patterns for Cognition

### Intent Classification
```python
intent = await llm.classify(
    prompt=f"Classify the intent of this message: '{user_input}'",
    options=["question", "request", "statement", "conversation"]
)
```

### Persistence Scoring  
```python
score = await llm.score(
    prompt=f"Rate the importance for long-term memory (0.0-1.0): '{user_input}'"
)
```

### Context Analysis
```python
context = await llm.complete(
    prompt=f"How does this message relate to the conversation? Message: '{user_input}' Recent context: {context_summary}"
)
```

## Provider Registry System

### Auto-Detection Priority
1. OpenAI (if OPENAI_API_KEY present)
2. Future providers (Anthropic, local models, etc.)
3. Fallback to simple heuristics if no LLM available

### Provider Factory
```python
def create_llm_provider(config: Dict[str, Any]) -> BaseLLMProvider:
    if config.get('openai_api_key'):
        return OpenAIProvider(config)
    # Future: Add other providers
    else:
        return LocalHeuristicProvider()  # Non-LLM fallback
```

## Implementation Progress

### ✅ Phase 1: OpenAI Foundation
- [ ] BaseLLMProvider interface
- [ ] OpenAIProvider implementation
- [ ] Environment configuration loading
- [ ] Basic error handling and retries
- [ ] Provider factory and auto-detection

### 🔄 Phase 2: Enhanced Features (Future)
- [ ] Multiple provider support (Anthropic, Ollama, etc.)
- [ ] Advanced rate limiting and caching
- [ ] Cost tracking and optimization
- [ ] Model selection strategies

### 📋 Phase 3: Advanced Capabilities (Future)
- [ ] Streaming responses for long completions
- [ ] Function calling for structured outputs
- [ ] Fine-tuned model support
- [ ] Local model integration (via Ollama, etc.)

## Design Principles

1. **Provider agnostic**: Cognition module doesn't know which LLM it's using
2. **Graceful fallback**: Always work even without LLM access
3. **Cost conscious**: Use efficient models and caching where appropriate
4. **Future-ready**: Easy to add new providers and capabilities
5. **Environment-driven**: Configuration through .env for security