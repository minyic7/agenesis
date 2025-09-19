# Cognition Module Design

## Start Simple: Intent Recognition + Context Awareness

### Current Focus: Process Every ImmediateMemory Entry

For each new input stored in ImmediateMemory, cognition should:

#### Core Processing
```
Input: ImmediateMemory (current focus) + WorkingMemory (recent context)
Process: Intent Recognition + Context Awareness
Output: CognitionResult (intent + context understanding)
```

### Phase 1: Basic Intent Recognition

#### Intent Types (Start Simple)
- **Question**: User asking for information
- **Request**: User asking for action/help  
- **Statement**: User sharing information
- **Conversation**: Social/casual interaction

#### Context Awareness (Start Simple)
- **New Topic**: Fresh conversation thread
- **Continuation**: Following up on previous messages  
- **Clarification**: Asking about something from working memory
- **Related**: Connected to recent context but new angle

### Core Classes

#### `CognitionResult`
- `intent`: Classified intent type
- `context_type`: How this relates to recent conversation
- `confidence`: How sure we are about the classification
- `relevant_memories`: Which working memories are important
- `summary`: Simple description of what user wants
- `persistence_score`: How important this is to remember long-term (0.0-1.0)

#### `BasicCognition`
- `process(immediate_memory, working_memory) -> CognitionResult`
- Simple keyword/pattern-based intent recognition
- Basic context relationship detection

## Implementation Progress

### ✅ Phase 1: Minimal Viable Cognition - COMPLETED
- [x] Intent classification (question/request/statement/conversation)
- [x] Context awareness (new/continuation/clarification/related)
- [x] CognitionResult output format with confidence and reasoning
- [x] LLM-enhanced cognition (structured JSON response)
- [x] Heuristic fallback when LLM unavailable
- [x] Basic persistence scoring (neutral default 0.5)
- [x] Simple keyword-based relevant memory detection
- [x] Pattern learning capability with SimplePatternLearning
- [x] Comprehensive test coverage (unit + integration tests)

### 🔄 Phase 2: Enhanced Understanding (Next Priority)
- [ ] **Improved Memory Retrieval**: Replace keyword matching with semantic similarity
  - [ ] Add embedding provider interface (OpenAI embeddings, sentence-transformers)
  - [ ] Implement embedding-based `_find_relevant_memories` method
  - [ ] Add cosine similarity for relevance scoring
  - [ ] Hybrid approach: embeddings + keyword matching + recency weighting
- [ ] Better intent recognition (ML-based or LLM-refined heuristics)
- [ ] Deeper context analysis with conversation topic detection
- [ ] User preference detection and adaptation

### 📋 Phase 3: Advanced Cognition (Future)
- [ ] Multi-turn reasoning and goal tracking
- [ ] Vector database integration for efficient memory search
- [ ] Learning from outcomes and user feedback
- [ ] Response strategy planning
- [ ] Advanced semantic reasoning with knowledge graphs

## Memory Management Strategy

### Working Memory (Always Store)
- **All interactions** go to working memory for conversational fluency
- Casual conversation, questions, everything
- Maintains natural conversation flow

### Persistent Memory (Selective Storage)
- **Only high-value interactions** based on `persistence_score`
- Threshold-based: Only store if score > 0.6 (configurable)

#### Persistence Scoring Strategy

**Hybrid Approach: Base Scoring + Optional LLM Enhancement**

##### Base Scoring (Heuristic Fallback - Placeholder)
```python
# TODO: Implement comprehensive base scoring system
# This is the critical foundation that must handle ALL situations gracefully
# Needs careful consideration of:
# - Cultural inclusivity
# - Edge cases and corner scenarios  
# - User behavior patterns
# - Context dependencies
# - Graceful degradation

def calculate_base_persistence_score(user_input, context):
    """
    Placeholder for sophisticated heuristic scoring
    Must be considerate, inclusive, and comprehensive
    """
    # TODO: Implement robust pattern analysis
    # TODO: Consider cultural and linguistic variations
    # TODO: Handle edge cases (empty input, special characters, etc.)
    # TODO: Context-aware scoring based on conversation flow
    # TODO: User behavior pattern recognition
    
    return 0.5  # Temporary neutral default
```

##### LLM-Enhanced Cognition (Single Structured Call)
```python
# Single comprehensive LLM call for all cognitive analysis
if llm_provider_available:
    cognition_result = await llm_provider.complete(
        prompt=f"""Analyze this user input and provide structured analysis:

User Input: "{user_input}"
Recent Context: {working_memory_summary}

Respond with JSON only:
{{
    "intent": "question|request|statement|conversation",
    "context_type": "new|continuation|clarification|related", 
    "persistence_score": 0.0-1.0,
    "summary": "brief description of what user wants",
    "relevant_memories": ["list of relevant context"],
    "reasoning": "why you classified it this way"
}}""",
        temperature=0.1,
        max_tokens=300
    )
    
    # Parse structured response
    try:
        analysis = json.loads(cognition_result)
        intent = analysis["intent"]
        context_type = analysis["context_type"]
        persistence_score = analysis["persistence_score"]
        summary = analysis["summary"]
    except (json.JSONDecodeError, KeyError):
        # Fallback to heuristics if JSON parsing fails
        intent, context_type, persistence_score, summary = heuristic_fallback(user_input)
else:
    # Direct heuristic processing when no LLM available
    intent, context_type, persistence_score, summary = heuristic_processing(user_input, recent_context)
```

**Benefits of Single Call Approach:**
- **Cost Efficient**: One API call instead of multiple separate calls
- **Context Aware**: LLM sees full picture for consistent analysis  
- **Future Ready**: Easy to add more analysis fields
- **Structured Output**: JSON format for reliable parsing

**LLM Provider Integration**
```python
from ..providers import create_llm_provider
import json

class BasicCognition:
    def __init__(self, config=None):
        self.llm_provider = create_llm_provider()  # Auto-detects from .env
        self.use_llm = isinstance(self.llm_provider, OpenAIProvider)
        
    async def process(self, immediate_memory, working_memory):
        user_input = immediate_memory.get_current().perception_result.content
        recent_context = self._summarize_working_memory(working_memory)
        
        if self.use_llm:
            return await self._process_with_llm(user_input, recent_context)
        else:
            return self._process_with_heuristics(user_input, recent_context)
            
    def _summarize_working_memory(self, working_memory):
        """Create brief summary of recent conversation for context"""
        recent = working_memory.get_recent(3)
        return "; ".join([r.perception_result.content[:50] for r in recent])
```

### Updated Agent Flow
```
1. User Input → Perception → PerceptionResult
2. Store in ImmediateMemory + WorkingMemory  
3. Cognition → CognitionResult (persistence_score based on user input)
4. Generate Agent Response (thinking + response)
5. Storage Decision:
   - WorkingMemory ← conversation unit (always)
   - PersistentMemory ← conversation unit (if persistence_score > 0.6)
```

### Conversation Unit Storage
```
ConversationUnit:
- user_input: PerceptionResult
- agent_thinking: str (internal reasoning)  
- agent_response: str
- persistence_score: float (from user input cognition)
- timestamp: datetime
```

### Storage Rules (Simple)
- **User input drives persistence**: If user input is worth remembering, store entire conversation
- **No response analysis**: Keep it simple - user input cognition determines everything
- **Accept edge cases**: Casual greetings with valuable responses won't persist (user learns to be explicit)

## Design Principles

1. **Every ImmediateMemory entry gets processed** - ensures understanding
2. **All interactions flow naturally** - working memory maintains context
3. **Selective long-term memory** - only remember what matters
4. **Human-like forgetting** - casual stuff fades, important stuff persists

## Current Implementation Status & TODOs

### ✅ What's Working Now
- **Basic Cognition**: Intent classification, context awareness, persistence scoring
- **LLM Integration**: Works with Anthropic and OpenAI providers with graceful fallback
- **Simple Memory Retrieval**: Keyword-based matching for relevant memories
- **Pattern Learning**: Basic analysis of user interaction patterns
- **Robust Testing**: Comprehensive test suite with real API integration

### ⚠️ Current Limitations & Known Issues

#### Memory Retrieval (Biggest Gap)
**Current Implementation**: Simple keyword overlap detection in `_find_relevant_memories`
```python
# Very basic - just checks if any words match
input_words = set(user_input.lower().split())
content_words = set(record.perception_result.content.lower().split())
if len(input_words.intersection(content_words)) > 0:
    relevant_ids.append(record.id)
```

**Problems:**
- No semantic understanding ("car" vs "automobile" won't match)
- No stemming ("run" vs "running" won't match)  
- No relevance scoring (all matches treated equally)
- No context weighting or recency bias
- Limited to exact word overlap

#### Persistence Scoring
**Current**: Neutral default (0.5) for heuristic mode, LLM-based for enhanced mode
**TODO**: Implement sophisticated heuristic scoring system

### 🎯 Next Sprint Priorities

#### 1. Enhanced Memory Retrieval (Critical)
- **Add Embedding Provider Interface**
  ```python
  # New module: agenesis/providers/embedding.py
  class BaseEmbeddingProvider(ABC):
      async def embed_text(self, text: str) -> List[float]
      async def embed_batch(self, texts: List[str]) -> List[List[float]]
  
  class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
      # Uses text-embedding-ada-002
  
  class LocalEmbeddingProvider(BaseEmbeddingProvider):  
      # Uses sentence-transformers
  ```

- **Semantic Memory Retrieval**
  ```python
  def _find_relevant_memories_semantic(self, working_memory, user_input: str) -> List[str]:
      # 1. Get embedding for current input
      # 2. Compare with stored memory embeddings  
      # 3. Use cosine similarity for relevance scoring
      # 4. Weight by recency and interaction frequency
      # 5. Return top-k most relevant memory IDs
  ```

- **Hybrid Approach**
  ```python
  def _find_relevant_memories_hybrid(self, working_memory, user_input: str) -> List[str]:
      # Combine multiple signals:
      # - Semantic similarity (embeddings)
      # - Keyword overlap (current method)
      # - Recency weighting
      # - Conversation thread continuity
      # - User interaction patterns
  ```

#### 2. Memory Storage Enhancement
- Add embedding storage to MemoryRecord
- Efficient similarity search (start simple, consider vector DB later)
- Configurable relevance thresholds

#### 3. Better Heuristic Scoring
- Implement comprehensive base persistence scoring
- Consider cultural inclusivity and edge cases
- Pattern-based importance detection

### 🔬 Technical Research Needed

#### Embedding Options Evaluation
1. **OpenAI Embeddings API** (text-embedding-ada-002)
   - Pros: High quality, consistent with LLM provider
   - Cons: API costs, requires internet
   
2. **Local Sentence Transformers** (all-MiniLM-L6-v2)
   - Pros: Free, offline, fast
   - Cons: Larger memory footprint, model download

3. **Hybrid Approach**
   - Use local embeddings as default
   - Fall back to API embeddings for enhanced accuracy
   - Cache embeddings for efficiency

#### Implementation Strategy
1. Start with OpenAI embeddings (align with existing LLM integration)
2. Add local embedding option for offline use
3. Implement smart caching and batch processing
4. Add configuration options for different use cases