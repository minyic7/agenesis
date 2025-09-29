# Cognition Module Design

## Overview

The Cognition module processes user input to understand intent, context, and determine whether interactions should persist in long-term memory. It forms the core intelligence layer between perception and action generation.

## Core Architecture

### Processing Pipeline
```
Input: ImmediateMemory (current focus) + WorkingMemory (recent context)
Process: Intent Recognition + Context Awareness + Binary Persistence Decision
Output: CognitionResult (intent + context + should_persist decision)
```

### Key Classes

#### `CognitionResult`
```python
@dataclass
class CognitionResult:
    intent: str                    # question, request, statement, conversation
    context_type: str             # new, continuation, clarification, related
    should_persist: bool          # whether to store in persistent memory
    summary: str                  # brief description of user intent
    relevant_memories: List[str]  # IDs of relevant memory records
    reasoning: str               # explanation of the analysis
    timestamp: Optional[datetime] = None
    memory_context: Optional[Dict[str, Any]] = None  # organized memory content
```

#### `BasicCognition`
- **LLM-Enhanced Mode**: Uses structured JSON prompts for intelligent analysis
- **Heuristic Fallback**: Pattern-based classification when LLM unavailable
- **Binary Decision Logic**: Simple yes/no for persistence instead of scoring

#### `SemanticCognition` (Enhanced)
- **Embedding-Based Search**: Uses OpenAI embeddings for semantic memory retrieval
- **Intelligent Caching**: 2x performance improvement with vector index caching
- **Memory Organization**: Structured context with working/persistent memory separation

## Implementation Status

### ✅ Completed Features

#### **Binary Persistence System**
- Replaced float scoring (0.0-1.0) with clear boolean decisions
- Simplified LLM prompts: "should_persist: true|false"
- Better decision clarity: store important interactions, skip casual conversation
- Eliminates magic number thresholds

#### **Semantic Search Implementation**
- **OpenAI Embeddings Integration**: text-embedding-3-small (1536 dimensions)
- **Vector Similarity Search**: SQLite-VSS for efficient similarity queries
- **Performance Optimization**: Intelligent caching reduces query time by ~50%
- **Fallback Strategy**: Graceful degradation to keyword matching

#### **Configuration Management**
All magic numbers eliminated with named constants:

**BasicCognitionConfig:**
```python
LLM_TEMPERATURE = 0.1        # Low temperature for consistent responses
LLM_MAX_TOKENS = 300         # Token limit for cognition responses
SUMMARY_MAX_LENGTH = 100     # Maximum length for summary text
```

**SemanticConfig:**
```python
WORKING_MEMORY_SEARCH_LIMIT = 1000    # Records to search in working memory
PERSISTENT_MEMORY_SEARCH_LIMIT = 1000  # Records to search in persistent memory
VECTOR_SEARCH_LIMIT = 10000            # High limit for database vector search
MAX_MEMORY_CONTEXT_ITEMS = 8           # Max relevant memories to organize
RECENT_CONTEXT_BOOST = 1.2             # 20% boost for recent working memory
DEFAULT_SIMILARITY_SCORE = 0.5         # Default when vector search unavailable
```

#### **Memory Context Organization**
```python
memory_context = {
    'focus': ['current user input'],           # Immediate focus
    'working': ['recent', 'conversation'],     # Session context
    'persistent': ['relevant', 'history'],    # Long-term relevant memories
    'has_memories': True                       # Whether context exists
}
```

## Current Capabilities

### Intent Classification
- **Question**: User seeking information
- **Request**: User asking for help/action
- **Statement**: User sharing information
- **Conversation**: Social/casual interaction

### Context Detection
- **New**: Fresh conversation thread
- **Continuation**: Following previous messages
- **Clarification**: Asking about recent context
- **Related**: Connected but new angle

### Persistence Logic

**Store in Persistent Memory (should_persist=True):**
- Important questions with learning value
- Explicit user preferences or requirements
- Meaningful statements with future relevance
- Actionable requests worth remembering

**Skip Persistent Storage (should_persist=False):**
- Casual greetings and social conversation
- One-off questions with no context
- Confirmations and acknowledgments
- Temporary clarifications

## Integration Points

### Memory Systems
- **ImmediateMemory**: Current user input being processed
- **WorkingMemory**: Recent conversation context (keyword + semantic search)
- **PersistentMemory**: Long-term knowledge (semantic search with vector indexing)

### LLM Providers
- **Primary**: OpenAI and Anthropic with structured JSON prompts
- **Fallback**: Pattern-based heuristics for offline operation
- **Error Handling**: Graceful degradation with partial parsing

### Embedding Provider
- **OpenAI text-embedding-3-small**: 1536-dimensional embeddings
- **Batch Processing**: Efficient embedding generation for multiple texts
- **Similarity Search**: Cosine similarity with configurable thresholds

## Performance Characteristics

### Semantic Search Performance
- **Vector Index Caching**: ~2x faster queries for repeated searches
- **Memory Limits**: Configurable search limits prevent performance degradation
- **Hybrid Retrieval**: Combines semantic similarity with recency weighting

### Processing Speed
- **LLM Mode**: ~1-2 seconds for OpenAI/Anthropic analysis
- **Heuristic Mode**: <100ms for pattern-based classification
- **Memory Retrieval**: <200ms for semantic search with caching

## Design Principles

1. **Binary Decisions**: Clear yes/no choices instead of ambiguous scoring
2. **Graceful Degradation**: Always functional, even without LLM/embeddings
3. **Memory Efficiency**: Intelligent caching and configurable limits
4. **Context Awareness**: Rich memory integration for informed decisions
5. **No Magic Numbers**: All configuration values explicitly named and documented

## Future Enhancements

### Potential Improvements
- **Multi-modal Input**: Support for images, audio, files
- **Conversation Threading**: Track topics across multiple turns
- **User Adaptation**: Learn individual user patterns and preferences
- **Advanced Reasoning**: Multi-step logic and goal decomposition
- **Performance Optimization**: Further caching and batching improvements

### Integration Opportunities
- **Persona System**: Context-aware cognition based on agent personality
- **Evolution Feedback**: Learn from interaction outcomes to improve decisions
- **External Knowledge**: Integration with documentation, APIs, and databases