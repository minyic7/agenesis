# Memory Module Design

## Overview

The Memory module implements a human-like attention and memory system with multiple storage types optimized for different access patterns and persistence requirements.

## Core Memory Architecture

### Memory Types by Priority

#### `ImmediateMemory` - Current Focus
- **Purpose**: Single current message being processed
- **Scope**: Highest attention priority, current interaction only
- **Lifecycle**: Cleared after each interaction
- **Storage**: In-memory only

#### `WorkingMemory` - Session Context
- **Purpose**: Recent conversation history and session state
- **Scope**: Current session's interactions for context continuity
- **Lifecycle**: Maintained during session, configurable capacity limits
- **Storage**: In-memory with configurable persistence

#### `PersistentMemory` - Long-term Knowledge
- **Purpose**: Important interactions and learned knowledge across sessions
- **Scope**: Cross-session knowledge and user relationship building
- **Lifecycle**: Permanent storage with selective retention
- **Storage**: SQLite database with vector search capabilities

## Implementation Status

### ✅ Completed Features

#### **Memory Record Structure**
```python
@dataclass
class MemoryRecord:
    id: str                                    # Unique identifier
    perception_result: PerceptionResult        # Original input data
    stored_at: datetime                        # Storage timestamp
    context: Dict[str, Any]                   # Session/conversation context
    metadata: Dict[str, Any]                  # Storage-specific information

    # Evolution support
    is_evolved_knowledge: bool = False         # Marks learned/enhanced memories
    evolution_metadata: Optional[Dict[str, Any]] = None

    # Semantic search support
    embedding: Optional[List[float]] = None    # OpenAI embedding vectors

    # Full interaction context
    agent_response: Optional[str] = None       # Agent's response for context
```

#### **Storage Backend Implementations**

**SQLiteMemory (Universal):**
- **Format**: SQLite database with optimized schema and vector search
- **Use Case**: Universal storage for all use cases - production, development, semantic search
- **Features**: Vector search, indexing, efficient queries, ACID compliance, intelligent caching
- **Benefits**: No configuration required, full semantic search capabilities out of the box, simplified architecture

#### **Semantic Search Integration**
- **Embedding Provider**: OpenAI text-embedding-3-small (1536 dimensions)
- **Vector Storage**: SQLite-VSS extension for efficient similarity search
- **Search Strategy**: Cosine similarity with configurable thresholds
- **Performance Optimization**: Intelligent caching reduces query time by ~50%

#### **Configuration Management**
All magic numbers eliminated with named constants:

**WorkingMemoryConfig:**
```python
DEFAULT_MAX_CAPACITY = 100  # Maximum records in working memory
```

## Storage Architecture

### Profile-Based Organization
```
~/.agenesis/
└── profiles/
    ├── project_alpha/
    │   └── memory.db          # SQLite storage (default)
    └── personal_assistant/
        └── memory.db          # SQLite storage (default)
```


**Benefits:**
- **User-scoped data**: Each user maintains separate agent profiles
- **Cross-project persistence**: Agent knowledge survives project changes
- **Profile isolation**: Multiple agents don't interfere with each other
- **Standard conventions**: Follows Unix home directory patterns

### Project Knowledge Integration

#### **Enhanced Memory Architecture for Long-term Work**
Project documentation and knowledge are stored as special types of persistent memories, leveraging existing infrastructure without requiring new storage systems.

**Project Knowledge as Evolved Memories:**
```python
project_memory = MemoryRecord(
    perception_result=doc_content,
    context={
        'source_type': 'project_knowledge',    # Distinguishes from conversations
        'document_type': 'requirements',       # Type classification
        'importance': 'high'                   # Priority for retrieval
    },
    is_evolved_knowledge=True,                # Marks as important knowledge
    evolution_metadata={                      # Learning context
        'knowledge_summary': 'API documentation',
        'learning_context': 'project_documentation',
        'future_relevance': 'API design decisions'
    }
)
```

**Integration Benefits:**
- **Unified Retrieval**: Cognition finds both project docs and conversation history seamlessly
- **Cross-session Persistence**: Project knowledge survives across all sessions
- **Evolution Compatible**: Can learn from conversations while maintaining project context
- **Profile Isolation**: Each project maintains separate knowledge base

## Current Capabilities

### Memory Storage Operations

**Store Operations:**
- **Immediate**: Direct storage with automatic ID generation
- **Working**: Capacity-managed storage with LRU eviction
- **Persistent**: Selective storage based on cognition decisions

**Retrieval Operations:**
- **By ID**: Direct record retrieval for specific memories
- **Recent Records**: Time-based retrieval with configurable limits
- **Semantic Search**: Vector similarity search with threshold filtering
- **Keyword Matching**: Fallback search for non-semantic deployments

### Project Knowledge Management

**Import Capabilities:**
```python
# Import from structured sources
await agent.import_project_knowledge([
    {'content': requirements_doc, 'type': 'requirements'},
    {'content': architecture_doc, 'type': 'architecture'},
    {'content': api_docs, 'type': 'documentation'}
])

# Import from files
await agent.import_from_files([
    'docs/requirements.md',
    {'path': 'docs/api.md', 'type': 'api_documentation'},
    {'path': 'ARCHITECTURE.md', 'importance': 'high'}
])
```

**Knowledge Types Supported:**
- **Requirements**: User stories, acceptance criteria, constraints
- **Architecture**: System design, patterns, technology decisions
- **Documentation**: API specs, coding standards, best practices
- **Domain Knowledge**: Business context, industry standards

### Vector Search Performance

**Optimization Features:**
- **Index Caching**: Intelligent caching of vector search indices
- **Batch Operations**: Efficient batch embedding generation
- **Similarity Thresholds**: Configurable relevance filtering
- **Fallback Strategy**: Graceful degradation to keyword search

**Performance Characteristics:**
- **Cache Hit**: <50ms for cached similarity searches
- **Cache Miss**: ~200ms for fresh vector searches
- **Batch Embedding**: ~100ms per text for OpenAI API
- **Keyword Fallback**: <10ms for simple text matching

## Integration Points

### Cognition Module
- **Memory Context**: Provides organized context structure to cognition
- **Relevant Memories**: Supplies semantically similar historical interactions
- **Cross-session Knowledge**: Enables learning from past sessions

### Evolution Module
- **Learning Storage**: Stores evolved knowledge as enhanced memories
- **Pattern Analysis**: Provides historical data for pattern extraction
- **Knowledge Enhancement**: Upgrades regular memories to evolved knowledge

### Agent Orchestration
- **Profile Management**: Handles agent-specific memory isolation
- **Project Context**: Maintains project-specific knowledge bases
- **Session Management**: Coordinates memory across interaction sessions

## Memory Lifecycle Management

### Working Memory Management
- **Capacity Control**: Automatic eviction when exceeding configured limits
- **LRU Strategy**: Least recently used records removed first
- **Session Persistence**: Optional persistence across session restarts

### Persistent Memory Management
- **Selective Storage**: Only stores interactions marked as important by cognition
- **Evolution Enhancement**: Upgrades memories when learning occurs
- **Cross-session Continuity**: Maintains knowledge across all agent sessions

### Embedding Management
- **Lazy Generation**: Embeddings generated on-demand for efficiency
- **Batch Processing**: Efficient batch operations for multiple records
- **Cache Invalidation**: Smart cache management for optimal performance

## Design Principles

1. **Attention-Based Access**: Higher priority memories get faster, more frequent access
2. **Profile Isolation**: Different agents maintain completely separate memory spaces
3. **Semantic Intelligence**: Vector search enables conceptual memory retrieval
4. **Graceful Degradation**: Always functional, even without embeddings or external APIs
5. **Project Integration**: Seamless incorporation of project knowledge into memory architecture

## Performance Optimization

### Caching Strategy
- **Vector Index Caching**: Reduces repeated search computation
- **Embedding Caching**: Avoids regenerating embeddings for known content
- **Query Result Caching**: Speeds up repeated similar queries

### Storage Efficiency
- **Selective Persistence**: Only important interactions stored long-term
- **Compression**: Efficient storage of large embedding vectors
- **Index Optimization**: Database indices for common query patterns

## Future Enhancements

### Advanced Memory Types
- **Episodic Memory**: Specific interaction sequences and experiences
- **Semantic Memory**: Abstract knowledge and learned concepts
- **Procedural Memory**: Learned skills and behavior patterns

### Enhanced Search Capabilities
- **Multi-modal Search**: Support for images, audio, and other media types
- **Temporal Search**: Time-aware memory retrieval and relevance
- **Cross-Memory Search**: Unified search across all memory types

### Performance Improvements
- **Distributed Storage**: Scale beyond single-machine limitations
- **Advanced Caching**: Multi-level caching strategies
- **Async Operations**: Non-blocking memory operations for better responsiveness