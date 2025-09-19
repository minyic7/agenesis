# Memory Module Design

## Human-Like Attention & Memory System

### Core Memory Types (by Attention Priority)

#### `ImmediateMemory` - Current Focus
- Single current message being processed
- Most relevant/important information RIGHT NOW
- Highest attention priority
- Cleared after each interaction

#### `WorkingMemory` - Recent Context  
- Current session's conversation history
- Recent relevant information
- Medium attention priority
- Some information may become irrelevant over time

#### `EpisodicMemory` - Past Experiences (Future)
- Specific past interactions/experiences
- Lower attention priority
- Retrieved only when highly relevant to current task

#### `SemanticMemory` - Learned Knowledge (Future)
- Abstract facts, patterns, skills learned
- Background knowledge
- Retrieved when conceptually relevant

### Attention-Based Retrieval System

#### `AttentionManager`
- **Relevance Scoring**: Rate information importance to current task
- **Context Filtering**: Pick most helpful info from each memory type
- **Attention Allocation**: Focus cognitive resources on high-value memories
- **Noise Reduction**: Filter out irrelevant information

#### Information Flow
```
Current Input → Relevance Analysis → 
  ↓
Priority 1: ImmediateMemory (current focus)
Priority 2: WorkingMemory (recent context - filtered)  
Priority 3: EpisodicMemory (relevant past - searched)
Priority 4: SemanticMemory (applicable knowledge - queried)
  ↓
Combined Relevant Context → Cognition
```

### Core Classes

#### `BaseMemory`
- `store(perception_result) -> memory_id`
- `retrieve(memory_id) -> PerceptionResult`
- `search(query) -> List[PerceptionResult]`

#### `MemoryStore`
- Unified interface for all memory types
- Route to appropriate memory type
- Handle cross-memory queries

### Storage Format

#### `MemoryRecord`
- `id`: Unique identifier
- `perception_result`: Original PerceptionResult
- `timestamp`: When stored
- `context`: Session/conversation context
- `metadata`: Storage-specific info

## Implementation Progress

### ✅ Phase 1: Basic Storage
- [x] BaseMemory interface
- [x] ImmediateMemory for current message focus
- [x] WorkingMemory implementation
- [x] MemoryRecord format
- [x] Simple storage/retrieval
- [x] FileMemory and SQLiteMemory for persistence

### 🔄 Phase 2: Attention-Based Retrieval
- [ ] AttentionManager for relevance scoring
- [ ] Context filtering from WorkingMemory
- [ ] Relevance-based memory prioritization
- [ ] Combined context assembly

### 📋 Phase 3: Advanced Memory Types (Future)
- [ ] EpisodicMemory with search
- [ ] SemanticMemory with pattern recognition
- [ ] Cross-memory relevance analysis
- [ ] Memory consolidation and forgetting