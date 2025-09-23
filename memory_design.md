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

### Storage Location Strategy

#### User Home Directory (`~/.agenesis/`)
The framework stores persistent memory in the user's home directory following Unix conventions:

```
~/.agenesis/
└── profiles/
    ├── user_profile_1/
    │   ├── records.jsonl      # File storage
    │   └── memory.db          # SQLite storage
    └── user_profile_2/
        └── records.jsonl
```

**Benefits of ~/.agenesis/ approach:**
- **User-scoped data**: Each user maintains their own agent profiles
- **Cross-project persistence**: Same agent profile works across different projects
- **Clean development**: No clutter in project directories
- **Standard practice**: Follows conventions like `.npm/`, `.pip/`, `.cache/`
- **Multi-project learning**: Agent knowledge persists and improves across codebases
- **Team collaboration**: No conflicts with team members' local agent states

**Agent Profile Isolation:**
- Each named agent gets its own profile directory
- Anonymous agents use no persistent storage
- Storage type (file vs SQLite) configurable per agent
- Profile data never conflicts between different agents

### Project Knowledge Integration

#### Leveraging Existing Memory Architecture for Project Work
The framework supports long-term project work by treating project documentation and knowledge as special types of persistent memories, leveraging the existing memory infrastructure without requiring new storage systems.

**Project Knowledge as Enhanced Memories:**
```python
# Project documents stored as MemoryRecord with special context
project_memory = MemoryRecord(
    perception_result=doc_content,           # Processed project document
    context={
        'source_type': 'project_knowledge',  # Distinguishes from conversations
        'document_type': 'requirements',     # requirements, architecture, etc.
        'importance': 'high'                 # Priority level for retrieval
    },
    is_evolved_knowledge=True,              # Marked as important knowledge
    reliability_multiplier=1.5              # Higher retrieval priority
)
```

**Integration Benefits:**
- **Unified retrieval**: Cognition module naturally finds both project docs and conversation history
- **Cross-session persistence**: Project knowledge survives across all sessions
- **Evolution compatibility**: Can still learn from conversations while maintaining project context
- **Profile isolation**: Each project profile maintains separate knowledge base
- **Storage flexibility**: Works with both file and SQLite storage backends

**Usage Pattern:**
```python
# 1. Create project-specific agent
agent = Agent(profile="ecommerce_rebuild", persona="technical_mentor")

# 2. Import project knowledge (optional)
await agent.import_project_knowledge([
    {'content': requirements_doc, 'type': 'requirements', 'boost': 1.5},
    {'content': architecture_doc, 'type': 'architecture', 'boost': 1.4}
])

# 3. Natural conversations with project context
response = await agent.process_input("How should we implement authentication?")
# → Agent retrieves relevant architecture docs + conversation history
```

**Project Knowledge Types:**
- **Requirements documents**: User stories, acceptance criteria, constraints
- **Architecture documentation**: System design, patterns, technology decisions  
- **Domain knowledge**: Business context, industry standards, best practices
- **Code documentation**: API specs, coding standards, existing patterns
- **Historical context**: Previous decisions, lessons learned, project evolution

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