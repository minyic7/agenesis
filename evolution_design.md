# Evolution Module Design

## Overview

The Evolution module enables agents to learn and improve through experience analysis. It processes interactions to extract valuable patterns and decides what knowledge should persist for future use.

## Core Architecture

### Learning Pipeline
```
Input: WorkingMemory (session history) + Agent Response + Context
Process: Value Analysis → Binary Learning Decision → Knowledge Enhancement
Output: EvolutionDecision (should_persist + learning metadata)
```

### Key Classes

#### `EvolutionDecision`
```python
@dataclass
class EvolutionDecision:
    should_persist: bool                      # Whether session contains valuable learning
    learning_type: Optional[str] = None       # pattern, preference, knowledge, skill
    learning_description: Optional[str] = None # What was learned
    future_application: Optional[str] = None  # How this helps future interactions
    rejection_reason: Optional[str] = None    # Why learning was rejected
```

#### `EvolvedKnowledge`
```python
@dataclass
class EvolvedKnowledge:
    knowledge_summary: str                    # Brief description of learning
    learning_context: str                     # Context where learning occurred
    future_relevance: str                     # How this applies to future situations
```

#### `EvolutionAnalyzer`
- **LLM-Enhanced Analysis**: Intelligent session evaluation for learning value
- **Binary Decision Making**: Clear yes/no decisions instead of complex scoring
- **Persona-Aware Learning**: Adapts analysis based on agent personality
- **Memory Enhancement**: Upgrades valuable interactions to evolved knowledge

## Implementation Status

### ✅ Completed Features

#### **Binary Learning Decisions**
- **Clear Analysis**: Simple yes/no for whether sessions contain valuable learning
- **LLM Evaluation**: Structured prompts assess learning value objectively
- **Context-Aware**: Considers full interaction context including agent responses
- **Rejection Reasoning**: Provides clear explanations when learning is rejected

#### **Persona Integration**
- **Context-Aware Analysis**: Evolution decisions consider agent personality
- **Personalized Learning**: Different agents focus on different types of learning
- **Behavioral Adaptation**: Learning reinforces agent-specific behaviors

#### **Memory Enhancement System**
- **Evolved Knowledge Creation**: Converts regular memories to enhanced knowledge
- **Metadata Enrichment**: Adds learning context and future relevance information
- **Cross-Session Persistence**: Enhanced memories survive across all agent sessions

#### **Configuration Management**
All magic numbers eliminated with named constants:

**EvolutionConfig:**
```python
ANALYSIS_TEMPERATURE = 0.1     # Low temperature for consistent analysis
MAX_ANALYSIS_TOKENS = 300      # Token limit for analysis responses
RECENT_RECORDS_LIMIT = 5       # Number of recent records to analyze
TRUNCATION_LENGTH = 50         # Length for string truncation in logs
```

## Current Capabilities

### Learning Value Assessment

**Valuable Learning Indicators:**
- **Pattern Recognition**: Recurring user preferences or behavior patterns
- **Knowledge Acquisition**: New information that enhances future responses
- **Skill Development**: Improved approaches to common tasks
- **Preference Learning**: User-specific preferences and communication styles

**Learning Rejection Criteria:**
- **Casual Conversation**: Social interactions without learning value
- **One-off Questions**: Isolated queries with no broader application
- **Error Interactions**: Failed responses that don't provide insight
- **Repetitive Content**: Already-known information without new insights

### Session Analysis Process

#### **Context Assembly**
```python
# Comprehensive session context for analysis
session_context = {
    'user_inputs': [...],           # All user messages in session
    'agent_responses': [...],       # All agent responses
    'interaction_flow': {...},      # Conversation progression
    'persona_context': {...}        # Agent personality information
}
```

#### **LLM-Enhanced Evaluation**
The evolution module uses structured LLM prompts to evaluate learning value:

```python
analysis_prompt = f"""
Analyze this interaction session for valuable learning:

Session Summary: {session_context}
Agent Persona: {persona_context}

Determine if this session contains valuable learning for future interactions.

Consider:
1. Does this session reveal user preferences or patterns?
2. Is there new knowledge that would improve future responses?
3. Are there behavioral insights worth remembering?

Respond with JSON only:
{{
    "should_persist": true|false,
    "learning_type": "pattern|preference|knowledge|skill",
    "learning_description": "what was learned",
    "future_application": "how this helps future interactions",
    "rejection_reason": "why rejected (if should_persist=false)"
}}
"""
```

### Integration with Agent Pipeline

#### **Evolution Analysis Trigger**
Evolution analysis occurs when:
- **Persistent Memory Available**: Only for named agents with long-term storage
- **Session Completion**: After meaningful interaction sequences
- **Learning Opportunities**: When cognition identifies potentially valuable interactions

#### **Memory Enhancement Process**
When learning is detected:
1. **Session Analysis**: Evaluate interaction sequence for learning value
2. **Knowledge Extraction**: Identify specific learnings and their applications
3. **Memory Enhancement**: Upgrade working memory records to evolved knowledge
4. **Persistent Storage**: Store enhanced memories with learning metadata

## Learning Categories

### Pattern Learning
- **User Behavior**: Recurring interaction patterns and preferences
- **Communication Style**: How users prefer to receive information
- **Topic Preferences**: Subject areas of particular interest
- **Context Patterns**: Situational factors that influence interactions

### Knowledge Learning
- **Domain Information**: New facts or information relevant to user context
- **Process Knowledge**: Step-by-step procedures and methodologies
- **Relationship Knowledge**: Connections between concepts and ideas
- **Historical Context**: Background information that informs future decisions

### Preference Learning
- **Response Style**: Preferred tone, length, and format of responses
- **Information Depth**: Desired level of detail and explanation
- **Interaction Pace**: Preferred speed and frequency of interactions
- **Tool Preferences**: Favored approaches to solving specific problems

## Performance Characteristics

### Analysis Speed
- **LLM Mode**: ~1-2 seconds for comprehensive session evaluation
- **Heuristic Mode**: <100ms for pattern-based decisions
- **Memory Enhancement**: ~200ms for evolved knowledge creation

### Learning Accuracy
- **Valuable Learning Detection**: High precision in identifying genuinely useful patterns
- **False Positive Reduction**: Effective filtering of casual conversation
- **Context Sensitivity**: Appropriate learning decisions based on agent persona

## Design Principles

1. **Binary Clarity**: Clear yes/no decisions for learning value
2. **Context Awareness**: Full interaction history considered in analysis
3. **Persona Integration**: Learning decisions align with agent personality
4. **Future-Focused**: Learning evaluated based on future utility
5. **No Magic Numbers**: All configuration values explicitly named

## Integration Points

### Cognition Module
- **Learning Triggers**: Receives persistence decisions that indicate learning opportunities
- **Context Enhancement**: Evolved knowledge improves future cognition quality
- **Pattern Recognition**: Historical patterns inform current interaction analysis

### Memory Systems
- **Memory Enhancement**: Upgrades regular memories to evolved knowledge status
- **Cross-Session Learning**: Maintains learned patterns across agent sessions
- **Context Retrieval**: Enhanced memories provide richer context for future interactions

### Agent Orchestration
- **Profile-Specific Learning**: Each agent profile maintains separate learning history
- **Session Management**: Coordinates learning across interaction sequences
- **Persona Integration**: Learning behavior adapts to agent personality

## Future Enhancements

### Advanced Learning Capabilities
- **Multi-Session Pattern Analysis**: Learning from patterns across multiple sessions
- **Collaborative Learning**: Sharing insights across different agent instances
- **Adaptive Learning**: Adjusting learning strategies based on success rates
- **Meta-Learning**: Learning about what types of learning are most valuable

### Enhanced Analysis
- **Outcome Tracking**: Measuring effectiveness of learned patterns
- **Learning Validation**: Confirming learning value through subsequent interactions
- **Pattern Refinement**: Improving learned patterns based on new evidence
- **Context Generalization**: Applying learned patterns to new situations

### Integration Opportunities
- **User Feedback**: Incorporating explicit feedback about learning quality
- **External Knowledge**: Integration with documentation and knowledge bases
- **Team Learning**: Sharing effective patterns across team agent instances