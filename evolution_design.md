# Evolution Module Design

## Purpose
The Evolution module enables the agent to learn, adapt, and improve over time through experience. It processes past interactions to extract lessons, identify patterns, and reinforce successful behaviors.

## Core Questions & Definitions

### What Does "Evolution" Mean for an AI Agent?

**Agent Evolution Definition**: Learning from past experiences to extract patterns, confirmed truths, and valuable information to prepare the agent for future usage.

**Core Evolution Process**:
1. **Experience Analysis**: Mining past interactions for valuable insights
2. **Pattern Extraction**: Identifying recurring themes, successful approaches, and failure modes
3. **Truth Confirmation**: Validating which insights are reliable across contexts
4. **Knowledge Preparation**: Organizing learned information for future application
5. **Proactive Readiness**: Using extracted knowledge to improve future responses

### How Do We Measure Evolution?

**Pattern Extraction Success:**
- Number of reliable patterns identified
- Consistency of patterns across interactions
- Accuracy of pattern predictions for new situations

**Truth Confirmation Quality:**
- Validation rate of extracted insights
- Cross-context reliability of learned truths
- Reduction in contradictory or false patterns

**Knowledge Application:**
- Successful application of past lessons to new situations
- Improved relevance of retrieved experiences
- Better preparedness for similar future contexts

**Observable Improvements:**
- More appropriate responses in familiar contexts
- Better anticipation of user needs based on history
- Reduced repetition of confirmed ineffective approaches
- Improved context matching using past experience patterns

## Evolution Data Sources

### **Past Experiences (Primary Source):**
1. **Complete Interaction Records**: User input → cognition → action → outcome
2. **Memory Usage Patterns**: Which memories were relevant and useful
3. **Successful Response Patterns**: What worked in specific contexts
4. **Failed Approaches**: What didn't work and should be avoided
5. **Context Transitions**: How conversations flow and evolve

### **Pattern Extraction Targets:**
1. **User Intent Patterns**: Recurring ways users express similar needs
2. **Context Clues**: Signals that indicate specific response approaches
3. **Successful Memory Retrieval**: Which past experiences prove most relevant
4. **Response Effectiveness**: What types of responses work in different situations
5. **Conversation Flow**: How interactions typically progress

### **Truth Confirmation Sources:**
1. **Repeated Validations**: Patterns that work consistently across multiple interactions
2. **Cross-Context Testing**: Truths that hold across different conversation types
3. **Long-term Outcomes**: Approaches that prove valuable over extended sessions

## Current Architecture Analysis

### Existing File Structure:
```
evolution/
├── experience_processor.py  # "extract lessons from experiences"
├── pattern_recognition.py   # "behavioral pattern identification"  
└── trait_reinforcement.py   # "strengthen/weaken traits based on outcomes"
```

### Critical Analysis:

**✅ What Works:**
- `experience_processor.py`: Good for analyzing past interactions
- `pattern_recognition.py`: Essential for identifying successful strategies
- `trait_reinforcement.py`: Important for strengthening/weakening behaviors

**❌ What's Missing:**
- **Skill Tracking**: How does the agent know what it's good/bad at?
- **Learning Integration**: How do lessons get applied to future interactions?
- **Evolution History**: Tracking what has changed and when
- **Performance Measurement**: Objective assessment of improvements
- **Adaptation Mechanisms**: How changes get implemented

**🤔 Potential Issues:**
- No clear connection to existing pipeline (perception/memory/cognition/action)
- No feedback loop mechanism
- Missing performance baseline and measurement
- Unclear how "traits" map to actual behaviors

## Simple Evolution Architecture 

**Focus**: Use LLM analysis to identify valuable learning from current session that should be persisted for future improvement.

### Core Concept:

**Learning-Enhanced Persistence**: Instead of complex pattern extraction, use an LLM call to analyze the current working memory + immediate focus to identify if there's valuable knowledge that would make the agent better in future interactions.

### Core Components:

#### **BaseEvolution** (abstract interface)
```python
class BaseEvolution(ABC):
    """Abstract base class for evolution analysis"""
    
    @abstractmethod
    async def analyze_for_learning(self, data_source: Any) -> EvolutionDecision:
        """Analyze data source for learning opportunities"""
        pass
```

#### **EvolutionAnalyzer** (main implementation)
```python
class EvolutionAnalyzer:
    """Main evolution analyzer supporting multiple data sources"""
    
    async def analyze_memory_session(self, immediate_memory, working_memory) -> EvolutionDecision
    async def analyze_feedback_data(self, feedback: FeedbackData) -> EvolutionDecision  # Future
    def create_evolved_knowledge_metadata(self, decision: EvolutionDecision) -> EvolvedKnowledge
    def should_trigger_analysis(self, trigger_type: str, context: Dict) -> bool
```

#### **MemoryEvolutionSource** (memory-specific analysis)
```python
class MemoryEvolutionSource:
    """Handles memory-specific evolution analysis"""
    
    async def analyze_session_content(self, immediate_memory, working_memory) -> EvolutionDecision
    def extract_session_summary(self, memories: List[MemoryRecord]) -> str
    def build_analysis_prompt(self, session_content: str) -> str
```

### Process Flow:
1. **Session Analysis**: LLM analyzes working memory + current focus
2. **Learning Detection**: Identifies if session contains valuable future knowledge
3. **Enhanced Persistence**: Mark valuable insights as "evolved knowledge" 
4. **Retrieval Priority**: Evolved knowledge gets higher reliability during retrieval

### Project Knowledge Integration
The evolution system supports both conversational learning and imported project knowledge, treating both as enhanced memories with different learning contexts.

**Dual Learning Sources:**
1. **Conversational Learning**: Real-time analysis of user interactions for preferences and patterns
2. **Project Knowledge**: Pre-imported documentation and context marked as evolved knowledge

**Enhanced Memory Classification:**
```python
# Conversation-derived evolved knowledge
conversation_memory = MemoryRecord(
    context={'source_type': 'conversation', 'learning_type': 'preference'},
    is_evolved_knowledge=True,
    evolution_metadata={'learning_context': 'user_preference'},
    reliability_multiplier=1.3  # Learned from interaction
)

# Project knowledge imported as evolved memory
project_memory = MemoryRecord(
    context={'source_type': 'project_knowledge', 'document_type': 'requirements'},
    is_evolved_knowledge=True,
    evolution_metadata={'learning_context': 'project_documentation'},
    reliability_multiplier=1.5  # Pre-verified important knowledge
)
```

**Unified Evolution Approach:**
- **Same storage format**: Both types use MemoryRecord with evolution metadata
- **Same retrieval boost**: reliability_multiplier enhances both conversation learning and project docs
- **Complementary learning**: Project knowledge provides context, conversations provide personalization
- **Consistent behavior**: Cognition module treats both as high-priority memories

### Data Structures:

#### **EvolutionDecision**
```python
@dataclass
class EvolutionDecision:
    should_persist: bool  # Whether this session contains valuable learning (default: False)
    learning_type: Optional[str] = None  # pattern, preference, knowledge, skill
    learning_description: Optional[str] = None  # What was learned
    confidence: float = 0.0  # How confident the LLM is about this learning
    future_application: Optional[str] = None  # How this could help in future interactions
    rejection_reason: Optional[str] = None  # Why learning was rejected (if should_persist=False)
```

#### **EvolvedKnowledge**
```python
@dataclass
class EvolvedKnowledge:
    knowledge_summary: str  # What valuable knowledge was identified
    learning_context: str  # The context in which this was learned
    future_relevance: str  # When this knowledge would be useful
    reliability_boost: float  # How much this increases retrieval priority (e.g., 1.5x)
    evolved_at: datetime  # When this knowledge was identified
```

#### **Enhanced MemoryRecord** (extends existing)
```python
# Add to existing MemoryRecord structure:
@dataclass 
class MemoryRecord:
    # ... existing fields ...
    is_evolved_knowledge: bool = False
    evolution_metadata: Optional[EvolvedKnowledge] = None
    reliability_multiplier: float = 1.0  # Higher for evolved knowledge
```

## Evolution Module Architecture (Independent)

### Separate Module Design:
Evolution is **decoupled** from memory and other modules to allow future expansion to multiple learning sources.

```
📁 agenesis/evolution/
   📄 __init__.py          # Module exports
   📄 base.py              # Abstract interfaces
   📄 analyzer.py          # Main EvolutionAnalyzer class
   📄 memory_source.py     # Memory-specific analysis (current implementation)
```

### Future Evolution Sources (Not Yet Implemented):
- External feedback analysis
- Performance metrics analysis  
- Cross-session pattern analysis
- Multi-agent learning
- User behavior analytics

## Integration with Existing Pipeline

### Simple Evolution Flow:
```
1. User Interaction → Complete Pipeline (perception/memory/cognition/action)
2. [At session boundaries or key moments]
3. EvolutionAnalyzer → Analyze memory session data
4. LLM Call → "Does this session contain valuable learning for future?"
5. If Yes → Generate EvolutionDecision with learning metadata
6. Agent → Pass evolution decision to memory for enhanced storage
7. Memory → Store with evolved knowledge markings
8. Future Retrieval → Evolved knowledge gets priority in memory searches
```

### When Evolution Analysis Triggers:
- **End of Session**: Before clearing working memory
- **Significant Interactions**: After high-confidence cognition results
- **User-Indicated Learning**: When users explicitly share important information
- **Pattern Detection**: When similar contexts repeat (optional)

### LLM Analysis Prompt (Example):
```
Analyze this conversation session for valuable learning:

Working Memory Context: [recent interactions]
Current Focus: [immediate memory content]

Critical Questions:
1. Does this session contain genuinely valuable patterns, preferences, or knowledge that would help the agent perform better in future similar situations?
2. Is this information specific and actionable enough to warrant long-term storage?
3. Would storing this information improve future interactions, or would it just add noise?

IMPORTANT: Be selective. Most casual conversations should NOT be marked for evolution learning. 
Only identify truly valuable insights like:
- Clear user preferences or requirements
- Successful problem-solving approaches
- Important context about user's work/interests
- Patterns that consistently work well

AVOID learning from:
- Casual small talk or greetings
- One-off questions without broader application
- Generic information easily available elsewhere
- Temporary context that won't be relevant later

Respond with structured analysis, and default to NO LEARNING unless there's clear value.
```

### Memory Enhancement:
- **Existing Memory Retrieval**: Enhanced with reliability multipliers
- **Evolved Knowledge Priority**: Higher weight in similarity calculations  
- **Learning Accumulation**: Evolved knowledge builds over time
- **Context Awareness**: Better matching of relevant past learning

## Simple Evolution Benefits

### 1. **Learning Accumulation**
- **Valuable Knowledge**: Important insights get preserved with higher reliability
- **Pattern Recognition**: LLM identifies recurring successful approaches
- **Context Wisdom**: Better understanding of when specific approaches work

### 2. **Enhanced Memory Retrieval**
- **Priority Boost**: Evolved knowledge gets higher relevance in searches
- **Reliability Scoring**: More trustworthy information is weighted higher
- **Learning Build-up**: Knowledge compounds over multiple sessions

### 3. **Future-Focused Learning**
- **Proactive Insight**: LLM identifies what will be useful later
- **Actionable Knowledge**: Focus on insights that improve future interactions
- **Practical Evolution**: Real improvement in agent capabilities

## Implementation Advantages

### 1. **Leverages Existing Infrastructure**
- **Uses Current Pipeline**: Builds on working memory + persistent memory
- **LLM Integration**: Uses existing LLM providers (no new dependencies)
- **Simple Extension**: Minimal changes to existing architecture

### 2. **LLM-Powered Intelligence**
- **Smart Analysis**: LLM can identify subtle patterns humans might miss
- **Context Understanding**: Better assessment of what's truly valuable
- **Natural Language**: Analysis in human-understandable terms

### 3. **Incremental Implementation**
- **Start Simple**: Begin with basic evolved knowledge marking
- **Gradual Enhancement**: Add retrieval prioritization later
- **Measurable Impact**: Easy to see if evolved knowledge helps

### 4. **Selective Learning (Critical)**
- **Default to NO**: Most interactions should NOT result in evolved knowledge
- **Quality over Quantity**: Better to learn fewer, high-value insights
- **Noise Prevention**: Avoid cluttering persistent memory with trivial information
- **LLM Filtering**: Let LLM be the intelligent filter for what's worth learning

## Success Criteria

### Phase 1: Basic Evolution (First Implementation):
- [ ] EvolutionAnalyzer can analyze working memory + immediate focus
- [ ] LLM successfully identifies valuable learning from sessions
- [ ] Memory records can be marked as "evolved knowledge"
- [ ] Enhanced MemoryRecord structure with reliability multipliers

### Phase 2: Enhanced Persistence:
- [ ] Evolved knowledge gets stored in persistent memory with special marking
- [ ] Memory retrieval prioritizes evolved knowledge appropriately
- [ ] Agent demonstrates improved responses using past learning
- [ ] Clear examples of knowledge accumulation over multiple sessions

### Phase 3: Measurable Learning:
- [ ] Observable improvement in similar context handling
- [ ] Reduced repetition of unsuccessful approaches
- [ ] Better anticipation of user needs based on evolved knowledge
- [ ] User-visible improvements in interaction quality

## Implementation Plan (Simplified)

### Immediate Next Steps:
1. **Enhance MemoryRecord**: Add evolved knowledge fields
2. **Create EvolutionAnalyzer**: Simple class with LLM analysis method
3. **Integration Point**: Add evolution analysis to agent session flow
4. **Basic Testing**: Verify LLM can identify valuable learning

### Agent Integration Example:
```python
class Agent:
    def __init__(self, profile: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        # ... existing initialization ...
        self.evolution = EvolutionAnalyzer(self.config.get('evolution', {}))
    
    async def process_input(self, text_input: str) -> str:
        # ... existing pipeline: perception → memory → cognition → action ...
        
        # Evolution analysis (decoupled)
        if self.has_persistent_memory:
            evolution_decision = await self.evolution.analyze_memory_session(
                self.immediate_memory, self.working_memory
            )
            
            # Default expectation: should_persist = False for most interactions
            if evolution_decision.should_persist:
                print(f"Learning detected: {evolution_decision.learning_description}")
                
                # Create evolved knowledge metadata
                evolved_metadata = self.evolution.create_evolved_knowledge_metadata(evolution_decision)
                
                # Pass to memory for enhanced storage (memory handles the details)
                self.memory.store_evolved_knowledge(
                    memories=self.working_memory.get_recent(3),
                    evolution_metadata=evolved_metadata
                )
            else:
                # Most common case - no learning needed
                if evolution_decision.rejection_reason:
                    print(f"No learning: {evolution_decision.rejection_reason}")
        
        return response
```

### Key Principle: **Selective Learning**
- **90%+ of interactions**: `should_persist = False` (casual conversation, simple Q&A)
- **<10% of interactions**: `should_persist = True` (valuable insights, preferences, patterns)
- **Quality over Quantity**: Better to miss some learning than pollute with noise

This approach is **simple, practical, and builds on existing infrastructure** while providing real learning capabilities!