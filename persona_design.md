# Persona System Design

## Overview
The persona system allows developers to customize agent personality, response style, and behavior patterns for their specific use cases. Personas act as a **cognitive framework** that influences how the agent thinks, focuses, and responds throughout the entire processing pipeline.

## Core Design Principles
1. **Static YAML Configuration** - Personas defined via YAML for predictable behavior
2. **Context Injection** - Persona creates context that flows through pipeline without tight coupling
3. **Optional Integration** - Modules can use persona context or ignore it entirely
4. **Predictable Behavior** - Same persona + same input = consistent behavior (no dynamic personas)
5. **Influence without Coupling** - Broad influence on agent behavior without module dependencies

## Final Architecture: Context Injection Pattern

### PersonaContext (Flow Object)
```python
@dataclass
class PersonaContext:
    """Context object that flows through the pipeline"""
    # Attention guidance
    focus_areas: List[str] = field(default_factory=list)
    priority_signals: List[str] = field(default_factory=list)
    
    # Memory preferences  
    relevance_boosts: Dict[str, float] = field(default_factory=dict)
    context_filters: List[str] = field(default_factory=list)
    
    # Thinking framework
    reasoning_approach: Optional[str] = None
    decision_criteria: List[str] = field(default_factory=list)
    
    # Content preferences
    response_structure: Optional[str] = None
    include_examples: bool = True
    detail_level: str = "normal"
    
    # System prompt additions
    system_additions: List[str] = field(default_factory=list)
```

### YAML Persona Template
```yaml
# personas/technical_mentor.yaml
name: "technical_mentor"
description: "Systematic problem-solving mentor for developers"

context_template:
  focus_areas: 
    - "error_analysis"
    - "performance_bottlenecks"
    - "architecture_patterns"
  
  priority_signals:
    - "error"
    - "slow" 
    - "scale"
    - "debug"
  
  relevance_boosts:
    technical_solutions: 1.5
    debugging_steps: 1.3
    best_practices: 1.4
  
  reasoning_approach: "systematic_problem_solving"
  
  decision_criteria:
    - "technical_correctness"
    - "maintainability"
    - "performance_impact"
  
  response_structure: "analysis_then_solution"
  include_examples: true
  detail_level: "comprehensive"
  
  system_additions:
    - "Always explain your reasoning step by step"
    - "Provide code examples when discussing technical concepts"
    - "Consider scalability and maintainability in your recommendations"
```

## Pipeline Integration

```python
class Agent:
    async def process_input(self, text_input: str) -> str:
        # 1. Generate persona context (or None if no persona)
        persona_context = self.persona.create_context(text_input) if self.persona else None
        
        # 2. Pass context through pipeline - modules use it optionally
        perception_result = self.perception.process(text_input, persona_context)
        
        # Store with context
        self.immediate_memory.store(perception_result, persona_context)
        self.working_memory.store(perception_result, persona_context)
        
        # Cognition with context
        cognition_result = await self.cognition.process(
            self.immediate_memory, self.working_memory, persona_context
        )
        
        # Action with context
        action_result = await self.action.generate_response(cognition_result, persona_context)
        
        return action_result.response_text
```

## Module Integration (Loosely Coupled)

### Optional Context Usage Pattern
```python
# Each module adds optional context parameter
class TextPerception:
    def process(self, text: str, context: Optional[PersonaContext] = None) -> PerceptionResult:
        features = self._extract_base_features(text)
        
        # Optional: Use persona context for focused extraction
        if context and context.focus_areas:
            focused_features = self._extract_focused_features(text, context.focus_areas)
            features.update(focused_features)
        
        return PerceptionResult(text=text, features=features)

class BasicCognition:
    async def process(self, immediate_memory, working_memory, context: Optional[PersonaContext] = None):
        base_prompt = self._build_base_prompt(immediate_memory, working_memory)
        
        # Optional: Add persona reasoning framework
        if context:
            if context.reasoning_approach:
                base_prompt += f"\nReasoning approach: {context.reasoning_approach}"
            if context.system_additions:
                base_prompt += "\n" + "\n".join(context.system_additions)
        
        return await self._process_with_llm(base_prompt)
```

**Impact on Modules:**
- **Perception**: Optional focused feature extraction
- **Memory**: Optional relevance boosting and context tagging
- **Cognition**: Optional reasoning framework and system prompt additions
- **Action**: Optional response structuring and content preferences
- **Evolution**: No changes needed

## Benefits

✅ **Influence without coupling** - Modules can ignore context completely  
✅ **Backwards compatible** - All context parameters are optional  
✅ **Predictable behavior** - Static YAML ensures consistent agent behavior  
✅ **Gradual adoption** - Implement context usage module by module  
✅ **Clean separation** - Persona logic stays in persona module  
✅ **Testable** - Easy to test with/without persona context  
✅ **Developer-friendly** - Simple YAML configuration for library users  

## Actual Implementation Structure

```
agenesis/
├── persona/
│   ├── __init__.py      # Public API exports
│   ├── base.py          # PersonaContext, BasePersona, DefaultPersona classes
│   └── loader.py        # YAML loading and caching
└── config/
    └── persona/         # Built-in persona YAML files
        ├── technical_mentor.yaml
        ├── customer_support.yaml
        ├── casual_assistant.yaml
        └── professional.yaml
```

## Usage Examples

```python
# Built-in persona from YAML
agent = Agent(profile="user123", persona="technical_mentor")

# Custom persona from file
agent = Agent(profile="user123", persona_config="./my_persona.yaml")

# Custom persona from dictionary
agent = Agent(profile="user123", persona={
    "name": "custom_assistant",
    "description": "My custom assistant",
    "context_template": {
        "focus_areas": ["custom_focus"],
        "detail_level": "normal"
    }
})

# Runtime switching
agent.set_persona("customer_support")

# No persona (backwards compatible)
agent = Agent(profile="user123")  # Works exactly as before
```

## Implementation Results (Tested and Verified)

### ✅ **Comprehensive Testing Completed**
- **39 persona tests** all passing
- **End-to-end integration** with 8-interaction conversation
- **All modules** successfully receiving and using persona context
- **Priority signal detection** working flawlessly
- **Memory storage** with persona context serialization
- **Response quality** clearly influenced by persona

### ✅ **Technical Mentor Persona Verified**
**Priority Signal Detection:**
- `performance` → Auto-adjusts detail level to comprehensive
- `slow` → Triggers performance optimization focus
- `error` → Activates debugging strategies
- `scale` → Enables architecture guidance
- `debug + bug` → Systematic troubleshooting approach

**Response Transformation:**
- Systematic numbered recommendations (1., 2., 3.)
- Technical depth: caching, indexing, connection pooling
- Mentoring tone: "I'm happy to help", "A few suggestions"
- Step-by-step problem solving approach
- Specific tool recommendations (New Relic, Redis, etc.)

### ✅ **Module Integration Confirmed**
1. **Perception**: Receives persona context, extracts technical features
2. **Memory**: Stores persona context in all three memory systems
3. **Cognition**: Uses persona for reasoning approach and system prompts
4. **Action**: Generates responses influenced by persona preferences
5. **Evolution**: Works alongside persona (no conflicts)

This design gives personas significant influence over agent behavior while maintaining clean module separation and predictable, testable behavior.