# Agenesis

**A human-like AI agent development framework with cognitive architecture**

Agenesis provides a modular framework for building AI agents that mimic human cognitive processes through perception, memory, cognition, action, and learning systems. Features include persona-based behavior customization, multi-tier memory architecture, and intelligent learning from interactions.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key and/or Anthropic API key (optional, has heuristic fallback)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd agenesis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (optional)
cp .env.example .env
# Edit .env with your API keys
```

### Basic Usage

```python
import asyncio
from agenesis.core import Agent

async def main():
    # Create a simple agent
    agent = Agent()
    response = await agent.process_input("Hello! How can you help me?")
    print(response)

    # Create agent with persistent memory
    agent = Agent(profile="my_user")
    response = await agent.process_input("Remember that I prefer Python over JavaScript")

    # Create agent with persona
    agent = Agent(profile="developer", persona="technical_mentor")
    response = await agent.process_input("I have performance issues in my web app")

# Run the example
asyncio.run(main())
```

### Available Personas

- **technical_mentor** - Systematic problem-solving mentor for developers
- **customer_support** - Helpful, patient customer service representative  
- **casual_assistant** - Friendly, relaxed, and approachable communication
- **professional** - Professional, courteous, and business-appropriate

### Custom Personas

Create custom personas with YAML configuration:

```yaml
# my_persona.yaml
name: "my_custom_assistant"
description: "Specialized assistant for my needs"

context_template:
  focus_areas:
    - "domain_expertise"
    - "specific_tasks"
  
  priority_signals:
    - "urgent"
    - "important"
  
  reasoning_approach: "step_by_step"
  detail_level: "comprehensive"
  
  system_additions:
    - "Always provide practical examples"
    - "Be encouraging and supportive"
```

```python
# Use custom persona
agent = Agent(profile="user", persona_config="./my_persona.yaml")
```

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run specific module tests
pytest tests/persona/
pytest tests/memory/

# Run with coverage
pytest --cov=agenesis
```

## 📁 Project Structure

```
agenesis/
├── agenesis/           # Main package
│   ├── core/          # Agent orchestration
│   ├── perception/    # Input processing
│   ├── memory/        # Memory systems
│   ├── cognition/     # Reasoning engine
│   ├── action/        # Response generation
│   ├── evolution/     # Learning system
│   ├── persona/       # Behavior customization
│   └── providers/     # LLM integrations
├── config/
│   └── persona/       # Built-in persona definitions
├── tests/             # Comprehensive test suite
└── demos/             # Example scripts
```

## 🎯 Key Features

### Human-Like Memory Architecture
- **Immediate Memory** - Current focus and attention
- **Working Memory** - Session context with attention management
- **Persistent Memory** - Long-term storage across sessions

### Persona System
- **Static YAML Configuration** - Predictable, customizable behavior
- **Context Injection** - Influences entire processing pipeline
- **Priority Signal Detection** - Auto-adjusts response detail level
- **Runtime Switching** - Change personas dynamically

### Learning & Evolution
- **Selective Learning** - Only valuable interactions are learned
- **Memory Enhancement** - Learned knowledge improves future responses
- **LLM + Heuristic Analysis** - Intelligent learning detection

### LLM Integration
- **Multi-Provider Support** - OpenAI and Anthropic
- **Graceful Fallback** - Works without API keys using heuristics
- **Async Processing** - Non-blocking API calls

## 🔧 Configuration

### Environment Variables
```bash
# Optional - enables LLM features
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Agent stores memory in ~/.agenesis/profiles/
# No additional configuration needed
```

### Agent Profiles
- **Anonymous agents** - No persistent memory
- **Named agents** - Persistent memory stored in user profile
- **Profile isolation** - Each profile maintains separate memory

## 📚 Examples

Check the `demos/` directory for comprehensive examples:
- `test_technical_mentor_scenario.py` - Advanced persona interaction
- `test_evolution_demo.py` - Learning and memory evolution
- `test_cognition_demo.py` - Reasoning and context building

## 🤝 Contributing

This is a research/development framework. See design documents (`*_design.md`) for architecture details.

## 📄 License

[Add your license here]