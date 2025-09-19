#!/usr/bin/env python3
"""
Comprehensive test scenario for technical_mentor persona.
Tests long conversation context and examines each module's output.
"""
import asyncio
import json
from dotenv import load_dotenv

from agenesis.core import Agent
from agenesis.persona import PersonaContext

# Load environment variables
load_dotenv()


async def test_technical_mentor_comprehensive():
    """Test technical mentor persona with detailed output inspection"""
    print("🧠 Technical Mentor Persona - Comprehensive Test\n")
    
    # Create agent with technical mentor persona
    agent = Agent(profile="tech_mentor_test", persona="technical_mentor")
    
    print("✅ Agent initialized:")
    profile_info = agent.get_profile_info()
    print(f"   Profile: {agent.profile}")
    print(f"   Persona: {profile_info['persona']['name']}")
    print(f"   Description: {profile_info['persona']['description']}")
    print(f"   Persistent memory: {profile_info['has_persistent_memory']}")
    print()
    
    # Long conversation scenario - technical debugging session
    conversation = [
        # 1. Initial problem description
        "Hi, I'm working on a Python web application and I'm experiencing some performance issues",
        
        # 2. More specific problem
        "The API response times are really slow, sometimes taking 5-10 seconds for simple queries",
        
        # 3. Technical details
        "I'm using Flask with SQLAlchemy ORM, PostgreSQL database, and the app is deployed on AWS EC2",
        
        # 4. Error mention (should trigger priority signals)
        "I'm also getting occasional timeout errors when the database queries take too long",
        
        # 5. Code question  
        "Here's my query code: User.query.filter_by(active=True).all() - is this efficient?",
        
        # 6. Architecture question
        "Should I be using database connection pooling? How do I optimize this?",
        
        # 7. Debugging approach
        "What's the best way to debug and profile these performance bottlenecks?",
        
        # 8. Scaling concerns (priority signal)
        "The app needs to scale to handle 1000+ concurrent users. What should I consider?"
    ]
    
    print("📝 Testing conversation with detailed module inspection:\n")
    
    for i, user_input in enumerate(conversation, 1):
        print(f"{'='*80}")
        print(f"🔄 INTERACTION #{i}")
        print(f"👤 User: {user_input}")
        print(f"{'='*80}")
        
        # === 1. PERSONA CONTEXT GENERATION ===
        print("\n1️⃣ PERSONA CONTEXT GENERATION:")
        persona_context = agent.persona.create_context(user_input)
        print(f"   🎯 Focus Areas: {persona_context.focus_areas}")
        print(f"   🚨 Priority Signals: {persona_context.priority_signals}")
        print(f"   📈 Relevance Boosts: {persona_context.relevance_boosts}")
        print(f"   🧠 Reasoning Approach: {persona_context.reasoning_approach}")
        print(f"   📊 Detail Level: {persona_context.detail_level}")
        print(f"   💡 System Additions: {len(persona_context.system_additions)} items")
        
        # Check priority signal detection
        signals_detected = [signal for signal in persona_context.priority_signals 
                          if signal.lower() in user_input.lower()]
        if signals_detected:
            print(f"   🚨 PRIORITY SIGNALS DETECTED: {signals_detected}")
            print(f"   📈 Detail Level Auto-Adjusted: {persona_context.detail_level}")
        
        # === 2. PERCEPTION MODULE ===
        print("\n2️⃣ PERCEPTION MODULE:")
        perception_result = agent.perception.process(user_input, persona_context)
        print(f"   📝 Content Length: {len(perception_result.content)} chars")
        print(f"   🏷️  Input Type: {perception_result.input_type}")
        print(f"   📊 Features Extracted:")
        for key, value in perception_result.features.items():
            print(f"      - {key}: {value}")
        
        # Check if persona influenced perception
        print(f"   🎭 Persona Context Provided: {persona_context is not None}")
        
        # === 3. MEMORY STORAGE ===
        print("\n3️⃣ MEMORY STORAGE:")
        memory_context = {"persona_context": persona_context.to_dict()} if persona_context else None
        
        # Store in all memory systems
        immediate_id = agent.immediate_memory.store(perception_result, memory_context)
        working_id = agent.working_memory.store(perception_result, memory_context)
        if agent.has_persistent_memory:
            persistent_id = agent.persistent_memory.store(perception_result, memory_context)
        
        print(f"   💾 Immediate Memory ID: {immediate_id}")
        print(f"   💾 Working Memory ID: {working_id}")
        if agent.has_persistent_memory:
            print(f"   💾 Persistent Memory ID: {persistent_id}")
        
        # Check memory contents
        current_record = agent.immediate_memory.get_current()
        print(f"   📄 Current Focus: {current_record.perception_result.content[:50]}...")
        print(f"   🎭 Persona Context Stored: {'persona_context' in current_record.context}")
        
        working_memories = agent.working_memory.get_recent(3)
        print(f"   📚 Working Memory Size: {len(working_memories)} items")
        
        # === 4. COGNITION MODULE ===
        print("\n4️⃣ COGNITION MODULE:")
        cognition_result = await agent.cognition.process(
            agent.immediate_memory, agent.working_memory, persona_context
        )
        
        print(f"   🧠 Intent: {cognition_result.intent}")
        print(f"   🎯 Confidence: {cognition_result.confidence}")
        print(f"   📝 Summary: {cognition_result.summary[:100]}...")
        print(f"   🔧 Context Type: {cognition_result.context_type}")
        print(f"   📊 Persistence Score: {cognition_result.persistence_score}")
        print(f"   🧠 Reasoning: {cognition_result.reasoning[:100]}...")
        print(f"   📚 Relevant Memories: {len(cognition_result.relevant_memories)} items")
        print(f"   🎭 Persona Context Used: {persona_context is not None}")
        
        # Check if persona influenced cognition
        if persona_context and persona_context.reasoning_approach:
            print(f"   🧭 Persona Reasoning: {persona_context.reasoning_approach}")
        
        # === 5. ACTION MODULE ===
        print("\n5️⃣ ACTION MODULE:")
        action_result = await agent.action.generate_response(cognition_result, persona_context)
        
        print(f"   📤 Response Length: {len(action_result.response_text)} chars")
        print(f"   🎯 Action Confidence: {action_result.confidence}")
        print(f"   🎭 Persona Context Used: {persona_context is not None}")
        if action_result.internal_reasoning:
            print(f"   🔧 Internal Reasoning: {action_result.internal_reasoning[:100]}...")
        
        # === 6. FINAL RESPONSE ===
        print("\n6️⃣ FINAL RESPONSE:")
        print(f"🤖 Agent: {action_result.response_text}")
        
        # === 7. MEMORY ANALYSIS ===
        print("\n7️⃣ MEMORY ANALYSIS:")
        session_context = agent.get_session_context(5)
        print(f"   📊 Session Context Items: {len(session_context)}")
        
        # Check for persona influence accumulation
        if len(working_memories) > 1:
            print("   🔄 Conversation History:")
            for idx, memory in enumerate(working_memories[-3:], 1):
                snippet = memory.perception_result.content[:30]
                has_persona = 'persona_context' in memory.context
                print(f"      {idx}. {snippet}... (persona: {has_persona})")
        
        print(f"\n{'='*80}\n")
        
        # Brief pause between interactions
        await asyncio.sleep(0.1)
    
    # === FINAL ANALYSIS ===
    print("📊 FINAL SESSION ANALYSIS:")
    final_profile = agent.get_profile_info()
    print(f"   👤 Profile: {final_profile['profile']}")
    print(f"   🎭 Persona Active: {final_profile['persona']['active']}")
    print(f"   📚 Session Size: {final_profile['session_size']} interactions")
    print(f"   🎯 Current Focus: {final_profile['current_focus']}")
    
    # Memory distribution analysis
    all_working_memories = agent.working_memory.get_recent(10)
    persona_memories = [m for m in all_working_memories if 'persona_context' in m.context]
    print(f"   💾 Total Working Memories: {len(all_working_memories)}")
    print(f"   🎭 Memories with Persona Context: {len(persona_memories)}")
    
    # Check persona consistency
    if persona_memories:
        sample_persona_data = persona_memories[0].context['persona_context']
        print(f"   📋 Persona Focus Areas: {sample_persona_data.get('focus_areas', [])}")
        print(f"   🚨 Persona Priority Signals: {sample_persona_data.get('priority_signals', [])}")
        print(f"   🧠 Persona Reasoning: {sample_persona_data.get('reasoning_approach')}")
    
    print("\n✅ Technical Mentor Persona Test Complete!")


async def test_persona_context_details():
    """Detailed examination of persona context generation"""
    print("\n🔬 DETAILED PERSONA CONTEXT ANALYSIS\n")
    
    agent = Agent(profile="context_test", persona="technical_mentor")
    
    test_inputs = [
        ("Normal query", "How do I connect to a database?"),
        ("Performance issue", "My app is running really slow"),
        ("Error case", "I'm getting database timeout errors"),
        ("Architecture question", "Should I use microservices for scaling?"),
        ("Debug request", "How do I debug memory leaks in Python?"),
        ("Multiple signals", "I have performance errors that need debugging for scale")
    ]
    
    for label, input_text in test_inputs:
        print(f"📝 {label}: '{input_text}'")
        context = agent.persona.create_context(input_text)
        
        print(f"   🎯 Focus Areas: {context.focus_areas}")
        print(f"   🚨 Priority Signals: {context.priority_signals}")
        print(f"   📊 Detail Level: {context.detail_level}")
        print(f"   🧠 Reasoning: {context.reasoning_approach}")
        
        # Check signal detection
        signals_found = [sig for sig in context.priority_signals 
                        if sig.lower() in input_text.lower()]
        print(f"   ✅ Signals Detected: {signals_found}")
        print(f"   📈 Context Has Content: {context.has_content()}")
        print()


if __name__ == "__main__":
    print("🧪 Starting Comprehensive Technical Mentor Persona Test\n")
    asyncio.run(test_technical_mentor_comprehensive())
    asyncio.run(test_persona_context_details())