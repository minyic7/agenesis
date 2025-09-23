"""
Complete flow integration test
Tests: Input → Perception → Memory (Immediate + Working + Persistent) → Agent
"""
import tempfile
import os
from pathlib import Path
import pytest

from agenesis.core import Agent


@pytest.mark.asyncio
async def test_complete_flow_anonymous_agent():
    """Test complete flow with anonymous agent (no persistent memory)"""
    agent = Agent()  # Anonymous agent
    
    # Verify initial state
    info = agent.get_profile_info()
    assert info['is_anonymous'] is True
    assert info['has_persistent_memory'] is False
    assert info['current_focus'] is False
    assert info['session_size'] == 0
    
    # Process first input - complete flow
    response1 = await agent.process_input("Hello, I'm working on a Python project")
    
    # Verify response
    assert isinstance(response1, str)
    assert len(response1) > 0
    
    # Check memory states after first input
    # 1. Immediate memory (current focus)
    current_focus = agent.get_current_focus()
    assert current_focus is not None
    assert current_focus.content == "Hello, I'm working on a Python project"
    
    # 2. Working memory (session context)
    session_context = agent.get_session_context()
    assert len(session_context) == 1
    assert session_context[0].content == "Hello, I'm working on a Python project"
    
    # 3. No persistent memory for anonymous agent
    assert agent.persistent_memory is None
    
    # Process second input
    response2 = await agent.process_input("Can you help me with error handling?")
    
    # Check memory states after second input
    # 1. Immediate memory should update to new focus
    current_focus = agent.get_current_focus()
    assert current_focus.content == "Can you help me with error handling?"
    
    # 2. Working memory should have both messages (newest first)
    session_context = agent.get_session_context()
    assert len(session_context) == 2
    assert session_context[0].content == "Can you help me with error handling?"  # newest
    assert session_context[1].content == "Hello, I'm working on a Python project"  # older
    
    # Process third input
    await agent.process_input("What about try-catch blocks?")
    
    # Session should have all 3 messages
    session_context = agent.get_session_context()
    assert len(session_context) == 3
    assert session_context[0].content == "What about try-catch blocks?"
    
    # Test session management
    agent.clear_focus()
    assert agent.get_current_focus() is None
    assert len(agent.get_session_context()) == 3  # Working memory unchanged
    
    agent.end_session()
    assert agent.get_current_focus() is None
    assert len(agent.get_session_context()) == 0  # Working memory cleared


@pytest.mark.asyncio
async def test_complete_flow_named_agent():
    """Test complete flow with named agent (with persistent memory)"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create named agent with custom storage location
        test_profile = "test_coding_assistant"
        
        # First session
        agent1 = Agent(profile=test_profile, config={'storage_type': 'sqlite'})
        
        # Verify named agent setup
        info = agent1.get_profile_info()
        assert info['is_anonymous'] is False
        assert info['profile'] == test_profile
        assert info['has_persistent_memory'] is True
        
        # Override storage location for test
        db_path = os.path.join(temp_dir, 'test_agent.db')
        from agenesis.memory import SQLiteMemory
        agent1.persistent_memory = SQLiteMemory({'db_path': db_path})
        
        # Process inputs in first session
        await agent1.process_input("I need help with Python debugging")
        await agent1.process_input("How do I use breakpoints?")
        await agent1.process_input("What about logging?")
        
        # Check memory states
        assert len(agent1.get_session_context()) == 3
        assert agent1.get_current_focus().content == "What about logging?"
        
        # End first session
        agent1.end_session()
        
        # Start second session with same profile (simulating app restart)
        agent2 = Agent(profile=test_profile, config={'storage_type': 'sqlite'})
        agent2.persistent_memory = SQLiteMemory({'db_path': db_path})
        
        # New session should start fresh for immediate/working memory
        assert agent2.get_current_focus() is None
        assert len(agent2.get_session_context()) == 0
        
        # But persistent memory should have previous data
        recent_persistent = agent2.persistent_memory.get_recent(3)
        assert len(recent_persistent) == 3
        assert recent_persistent[0].perception_result.content == "What about logging?"
        assert recent_persistent[1].perception_result.content == "How do I use breakpoints?"
        assert recent_persistent[2].perception_result.content == "I need help with Python debugging"
        
        # Continue with new session
        await agent2.process_input("Now I want to learn about testing")
        
        # Check current session vs persistent history
        assert len(agent2.get_session_context()) == 1  # Only current session
        recent_persistent = agent2.persistent_memory.get_recent(4) 
        assert len(recent_persistent) == 4  # All historical data


@pytest.mark.asyncio
async def test_memory_attention_hierarchy():
    """Test the attention hierarchy: Immediate > Working > Persistent"""
    agent = Agent(profile="attention_test")
    
    # Override persistent memory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        from agenesis.memory import SQLiteMemory
        db_path = os.path.join(temp_dir, 'attention_test.db')
        agent.persistent_memory = SQLiteMemory({'db_path': db_path})
        
        # Simulate attention-based processing
        inputs = [
            "I'm learning Python",
            "Can you explain functions?", 
            "What about lambda functions?",
            "How do decorators work?",
            "Show me async/await"
        ]
        
        for i, text in enumerate(inputs):
            await agent.process_input(text)
            
            # After each input, check attention hierarchy
            
            # 1. Immediate Memory (highest priority) - current focus
            current = agent.get_current_focus()
            assert current.content == text
            
            # 2. Working Memory (medium priority) - recent session context
            recent_working = agent.get_session_context(3)  # Last 3 from session
            assert len(recent_working) == min(i + 1, 3)
            assert recent_working[0].content == text  # Most recent first
            
            # 3. Persistent Memory (lower priority) - all historical context  
            all_persistent = agent.persistent_memory.get_recent(10)
            assert len(all_persistent) == i + 1
            assert all_persistent[0].perception_result.content == text
        
        # Verify final state - agent has layered memory access
        assert agent.get_current_focus().content == "Show me async/await"
        assert len(agent.get_session_context()) == 5
        assert len(agent.persistent_memory.get_recent(10)) == 5


if __name__ == "__main__":
    # Run the tests manually
    print("Running complete flow tests...")
    
    import asyncio

    async def run_tests():
        print("1. Testing anonymous agent flow...")
        await test_complete_flow_anonymous_agent()
        print("✅ Anonymous agent test passed")

        print("2. Testing named agent flow...")
        await test_complete_flow_named_agent()
        print("✅ Named agent test passed")

        print("3. Testing memory attention hierarchy...")
        await test_memory_attention_hierarchy()
        print("✅ Memory attention hierarchy test passed")

    asyncio.run(run_tests())
    
    print("\n🎉 All complete flow tests passed!")
    print("Integration working: Input → Perception → Memory → Agent")