import tempfile
import shutil
from pathlib import Path
import pytest

from agenesis.core import Agent


@pytest.mark.asyncio
async def test_anonymous_agent():
    """Test anonymous agent has no persistent memory"""
    agent = Agent()  # No profile
    
    # Check agent info
    info = agent.get_profile_info()
    assert info['is_anonymous'] is True
    assert info['has_persistent_memory'] is False
    assert info['profile'] is None
    assert info['storage_location'] is None
    
    # Process input
    response = await agent.process_input("Hello world")
    assert isinstance(response, str)
    assert len(response) > 0
    
    # Should have immediate and working memory
    assert agent.get_current_focus() is not None
    assert len(agent.get_session_context()) == 1
    
    # But no persistent memory
    assert agent.persistent_memory is None


@pytest.mark.asyncio
async def test_named_agent_profile():
    """Test named agent has persistent memory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Override home directory for test
        test_home = Path(temp_dir)
        
        # Create agent with profile
        agent = Agent(profile="test_agent")
        
        # Manually set storage to test directory
        agent.persistent_memory = None  # Disable for this test
        agent.has_persistent_memory = False
        
        # Check basic profile info
        info = agent.get_profile_info()
        assert info['is_anonymous'] is False
        assert info['profile'] == "test_agent"
        
        # Process input
        response = await agent.process_input("Test message")
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Should have focus and session context
        assert agent.get_current_focus() is not None
        assert len(agent.get_session_context()) == 1


@pytest.mark.asyncio
async def test_agent_session_management():
    """Test agent session lifecycle"""
    agent = Agent()
    
    # Process multiple inputs
    await agent.process_input("Message 1")
    await agent.process_input("Message 2")
    
    # Should have current focus and session context
    assert agent.get_current_focus().content == "Message 2"
    assert len(agent.get_session_context()) == 2
    
    # Clear focus only
    agent.clear_focus()
    assert agent.get_current_focus() is None
    assert len(agent.get_session_context()) == 2  # Session context remains
    
    # End session
    agent.end_session()
    assert agent.get_current_focus() is None
    assert len(agent.get_session_context()) == 0


@pytest.mark.asyncio
async def test_agent_memory_flow():
    """Test complete memory flow: immediate → working"""
    agent = Agent()
    
    # Process first input
    await agent.process_input("First input")
    
    # Check immediate memory (current focus)
    current = agent.get_current_focus()
    assert current.content == "First input"
    
    # Check working memory (session context)
    context = agent.get_session_context()
    assert len(context) == 1
    assert context[0].content == "First input"
    
    # Process second input
    await agent.process_input("Second input")
    
    # Immediate memory should update to new focus
    current = agent.get_current_focus()
    assert current.content == "Second input"
    
    # Working memory should have both
    context = agent.get_session_context()
    assert len(context) == 2
    assert context[0].content == "Second input"  # Most recent first
    assert context[1].content == "First input"