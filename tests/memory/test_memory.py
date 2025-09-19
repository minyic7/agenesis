from agenesis.memory import ImmediateMemory, WorkingMemory
from agenesis.perception import TextPerception


def test_immediate_memory_focus():
    """Test that ImmediateMemory holds single focus"""
    memory = ImmediateMemory()
    perception = TextPerception()
    
    # Store first input
    result1 = perception.process("First message")
    memory_id1 = memory.store(result1)
    
    assert memory.has_focus()
    assert memory.get_current().id == memory_id1
    
    # Store second input - should replace first
    result2 = perception.process("Second message") 
    memory_id2 = memory.store(result2)
    
    assert memory.get_current().id == memory_id2
    assert memory.get_current().perception_result.content == "Second message"
    
    # Clear focus
    memory.clear()
    assert not memory.has_focus()
    assert memory.get_current() is None


def test_working_memory_session():
    """Test that WorkingMemory maintains session context"""
    memory = WorkingMemory()
    perception = TextPerception()
    
    # Store multiple inputs
    result1 = perception.process("Message 1")
    result2 = perception.process("Message 2") 
    result3 = perception.process("Message 3")
    
    memory.store(result1)
    memory.store(result2)
    memory.store(result3)
    
    # Get recent should return in reverse chronological order
    recent = memory.get_recent(2)
    assert len(recent) == 2
    assert recent[0].perception_result.content == "Message 3"
    assert recent[1].perception_result.content == "Message 2"
    
    # Check total size
    assert memory.size() == 3


def test_working_memory_capacity():
    """Test WorkingMemory capacity limits"""
    memory = WorkingMemory({'max_capacity': 2})
    perception = TextPerception()
    
    # Fill beyond capacity
    for i in range(3):
        result = perception.process(f"Message {i+1}")
        memory.store(result)
    
    # Should only keep last 2
    assert memory.size() == 2
    
    # get_all() returns chronological order (oldest first)
    all_records = memory.get_all()
    assert all_records[0].perception_result.content == "Message 2"
    assert all_records[1].perception_result.content == "Message 3"
    
    # get_recent() returns reverse chronological (newest first)
    recent = memory.get_recent(2)
    assert recent[0].perception_result.content == "Message 3"
    assert recent[1].perception_result.content == "Message 2"