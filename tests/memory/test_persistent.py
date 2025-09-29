import tempfile
import os
from pathlib import Path

from agenesis.memory import SQLiteMemory
from agenesis.perception import TextPerception


def test_sqlite_memory_persistence():
    """Test SQLiteMemory stores and retrieves correctly"""
    perception = TextPerception()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, 'test_memory.db')
        
        # Create memory with temporary database
        memory = SQLiteMemory({'db_path': db_path})
        
        # Store some data
        result1 = perception.process("Test message 1")
        result2 = perception.process("Test message 2")
        
        memory_id1 = memory.store(result1)
        memory_id2 = memory.store(result2)
        
        # Verify storage
        assert os.path.exists(db_path)
        
        # Retrieve by ID
        retrieved1 = memory.retrieve(memory_id1)
        assert retrieved1 is not None
        assert retrieved1.perception_result.content == "Test message 1"
        
        # Get recent records
        recent = memory.get_recent(2)
        assert len(recent) == 2
        assert recent[0].perception_result.content == "Test message 2"  # newest first
        assert recent[1].perception_result.content == "Test message 1"


def test_sqlite_memory_persistence_across_instances():
    """Test SQLiteMemory persists data across different instances"""
    perception = TextPerception()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, 'persistent_test.db')
        
        # First instance - store data
        memory1 = SQLiteMemory({'db_path': db_path})
        result = perception.process("Persistent message")
        memory_id = memory1.store(result)
        
        # Second instance - should load existing data
        memory2 = SQLiteMemory({'db_path': db_path})
        retrieved = memory2.retrieve(memory_id)
        
        assert retrieved is not None
        assert retrieved.perception_result.content == "Persistent message"
        
        # Should also appear in recent
        recent = memory2.get_recent(1)
        assert len(recent) == 1
        assert recent[0].perception_result.content == "Persistent message"


