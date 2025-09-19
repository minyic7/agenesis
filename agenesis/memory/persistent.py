import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any
from .base import BaseMemory, MemoryRecord, PerceptionResult


class FileMemory(BaseMemory):
    """Simple file-based persistent memory storage"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.storage_dir = Path(self.config.get('storage_dir', '.agenesis/memory'))
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_files_exist()
    
    def _ensure_files_exist(self):
        """Ensure storage files exist"""
        if not (self.storage_dir / 'records.jsonl').exists():
            (self.storage_dir / 'records.jsonl').touch()
    
    def store(self, perception_result: PerceptionResult, context: Optional[Dict[str, Any]] = None) -> str:
        """Store perception result to file"""
        record = MemoryRecord(
            id="",  # Generated in __post_init__
            perception_result=perception_result,
            stored_at=None,  # Set in __post_init__
            context=self._create_context(context),
            metadata=self._create_metadata()
        )
        
        # Append to JSONL file
        record_data = self._serialize_record(record)
        with open(self.storage_dir / 'records.jsonl', 'a') as f:
            f.write(json.dumps(record_data) + '\n')
        
        return record.id
    
    def retrieve(self, memory_id: str) -> Optional[MemoryRecord]:
        """Retrieve memory record by ID"""
        with open(self.storage_dir / 'records.jsonl', 'r') as f:
            for line in f:
                if line.strip():
                    record_data = json.loads(line)
                    if record_data['id'] == memory_id:
                        return self._deserialize_record(record_data)
        return None
    
    def get_recent(self, count: int = 10) -> List[MemoryRecord]:
        """Get most recent memory records"""
        records = []
        with open(self.storage_dir / 'records.jsonl', 'r') as f:
            for line in f:
                if line.strip():
                    record_data = json.loads(line)
                    records.append(self._deserialize_record(record_data))
        
        # Sort by timestamp and return most recent
        records.sort(key=lambda r: r.stored_at, reverse=True)
        return records[:count]
    
    def _serialize_record(self, record: MemoryRecord) -> Dict[str, Any]:
        """Convert MemoryRecord to JSON-serializable dict"""
        return {
            'id': record.id,
            'perception_result': {
                'content': record.perception_result.content,
                'input_type': record.perception_result.input_type.value,
                'metadata': record.perception_result.metadata,
                'features': record.perception_result.features,
                'timestamp': record.perception_result.timestamp.isoformat()
            },
            'stored_at': record.stored_at.isoformat(),
            'context': record.context,
            'metadata': record.metadata
        }
    
    def _deserialize_record(self, data: Dict[str, Any]) -> MemoryRecord:
        """Convert JSON dict back to MemoryRecord"""
        # TODO: Properly reconstruct PerceptionResult with InputType enum
        # For now, placeholder implementation
        from ..perception.base import PerceptionResult, InputType
        from datetime import datetime
        
        perception_result = PerceptionResult(
            content=data['perception_result']['content'],
            input_type=InputType(data['perception_result']['input_type']),
            metadata=data['perception_result']['metadata'],
            features=data['perception_result']['features'],
            timestamp=datetime.fromisoformat(data['perception_result']['timestamp'])
        )
        
        return MemoryRecord(
            id=data['id'],
            perception_result=perception_result,
            stored_at=datetime.fromisoformat(data['stored_at']),
            context=data['context'],
            metadata=data['metadata']
        )


class SQLiteMemory(BaseMemory):
    """SQLite-based persistent memory storage"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        db_path = self.config.get('db_path', '.agenesis/memory.db')
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database and tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS memory_records (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    input_type TEXT NOT NULL,
                    perception_metadata TEXT NOT NULL,
                    perception_features TEXT NOT NULL,
                    perception_timestamp TEXT NOT NULL,
                    stored_at TEXT NOT NULL,
                    context TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
            ''')
            
            # Index for recent queries
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_stored_at 
                ON memory_records(stored_at DESC)
            ''')
    
    def store(self, perception_result: PerceptionResult, context: Optional[Dict[str, Any]] = None) -> str:
        """Store perception result to SQLite"""
        record = MemoryRecord(
            id="",  # Generated in __post_init__
            perception_result=perception_result,
            stored_at=None,  # Set in __post_init__
            context=self._create_context(context),
            metadata=self._create_metadata()
        )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO memory_records 
                (id, content, input_type, perception_metadata, perception_features, 
                 perception_timestamp, stored_at, context, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.id,
                record.perception_result.content,
                record.perception_result.input_type.value,
                json.dumps(record.perception_result.metadata),
                json.dumps(record.perception_result.features),
                record.perception_result.timestamp.isoformat(),
                record.stored_at.isoformat(),
                json.dumps(record.context),
                json.dumps(record.metadata)
            ))
        
        return record.id
    
    def retrieve(self, memory_id: str) -> Optional[MemoryRecord]:
        """Retrieve memory record by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                'SELECT * FROM memory_records WHERE id = ?', 
                (memory_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return self._row_to_record(row)
            return None
    
    def get_recent(self, count: int = 10) -> List[MemoryRecord]:
        """Get most recent memory records"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                'SELECT * FROM memory_records ORDER BY stored_at DESC LIMIT ?',
                (count,)
            )
            
            return [self._row_to_record(row) for row in cursor.fetchall()]
    
    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        """Convert SQLite row to MemoryRecord"""
        from ..perception.base import PerceptionResult, InputType
        from datetime import datetime
        
        perception_result = PerceptionResult(
            content=row['content'],
            input_type=InputType(row['input_type']),
            metadata=json.loads(row['perception_metadata']),
            features=json.loads(row['perception_features']),
            timestamp=datetime.fromisoformat(row['perception_timestamp'])
        )
        
        return MemoryRecord(
            id=row['id'],
            perception_result=perception_result,
            stored_at=datetime.fromisoformat(row['stored_at']),
            context=json.loads(row['context']),
            metadata=json.loads(row['metadata'])
        )