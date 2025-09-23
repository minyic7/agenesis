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
                    metadata TEXT NOT NULL,
                    is_evolved_knowledge INTEGER DEFAULT 0,
                    evolution_metadata TEXT,
                    reliability_multiplier REAL DEFAULT 1.0,
                    embedding TEXT
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

        return self._store_record_to_db(record)

    def store_record(self, memory_record: MemoryRecord) -> str:
        """Store a complete memory record - preserves all fields including embedding"""
        # Update context and metadata but preserve other fields
        memory_record.context.update(self._create_context(memory_record.context))
        memory_record.metadata.update(self._create_metadata())

        return self._store_record_to_db(memory_record)

    def _store_record_to_db(self, record: MemoryRecord) -> str:
        """Internal method to store record to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO memory_records
                (id, content, input_type, perception_metadata, perception_features,
                 perception_timestamp, stored_at, context, metadata,
                 is_evolved_knowledge, evolution_metadata, reliability_multiplier, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.id,
                record.perception_result.content,
                record.perception_result.input_type.value,
                json.dumps(record.perception_result.metadata),
                json.dumps(record.perception_result.features),
                record.perception_result.timestamp.isoformat(),
                record.stored_at.isoformat(),
                json.dumps(record.context),
                json.dumps(record.metadata),
                1 if record.is_evolved_knowledge else 0,
                json.dumps(record.evolution_metadata) if record.evolution_metadata else None,
                record.reliability_multiplier,
                json.dumps(record.embedding) if record.embedding else None
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

        # Handle evolution metadata and embedding
        evolution_metadata = None
        if row['evolution_metadata']:
            evolution_metadata = json.loads(row['evolution_metadata'])

        embedding = None
        if row['embedding']:
            embedding = json.loads(row['embedding'])

        return MemoryRecord(
            id=row['id'],
            perception_result=perception_result,
            stored_at=datetime.fromisoformat(row['stored_at']),
            context=json.loads(row['context']),
            metadata=json.loads(row['metadata']),
            is_evolved_knowledge=bool(row['is_evolved_knowledge']),
            evolution_metadata=evolution_metadata,
            reliability_multiplier=row['reliability_multiplier'],
            embedding=embedding
        )

    def update_embedding(self, memory_id: str, embedding: List[float]) -> bool:
        """Update the embedding for a specific memory record"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    'UPDATE memory_records SET embedding = ? WHERE id = ?',
                    (json.dumps(embedding), memory_id)
                )
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Failed to update embedding for {memory_id}: {e}")
            return False

    def get_records_without_embeddings(self, limit: int = 100) -> List[MemoryRecord]:
        """Get records that don't have embeddings yet"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                'SELECT * FROM memory_records WHERE embedding IS NULL ORDER BY stored_at DESC LIMIT ?',
                (limit,)
            )

            return [self._row_to_record(row) for row in cursor.fetchall()]

    def batch_update_embeddings(self, embeddings_data: List[tuple]) -> int:
        """Batch update embeddings for multiple records

        Args:
            embeddings_data: List of (memory_id, embedding) tuples

        Returns:
            Number of records updated
        """
        if not embeddings_data:
            return 0

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.executemany(
                    'UPDATE memory_records SET embedding = ? WHERE id = ?',
                    [(json.dumps(embedding), memory_id) for memory_id, embedding in embeddings_data]
                )
                return cursor.rowcount
        except Exception as e:
            print(f"Failed to batch update embeddings: {e}")
            return 0

    def get_embedding_statistics(self) -> Dict[str, int]:
        """Get statistics about embeddings in the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT COUNT(*) as total, SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) as with_embeddings FROM memory_records'
            )
            row = cursor.fetchone()
            total = row[0]
            with_embeddings = row[1]

            return {
                'total_records': total,
                'records_with_embeddings': with_embeddings,
                'records_without_embeddings': total - with_embeddings,
                'embedding_coverage': (with_embeddings / total * 100) if total > 0 else 0
            }