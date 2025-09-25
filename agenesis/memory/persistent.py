import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from datetime import datetime
from .base import BaseMemory, MemoryRecord, PerceptionResult

try:
    import sqlite_vss
    VSS_AVAILABLE = True
except ImportError:
    VSS_AVAILABLE = False


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
    """SQLite-based persistent memory storage with vector search support"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        db_path = self.config.get('db_path', '.agenesis/memory.db')
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self.vector_search_enabled = VSS_AVAILABLE and self.config.get('enable_vector_search', True)

        # Phase 1.5: Index caching for performance
        self._embedding_cache = None  # numpy array of embeddings
        self._rowid_cache = None      # corresponding rowids
        self._cache_record_count = 0  # number of records when cache was built
        self._cache_built_at = None   # timestamp when cache was built
        self.enable_index_caching = self.config.get('enable_index_caching', True)

        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database and tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Enable sqlite-vss extension if available
            if self.vector_search_enabled:
                try:
                    conn.enable_load_extension(True)
                    sqlite_vss.load(conn)
                    print("🔍 Vector search enabled with sqlite-vss")
                except Exception as e:
                    print(f"⚠️ Failed to load sqlite-vss, falling back to in-memory search: {e}")
                    self.vector_search_enabled = False

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
                    embedding TEXT,
                    agent_response TEXT
                )
            ''')

            # Index for recent queries
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_stored_at
                ON memory_records(stored_at DESC)
            ''')

            # Initialize vector search table if enabled
            if self.vector_search_enabled:
                try:
                    # Create virtual table for vector similarity search
                    # Using text-embedding-3-small dimension (1536)
                    conn.execute('''
                        CREATE VIRTUAL TABLE IF NOT EXISTS vss_memory_embeddings USING vss0(
                            embedding(1536)
                        )
                    ''')

                    # Create trigger to automatically populate vector table when embeddings are inserted/updated
                    conn.execute('''
                        CREATE TRIGGER IF NOT EXISTS memory_embedding_insert
                        AFTER INSERT ON memory_records
                        WHEN NEW.embedding IS NOT NULL
                        BEGIN
                            INSERT INTO vss_memory_embeddings(rowid, embedding)
                            VALUES (NEW.rowid, NEW.embedding);
                        END
                    ''')

                    conn.execute('''
                        CREATE TRIGGER IF NOT EXISTS memory_embedding_update
                        AFTER UPDATE OF embedding ON memory_records
                        WHEN NEW.embedding IS NOT NULL
                        BEGIN
                            INSERT OR REPLACE INTO vss_memory_embeddings(rowid, embedding)
                            VALUES (NEW.rowid, NEW.embedding);
                        END
                    ''')

                except Exception as e:
                    print(f"⚠️ Failed to initialize vector search tables: {e}")
                    self.vector_search_enabled = False
    
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
            # Load sqlite-vss if enabled
            if self.vector_search_enabled:
                conn.enable_load_extension(True)
                sqlite_vss.load(conn)
            conn.execute('''
                INSERT INTO memory_records
                (id, content, input_type, perception_metadata, perception_features,
                 perception_timestamp, stored_at, context, metadata,
                 is_evolved_knowledge, evolution_metadata, reliability_multiplier, embedding, agent_response)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                json.dumps(record.embedding) if record.embedding else None,
                record.agent_response
            ))

        # Phase 1.5: Invalidate cache after storing new record
        self._invalidate_cache()

        return record.id
    
    def retrieve(self, memory_id: str) -> Optional[MemoryRecord]:
        """Retrieve memory record by ID"""
        with sqlite3.connect(self.db_path) as conn:
            # Load sqlite-vss if enabled
            if self.vector_search_enabled:
                conn.enable_load_extension(True)
                sqlite_vss.load(conn)
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
            # Load sqlite-vss if enabled
            if self.vector_search_enabled:
                conn.enable_load_extension(True)
                sqlite_vss.load(conn)
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
            embedding=embedding,
            agent_response=row['agent_response']
        )

    def update_embedding(self, memory_id: str, embedding: List[float]) -> bool:
        """Update the embedding for a specific memory record"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    'UPDATE memory_records SET embedding = ? WHERE id = ?',
                    (json.dumps(embedding), memory_id)
                )
                success = cursor.rowcount > 0

                # Phase 1.5: Invalidate cache after updating embedding
                if success:
                    self._invalidate_cache()

                return success
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
                updated_count = cursor.rowcount

                # Phase 1.5: Invalidate cache after batch update
                if updated_count > 0:
                    self._invalidate_cache()

                return updated_count
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
            total = row[0] or 0
            with_embeddings = row[1] or 0

            return {
                'total_records': total,
                'records_with_embeddings': with_embeddings,
                'records_without_embeddings': total - with_embeddings,
                'embedding_coverage': (with_embeddings / total * 100) if total > 0 else 0
            }

    def vector_similarity_search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        min_similarity: float = 0.1
    ) -> List[Tuple[MemoryRecord, float]]:
        """Perform vector similarity search using database-level operations

        Args:
            query_embedding: The query vector to search for
            limit: Maximum number of results to return
            min_similarity: Minimum similarity threshold (0.0 to 1.0)

        Returns:
            List of (MemoryRecord, similarity_score) tuples, sorted by similarity
        """
        if not self.vector_search_enabled:
            print("⚠️ Vector search not enabled, falling back to get_recent")
            return [(record, 0.0) for record in self.get_recent(limit)]

        # Phase 1.5: Use cached similarity search if enabled
        if self.enable_index_caching:
            try:
                return self._cached_similarity_search(query_embedding, limit, min_similarity)
            except Exception as e:
                print(f"⚠️ Cached similarity search failed: {e}")
                print("   Falling back to sqlite-vss search")
                # Fall through to original sqlite-vss implementation

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Enable sqlite-vss extension for this connection
                conn.enable_load_extension(True)
                sqlite_vss.load(conn)
                conn.row_factory = sqlite3.Row

                # Convert query embedding to JSON format expected by sqlite-vss
                query_embedding_json = json.dumps(query_embedding)

                # Perform vector similarity search
                # sqlite-vss uses cosine distance, we convert to similarity (1 - distance)
                # Use subquery approach since LIMIT must be in vss_search
                cursor = conn.execute('''
                    SELECT m.*, (1 - v.distance) as similarity_score
                    FROM (
                        SELECT rowid, distance
                        FROM vss_memory_embeddings
                        WHERE vss_search(embedding, json(?))
                        ORDER BY distance ASC
                        LIMIT ?
                    ) v
                    JOIN memory_records m ON m.rowid = v.rowid
                    WHERE (1 - v.distance) >= ?
                    ORDER BY v.distance ASC
                ''', (query_embedding_json, limit, min_similarity))

                results = []
                for row in cursor.fetchall():
                    memory_record = self._row_to_record(row)
                    similarity_score = row['similarity_score']
                    results.append((memory_record, similarity_score))

                return results

        except Exception as e:
            print(f"⚠️ Vector similarity search failed: {e}")
            print("   Falling back to recent records")
            return [(record, 0.0) for record in self.get_recent(limit)]

    def migrate_existing_embeddings(self) -> int:
        """Migrate existing embeddings to the vector search table

        This is useful for databases that already have embeddings but need
        to populate the vector search table.

        Returns:
            Number of embeddings migrated
        """
        if not self.vector_search_enabled:
            print("⚠️ Vector search not enabled, skipping migration")
            return 0

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Enable sqlite-vss extension
                conn.enable_load_extension(True)
                sqlite_vss.load(conn)

                # Find records with embeddings that aren't in the vector table
                cursor = conn.execute('''
                    SELECT m.rowid, m.embedding
                    FROM memory_records m
                    LEFT JOIN vss_memory_embeddings v ON m.rowid = v.rowid
                    WHERE m.embedding IS NOT NULL
                    AND v.rowid IS NULL
                ''')

                migrated_count = 0
                for row in cursor.fetchall():
                    rowid, embedding_json = row
                    try:
                        # Insert into vector table
                        conn.execute('''
                            INSERT INTO vss_memory_embeddings(rowid, embedding)
                            VALUES (?, ?)
                        ''', (rowid, embedding_json))
                        migrated_count += 1
                    except Exception as e:
                        print(f"⚠️ Failed to migrate embedding for rowid {rowid}: {e}")

                conn.commit()
                print(f"📊 Migrated {migrated_count} embeddings to vector search table")
                return migrated_count

        except Exception as e:
            print(f"⚠️ Embedding migration failed: {e}")
            return 0

    def get_vector_search_info(self) -> Dict[str, Any]:
        """Get information about vector search capabilities"""
        info = {
            'vector_search_enabled': self.vector_search_enabled,
            'sqlite_vss_available': VSS_AVAILABLE,
        }

        if self.vector_search_enabled:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.enable_load_extension(True)
                    sqlite_vss.load(conn)

                    # Get vector table statistics
                    cursor = conn.execute('SELECT COUNT(*) FROM vss_memory_embeddings')
                    vector_count = cursor.fetchone()[0]
                    info['vector_records_count'] = vector_count

            except Exception as e:
                info['vector_search_error'] = str(e)

        return info

    def _needs_cache_rebuild(self) -> bool:
        """Check if the embedding cache needs to be rebuilt"""
        if not self.enable_index_caching:
            return False

        # Always rebuild if cache doesn't exist
        if self._embedding_cache is None:
            return True

        # Check if new records have been added
        current_count = self._get_current_embedding_count()
        if current_count > self._cache_record_count:
            return True

        return False

    def _get_current_embedding_count(self) -> int:
        """Get current number of records with embeddings"""
        with sqlite3.connect(self.db_path) as conn:
            if self.vector_search_enabled:
                conn.enable_load_extension(True)
                sqlite_vss.load(conn)
            cursor = conn.execute('SELECT COUNT(*) FROM memory_records WHERE embedding IS NOT NULL')
            return cursor.fetchone()[0]

    def _rebuild_embedding_cache(self):
        """Rebuild the in-memory embedding cache"""
        if not self.enable_index_caching:
            return

        print("🔄 Rebuilding embedding cache...")
        start_time = datetime.now()

        with sqlite3.connect(self.db_path) as conn:
            if self.vector_search_enabled:
                conn.enable_load_extension(True)
                sqlite_vss.load(conn)

            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT rowid, embedding
                FROM memory_records
                WHERE embedding IS NOT NULL
                ORDER BY rowid
            ''')

            embeddings = []
            rowids = []

            for row in cursor.fetchall():
                embedding = json.loads(row['embedding'])
                embeddings.append(embedding)
                rowids.append(row['rowid'])

            if embeddings:
                self._embedding_cache = np.array(embeddings, dtype=np.float32)
                self._rowid_cache = np.array(rowids)
                self._cache_record_count = len(embeddings)
                self._cache_built_at = datetime.now()

                build_time = (self._cache_built_at - start_time).total_seconds() * 1000
                print(f"✅ Cache rebuilt: {len(embeddings)} embeddings in {build_time:.1f}ms")
            else:
                self._embedding_cache = None
                self._rowid_cache = None
                self._cache_record_count = 0
                self._cache_built_at = None

    def _invalidate_cache(self):
        """Invalidate the embedding cache (called after inserts)"""
        if self.enable_index_caching:
            # Don't immediately clear cache, just mark it as stale
            # It will be rebuilt on next search if needed
            pass

    def _cached_similarity_search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        min_similarity: float = 0.1
    ) -> List[Tuple[MemoryRecord, float]]:
        """Perform similarity search using cached embeddings (Phase 1.5 optimization)"""

        # Check if we need to rebuild cache
        if self._needs_cache_rebuild():
            self._rebuild_embedding_cache()

        # If no cache available (no embeddings), return empty
        if self._embedding_cache is None:
            return []

        # Convert query to numpy array
        query_array = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        # Compute cosine similarities efficiently with numpy
        # Normalize vectors for cosine similarity
        query_norm = query_array / np.linalg.norm(query_array)
        cache_norms = self._embedding_cache / np.linalg.norm(self._embedding_cache, axis=1, keepdims=True)

        # Cosine similarity = dot product of normalized vectors
        similarities = np.dot(cache_norms, query_norm.T).flatten()

        # Filter by minimum similarity
        valid_indices = similarities >= min_similarity
        valid_similarities = similarities[valid_indices]
        valid_rowids = self._rowid_cache[valid_indices]

        # Get top results
        if len(valid_similarities) > limit:
            top_indices = np.argpartition(valid_similarities, -limit)[-limit:]
            top_indices = top_indices[np.argsort(valid_similarities[top_indices])][::-1]
            valid_similarities = valid_similarities[top_indices]
            valid_rowids = valid_rowids[top_indices]
        else:
            # Sort all results by similarity (descending)
            sorted_indices = np.argsort(valid_similarities)[::-1]
            valid_similarities = valid_similarities[sorted_indices]
            valid_rowids = valid_rowids[sorted_indices]

        # Fetch actual memory records for the top results
        results = []
        if len(valid_rowids) > 0:
            with sqlite3.connect(self.db_path) as conn:
                if self.vector_search_enabled:
                    conn.enable_load_extension(True)
                    sqlite_vss.load(conn)
                conn.row_factory = sqlite3.Row

                # Query for all matching rowids
                placeholders = ','.join(['?'] * len(valid_rowids))
                cursor = conn.execute(f'''
                    SELECT rowid, * FROM memory_records
                    WHERE rowid IN ({placeholders})
                ''', valid_rowids.tolist())

                # Create a mapping from rowid to record
                rowid_to_record = {}
                for row in cursor.fetchall():
                    record_rowid = row[0]  # rowid is first column
                    rowid_to_record[record_rowid] = self._row_to_record(row)

                # Build results in the correct order
                for rowid, similarity in zip(valid_rowids, valid_similarities):
                    rowid_key = int(rowid)
                    if rowid_key in rowid_to_record:
                        results.append((rowid_to_record[rowid_key], float(similarity)))

        return results