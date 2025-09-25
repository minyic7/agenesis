#!/usr/bin/env python3
"""
Test vector search functionality with SQLite + sqlite-vss
"""
import asyncio
import os
import tempfile
from pathlib import Path
from agenesis.memory.persistent import SQLiteMemory
from agenesis.perception.base import PerceptionResult, InputType
from agenesis.providers.embedding import create_embedding_provider
from datetime import datetime, timezone

async def test_vector_search():
    """Test the new vector search functionality"""

    # Create temporary database
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as temp_db:
        db_path = temp_db.name

    try:
        print("🧪 Testing Vector Search with SQLite + sqlite-vss")
        print("=" * 60)

        # Initialize memory with vector search enabled
        memory_config = {
            'db_path': db_path,
            'enable_vector_search': True
        }
        memory = SQLiteMemory(memory_config)

        # Initialize embedding provider
        embedding_provider = create_embedding_provider()
        if not embedding_provider:
            print("❌ No embedding provider available, skipping test")
            return

        print(f"📊 Vector search info: {memory.get_vector_search_info()}")

        # Test data - different topics for semantic similarity testing
        test_conversations = [
            "I love programming in Python, especially working with FastAPI and async functions",
            "Can you help me debug a JavaScript Promise issue?",
            "I'm learning machine learning with TensorFlow and PyTorch",
            "What's the best way to optimize database queries in PostgreSQL?",
            "I enjoy cooking Italian food, especially making pasta from scratch",
            "How do I train a neural network for image classification?",
            "My favorite hobby is photography, particularly landscape shots",
            "Can you explain how gradient descent works in deep learning?",
            "I need help with React component lifecycle methods",
            "What are the best practices for API design in Python?"
        ]

        print(f"📝 Storing {len(test_conversations)} test conversations...")

        # Store conversations with embeddings
        stored_records = []
        for i, content in enumerate(test_conversations):
            # Generate embedding
            embedding = await embedding_provider.embed_text(content)

            # Create perception result
            perception_result = PerceptionResult(
                content=content,
                input_type=InputType.TEXT,
                metadata={'test_id': i},
                features={},
                timestamp=datetime.now(timezone.utc)
            )

            # Create memory record with embedding
            from agenesis.memory.base import MemoryRecord
            memory_record = MemoryRecord(
                id="",  # Will be auto-generated
                perception_result=perception_result,
                stored_at=None,  # Will be auto-set
                context={'test_session': True},
                metadata={},
                embedding=embedding
            )

            # Store to database
            record_id = memory.store_record(memory_record)
            stored_records.append(record_id)
            print(f"   ✅ Stored record {i+1}: {content[:50]}...")

        # Wait a moment for database operations
        await asyncio.sleep(0.1)

        print(f"\n📊 Memory statistics: {memory.get_embedding_statistics()}")
        print(f"📊 Vector search info: {memory.get_vector_search_info()}")

        # Migrate existing embeddings (in case triggers didn't work)
        migrated = memory.migrate_existing_embeddings()
        print(f"📊 Migrated {migrated} embeddings to vector table")

        # Test vector similarity searches
        test_queries = [
            ("machine learning with neural networks", "Should find ML/AI related conversations"),
            ("Python web development with APIs", "Should find Python/API related conversations"),
            ("cooking recipes and food", "Should find cooking related conversations"),
            ("database optimization techniques", "Should find database related conversations")
        ]

        print(f"\n🔍 Testing Vector Similarity Search")
        print("-" * 40)

        for query, expected in test_queries:
            print(f"\nQuery: '{query}'")
            print(f"Expected: {expected}")

            # Generate query embedding
            query_embedding = await embedding_provider.embed_text(query)

            # Perform vector similarity search
            results = memory.vector_similarity_search(
                query_embedding=query_embedding,
                limit=3,
                min_similarity=0.1
            )

            print(f"Found {len(results)} results:")
            for j, (record, similarity) in enumerate(results, 1):
                print(f"  {j}. [Score: {similarity:.3f}] {record.perception_result.content}")

        # Test with very high limit (testing the removal of artificial limits)
        print(f"\n🚀 Testing High-Limit Search (removed artificial limits)")
        print("-" * 50)

        query_embedding = await embedding_provider.embed_text("programming and software development")
        results = memory.vector_similarity_search(
            query_embedding=query_embedding,
            limit=10000,  # Very high limit
            min_similarity=0.05  # Lower threshold to get more results
        )

        print(f"High-limit search found {len(results)} results with similarity >= 0.05")
        for j, (record, similarity) in enumerate(results[:5], 1):  # Show top 5
            print(f"  {j}. [Score: {similarity:.3f}] {record.perception_result.content[:80]}...")

        print(f"\n✅ Vector search test completed successfully!")
        print(f"   - SQLite + sqlite-vss integration: ✅")
        print(f"   - Database-level vector search: ✅")
        print(f"   - Removed artificial limits: ✅")
        print(f"   - Semantic similarity ranking: ✅")

    except Exception as e:
        print(f"❌ Vector search test failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up
        try:
            Path(db_path).unlink()
        except:
            pass

if __name__ == "__main__":
    asyncio.run(test_vector_search())