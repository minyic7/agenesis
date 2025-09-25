"""
Semantic Cognition Module with OpenAI Embedding-based Memory Retrieval
"""
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from .basic import BasicCognition
from ..providers import create_embedding_provider, EmbeddingUtils


class SemanticCognition(BasicCognition):
    """Enhanced cognition with semantic memory retrieval using OpenAI embeddings"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        config = config or {}

        # Initialize embedding provider
        self.embedding_provider = create_embedding_provider(config.get('embedding', {}))
        self.use_semantic_search = self.embedding_provider is not None

        # Semantic search configuration
        self.max_relevant_memories = config.get('max_relevant_memories', 5)
        self.min_similarity_threshold = config.get('min_similarity_threshold', 0.1)
        self.include_persistent_memory = config.get('include_persistent_memory', True)

        if not self.use_semantic_search:
            print("Warning: No embedding provider available. Falling back to keyword matching.")

    async def process(self, immediate_memory, working_memory, context: Any = None, persistent_memory=None) -> 'CognitionResult':
        """Enhanced processing with semantic memory retrieval"""
        user_input = self._get_user_input(immediate_memory)

        if not user_input:
            return self._create_empty_result()

        recent_context = self._summarize_working_memory(working_memory)

        # Use semantic search if available, otherwise fall back to keyword matching
        if self.use_semantic_search:
            relevant_memories = await self._find_relevant_memories_semantic(
                working_memory, user_input, persistent_memory
            )
        else:
            relevant_memories = self._find_relevant_memories(working_memory, user_input)

        if self.use_llm:
            try:
                return await self._process_with_llm(user_input, recent_context, relevant_memories)
            except Exception as e:
                print(f"LLM processing failed, falling back to heuristics: {e}")
                return self._process_with_heuristics(user_input, recent_context, relevant_memories)
        else:
            return self._process_with_heuristics(user_input, recent_context, relevant_memories)

    async def _find_relevant_memories_semantic(
        self,
        working_memory,
        user_input: str,
        persistent_memory=None
    ) -> List[str]:
        """Find relevant memories using semantic similarity with database-level vector search"""

        if not self.embedding_provider:
            return self._find_relevant_memories(working_memory, user_input)

        try:
            # Generate embedding for current user input
            query_embedding = await self.embedding_provider.embed_text(user_input)

            relevant_memory_ids = []

            # Search working memory (in-memory for recent conversations)
            working_records = working_memory.get_recent(1000)  # Increased from 100 for better performance
            working_memories_with_embeddings = [
                {
                    'id': record.id,
                    'embedding': record.embedding,
                    'source': 'working'
                }
                for record in working_records
                if record.embedding is not None and len(record.embedding) > 0
            ]

            if working_memories_with_embeddings:
                # In-memory search for working memory (small dataset)
                embeddings = [mem['embedding'] for mem in working_memories_with_embeddings]
                similar_indices = EmbeddingUtils.find_most_similar(
                    query_embedding=query_embedding,
                    candidate_embeddings=embeddings,
                    top_k=self.max_relevant_memories // 2,  # Split quota between working and persistent
                    min_similarity=self.min_similarity_threshold
                )

                for embedding_idx, similarity_score in similar_indices:
                    memory = working_memories_with_embeddings[embedding_idx]
                    final_score = similarity_score * 1.2  # 20% boost for recent context
                    relevant_memory_ids.append((memory['id'], final_score))

            # Search persistent memory using database-level vector search
            if persistent_memory and self.include_persistent_memory:
                # Check if persistent memory supports vector search
                if hasattr(persistent_memory, 'vector_similarity_search'):
                    try:
                        # Use database-level vector search - NO LIMIT!
                        persistent_results = persistent_memory.vector_similarity_search(
                            query_embedding=query_embedding,
                            limit=10000,  # Very high limit - let similarity threshold filter
                            min_similarity=self.min_similarity_threshold
                        )

                        # Add persistent memory results
                        for memory_record, similarity_score in persistent_results:
                            relevant_memory_ids.append((memory_record.id, similarity_score))

                        print(f"🔍 Found {len(persistent_results)} relevant memories from persistent storage")

                    except Exception as e:
                        print(f"⚠️ Database vector search failed, falling back to recent records: {e}")
                        # Fallback to recent records if vector search fails
                        persistent_records = persistent_memory.get_recent(1000)  # Increased from 100 for better performance
                        for record in persistent_records:
                            if record.embedding is not None:
                                relevant_memory_ids.append((record.id, 0.5))  # Default similarity score

                else:
                    print("⚠️ Persistent memory doesn't support vector search, using recent records")
                    # Fallback for persistent memory without vector search
                    persistent_records = persistent_memory.get_recent(1000)  # Increased from 100 for better performance
                    for record in persistent_records:
                        relevant_memory_ids.append((record.id, 0.3))  # Lower default score

            # Sort by final score and return just the IDs
            relevant_memory_ids.sort(key=lambda x: x[1], reverse=True)

            # Limit to max_relevant_memories for final results
            final_ids = [memory_id for memory_id, score in relevant_memory_ids[:self.max_relevant_memories]]

            print(f"📋 Selected {len(final_ids)} most relevant memories (from {len(relevant_memory_ids)} candidates)")
            return final_ids

        except Exception as e:
            print(f"Semantic memory retrieval failed: {e}")
            return self._find_relevant_memories(working_memory, user_input)


    def get_semantic_search_info(self) -> Dict[str, Any]:
        """Get information about semantic search capabilities"""
        return {
            'semantic_search_enabled': self.use_semantic_search,
            'embedding_provider': type(self.embedding_provider).__name__ if self.embedding_provider else None,
            'embedding_dimension': self.embedding_provider.get_dimension() if self.embedding_provider else None,
            'max_relevant_memories': self.max_relevant_memories,
            'min_similarity_threshold': self.min_similarity_threshold,
            'include_persistent_memory': self.include_persistent_memory
        }