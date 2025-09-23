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
        """Find relevant memories using semantic similarity with OpenAI embeddings"""

        if not self.embedding_provider:
            # Fallback to keyword matching
            return self._find_relevant_memories(working_memory, user_input)

        try:
            # Generate embedding for current user input
            query_embedding = await self.embedding_provider.embed_text(user_input)

            # Collect candidate memories from working memory and persistent memory
            candidate_memories = []

            # Get working memory records (recent conversation)
            working_records = working_memory.get_recent(20)  # Get more for better search
            for record in working_records:
                candidate_memories.append({
                    'id': record.id,
                    'content': record.perception_result.content,
                    'embedding': record.embedding,
                    'source': 'working',
                    'record': record
                })

            # Get persistent memory records if available and enabled
            if persistent_memory and self.include_persistent_memory:
                persistent_records = persistent_memory.get_recent(50)  # Get more historical context
                for record in persistent_records:
                    candidate_memories.append({
                        'id': record.id,
                        'content': record.perception_result.content,
                        'embedding': record.embedding,
                        'source': 'persistent',
                        'record': record
                    })

            # All memories should already have embeddings (embedded before storage)
            # Filter out any without embeddings (edge case for old data)
            memories_with_embeddings = [
                mem for mem in candidate_memories
                if mem['embedding'] is not None and len(mem['embedding']) > 0
            ]

            if len(memories_with_embeddings) < len(candidate_memories):
                missing_count = len(candidate_memories) - len(memories_with_embeddings)
                print(f"⚠️ Skipping {missing_count} memories without embeddings (old data)")

            # Find most similar memories
            embeddings = [mem['embedding'] for mem in memories_with_embeddings]

            if not embeddings:
                # No embeddings available, fallback to keyword matching
                return self._find_relevant_memories(working_memory, user_input)

            # Get similarity scores
            similar_indices = EmbeddingUtils.find_most_similar(
                query_embedding=query_embedding,
                candidate_embeddings=embeddings,
                top_k=self.max_relevant_memories,
                min_similarity=self.min_similarity_threshold
            )

            # Convert to memory IDs with priority weighting
            relevant_memory_ids = []
            for embedding_idx, similarity_score in similar_indices:
                memory = memories_with_embeddings[embedding_idx]

                # Apply reliability multiplier for evolved knowledge
                final_score = similarity_score * memory['record'].reliability_multiplier

                # Prioritize working memory over persistent memory for recency
                if memory['source'] == 'working':
                    final_score *= 1.2  # 20% boost for recent context

                relevant_memory_ids.append((memory['id'], final_score))

            # Sort by final score and return just the IDs
            relevant_memory_ids.sort(key=lambda x: x[1], reverse=True)
            return [memory_id for memory_id, score in relevant_memory_ids]

        except Exception as e:
            print(f"Semantic memory retrieval failed: {e}")
            # Fallback to keyword matching
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