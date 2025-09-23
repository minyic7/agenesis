"""
OpenAI Embedding Provider for Semantic Search
"""
import asyncio
import math
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import openai
import os
from openai import AsyncOpenAI


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of embeddings"""
        pass


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider using text-embedding-3-small"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model = self.config.get('model', 'text-embedding-3-small')
        self.max_retries = self.config.get('max_retries', 3)
        self.timeout = self.config.get('timeout', 30)

        # Initialize OpenAI client
        api_key = self.config.get('api_key') or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key is required for embedding provider")

        self.client = AsyncOpenAI(
            api_key=api_key,
            timeout=self.timeout,
            max_retries=self.max_retries
        )

        # Cache dimension based on model
        self._dimension = self._get_model_dimension()

    def _get_model_dimension(self) -> int:
        """Get embedding dimension for the model"""
        model_dimensions = {
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072,
            'text-embedding-ada-002': 1536
        }
        return model_dimensions.get(self.model, 1536)

    def get_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self._dimension

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if not text.strip():
            # Return zero vector for empty text
            return [0.0] * self.get_dimension()

        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=[text],
                encoding_format="float"
            )

            return response.data[0].embedding

        except Exception as e:
            print(f"OpenAI embedding failed for text: {text[:50]}... Error: {e}")
            # Return zero vector on failure
            return [0.0] * self.get_dimension()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (batch processing)"""
        if not texts:
            return []

        # Filter out empty texts and track indices
        non_empty_texts = []
        text_indices = []

        for i, text in enumerate(texts):
            if text.strip():
                non_empty_texts.append(text)
                text_indices.append(i)

        if not non_empty_texts:
            # All texts are empty, return zero vectors
            return [[0.0] * self.get_dimension()] * len(texts)

        try:
            # OpenAI supports batch embedding
            response = await self.client.embeddings.create(
                model=self.model,
                input=non_empty_texts,
                encoding_format="float"
            )

            # Create result array with zero vectors for empty texts
            results = [[0.0] * self.get_dimension()] * len(texts)

            # Fill in actual embeddings for non-empty texts
            for i, embedding_data in enumerate(response.data):
                original_index = text_indices[i]
                results[original_index] = embedding_data.embedding

            return results

        except Exception as e:
            print(f"OpenAI batch embedding failed for {len(texts)} texts. Error: {e}")
            # Return zero vectors for all texts on failure
            return [[0.0] * self.get_dimension()] * len(texts)


class EmbeddingUtils:
    """Utility functions for working with embeddings"""

    @staticmethod
    def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        if not embedding1 or not embedding2 or len(embedding1) != len(embedding2):
            return 0.0

        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))

        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in embedding1))
        magnitude2 = math.sqrt(sum(b * b for b in embedding2))

        # Handle zero vectors
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        # Calculate cosine similarity
        return dot_product / (magnitude1 * magnitude2)

    @staticmethod
    def find_most_similar(
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> List[tuple]:
        """
        Find most similar embeddings to query

        Returns:
            List of (index, similarity_score) tuples, sorted by similarity (highest first)
        """
        if not query_embedding or not candidate_embeddings:
            return []

        similarities = []
        for i, candidate in enumerate(candidate_embeddings):
            similarity = EmbeddingUtils.cosine_similarity(query_embedding, candidate)
            if similarity >= min_similarity:
                similarities.append((i, similarity))

        # Sort by similarity (highest first) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


def create_embedding_provider(config: Optional[Dict[str, Any]] = None) -> Optional[BaseEmbeddingProvider]:
    """Factory function to create embedding provider"""
    config = config or {}

    # Try OpenAI first
    if os.getenv('OPENAI_API_KEY') or config.get('api_key'):
        try:
            return OpenAIEmbeddingProvider(config)
        except Exception as e:
            print(f"Failed to create OpenAI embedding provider: {e}")

    # No provider available
    print("Warning: No embedding provider available. Set OPENAI_API_KEY for semantic search.")
    return None