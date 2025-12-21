"""
Embedding module using Gemini Embedding 001 via OpenRouter API.

This module provides embedding functionality for the RAG system using Google's
Gemini Embedding model accessed through OpenRouter's API.

Key features:
- Uses google/gemini-embedding-001 model via OpenRouter
- Truncates to 768 dimensions (Matryoshka) for pgvector compatibility
- Uses task_type: RETRIEVAL_DOCUMENT for saving, RETRIEVAL_QUERY for searching
- Includes retry logic for API rate limits
"""

import os
import logging
import time
from typing import List
import numpy as np
import requests

logger = logging.getLogger(__name__)

# Configuration
OPENROUTER_API_URL = 'https://openrouter.ai/api/v1/embeddings'
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'google/gemini-embedding-001')
EMBEDDING_DIMENSIONS = int(os.environ.get('EMBEDDING_DIMENSIONS', '768'))

# Retry configuration
MAX_RETRIES = 5
INITIAL_BACKOFF = 2  # seconds


class EmbeddingError(Exception):
    """Custom exception for embedding-related errors."""
    pass


def _get_headers() -> dict:
    """Get headers for OpenRouter API requests."""
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        raise EmbeddingError("OPENROUTER_API_KEY environment variable not set")

    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://chat-assist.bitcoinjungle.app",
        "X-Title": "Bitcoin Beatriz RAG"
    }


def truncate_embedding(embedding: List[float], dimensions: int = EMBEDDING_DIMENSIONS) -> List[float]:
    """
    Truncate embedding to specified dimensions (Matryoshka).

    Gemini embeddings support Matryoshka representation learning,
    allowing truncation to smaller dimensions while preserving quality.
    """
    if len(embedding) <= dimensions:
        return embedding
    return embedding[:dimensions]


def _call_embedding_api(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
    """
    Call the OpenRouter embedding API with retry logic.

    Args:
        text: Text to embed
        task_type: Either "RETRIEVAL_DOCUMENT" or "RETRIEVAL_QUERY"

    Returns:
        List of floats representing the embedding
    """
    result = _call_embedding_api_batch([text], task_type)
    return result[0]


def _call_embedding_api_batch(texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
    """
    Call the OpenRouter embedding API with batch support and retry logic.

    Args:
        texts: List of texts to embed
        task_type: Either "RETRIEVAL_DOCUMENT" or "RETRIEVAL_QUERY"

    Returns:
        List of embeddings (each a list of floats)
    """
    headers = _get_headers()
    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts,
        "dimensions": EMBEDDING_DIMENSIONS,
        "task_type": task_type
    }

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                OPENROUTER_API_URL,
                json=payload,
                headers=headers,
                timeout=60
            )

            # Handle rate limiting
            if response.status_code == 429:
                retry_after = response.headers.get('Retry-After')
                if retry_after:
                    wait_time = int(retry_after)
                else:
                    wait_time = INITIAL_BACKOFF * (2 ** attempt)
                logger.warning(f"Rate limit hit, waiting {wait_time}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            data = response.json()

            # Sort by index to ensure correct order
            embeddings_data = sorted(data['data'], key=lambda x: x['index'])
            return [truncate_embedding(e['embedding'], EMBEDDING_DIMENSIONS) for e in embeddings_data]

        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                wait_time = INITIAL_BACKOFF * (2 ** attempt)
                logger.warning(f"Request failed: {e}. Retrying in {wait_time}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait_time)
            continue

    raise EmbeddingError(f"Embedding API error after {MAX_RETRIES} attempts: {last_error}")


def compute_embedding_for_document(text: str) -> np.ndarray:
    """
    Compute embedding for a document (saving to database).

    Uses task_type: RETRIEVAL_DOCUMENT which optimizes the embedding
    for being retrieved by a search query.

    Args:
        text: Text content to embed

    Returns:
        numpy array of shape (768,) containing the embedding
    """
    if text is None:
        text = ""

    if not text.strip():
        logger.warning("Empty text provided for embedding, returning zero vector")
        return np.zeros(EMBEDDING_DIMENSIONS, dtype=np.float32)

    embedding = _call_embedding_api(text, task_type="RETRIEVAL_DOCUMENT")
    return np.array(embedding, dtype=np.float32)


def compute_embedding_for_query(text: str) -> np.ndarray:
    """
    Compute embedding for a search query.

    Uses task_type: RETRIEVAL_QUERY which optimizes the embedding
    for finding relevant documents.

    Args:
        text: Query text to embed

    Returns:
        numpy array of shape (768,) containing the embedding
    """
    if text is None:
        text = ""

    if not text.strip():
        logger.warning("Empty query provided for embedding, returning zero vector")
        return np.zeros(EMBEDDING_DIMENSIONS, dtype=np.float32)

    embedding = _call_embedding_api(text, task_type="RETRIEVAL_QUERY")
    return np.array(embedding, dtype=np.float32)


def compute_embeddings_batch(
    texts: List[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    batch_size: int = 50
) -> List[np.ndarray]:
    """
    Compute embeddings for a batch of texts using batch API calls.

    Args:
        texts: List of texts to embed
        task_type: Either "RETRIEVAL_DOCUMENT" or "RETRIEVAL_QUERY"
        batch_size: Number of texts per API call (default 50)

    Returns:
        List of numpy arrays, each of shape (768,)
    """
    embeddings = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        # Filter out empty texts, keep track of indices
        non_empty = [(j, t) for j, t in enumerate(batch) if t and t.strip()]

        if non_empty:
            indices, batch_texts = zip(*non_empty)
            batch_embeddings = _call_embedding_api_batch(list(batch_texts), task_type)

            # Build result with zeros for empty texts
            result = [np.zeros(EMBEDDING_DIMENSIONS, dtype=np.float32)] * len(batch)
            for idx, emb in zip(indices, batch_embeddings):
                result[idx] = np.array(emb, dtype=np.float32)
            embeddings.extend(result)
        else:
            embeddings.extend([np.zeros(EMBEDDING_DIMENSIONS, dtype=np.float32)] * len(batch))

        logger.info(f"Processed {min(i + batch_size, total)}/{total} embeddings")

    return embeddings


# For backward compatibility - expose the dimension constant
VECTOR_DIMENSIONS = EMBEDDING_DIMENSIONS
