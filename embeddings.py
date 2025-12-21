"""
Embedding module using Gemini Embedding 001 via OpenRouter API.

This module provides embedding functionality for the RAG system using Google's
Gemini Embedding model accessed through OpenRouter's API (OpenAI SDK compatible).

Key features:
- Uses google/gemini-embedding-001 model via OpenRouter
- Truncates to 768 dimensions (Matryoshka) for pgvector compatibility
- Uses task_type: RETRIEVAL_DOCUMENT for saving, RETRIEVAL_QUERY for searching
- Includes retry logic for API rate limits
"""

import os
import logging
from typing import List
import numpy as np
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

logger = logging.getLogger(__name__)

# Configuration
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'google/gemini-embedding-001')
EMBEDDING_DIMENSIONS = int(os.environ.get('EMBEDDING_DIMENSIONS', '768'))


class EmbeddingError(Exception):
    """Custom exception for embedding-related errors."""
    pass


class RateLimitError(Exception):
    """Exception for API rate limit errors."""
    pass


def get_openrouter_client() -> OpenAI:
    """Create and return an OpenAI client configured for OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise EmbeddingError("OPENROUTER_API_KEY environment variable not set")

    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        default_headers={
            "HTTP-Referer": "https://chat-assist.bitcoinjungle.app",
            "X-Title": "Bitcoin Beatriz RAG"
        }
    )


def truncate_embedding(embedding: List[float], dimensions: int = EMBEDDING_DIMENSIONS) -> List[float]:
    """
    Truncate embedding to specified dimensions (Matryoshka).

    Gemini embeddings support Matryoshka representation learning,
    allowing truncation to smaller dimensions while preserving quality.
    """
    if len(embedding) <= dimensions:
        return embedding
    return embedding[:dimensions]


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type((RateLimitError, httpx.TimeoutException)),
    before_sleep=lambda retry_state: logger.warning(
        f"Retrying embedding request (attempt {retry_state.attempt_number})"
    )
)
def _call_embedding_api(
    client: OpenAI,
    text: str,
    task_type: str = "RETRIEVAL_DOCUMENT"
) -> List[float]:
    """
    Call the OpenRouter embedding API with retry logic.

    Args:
        client: OpenAI client configured for OpenRouter
        text: Text to embed
        task_type: Either "RETRIEVAL_DOCUMENT" or "RETRIEVAL_QUERY"

    Returns:
        List of floats representing the embedding
    """
    try:
        # OpenRouter uses the OpenAI API format
        # For Gemini embeddings, we pass task_type in the extra_body
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
            extra_body={
                "task_type": task_type,
                "dimensions": EMBEDDING_DIMENSIONS
            }
        )

        embedding = response.data[0].embedding
        return truncate_embedding(embedding, EMBEDDING_DIMENSIONS)

    except Exception as e:
        error_str = str(e).lower()
        if 'rate limit' in error_str or '429' in error_str:
            logger.warning(f"Rate limit hit: {e}")
            raise RateLimitError(str(e))
        raise EmbeddingError(f"Embedding API error: {e}")


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

    client = get_openrouter_client()
    embedding = _call_embedding_api(client, text, task_type="RETRIEVAL_DOCUMENT")
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

    client = get_openrouter_client()
    embedding = _call_embedding_api(client, text, task_type="RETRIEVAL_QUERY")
    return np.array(embedding, dtype=np.float32)


def compute_embedding(text: str, is_query: bool = False) -> np.ndarray:
    """
    Unified embedding function that handles both documents and queries.

    This is a drop-in replacement for the old SentenceTransformer-based
    compute_embedding function.

    Args:
        text: Text to embed
        is_query: If True, use RETRIEVAL_QUERY task type; else RETRIEVAL_DOCUMENT

    Returns:
        numpy array of shape (768,) containing the embedding
    """
    if is_query:
        return compute_embedding_for_query(text)
    return compute_embedding_for_document(text)


def compute_embeddings_batch(
    texts: List[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    batch_size: int = 100
) -> List[np.ndarray]:
    """
    Compute embeddings for a batch of texts.

    Note: OpenRouter may not support batch embeddings natively,
    so this processes texts one at a time with progress logging.

    Args:
        texts: List of texts to embed
        task_type: Either "RETRIEVAL_DOCUMENT" or "RETRIEVAL_QUERY"
        batch_size: Not used currently, kept for API compatibility

    Returns:
        List of numpy arrays, each of shape (768,)
    """
    embeddings = []
    total = len(texts)

    for i, text in enumerate(texts):
        if i > 0 and i % 10 == 0:
            logger.info(f"Processed {i}/{total} embeddings")

        if task_type == "RETRIEVAL_QUERY":
            embedding = compute_embedding_for_query(text)
        else:
            embedding = compute_embedding_for_document(text)
        embeddings.append(embedding)

    logger.info(f"Completed {total} embeddings")
    return embeddings


# For backward compatibility - expose the dimension constant
VECTOR_DIMENSIONS = EMBEDDING_DIMENSIONS
