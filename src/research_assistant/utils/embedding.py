"""
This utility provides functions for generating text embeddings using liteLLM
to connect with various embedding providers like Gemini.
"""

from __future__ import annotations
import logging
import os
from litellm import embedding
from typing import List

logger = logging.getLogger(__name__)

# Ensure the necessary API key is set in the environment
if "GEMINI_API_KEY" not in os.environ:
    logger.warning(
        "GEMINI_API_KEY environment variable not set. Embedding calls may fail."
    )


class EmbeddingError(Exception):
    """Custom exception for embedding-related errors."""

    pass


def get_embedding(text: str, model_name: str = "gemini/embedding-001") -> list[float]:
    """
    Generates an embedding for a single text string using liteLLM.

    Args:
        text: The input text to embed.
        model_name: The name of the embedding model to use (e.g., "gemini/embedding-001").

    Returns:
        A list of floats representing the embedding.

    Raises:
        EmbeddingError: If the embedding generation fails.
    """
    try:
        logger.debug(f"Generating embedding with model: {model_name}")
        response = embedding(model=model_name, input=[text])
        # The response object contains a list of embedding data objects.
        # We take the first one and get its 'embedding' attribute.
        embedding_vector = response.data[0]["embedding"]
        return embedding_vector
    except Exception as e:
        logger.error(f"Error generating embedding for text with liteLLM: {e}")
        raise EmbeddingError(f"liteLLM embedding generation failed: {e}") from e


def batch_get_embeddings(
    texts: list[str], model_name: str = "gemini/embedding-001", batch_size: int = 99
) -> list[list[float]]:
    """
    Generates embeddings for a batch of text strings using liteLLM,
    respecting the provider's batch size limits.

    Args:
        texts: A list of text strings to embed.
        model_name: The name of the embedding model to use.
        batch_size: The number of texts to process in a single API call.

    Returns:
        A list of embeddings, where each embedding is a list of floats.

    Raises:
        EmbeddingError: If the batch embedding generation fails.
    """
    if not texts:
        return []

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            logger.info(
                f"Generating batch embeddings for {len(batch)} texts with model: {model_name}"
            )
            response = embedding(model=model_name, input=batch)
            # Extract the embedding vector from each data object in the response
            embedding_vectors = [data["embedding"] for data in response.data]
            all_embeddings.extend(embedding_vectors)
        except Exception as e:
            logger.error(f"Error generating batch embeddings with liteLLM: {e}")
            raise EmbeddingError(
                f"liteLLM batch embedding generation failed: {e}"
            ) from e

    return all_embeddings
