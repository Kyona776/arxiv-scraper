import hashlib
import json
from pathlib import Path

import numpy as np


class EmbeddingCache:
    """
    Manages caching of paper embeddings to disk to avoid re-computation.

    This class provides a simple file-based cache for storing and retrieving
    NumPy array embeddings. Embeddings are keyed by their arXiv ID.
    The cache is stored in a structured directory to avoid having too many
    files in a single folder.
    """

    def __init__(self, cache_dir: Path | str = "cache/embeddings"):
        """
        Initializes the EmbeddingCache.

        Args:
            cache_dir: The root directory where embeddings will be stored.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, arxiv_id: str) -> Path:
        """
        Generates a structured path for a given arXiv ID.

        Example: '2301.12345' -> cache_dir/2301/12345.npy

        Args:
            arxiv_id: The arXiv ID of the paper.

        Returns:
            The full path to the cache file.
        """
        parts = arxiv_id.split(".")
        if len(parts) != 2:
            # Handle cases where arxiv_id might not be in the expected format
            # by using a hash for the directory structure.
            hashed_id = hashlib.sha1(arxiv_id.encode()).hexdigest()
            dir_name = hashed_id[:2]
            file_name = hashed_id[2:]
        else:
            dir_name = parts[0]
            file_name = parts[1]

        cache_subdir = self.cache_dir / dir_name
        cache_subdir.mkdir(exist_ok=True)
        return cache_subdir / f"{file_name}.npy"

    def save_embedding(self, arxiv_id: str, vector: np.ndarray) -> None:
        """
        Saves an embedding vector to the cache.

        Args:
            arxiv_id: The arXiv ID to use as the cache key.
            vector: The NumPy array of the embedding to save.
        """
        cache_path = self._get_cache_path(arxiv_id)
        np.save(cache_path, vector)

    def load_embedding(self, arxiv_id: str) -> np.ndarray | None:
        """
        Loads an embedding vector from the cache.

        Args:
            arxiv_id: The arXiv ID of the embedding to load.

        Returns:
            The loaded NumPy array, or None if the cache entry does not exist.
        """
        cache_path = self._get_cache_path(arxiv_id)
        if cache_path.exists():
            return np.load(cache_path)
        return None

    def cache_exists(self, arxiv_id: str) -> bool:
        """
        Checks if an embedding for the given arXiv ID exists in the cache.

        Args:
            arxiv_id: The arXiv ID to check.

        Returns:
            True if the cache entry exists, False otherwise.
        """
        cache_path = self._get_cache_path(arxiv_id)
        return cache_path.exists()


if __name__ == "__main__":
    # Example usage and self-test
    cache = EmbeddingCache(cache_dir="cache/test_embeddings")

    # 1. Create dummy data
    test_id_1 = "2301.12345"
    test_vector_1 = np.random.rand(768)

    test_id_2 = "hep-th/9901001"
    test_vector_2 = np.random.rand(768)

    # 2. Test saving
    print(f"Saving embedding for {test_id_1}...")
    cache.save_embedding(test_id_1, test_vector_1)
    print(f"Saving embedding for {test_id_2}...")
    cache.save_embedding(test_id_2, test_vector_2)

    # 3. Test existence check
    assert cache.cache_exists(test_id_1)
    assert cache.cache_exists(test_id_2)
    assert not cache.cache_exists("0000.00000")
    print("Cache existence checks passed.")

    # 4. Test loading
    print(f"Loading embedding for {test_id_1}...")
    loaded_vector_1 = cache.load_embedding(test_id_1)
    assert loaded_vector_1 is not None
    assert np.array_equal(test_vector_1, loaded_vector_1)
    print("Loading and data integrity for test_id_1 passed.")

    print(f"Loading embedding for {test_id_2}...")
    loaded_vector_2 = cache.load_embedding(test_id_2)
    assert loaded_vector_2 is not None
    assert np.array_equal(test_vector_2, loaded_vector_2)
    print("Loading and data integrity for test_id_2 passed.")

    # 5. Test loading non-existent
    non_existent = cache.load_embedding("0000.00000")
    assert non_existent is None
    print("Loading non-existent key test passed.")

    print("\nAll tests passed!")
    print(f"Cache location: {cache.cache_dir.resolve()}")
