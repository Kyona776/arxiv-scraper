import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from typing import cast

# Adjust the import path based on your project structure
from src.research_assistant.nodes import EmbeddingFilterNode
from src.research_assistant.utils.arxiv_search import ArxivPaper


class TestEmbeddingFilterNode(unittest.TestCase):
    def setUp(self):
        """Set up common test data."""
        self.node = EmbeddingFilterNode()
        # Create a list of ArxivPaper objects with all required fields
        self.papers = [
            ArxivPaper(
                title=f"Paper {i}",
                abstract=f"Abstract {i}",
                arxiv_id=f"id{i}",
                authors=[f"Author {i}"],
                published_date="2024-01-01",
                categories=["cs.AI"],
                pdf_url=f"http://arxiv.org/pdf/id{i}",
                abstract_url=f"http://arxiv.org/abs/id{i}",
            )
            for i in range(10)
        ]
        # Create a sample DataFrame with unsorted scores
        self.scores = np.array([0.1, 0.9, 0.3, 0.7, 0.5, 0.6, 0.4, 0.8, 0.2, 0.0])
        self.df = pd.DataFrame([p.to_dict() for p in self.papers])
        self.df["embedding_score"] = self.scores

    def test_top_n_selection_with_partition(self):
        """Test if the top N items are correctly selected and sorted."""
        shared_state: dict[str, dict[str, int | pd.DataFrame]] = {
            "query": {"top_n_filter": 3},
        }

        self.node.post(shared_state, None, self.df)
        result_df = shared_state["filtered_df"]

        # 1. Check if the length is correct
        self.assertEqual(len(result_df), 3)

        # 2. Check if the scores are the top 3 scores
        expected_top_scores = [0.9, 0.8, 0.7]
        actual_scores = cast(pd.Series, result_df["embedding_score"]).tolist()
        self.assertListEqual(actual_scores, expected_top_scores)

        # 3. Check if the IDs match the top scores
        expected_ids = ["id1", "id7", "id3"]
        actual_ids = cast(pd.Series, result_df["arxiv_id"]).tolist()
        self.assertListEqual(actual_ids, expected_ids)

    def test_top_n_selection_when_n_is_large(self):
        """Test the case where top_n is greater than or equal to the number of items."""
        shared_state: dict[str, dict[str, int | pd.DataFrame]] = {
            "query": {"top_n_filter": 15},  # n > number of papers
        }

        self.node.post(shared_state, None, self.df)
        result_df = shared_state["filtered_df"]

        # 1. Check if the length is correct (should be all papers)
        self.assertEqual(len(result_df), 10)

        # 2. Check if the result is sorted correctly
        expected_sorted_scores = sorted(self.scores, reverse=True)
        actual_scores = cast(pd.Series, result_df["embedding_score"]).tolist()
        self.assertListEqual(actual_scores, list(expected_sorted_scores))


class TestEmbeddingFilterNodeExec(unittest.TestCase):
    def setUp(self):
        """Set up common test data."""
        self.node = EmbeddingFilterNode()
        self.papers = [
            ArxivPaper(
                arxiv_id=f"id{i}",
                title=f"Title {i}",
                authors=[f"Author {i}"],
                abstract=f"Abstract {i}",
                published_date="2024-01-01",
                categories=["cs.AI"],
                pdf_url=f"http://arxiv.org/pdf/id{i}",
                abstract_url=f"http://arxiv.org/abs/id{i}",
            )
            for i in range(3)
        ]
        self.query_text = "test query"
        self.prep_data = (self.query_text, self.papers)

    @patch("src.research_assistant.nodes.batch_get_embeddings")
    @patch("src.research_assistant.nodes.EmbeddingCache")
    def test_exec_all_cache_miss(self, MockEmbeddingCache, mock_batch_get_embeddings):
        """Test the scenario where no embeddings are in the cache."""
        mock_cache_instance = MockEmbeddingCache.return_value
        mock_cache_instance.load_embedding.return_value = None

        query_embedding = [[0.1] * 8]
        # Simulate batching by providing a list of lists, where each inner list is a batch result
        paper_embeddings_batches = [[[0.2] * 8], [[0.3] * 8], [[0.4] * 8]]

        # side_effect will yield one item per call
        mock_batch_get_embeddings.side_effect = [
            query_embedding
        ] + paper_embeddings_batches

        # To test chunking, we need to mock load_config
        with patch(
            "src.research_assistant.nodes.load_config",
            return_value={"research_assistant": {"embedding_batch_size": 1}},
        ):
            result_df = self.node.exec(self.prep_data)

        self.assertEqual(mock_cache_instance.load_embedding.call_count, 3)
        # 1 call for query + 3 calls for each paper in its own chunk
        self.assertEqual(mock_batch_get_embeddings.call_count, 4)
        self.assertEqual(mock_cache_instance.save_embedding.call_count, 3)
        saved_arg = mock_cache_instance.save_embedding.call_args_list[0].args[1]
        self.assertIsInstance(saved_arg, np.ndarray)
        self.assertEqual(len(result_df), 3)
        self.assertTrue("embedding_score" in result_df.columns)

    @patch("src.research_assistant.nodes.batch_get_embeddings")
    @patch("src.research_assistant.nodes.EmbeddingCache")
    def test_exec_all_cache_hit(self, MockEmbeddingCache, mock_batch_get_embeddings):
        """Test the scenario where all embeddings are found in the cache."""
        mock_cache_instance = MockEmbeddingCache.return_value
        cached_embeddings = [np.random.rand(8) for _ in range(3)]
        mock_cache_instance.load_embedding.side_effect = cached_embeddings

        query_embedding = [[0.1] * 8]
        mock_batch_get_embeddings.return_value = query_embedding

        result_df = self.node.exec(self.prep_data)

        self.assertEqual(mock_cache_instance.load_embedding.call_count, 3)
        self.assertEqual(mock_batch_get_embeddings.call_count, 1)
        mock_cache_instance.save_embedding.assert_not_called()
        self.assertEqual(len(result_df), 3)

    @patch("src.research_assistant.nodes.batch_get_embeddings")
    @patch("src.research_assistant.nodes.EmbeddingCache")
    def test_exec_partial_cache_hit(
        self, MockEmbeddingCache, mock_batch_get_embeddings
    ):
        """Test a mix of cached and non-cached embeddings."""
        mock_cache_instance = MockEmbeddingCache.return_value
        cached_embedding = np.random.rand(8)
        mock_cache_instance.load_embedding.side_effect = [cached_embedding, None, None]

        query_embedding = [[0.1] * 8]
        # Only the 2 missing papers will be chunked
        paper_embeddings_batches = [[[0.2] * 8], [[0.3] * 8]]
        mock_batch_get_embeddings.side_effect = [
            query_embedding
        ] + paper_embeddings_batches

        with patch(
            "src.research_assistant.nodes.load_config",
            return_value={"research_assistant": {"embedding_batch_size": 1}},
        ):
            result_df = self.node.exec(self.prep_data)

        self.assertEqual(mock_cache_instance.load_embedding.call_count, 3)
        # 1 for query + 2 for the two missing papers
        self.assertEqual(mock_batch_get_embeddings.call_count, 3)
        self.assertEqual(mock_cache_instance.save_embedding.call_count, 2)
        self.assertEqual(len(result_df), 3)


if __name__ == "__main__":
    unittest.main()
