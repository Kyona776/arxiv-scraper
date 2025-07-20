"""
This module defines the PocketFlow Nodes for the Research Assistant workflow.
Each node represents a distinct step in the data processing pipeline,
from initial query parsing to final result storage and interactive refinement.
"""

from __future__ import annotations
import logging
import pandas as pd
import numpy as np
from pocketflow import Node
from pathlib import Path
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import questionary
import json
import re
from datetime import datetime
from src.models.error import CallLLMError

from .utils import (
    arxiv_search,
    chunk_date_range,
    batch_get_embeddings,
    call_llm,
    create_project_directory,
    save_csv_version,
    generate_project_markdown,
    load_config,
)
from .utils.arxiv_search import ArxivPaper
from .utils.embedding_cache import EmbeddingCache
from .utils.llm_api import call_llm
from .utils.config_loader import load_config

logger = logging.getLogger(__name__)


# ==============================================================================
# Flow 0: Interactive Query Construction Nodes
# ==============================================================================


class InitialQueryNode(Node):
    """
    Captures the user's initial high-level research goal, including
    keywords and a semantic description, via an interactive prompt.
    """

    def exec(self, prep_res: Any) -> dict[str, Any]:  # type: ignore
        # This will be implemented with `questionary`
        logger.info("Prompting user for initial query...")

        query_text = questionary.text(
            "First, what are the core keywords for your research?",
            validate=lambda text: True
            if len(text) > 0
            else "Please enter at least one keyword.",
        ).ask()

        description = questionary.text(
            "Great. Now, could you briefly describe your research goal or what you're looking for?",
            validate=lambda text: True
            if len(text) > 10
            else "Please provide a bit more detail.",
        ).ask()

        # Handle case where user cancels (ctrl+c)
        if not query_text or not description:
            logger.warning("User cancelled the query input. Aborting.")
            # We can return a special action to stop the flow gracefully
            # For now, we'll let it proceed and fail downstream, or handle in post.
            return {}

        initial_query = {
            "query_text": query_text,
            "description": description,
        }
        return initial_query

    def post(self, shared: dict, prep_res: Any, exec_res: dict[str, Any]):  # type: ignore
        shared["initial_query"] = exec_res
        logger.info(f"Captured initial query: {exec_res}")
        return "default"


class InteractiveClarificationNode(Node):
    """
    Engages in a dialogue with the user to clarify intent, suggest
    keyword expansions, and define exclusion criteria.
    """

    def prep(self, shared: dict) -> dict[str, Any]:  # type: ignore
        return shared["initial_query"]

    def exec(self, initial_query: dict[str, Any]) -> dict[str, Any]:  # type: ignore
        logger.info("Clarifying user intent via interactive dialogue...")

        # 1. Generate suggestions with LLM
        config = load_config()
        model_name = (
            config.get("research_assistant", {})
            .get("default_model", {})
            .get("name", None)
        )
        provider = (
            config.get("research_assistant", {})
            .get("default_model", {})
            .get("provider", None)
        )

        prompt = self._create_clarification_prompt(initial_query)

        try:
            response = call_llm(
                prompt,
                model_name,
                response_format={"type": "json_object"},
                provider=provider,
            )
            content = (
                response.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            )
            suggestions = json.loads(content)
        except (json.JSONDecodeError, IndexError) as e:
            logger.error(f"Failed to parse LLM suggestions: {e}")
            suggestions = {}

        # 2. Interact with user using questionary
        intent = initial_query.copy()

        # Combine all questions into a single form
        answers = questionary.form(
            start_date=questionary.text(
                "Enter the start date for the search (YYYY-MM-DD):",
                validate=lambda text: True
                if re.match(r"^\d{4}-\d{2}-\d{2}$", text)
                else "Please use YYYY-MM-DD format.",
            ),
            end_date=questionary.text(
                "Enter the end date for the search (YYYY-MM-DD):",
                validate=lambda text: True
                if re.match(r"^\d{4}-\d{2}-\d{2}$", text)
                else "Please use YYYY-MM-DD format.",
            ),
            max_candidates=questionary.text(
                "What is the maximum number of papers to fetch initially? (e.g., 5000)",
                default="5000",
                validate=lambda text: text.isdigit() and int(text) > 0,
            ),
            top_n_filter=questionary.text(
                "How many papers should be analyzed deeply after initial filtering? (e.g., 100)",
                default="100",
                validate=lambda text: text.isdigit() and int(text) > 0,
            ),
            final_n=questionary.text(
                "How many papers should be in the final report after LLM analysis? (e.g., 10)",
                default="10",
                validate=lambda text: text.isdigit() and int(text) > 0,
            ),
            exclude_types=questionary.checkbox(
                "Should I exclude any of these paper types?",
                choices=suggestions.get("exclude_types", []),
            )
            if suggestions.get("exclude_types")
            else None,
            add_keywords=questionary.checkbox(
                "Should I add any of these related keywords to the search?",
                choices=suggestions.get("add_keywords", []),
            )
            if suggestions.get("add_keywords")
            else None,
        ).ask()

        if not answers:
            logger.warning("User cancelled the form. Aborting.")
            return {}

        # Update intent with answers, filtering out None values from cancelled optional prompts
        intent.update({k: v for k, v in answers.items() if v is not None})

        # Combine and deduplicate keywords
        initial_keywords_str = intent.get("query_text", "")
        initial_keywords_list = [
            k.strip() for k in initial_keywords_str.split(",") if k.strip()
        ]
        added_keywords = answers.get("add_keywords", [])

        # Combine, deduplicate using a set, and then sort for consistent ordering
        all_keywords = sorted(list(set(initial_keywords_list + added_keywords)))

        # Update the main query_text with the cleaned, combined list
        intent["query_text"] = ", ".join(all_keywords)

        # Type conversion for numeric fields
        intent["max_candidates"] = int(intent.get("max_candidates", 5000))
        intent["top_n_filter"] = int(intent.get("top_n_filter", 100))
        intent["final_n"] = int(intent.get("final_n", 10))

        # Rename 'add_keywords' to 'extra_keywords' for consistency in project metadata
        if "add_keywords" in intent:
            intent["extra_keywords"] = intent.pop("add_keywords")
        else:
            intent["extra_keywords"] = []

        logger.info(f"Finalized user intent: {intent}")
        return intent

    def _create_clarification_prompt(self, query: dict[str, str]) -> str:
        return f"""
        You are an AI research assistant. Your goal is to help a user refine their search query.
        Based on the user's initial input, suggest potential refinements.

        User's Keywords: "{query['query_text']}"
        User's Description: "{query['description']}"

        Please provide suggestions as a JSON object with two optional keys:
        1. "exclude_types": A list of common paper types to ask the user if they want to exclude (e.g., "review", "survey", "meta-analysis").
        2. "add_keywords": A list of 3-5 related keywords or concepts that could broaden the search.

        Example Output:
        {{
            "exclude_types": ["review", "survey"],
            "add_keywords": ["graph neural networks", "semantic retrieval", "vector databases"]
        }}

        JSON:
        """

    def post(self, shared: dict, prep_res: Any, exec_res: dict[str, Any]):  # type: ignore
        shared["finalized_intent"] = exec_res
        return "default"


class KeywordExtractionNode(Node):
    """
    Extracts granular 1-word and 2-word keywords from the user's description
    using an LLM to augment the keyword list.
    """

    def prep(self, shared: dict[str, Any]) -> dict[str, Any]:
        return shared["finalized_intent"]

    def exec(self, finalized_intent: dict[str, Any]) -> dict[str, Any]:  # type: ignore
        logger.info("Extracting granular keywords from description...")

        config = load_config()
        model_name = config["research_assistant"]["default_model"]["name"]
        provider = config["research_assistant"]["default_model"]["provider"]

        prompt = self._create_extraction_prompt(finalized_intent)

        schema = {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of 1-word and 2-word keywords.",
                }
            },
            "required": ["keywords"],
        }

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "keyword_extractor",
                "strict": True,
                "schema": schema,
            },
        }

        try:
            response = call_llm(
                prompt, model_name, response_format=response_format, provider=provider
            )
            data = json.loads(response["choices"][0]["message"]["content"])

            extracted_keywords = data.get("keywords", [])

            # Combine with existing keywords
            original_keywords = finalized_intent["query_text"].split(", ")

            all_keywords = set(
                k.strip().lower() for k in original_keywords if k.strip()
            )
            all_keywords.update(
                k.strip().lower() for k in extracted_keywords if k.strip()
            )

            # Update finalized_intent
            updated_intent = finalized_intent.copy()
            updated_intent["query_text"] = ", ".join(sorted(list(all_keywords)))

            logger.info(
                f"Extracted and combined keywords: {updated_intent['query_text']}"
            )
            return updated_intent

        except (json.JSONDecodeError, IndexError, CallLLMError, KeyError) as e:
            logger.error(
                f"Failed to extract keywords via LLM ({e}). Using original keywords only."
            )
            return finalized_intent

    def _create_extraction_prompt(self, intent: dict[str, Any]) -> str:
        description = intent["description"]
        keywords = intent["query_text"]
        return f"""
        You are a research assistant specializing in Natural Language Processing.
        Your task is to extract significant 1-word and 2-word (bigram) technical keywords from the user's research goal description.

        User's Goal Description: "{description}"
        User's Provided Keywords: "{keywords}"

        Instructions:
        1. Read the user's description and identify the most important technical terms and concepts.
        2. Extract these terms as single words or two-word phrases.
        3. Do not extract generic words. Focus on terms relevant for a literature search.
        4. Return a JSON object with a single key "keywords", containing a list of the extracted keyword strings.

        Example:
        Description: "I am looking for papers on using retrieval augmented generation with knowledge graphs for enterprise search."
        Provided Keywords: "RAG, enterprise search"
        Output:
        {{
          "keywords": ["retrieval augmented generation", "knowledge graphs", "enterprise search", "RAG"]
        }}

        JSON:
        """

    def post(self, shared: dict, prep_res: dict[str, Any], exec_res: dict[str, Any]):  # type: ignore
        # The exec_res is the updated finalized_intent
        shared["finalized_intent"] = exec_res
        return "default"


class KeywordExpansionNode(Node):
    """
    Expands a list of keywords by generating broader, more general concepts
    for each, using an LLM. This creates a wider net for the search.
    """

    def prep(self, shared: dict[str, Any]) -> dict[str, Any]:
        return shared["finalized_intent"]

    def exec(self, finalized_intent: dict[str, Any]) -> list[str]:  # type: ignore
        logger.info("Expanding keywords using LLM...")

        config = load_config()
        model_name = config["research_assistant"]["default_model"]["name"]
        provider = config["research_assistant"]["default_model"]["provider"]

        prompt = self._create_expansion_prompt(finalized_intent)

        schema = {
            "type": "object",
            "properties": {
                "expansions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "original": {"type": "string"},
                            "broader": {"type": "string"},
                        },
                        "required": ["original", "broader"],
                    },
                }
            },
            "required": ["expansions"],
        }

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "keyword_expander",
                "strict": True,
                "schema": schema,
            },
        }

        try:
            response = call_llm(
                prompt, model_name, response_format=response_format, provider=provider
            )
            data = json.loads(response["choices"][0]["message"]["content"])

            original_keywords = finalized_intent["query_text"].split(", ")
            expanded_keywords = {item["broader"] for item in data.get("expansions", [])}

            # Combine and deduplicate
            all_keywords = set(original_keywords)
            all_keywords.update(expanded_keywords)

            logger.info(f"Expanded keywords to {len(all_keywords)} unique terms.")
            return list(all_keywords)

        except (json.JSONDecodeError, IndexError, CallLLMError, KeyError) as e:
            logger.error(
                f"Failed to expand keywords via LLM ({e}). Using original keywords only."
            )
            # Fallback to original keywords if LLM fails
            return finalized_intent["query_text"].split(", ")

    def _create_expansion_prompt(self, intent: dict[str, Any]) -> str:
        keywords = intent["query_text"]
        return f"""
        You are a research domain expert. For each keyword provided by the user, identify one single, broader, more general parent concept.

        User Keywords: "{keywords}"
        User Goal: "{intent['description']}"

        Instructions:
        1. Analyze each keyword.
        2. Determine a more general parent term. (e.g., for "graphRAG", the broader term is "RAG"; for "multi-agent systems", it's "agentic systems").
        3. Return a JSON object with a key "expansions", containing a list of objects, each with "original" and "broader" keys.

        Example:
        {{
          "expansions": [
            {{ "original": "graphRAG", "broader": "RAG" }},
            {{ "original": "multi-agent systems", "broader": "agentic systems" }}
          ]
        }}

        JSON:
        """

    def post(self, shared: dict, prep_res: dict[str, Any], exec_res: list[str]):  # type: ignore
        shared["expanded_keywords"] = exec_res
        return "default"


# ==============================================================================
# Flow 1: Core Search and Filtering Nodes
# ==============================================================================


class ParseQueryNode(Node):
    """Parses the initial user query and parameters."""

    def prep(self, shared: dict[str, Any]) -> dict[str, Any]:  # type: ignore
        """Loads the raw query from the shared store."""
        logger.info("Preparing to parse initial query.")
        return shared.get("initial_query", {})

    def exec(self, raw_query: dict[str, Any]) -> dict[str, Any]:  # type: ignore
        """
        Validates and structures the raw query.

        Args:
            raw_query: Raw dictionary of query parameters from prep.

        Returns:
            A structured and validated query dictionary.
        """
        logger.info("Parsing initial user query.")
        # In a real implementation, add validation logic here (e.g., with Pydantic)
        # For now, we assume the query is valid.
        structured_query = {
            "text": raw_query.get("text", ""),
            "start_date": raw_query.get("start_date"),
            "end_date": raw_query.get("end_date"),
            "max_candidates": raw_query.get("max_candidates", 5000),
            "top_n_filter": raw_query.get("top_n_filter", 100),
        }
        if not all([structured_query["start_date"], structured_query["end_date"]]):
            raise ValueError("Start and end dates are required.")
        logger.debug(f"Parsed query: {structured_query}")
        return structured_query

    def post(self, shared: dict, prep_res: Any, exec_res: dict[str, Any]):  # type: ignore
        """Saves the structured query to the shared store."""
        shared["query"] = exec_res
        return "default"


class BatchArxivNode(Node):
    """
    Performs a parallelized search on the arXiv API using a list of
    individual keywords, including date and exclusion filters in the query.
    """

    def prep(self, shared: dict) -> dict:  # type: ignore
        """
        Prepares the keywords and other parameters for the parallel search.
        """
        finalized_intent = shared.get("finalized_intent", {})
        return {
            "keywords": shared.get("expanded_keywords", []),
            "max_candidates": finalized_intent.get("max_candidates", 1000),
            "start_date": finalized_intent.get("start_date"),
            "end_date": finalized_intent.get("end_date"),
            "exclude_types": finalized_intent.get("exclude_types", []),
        }

    def exec(self, prep_res: dict) -> List[ArxivPaper]:  # type: ignore
        """
        Executes a search for each keyword in parallel and fetches paper details.
        """
        keywords = prep_res["keywords"]
        max_candidates = prep_res["max_candidates"]
        start_date = prep_res["start_date"]
        end_date = prep_res["end_date"]
        exclude_types = prep_res["exclude_types"]

        if not keywords:
            logger.warning("No keywords to search. Skipping.")
            return []

        # Create exclusion string
        exclusion_str = ""
        if exclude_types:
            exclusion_parts = [f'all:"{ex}"' for ex in exclude_types]
            exclusion_str = f" ANDNOT ({' OR '.join(exclusion_parts)})"

        # Create date string
        date_query_part = ""
        if start_date and end_date:
            try:
                # Validate and format dates for arXiv query
                start_fmt = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d")
                end_fmt = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%d")
                date_query_part = f" AND submittedDate:[{start_fmt} TO {end_fmt}]"
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid start or end date provided ({start_date}, {end_date}). "
                    "Skipping date filtering in query."
                )

        # Limit max candidates per keyword to avoid overwhelming the API
        max_per_keyword = max(1, max_candidates // len(keywords))

        all_papers_dict: Dict[str, ArxivPaper] = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_keyword = {
                executor.submit(
                    arxiv_search,
                    f'all:"{kw.strip()}"{exclusion_str}{date_query_part}',
                    max_per_keyword,
                ): kw
                for kw in keywords
            }

            for future in as_completed(future_to_keyword):
                keyword = future_to_keyword[future]
                try:
                    papers = future.result()
                    for paper in papers:
                        if paper.arxiv_id not in all_papers_dict:
                            all_papers_dict[paper.arxiv_id] = paper
                    logger.info(f"Found {len(papers)} papers for keyword: '{keyword}'")
                except Exception as e:
                    logger.error(f"Error fetching papers for keyword '{keyword}': {e}")

        all_papers = list(all_papers_dict.values())
        logger.info(
            f"Aggregated {len(all_papers)} unique papers from {len(keywords)} keywords."
        )
        # Sort by published date as the parallel fetching might mess up the order
        all_papers.sort(key=lambda p: p.published_date, reverse=True)
        return all_papers

    def post(self, shared: dict, prep_res: Any, exec_res: list[ArxivPaper]):  # type: ignore
        """Saves the final candidate papers to the shared store."""
        shared["candidates"] = exec_res
        logger.info(
            f"Stored {len(exec_res)} unique papers as candidates for embedding filter."
        )
        return "default"


class EmbeddingFilterNode(Node):
    """
    Filters candidate papers based on embedding similarity between the
    user's query and the paper's title and abstract.
    """

    def prep(self, shared: dict) -> tuple[str, list[ArxivPaper]]:  # type: ignore
        """Prepares the query and papers for embedding."""
        # Direct access is preferred to reveal errors from missing keys.
        finalized_intent = shared["finalized_intent"]
        query_text = finalized_intent["description"]
        candidates = shared["candidates"]

        if not query_text:
            logger.warning("Query text description is empty.")
        if not candidates:
            logger.warning("No candidate papers found for embedding.")

        return query_text, candidates

    def exec(self, prep_data: tuple[str, list[ArxivPaper]]) -> pd.DataFrame:  # type: ignore
        """
        Computes embeddings, calculates similarity scores, and filters papers.
        """
        cache = EmbeddingCache()
        query_text, papers = prep_data
        if not papers:
            logger.warning("No papers to filter. Skipping embedding.")
            return pd.DataFrame()

        logger.info(f"Processing embeddings for {len(papers)} papers...")

        # --- Efficiently load from cache and identify what's missing ---
        all_paper_embeddings: list[np.ndarray | None] = [None] * len(papers)
        missing_indices: list[int] = []
        for i, paper in enumerate(papers):
            cached_embedding = cache.load_embedding(paper.arxiv_id)
            if cached_embedding is not None:
                all_paper_embeddings[i] = cached_embedding
            else:
                missing_indices.append(i)

        logger.info(f"Found {len(papers) - len(missing_indices)} embeddings in cache.")
        logger.info(
            f"Need to generate embeddings for {len(missing_indices)} new papers."
        )

        # --- Generate embeddings only for missing papers ---
        if missing_indices:
            texts_to_embed = [
                f"{papers[i].title}. {papers[i].abstract}" for i in missing_indices
            ]

            config = load_config()
            chunk_size = config.get("research_assistant", {}).get(
                "embedding_batch_size", 128
            )

            logger.info(f"Generating missing embeddings in chunks of {chunk_size}...")

            newly_generated_embeddings: list[np.ndarray | None] = []
            for i in range(0, len(texts_to_embed), chunk_size):
                chunk_texts = texts_to_embed[i : i + chunk_size]
                try:
                    # batch_get_embeddings returns list[list[float]], convert to list[np.ndarray]
                    embedding_lists = batch_get_embeddings(chunk_texts)
                    chunk_embeddings = [np.array(e) for e in embedding_lists]
                    newly_generated_embeddings.extend(chunk_embeddings)
                except Exception as e:
                    logger.error(f"Error processing embedding chunk: {e}")
                    # On error, append None for each item in the chunk to maintain list size
                    newly_generated_embeddings.extend([None] * len(chunk_texts))

            # --- Fill in the missing embeddings and update cache ---
            for i, embedding in enumerate(newly_generated_embeddings):
                original_index = missing_indices[i]
                all_paper_embeddings[original_index] = embedding
                # Cache the newly generated embedding for future use
                if (
                    embedding is not None and embedding.size > 0
                ):  # Avoid caching empty/failed embeddings
                    cache.save_embedding(papers[original_index].arxiv_id, embedding)

        # Filter out any papers for which embedding failed
        valid_embeddings: list[np.ndarray] = []
        valid_papers: list[ArxivPaper] = []
        for i, emb in enumerate(all_paper_embeddings):
            if emb is not None and emb.size > 0:
                valid_embeddings.append(emb)
                valid_papers.append(papers[i])

        if len(valid_papers) != len(papers):
            logger.warning(
                f"Could not generate embeddings for {len(papers) - len(valid_papers)} papers. They will be excluded."
            )

        if not valid_papers:
            logger.error("All embeddings failed. Cannot proceed.")
            return pd.DataFrame()

        # Embed the query
        query_embedding_list = batch_get_embeddings([query_text])[0]
        query_embedding = np.array(query_embedding_list)

        # Compute cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity

        similarities = cosine_similarity([query_embedding], valid_embeddings)[0]

        # Create a DataFrame with results
        df = pd.DataFrame([p.to_dict() for p in valid_papers])
        df["embedding_score"] = similarities

        logger.info("Computed embedding scores for all valid candidate papers.")
        return df

    def post(self, shared: dict, prep_res: Any, exec_res: pd.DataFrame):  # type: ignore
        """Filters down to the top N papers using efficient partitioning."""
        if exec_res.empty:
            shared["filtered_df"] = pd.DataFrame()
            return "default"

        # The user's desired number of papers to analyze after this filter step.
        # This now comes from the finalized_intent.
        import numpy as np

        top_n = shared["finalized_intent"]["top_n_filter"]

        # Ensure top_n is not larger than the number of available papers
        if top_n >= len(exec_res):
            sorted_df = exec_res.sort_values(by="embedding_score", ascending=False)
            shared["top_k_df"] = sorted_df
            logger.info(
                f"Number of papers ({len(exec_res)}) is less than or equal to top_n ({top_n}). Returning all sorted papers."
            )
            return "default"

        # Use np.argpartition to find the indices of the top N scores without a full sort
        # We partition to find the k-th largest element, where k = -top_n
        partition_indices = np.argpartition(
            exec_res["embedding_score"].to_numpy(), -top_n
        )[-top_n:]

        # Select the top N rows using the indices and then sort only this small subset
        top_k_df = exec_res.iloc[partition_indices].sort_values(
            by="embedding_score", ascending=False
        )

        shared["top_k_df"] = top_k_df
        logger.info(f"Filtered down to top {len(top_k_df)} papers using partitioning.")
        return "default"


class LLMAnalysisNode(Node):
    """
    Performs a deep analysis of the top candidate papers using a powerful
    LLM to score relevance and provide justifications.
    """

    def __init__(self, max_workers: int = 10):
        super().__init__()  # Call the parent constructor
        self.max_workers = max_workers

    def prep(self, shared: dict[str, Any]) -> dict[str, Any]:  # type: ignore
        """Prepares the data needed for LLM analysis."""

        # The query for analysis should be the user's detailed description.
        query = shared["finalized_intent"]["description"]
        top_k_df = shared["top_k_df"]

        # Guard against empty DataFrame
        if top_k_df is None or top_k_df.empty:
            logger.warning("No papers available for LLM analysis.")
            raise ValueError("No papers available for LLM analysis.")

        if not query:
            logger.warning("No query description found for LLM Analysis.")

        config = load_config()
        model_name = (
            config.get("research_assistant", {})
            .get("default_model", {})
            .get("name", None)
        )
        provider = (
            config.get("research_assistant", {})
            .get("default_model", {})
            .get("provider", None)
        )

        if not model_name:
            raise ValueError("No model name found for LLM Analysis.")

        return {
            "rows": [row for _, row in top_k_df.iterrows()],
            "query": query,
            "model_name": model_name,
            "provider": provider,
        }

    def exec(self, prep_res: dict[str, Any]) -> list[dict[str, Any]]:  # type: ignore
        """Processes each paper in parallel to get LLM analysis."""
        rows = prep_res["rows"]
        query = prep_res["query"]
        model_name = prep_res["model_name"]
        provider = prep_res["provider"]

        if not rows or not query or not model_name:
            logger.info(
                "Insufficient data for LLM analysis (missing rows, query, or model_name)."
            )
            return []

        analyses = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_paper = {
                executor.submit(
                    self.analyze_single_paper,
                    row,
                    query,
                    model_name,
                    provider=provider,
                ): row
                for row in rows
            }
            for future in as_completed(future_to_paper):
                row = future_to_paper[future]
                try:
                    analysis_result = future.result()
                    # Combine original row data with analysis result
                    combined_result = row.to_dict()
                    if analysis_result and "relevance_score" in analysis_result:
                        combined_result.update(analysis_result)
                    else:
                        # Provide default values if analysis failed or was incomplete
                        combined_result.update(
                            {
                                "relevance_score": 0,
                                "justification": "LLM analysis failed or returned invalid format.",
                            }
                        )
                    analyses.append(combined_result)
                except Exception as e:
                    logger.error(
                        f"Error analyzing paper {row.get('id', 'N/A')}: {e}",
                        exc_info=True,
                    )
        logger.info(f"Completed LLM analysis for {len(analyses)} papers.")
        return analyses

    def analyze_single_paper(
        self, row: pd.Series, query: str, model_name: str, **kwargs
    ) -> dict[str, Any] | None:
        """Analyzes a single paper and returns the structured analysis."""
        prompt = f"""Analyze the following paper based on the user's research query.
        User Query: {query}

        Paper Title: {row['title']}
        Authors: {row['authors']}
        Published: {row['published_date']}
        Abstract: {row['abstract']}

        Please provide a relevance score from 0 to 100 and a brief justification.
        Respond in JSON format with keys: "relevance_score" (int) and "justification" (str).
        Example: {{"relevance_score": 85, "justification": "This paper directly addresses the core concepts..."}}
        """
        response = None
        try:
            response = call_llm(
                prompt,
                model_name=model_name,
                response_format={"type": "json_object"},
                **kwargs,
            )
            content = (
                response.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            )
            return json.loads(content)
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.error(
                f"Failed to parse LLM response for paper {row.get('id', 'N/A')}: {e}\nResponse: {response}"
            )
            return None

    def post(
        self, shared: dict[str, Any], prep_res: Any, exec_res: list[dict[str, Any]]
    ):  # type: ignore
        """Saves the final analyzed DataFrame."""
        if not exec_res:
            shared["final_df"] = pd.DataFrame()
            return

        final_df = pd.DataFrame(exec_res)
        # Sort by relevance_score if the column exists
        if "relevance_score" in final_df.columns:
            final_df = final_df.sort_values(
                by="relevance_score", ascending=False
            ).reset_index(drop=True)

        shared["final_df"] = final_df


class LLMFilterNode(Node):
    """Filters the LLM-analyzed papers down to the final requested number."""

    def prep(self, shared: dict[str, Any]) -> dict[str, Any]:
        """Prepares the analyzed DataFrame and the final count 'n'."""
        return {
            "analyzed_df": shared["final_df"]
            if "final_df" in shared
            else pd.DataFrame(),
            "final_n": shared["finalized_intent"]["final_n"],
        }

    def exec(self, prep_res: dict[str, Any]) -> pd.DataFrame:
        """Filters the DataFrame to the top 'n' results based on LLM score."""
        df = prep_res.get("analyzed_df")
        final_n = prep_res.get("final_n")

        if df is None or df.empty:
            logger.warning("No analyzed data to filter. Returning empty DataFrame.")
            return pd.DataFrame()

        if final_n is None:
            logger.warning("'final_n' not specified. Returning all analyzed papers.")
            return df

        if "relevance_score" not in df.columns:
            logger.error(
                "Cannot filter by LLM score: 'relevance_score' column not found."
            )
            return df

        # Ensure final_n is not larger than the number of available papers
        if final_n >= len(df):
            logger.info(
                f"Number of analyzed papers ({len(df)}) is less than or equal to "
                f"final_n ({final_n}). Returning all papers."
            )
            return df

        # Sort by score and take the top N
        final_df = df.sort_values(by="relevance_score", ascending=False).head(final_n)
        logger.info(
            f"Filtered down to the final {len(final_df)} papers based on LLM analysis."
        )
        return final_df

    def post(self, shared: dict[str, Any], prep_res: Any, exec_res: pd.DataFrame):
        """Updates the shared store with the final, filtered DataFrame."""
        shared["final_df"] = exec_res
        return "default"


class SaveResultsNode(Node):
    """Initializes the project and saves the first version of results."""

    def prep(self, shared: dict[str, Any]) -> dict[str, Any]:  # type: ignore
        """Passes all necessary info to the exec method."""
        return {
            "final_df": shared["final_df"] if "final_df" in shared else None,
            "top_k_df": shared["top_k_df"] if "top_k_df" in shared else None,
            "finalized_intent": shared["finalized_intent"],
        }

    def exec(self, prep_res: dict[str, Any]) -> str:  # type: ignore
        """Creates project structure and saves initial files."""
        finalized_intent = prep_res["finalized_intent"]
        final_df = prep_res["final_df"] if "final_df" in prep_res else None

        df_to_save = final_df
        if final_df is None or final_df.empty:
            logger.warning(
                "LLM Analysis did not return any results. Falling back to pre-analysis data."
            )
            top_k_df = prep_res["top_k_df"] if "top_k_df" in prep_res else None
            df_to_save = top_k_df

        if df_to_save is None or finalized_intent is None or df_to_save.empty:
            logger.warning("Not enough data to save. Skipping project creation.")
            return ""

        # --- Generate a meaningful project name ---
        import re

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        core_keywords = finalized_intent["query_text"]
        # Sanitize keywords to be filesystem-friendly
        sanitized_keywords = re.sub(r"[^\w\s-]", "", core_keywords).strip()
        sanitized_keywords = re.sub(r"[-\s]+", "_", sanitized_keywords).lower()
        project_name_slug = f"{timestamp}_{sanitized_keywords[:50]}"

        config = load_config()
        base_path = config.get("research_assistant", {}).get(
            "projects_dir", "output/research_projects"
        )
        project_path = create_project_directory(base_path, project_name_slug)

        # Save CSV
        version = 1
        csv_path = project_path / f"results_v{version}.csv"
        save_csv_version(df_to_save, csv_path)
        logger.info(f"Saved CSV version {version} to {csv_path}")

        # Generate Markdown
        md_content = generate_project_markdown(
            project_name_slug,
            finalized_intent["description"],
            finalized_intent,
        )
        md_path = project_path / "project.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info(f"Generated project markdown at {md_path}")

        logger.info(f"Project initialized at: {project_path}")
        return str(project_path)

    def post(self, shared: dict, prep_res: Any, exec_res: str):  # type: ignore
        """Stores the project path in the shared store."""
        shared["project_path"] = exec_res
        logger.info(f"Project initialized at: {exec_res}")
        return "default"


# ==============================================================================
# Flow 2: Interactive Refinement Flow Nodes (Skeletons)
# ==============================================================================


class LoadProjectNode(Node):
    """Loads an existing research project for refinement."""

    # To be implemented
    pass


class PromptUserNode(Node):
    """Prompts the user for a refinement query via an interactive CLI."""

    # To be implemented
    pass


class InterpretQueryNode(Node):
    """Uses an LLM to interpret the user's natural language refinement query."""

    # To be implemented
    pass


class RescoreNode(Node):
    """Re-scores the current paper list based on the refinement query."""

    # To be implemented
    pass


class MetaAnalysisNode(Node):
    """Generates meta-analysis insights based on the latest refinement."""

    # To be implemented
    pass


class UpdateProjectNode(Node):
    """Saves the new version of results and updates the project markdown file."""

    # To be implemented
    pass
