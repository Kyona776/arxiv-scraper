"""
This utility provides functions to search for papers on arXiv.
"""

from __future__ import annotations
import arxiv
from arxiv.arxiv import UnexpectedEmptyPageError
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Any
import feedparser
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.parse


logger = logging.getLogger(__name__)


@dataclass
class ArxivPaper:
    """A data class to hold structured information about an arXiv paper."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published_date: str
    categories: list[str]
    pdf_url: str
    abstract_url: str

    def to_dict(self) -> dict[str, Any]:
        """Converts the paper data to a dictionary, formatting list fields."""
        data = asdict(self)
        data["authors"] = ", ".join(self.authors)
        data["categories"] = ", ".join(self.categories)
        return data


def _fetch_papers_by_ids(id_list: list[str]) -> list[ArxivPaper]:
    """Fetches full paper details for a given list of arXiv IDs."""
    search = arxiv.Search(id_list=id_list)
    papers = []
    try:
        for result in search.results():
            papers.append(
                ArxivPaper(
                    arxiv_id=result.entry_id.split("/")[-1],
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    abstract=result.summary,
                    published_date=result.published.strftime("%Y-%m-%d"),
                    categories=result.categories,
                    pdf_url=result.pdf_url if result.pdf_url else "",
                    abstract_url=result.entry_id,
                )
            )
    except Exception as e:
        logger.error(f"Error fetching details for IDs {id_list}: {e}")
    return papers


def arxiv_search(query: str, max_results: int = 1000) -> list[ArxivPaper]:
    """
    Performs a search on arXiv, fetching all result IDs first and then
    fetching paper details in parallel batches. This version is hardened
    against pagination errors from the arxiv library.

    Args:
        query: The search query string for the arXiv API.
        max_results: The maximum number of results to retrieve.

    Returns:
        A list of ArxivPaper objects.
    """
    logger.info(f"Searching arXiv with query: '{query}'")

    # 1. Get all result IDs first, with robust error handling for pagination.
    all_ids = []
    try:
        search = arxiv.Search(
            query=query,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        results_generator = search.results()

        for result in results_generator:
            all_ids.append(result.entry_id.split("/")[-1])
            if len(all_ids) >= max_results:
                break

        total_results = len(all_ids)
        logger.info(f"Found {total_results} total paper IDs.")

    except UnexpectedEmptyPageError as e:
        logger.warning(
            f"Pagination ended unexpectedly early ({e}). "
            f"Proceeding with {len(all_ids)} papers found so far."
        )
    except Exception as e:
        # A more robust check in case the specific exception is not caught
        if "Page of results was unexpectedly empty" in str(e):
            logger.warning(
                f"Pagination ended unexpectedly early (caught in generic Exception: {e}). "
                f"Proceeding with {len(all_ids)} papers found so far."
            )
        else:
            logger.error(f"Failed to fetch paper IDs from arXiv: {e}")
            return []

    if not all_ids:
        return []

    # Limit the results if necessary (in case loop broke early)
    if len(all_ids) > max_results:
        all_ids = all_ids[:max_results]

    # 2. Fetch paper details in parallel chunks.
    chunk_size = 200  # arxiv.Search by id_list is efficient with larger chunks
    id_chunks = [
        all_ids[i : i + chunk_size] for i in range(0, len(all_ids), chunk_size)
    ]

    all_papers = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_chunk = {
            executor.submit(_fetch_papers_by_ids, chunk): chunk for chunk in id_chunks
        }
        for future in as_completed(future_to_chunk):
            try:
                papers_from_chunk = future.result()
                all_papers.extend(papers_from_chunk)
            except Exception as e:
                chunk = future_to_chunk[future]
                logger.error(
                    f"Error processing future for ID chunk {chunk[:2]}...: {e}"
                )

    logger.info(f"Successfully fetched details for {len(all_papers)} papers.")

    # Sort by published date as the parallel fetching might mess up the order
    all_papers.sort(key=lambda p: p.published_date, reverse=True)

    return all_papers


def chunk_date_range(
    start_date_str: str, end_date_str: str, days_per_chunk: int = 30
) -> list[tuple[str, str]]:
    """
    Splits a date range into smaller chunks.

    Args:
        start_date_str: The start date in 'YYYY-MM-DD' format.
        end_date_str: The end date in 'YYYY-MM-DD' format.
        days_per_chunk: The number of days each chunk should span.

    Returns:
        A list of tuples, where each tuple contains the start and end date
        of a chunk.
    """
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    date_chunks = []
    current_start = start_date
    while current_start <= end_date:
        current_end = current_start + timedelta(days=days_per_chunk - 1)
        if current_end > end_date:
            current_end = end_date
        date_chunks.append(
            (current_start.strftime("%Y-%m-%d"), current_end.strftime("%Y-%m-%d"))
        )
        current_start = current_end + timedelta(days=1)

    return date_chunks
