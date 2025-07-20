"""
This module contains utility functions for the Research Assistant feature.
These utilities handle tasks such as API interactions, data processing, and file management,
abstracting them away from the core PocketFlow logic.
"""

from .config_loader import load_config, ConfigError
from .arxiv_search import arxiv_search, chunk_date_range, ArxivPaper
from .embedding import get_embedding, batch_get_embeddings, EmbeddingError
from .llm_api import call_llm, CallLLMError
from .file_io import (
    create_project_directory,
    save_csv_version,
    load_csv_version,
    get_latest_version,
    ProjectIOError,
)
from .markdown import (
    parse_project_markdown,
    generate_project_markdown,
    update_project_markdown,
    MarkdownParseError,
)

__all__ = [
    # arxiv_search
    "arxiv_search",
    "chunk_date_range",
    "ArxivPaper",
    # embedding
    "get_embedding",
    "batch_get_embeddings",
    "EmbeddingError",
    # llm_api
    "call_llm",
    "CallLLMError",
    # file_io
    "create_project_directory",
    "save_csv_version",
    "load_csv_version",
    "get_latest_version",
    "ProjectIOError",
    # markdown
    "parse_project_markdown",
    "generate_project_markdown",
    "update_project_markdown",
    "MarkdownParseError",
]

__all__ += [
    # config_loader
    "load_config",
    "ConfigError",
]
 