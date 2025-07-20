"""
This utility handles parsing and generation of the project's state-tracking
markdown file (`project.md`), including its YAML frontmatter and content sections.
"""

from __future__ import annotations
import logging
import yaml
import frontmatter
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class MarkdownParseError(Exception):
    """Custom exception for errors during markdown file parsing."""

    pass


def parse_project_markdown(file_path: Path) -> tuple[dict[str, Any], str]:
    """
    Parses a project markdown file, separating frontmatter and content.

    Args:
        file_path: The path to the `project.md` file.

    Returns:
        A tuple containing the metadata dictionary and the markdown content string.

    Raises:
        MarkdownParseError: If the file cannot be read or parsed.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            post = frontmatter.load(f)
        return post.metadata, post.content
    except (IOError, yaml.YAMLError) as e:
        logger.error(f"Failed to parse markdown file at {file_path}: {e}")
        raise MarkdownParseError(f"Markdown parsing failed: {e}") from e


def generate_project_markdown(
    project_name_slug: str, initial_query: str, finalized_intent: dict[str, Any] | None
) -> str:
    """
    Generates the complete markdown string for a new project.md file.

    Args:
        project_name_slug: The unique, slugified name of the project.
        initial_query: The final, combined query string that was executed.
        finalized_intent: The dictionary containing the user's full intent.

    Returns:
        A string containing the full content of project.md (YAML + markdown).
    """
    from datetime import datetime

    metadata = {
        "project_name": project_name_slug,
        "created_at": datetime.now().isoformat(),
        "status": "active",
        "executed_query": initial_query,
        "user_intent": finalized_intent if finalized_intent else {},
    }

    content = f"""
# Research Project: {project_name_slug}

## Paper List (Version History)
- [v1 - Initial Results](./results_v1.csv)

## LLM Memory
- Initial analysis complete.
"""
    post = frontmatter.Post(content, **metadata)
    return frontmatter.dumps(post)


def update_project_markdown(
    file_path: Path,
    metadata_update: dict[str, Any] = None,
    content_append: str = None,
) -> None:
    """
    Updates an existing project markdown file.

    It can update the YAML frontmatter and/or append text to the main content.

    Args:
        file_path: The path to the `project.md` file.
        metadata_update: A dictionary of metadata fields to update.
        content_append: A string to append to the markdown content.

    Raises:
        MarkdownParseError: If the file cannot be read or written.
    """
    try:
        # Reading is still done in text mode
        with open(file_path, "r", encoding="utf-8") as f:
            post = frontmatter.load(f)

        if metadata_update:
            post.metadata.update(metadata_update)
            logger.debug("Updated markdown metadata.")

        if content_append:
            post.content += f"\n\n{content_append}"
            logger.debug("Appended content to markdown.")

        # Open in binary write mode ('wb') for dumping
        with open(file_path, "wb") as f:
            frontmatter.dump(post, f)
        logger.info(f"Updated project markdown at {file_path}")

    except (IOError, yaml.YAMLError) as e:
        logger.error(f"Failed to update markdown file at {file_path}: {e}")
        raise MarkdownParseError(f"Markdown update failed: {e}") from e
