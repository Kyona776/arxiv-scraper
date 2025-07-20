"""
This utility handles file input/output operations for Research Assistant projects,
including directory management and versioned CSV handling.
"""

from __future__ import annotations
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Any
import os

logger = logging.getLogger(__name__)


class ProjectIOError(Exception):
    """Custom exception for project I/O errors."""

    pass


def create_project_directory(base_path: str, project_name: str) -> Path:
    """
    Creates a new directory for a research project.

    Args:
        base_path: The base directory where projects are stored.
        project_name: The name of the project.

    Returns:
        The path to the newly created project directory.

    Raises:
        ProjectIOError: If the directory cannot be created.
    """
    project_path = Path(base_path) / project_name
    try:
        project_path.mkdir(parents=True, exist_ok=True)
        (project_path / "logs").mkdir(exist_ok=True)
        logger.info(f"Project directory created at: {project_path}")
        return project_path
    except OSError as e:
        logger.error(f"Failed to create project directory at {project_path}: {e}")
        raise ProjectIOError(f"Directory creation failed: {e}") from e


def save_csv_version(df: pd.DataFrame, file_path: Path) -> str:
    """
    Saves a DataFrame to a CSV file at the specified full path.

    Args:
        df: The pandas DataFrame to save.
        file_path: The full path (including filename) to save the CSV.

    Returns:
        The string representation of the file path where the CSV was saved.

    Raises:
        ProjectIOError: If the CSV file cannot be saved.
    """
    try:
        df.to_csv(file_path, index=False, encoding="utf-8")
        logger.info(f"Saved CSV to {file_path}")
        return str(file_path)
    except (IOError, TypeError) as e:
        logger.error(f"Failed to save CSV to {file_path}: {e}")
        raise ProjectIOError(f"Failed to save CSV: {e}") from e


def load_csv_version(project_path: Path, version: int = 1) -> Optional[pd.DataFrame]:
    """
    Loads a specific version of a CSV file.

    Args:
        project_path: The path to the project directory.
        version: The version number to load.

    Returns:
        A pandas DataFrame with the loaded data, or None if not found.
    """
    file_path = project_path / f"results_v{version}.csv"
    if not file_path.exists():
        logger.warning(f"CSV version {version} not found at {file_path}")
        return None

    try:
        df = pd.read_csv(file_path, encoding="utf-8")
        logger.info(f"Loaded CSV version {version} from {file_path}")
        return df
    except (IOError, pd.errors.ParserError) as e:
        logger.error(f"Failed to load CSV from {file_path}: {e}")
        raise ProjectIOError(f"CSV loading failed: {e}") from e


def get_latest_version(project_path: Path) -> int:
    """
    Finds the latest version number of the results CSV in a project.

    Args:
        project_path: The path to the project directory.

    Returns:
        The latest version number found, or 0 if no versions exist.
    """
    try:
        csv_files = list(project_path.glob("results_v*.csv"))
        if not csv_files:
            return 0

        versions = [int(f.stem.split("_v")[1]) for f in csv_files]
        return max(versions)
    except (ValueError, IndexError) as e:
        logger.error(f"Could not determine latest version in {project_path}: {e}")
        return 0
