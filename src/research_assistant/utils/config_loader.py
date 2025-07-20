"""
This utility provides a function to load the main application configuration
from a YAML file.
"""

from __future__ import annotations
import yaml
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

_CONFIG = None


class ConfigError(Exception):
    """Custom exception for configuration loading errors."""

    pass


def load_config(config_path: str = "config/config.yaml") -> dict[str, Any]:
    """
    Loads the YAML configuration file.

    It caches the configuration in memory to avoid repeated file reads.

    Args:
        config_path: The path to the YAML configuration file.

    Returns:
        A dictionary containing the application configuration.

    Raises:
        ConfigError: If the configuration file cannot be found or parsed.
    """
    global _CONFIG
    if _CONFIG:
        return _CONFIG

    path = Path(config_path)
    if not path.exists():
        logger.error(f"Configuration file not found at: {path}")
        raise ConfigError(f"Config file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            _CONFIG = yaml.safe_load(f)
        logger.info(f"Configuration loaded successfully from {path}")
        return _CONFIG
    except (IOError, yaml.YAMLError) as e:
        logger.error(f"Failed to load or parse config file at {path}: {e}")
        raise ConfigError(f"Config loading/parsing failed: {e}") from e
