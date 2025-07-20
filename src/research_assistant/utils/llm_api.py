"""
This utility provides a unified function to interact with LLM APIs,
leveraging the existing OpenRouter manager for model routing.
"""

from __future__ import annotations
import logging
from typing import Any, Dict
from src.openrouter_manager import OpenRouterManager
from .config_loader import load_config
from src.models.error import CallLLMError

logger = logging.getLogger(__name__)

# Initialize the manager once to reuse its state and cache.
LLM_PROVIDER = None


def _initialize_provider() -> OpenRouterManager:
    """Initializes and returns the OpenRouterManager."""
    global LLM_PROVIDER
    if LLM_PROVIDER is None:
        logger.info("Initializing OpenRouterManager...")
        config = load_config()
        # Pass the 'openrouter' section of the config to the manager
        LLM_PROVIDER = OpenRouterManager(config.get("llm", {}).get("openrouter", {}))
        # The original manager does not have a load_models method,
        # so this line is removed. Model loading is implicit.
        logger.info("OpenRouterManager initialized.")
    return LLM_PROVIDER


def call_llm(prompt: str, model_name: str, **kwargs: Any) -> dict[str, Any]:
    """
    Calls an LLM through the OpenRouterManager with a given prompt and model.

    This function acts as a simplified, unified interface to the more complex
    extraction logic that might exist in `LLMExtractor`.

    Args:
        prompt: The text prompt to send to the LLM.
        model_name: The identifier of the model to use (e.g., 'anthropic/claude-3-sonnet').
        **kwargs: Additional keyword arguments for the LLM provider's API.

    Returns:
        A dictionary containing the LLM's response.

    Raises:
        CallLLMError: If the API call fails for any reason.
    """
    provider = _initialize_provider()

    try:
        logger.debug(f"Calling LLM '{model_name}' with prompt: {prompt[:100]}...")

        # The get_completion method has been added to the OpenRouterManager.
        # This utility now directly calls it.
        response = provider.get_completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

        if not response:
            raise CallLLMError("Received an empty response from the LLM provider.")

        logger.debug(f"Received response from '{model_name}'.")
        return response

    except Exception as e:
        logger.error(f"LLM API call to '{model_name}' failed: {e}")
        raise CallLLMError(f"LLM call failed: {e}") from e
