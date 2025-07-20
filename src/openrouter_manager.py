"""
OpenRouter API Management
Centralized management for OpenRouter API operations including providers, models, and endpoints
"""

import os
import time
import json
import requests
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from loguru import logger
from pathlib import Path
from src.models.error import CallLLMError

try:
    from .models.openrouter_models import (
        Provider,
        Model,
        ModelEndpoints,
        Endpoint,
        ModelFilter,
        OpenRouterStats,
        ModelPricing,
    )
except ImportError:
    from models.openrouter_models import (
        Provider,
        Model,
        ModelEndpoints,
        Endpoint,
        ModelFilter,
        OpenRouterStats,
        ModelPricing,
    )


@dataclass
class CacheEntry:
    """Cache entry for API responses"""

    data: Any
    timestamp: float
    expires_at: float

    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return time.time() > self.expires_at


class OpenRouterManager:
    """OpenRouter API management and caching"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get("api_key") or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = config.get("base_url", "https://openrouter.ai/api/v1")
        self.site_url = config.get("site_url", "https://arxiv-scraper.local")
        self.site_name = config.get("site_name", "ArXiv Scraper")
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)

        # Caching settings
        self.cache_enabled = config.get("cache_enabled", True)
        self.cache_ttl = config.get("cache_ttl", 3600)  # 1 hour
        self.cache_dir = Path(config.get("cache_dir", "./cache/openrouter"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self._cache: Dict[str, CacheEntry] = {}

        if not self.api_key:
            logger.warning(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable."
            )

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for OpenRouter API requests"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
        }

    def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make a request to OpenRouter API with retry logic"""
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = self._get_headers()

        for attempt in range(self.max_retries):
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=json_data,
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    logger.warning(
                        f"Rate limit exceeded. Retrying in {self.retry_delay} seconds..."
                    )
                    time.sleep(self.retry_delay * (2**attempt))
                    continue
                else:
                    response.raise_for_status()

            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error on attempt {attempt + 1}: {e}")
                if e.response is not None:
                    logger.error(f"Error Response Body: {e.response.text}")
                if 400 <= e.response.status_code < 500:
                    logger.warning("Client error, not retrying.")
                    raise  # Re-raise immediately for client-side errors
                # For server errors (5xx), continue to retry
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed on attempt {attempt + 1}: {e}")

            if attempt < self.max_retries - 1:
                wait_time = self.retry_delay * (2**attempt)
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        logger.error(
            f"Failed to make request to {endpoint} after {self.max_retries} attempts."
        )
        raise CallLLMError(f"API request failed after {self.max_retries} retries.")

    def get_completion(
        self, model: str, messages: list[dict], **kwargs: Any
    ) -> dict[str, Any]:
        """
        Get a chat completion from a specified model.

        Args:
            model: The ID of the model to use.
            messages: A list of message objects, following OpenAI's format.
            **kwargs: Additional parameters for the chat completion API.

        Returns:
            The JSON response from the API.
        """
        endpoint = "/chat/completions"
        
        payload = {"model": model, "messages": messages, **kwargs}

        # This type of request should typically not be cached as prompts are unique.
        # If caching is desired, a more sophisticated key generation based on content is needed.
        return self._make_request(endpoint, method="POST", json_data=payload)

    def _get_cached_or_fetch(self, cache_key: str, fetch_func, *args, **kwargs) -> Any:
        """Get data from cache or fetch from API"""
        if not self.cache_enabled:
            return fetch_func(*args, **kwargs)

        # Check in-memory cache first
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if not entry.is_expired():
                logger.debug(f"Using cached data for {cache_key}")
                return entry.data
            else:
                del self._cache[cache_key]

        # Check file cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    cached_data = json.load(f)
                    entry = CacheEntry(
                        data=cached_data["data"],
                        timestamp=cached_data["timestamp"],
                        expires_at=cached_data["expires_at"],
                    )
                    if not entry.is_expired():
                        logger.debug(f"Using file cached data for {cache_key}")
                        self._cache[cache_key] = entry
                        return entry.data
                    else:
                        cache_file.unlink()
            except (json.JSONDecodeError, KeyError, OSError):
                logger.warning(f"Invalid cache file for {cache_key}, removing")
                cache_file.unlink(missing_ok=True)

        # Fetch from API
        logger.debug(f"Fetching fresh data for {cache_key}")
        data = fetch_func(*args, **kwargs)

        # Cache the result
        now = time.time()
        entry = CacheEntry(data=data, timestamp=now, expires_at=now + self.cache_ttl)
        self._cache[cache_key] = entry

        # Save to file cache
        try:
            with open(cache_file, "w") as f:
                # Convert dataclasses to dicts for JSON serialization
                serializable_data = data
                if hasattr(data, "__dict__"):
                    serializable_data = asdict(data)
                elif isinstance(data, list) and data and hasattr(data[0], "__dict__"):
                    serializable_data = [asdict(item) for item in data]

                json.dump(
                    {
                        "data": serializable_data,
                        "timestamp": entry.timestamp,
                        "expires_at": entry.expires_at,
                    },
                    f,
                )
        except (OSError, TypeError) as e:
            logger.warning(f"Failed to save cache file for {cache_key}: {e}")

        return data

    def clear_cache(self, cache_key: Optional[str] = None):
        """Clear cache entries"""
        if cache_key:
            # Clear specific cache entry
            self._cache.pop(cache_key, None)
            cache_file = self.cache_dir / f"{cache_key}.json"
            cache_file.unlink(missing_ok=True)
        else:
            # Clear all cache
            self._cache.clear()
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink(missing_ok=True)

        logger.info(f"Cleared cache: {cache_key or 'all'}")

    def is_available(self) -> bool:
        """Check if OpenRouter API is available"""
        return bool(self.api_key)

    # Provider Management Methods
    def get_providers(self) -> List[Provider]:
        """Get all available providers"""

        def fetch_providers():
            response = self._make_request("/providers")
            providers = []
            for provider_data in response.get("data", []):
                providers.append(Provider.from_dict(provider_data))
            return [asdict(p) for p in providers]

        providers_data = self._get_cached_or_fetch("providers", fetch_providers)
        return [Provider.from_dict(p) for p in providers_data]

    def get_provider_by_slug(self, slug: str) -> Optional[Provider]:
        """Get provider by slug"""
        providers = self.get_providers()
        for provider in providers:
            if provider.slug == slug:
                return provider
        return None

    def check_provider_status(self, slug: str) -> Dict[str, Any]:
        """Check provider status"""
        provider = self.get_provider_by_slug(slug)
        if not provider:
            return {"exists": False, "status": "not_found"}

        return {
            "exists": True,
            "status": provider.status,
            "may_log_prompts": provider.may_log_prompts,
            "may_train_on_data": provider.may_train_on_data,
            "moderated_by_openrouter": provider.moderated_by_openrouter,
            "needs_moderation": provider.needs_moderation,
        }

    # Model Management Methods
    def get_models(self, category: Optional[str] = None) -> List[Model]:
        """Get all available models"""

        def fetch_models():
            params = {}
            if category:
                params["category"] = category

            response = self._make_request("/models", params=params)
            models = []
            for model_data in response.get("data", []):
                models.append(Model.from_dict(model_data))
            return [asdict(m) for m in models]

        cache_key = f"models_{category or 'all'}"
        models_data = self._get_cached_or_fetch(cache_key, fetch_models)
        return [Model.from_dict(m) for m in models_data]

    def get_model_by_id(self, model_id: str) -> Optional[Model]:
        """Get model by ID"""
        models = self.get_models()
        for model in models:
            if model.id == model_id:
                return model
        return None

    def validate_model(self, model_id: str) -> bool:
        """Check if model exists and is available"""
        model = self.get_model_by_id(model_id)
        return model is not None

    def get_models_by_provider(self, provider_slug: str) -> List[Model]:
        """Get models by provider"""
        models = self.get_models()
        provider_models = []

        for model in models:
            if (
                model.top_provider
                and model.top_provider.get("name", "").lower() == provider_slug.lower()
            ):
                provider_models.append(model)

        return provider_models

    def filter_models(self, filter_criteria: ModelFilter) -> List[Model]:
        """Filter models based on criteria"""
        models = self.get_models()
        filtered = []

        for model in models:
            if filter_criteria.matches(model):
                filtered.append(model)

        return filtered

    # Endpoint Management Methods
    def get_model_endpoints(self, model_id: str) -> Optional[ModelEndpoints]:
        """Get endpoints for a specific model"""

        def fetch_endpoints():
            # Parse model ID to get author and slug
            if "/" in model_id:
                author, slug = model_id.split("/", 1)
            else:
                # Try to find the model first
                model = self.get_model_by_id(model_id)
                if not model:
                    return None
                if "/" in model.id:
                    author, slug = model.id.split("/", 1)
                else:
                    return None

            try:
                response = self._make_request(f"/models/{author}/{slug}/endpoints")
                return ModelEndpoints.from_dict(response)
            except Exception as e:
                logger.error(f"Failed to get endpoints for {model_id}: {e}")
                return None

        cache_key = f"endpoints_{model_id.replace('/', '_')}"
        return self._get_cached_or_fetch(cache_key, fetch_endpoints)

    def get_best_endpoint(
        self, model_id: str, criteria: str = "cost"
    ) -> Optional[Endpoint]:
        """Get the best endpoint for a model based on criteria"""
        endpoints_info = self.get_model_endpoints(model_id)
        if not endpoints_info or not endpoints_info.endpoints:
            return None

        endpoints = endpoints_info.endpoints

        if criteria == "cost":
            # Find endpoint with lowest cost
            best_endpoint = None
            best_cost = float("inf")

            for endpoint in endpoints:
                if endpoint.pricing:
                    avg_cost = (
                        endpoint.pricing.prompt + endpoint.pricing.completion
                    ) / 2
                    if avg_cost < best_cost:
                        best_cost = avg_cost
                        best_endpoint = endpoint

            return best_endpoint

        elif criteria == "latency":
            # Find endpoint with lowest latency
            best_endpoint = None
            best_latency = float("inf")

            for endpoint in endpoints:
                if endpoint.latency and endpoint.latency < best_latency:
                    best_latency = endpoint.latency
                    best_endpoint = endpoint

            return best_endpoint

        elif criteria == "uptime":
            # Find endpoint with highest uptime
            best_endpoint = None
            best_uptime = 0

            for endpoint in endpoints:
                if endpoint.uptime and endpoint.uptime > best_uptime:
                    best_uptime = endpoint.uptime
                    best_endpoint = endpoint

            return best_endpoint

        # Default: return first active endpoint
        for endpoint in endpoints:
            if endpoint.status == "active":
                return endpoint

        return endpoints[0] if endpoints else None

    def check_endpoint_health(
        self, model_id: str, endpoint_name: str
    ) -> Dict[str, Any]:
        """Check endpoint health status"""
        endpoints_info = self.get_model_endpoints(model_id)
        if not endpoints_info:
            return {"exists": False, "status": "model_not_found"}

        for endpoint in endpoints_info.endpoints:
            if endpoint.name == endpoint_name:
                return {
                    "exists": True,
                    "status": endpoint.status,
                    "uptime": endpoint.uptime,
                    "latency": endpoint.latency,
                    "throughput": endpoint.throughput,
                }

        return {"exists": False, "status": "endpoint_not_found"}

    # Utility Methods
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive model information"""
        model = self.get_model_by_id(model_id)
        if not model:
            return {"exists": False}

        endpoints_info = self.get_model_endpoints(model_id)

        return {
            "exists": True,
            "model": asdict(model),
            "endpoints": [asdict(ep) for ep in endpoints_info.endpoints]
            if endpoints_info
            else [],
            "best_endpoint": asdict(self.get_best_endpoint(model_id))
            if self.get_best_endpoint(model_id)
            else None,
        }

    def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple models"""
        comparison = {}

        for model_id in model_ids:
            model = self.get_model_by_id(model_id)
            if model:
                comparison[model_id] = {
                    "name": model.name,
                    "context_length": model.context_length,
                    "pricing": asdict(model.pricing) if model.pricing else None,
                    "provider": model.top_provider.get("name")
                    if model.top_provider
                    else None,
                }
            else:
                comparison[model_id] = {"exists": False}

        return comparison

    def get_recommendations(
        self, task_type: str = "general", budget: Optional[float] = None
    ) -> List[Model]:
        """Get model recommendations based on task type and budget"""
        models = self.get_models()

        # Filter by task type
        if task_type == "coding":
            # Prefer models good at coding
            preferred_models = [
                m for m in models if "code" in m.name.lower() or "gpt-4" in m.id.lower()
            ]
        elif task_type == "conversation":
            # Prefer models good at conversation
            preferred_models = [
                m for m in models if "claude" in m.id.lower() or "gpt" in m.id.lower()
            ]
        elif task_type == "analysis":
            # Prefer models with large context
            preferred_models = [m for m in models if m.context_length > 32000]
        else:
            preferred_models = models

        # Filter by budget
        if budget:
            budget_models = []
            for model in preferred_models:
                if model.pricing:
                    avg_cost = (model.pricing.prompt + model.pricing.completion) / 2
                    if avg_cost <= budget:
                        budget_models.append(model)
            preferred_models = budget_models

        # Sort by some criteria (e.g., cost, context length)
        preferred_models.sort(
            key=lambda m: (
                (m.pricing.prompt + m.pricing.completion) / 2 if m.pricing else 0,
                -m.context_length,
            )
        )

        return preferred_models[:10]  # Return top 10 recommendations


def create_open_router_manager(config: Dict[str, Any]) -> OpenRouterManager:
    """Create OpenRouter manager from configuration"""
    return OpenRouterManager(config.get("openrouter", {}))
