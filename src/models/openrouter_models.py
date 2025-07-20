"""
OpenRouter API data models
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class Provider:
    """OpenRouter provider information"""
    name: str
    slug: str
    privacy_policy_url: Optional[str] = None
    terms_of_service_url: Optional[str] = None
    status_page_url: Optional[str] = None
    may_log_prompts: bool = False
    may_train_on_data: bool = False
    moderated_by_openrouter: bool = False
    needs_moderation: bool = False
    status: str = "active"
    description: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Provider':
        """Create Provider from API response data"""
        return cls(
            name=data.get('name', ''),
            slug=data.get('slug', ''),
            privacy_policy_url=data.get('privacy_policy_url'),
            terms_of_service_url=data.get('terms_of_service_url'),
            status_page_url=data.get('status_page_url'),
            may_log_prompts=data.get('may_log_prompts', False),
            may_train_on_data=data.get('may_train_on_data', False),
            moderated_by_openrouter=data.get('moderated_by_openrouter', False),
            needs_moderation=data.get('needs_moderation', False),
            status=data.get('status', 'active'),
            description=data.get('description')
        )


@dataclass
class ModelPricing:
    """Model pricing information"""
    prompt: float  # Cost per token for input
    completion: float  # Cost per token for output
    image: Optional[float] = None  # Cost per image (if applicable)
    request: Optional[float] = None  # Cost per request (if applicable)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelPricing':
        """Create ModelPricing from API response data"""
        return cls(
            prompt=float(data.get('prompt', 0)),
            completion=float(data.get('completion', 0)),
            image=float(data.get('image', 0)) if data.get('image') else None,
            request=float(data.get('request', 0)) if data.get('request') else None
        )


@dataclass
class ModelArchitecture:
    """Model architecture information"""
    modality: str  # e.g., "text->text", "text+image->text"
    tokenizer: Optional[str] = None
    instruct_type: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelArchitecture':
        """Create ModelArchitecture from API response data"""
        return cls(
            modality=data.get('modality', 'text->text'),
            tokenizer=data.get('tokenizer'),
            instruct_type=data.get('instruct_type')
        )


@dataclass
class Model:
    """OpenRouter model information"""
    id: str
    name: str
    description: Optional[str] = None
    context_length: int = 0
    pricing: Optional[ModelPricing] = None
    architecture: Optional[ModelArchitecture] = None
    top_provider: Optional[Dict[str, Any]] = None
    per_request_limits: Optional[Dict[str, Any]] = None
    created: Optional[datetime] = None
    updated: Optional[datetime] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Model':
        """Create Model from API response data"""
        # Parse pricing
        pricing = None
        if 'pricing' in data and data['pricing']:
            pricing = ModelPricing.from_dict(data['pricing'])
        
        # Parse architecture
        architecture = None
        if 'architecture' in data and data['architecture']:
            architecture = ModelArchitecture.from_dict(data['architecture'])
        
        # Parse dates
        created = None
        if 'created' in data:
            try:
                created = datetime.fromisoformat(data['created'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass
        
        updated = None
        if 'updated' in data:
            try:
                updated = datetime.fromisoformat(data['updated'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass
        
        return cls(
            id=data.get('id', ''),
            name=data.get('name', ''),
            description=data.get('description'),
            context_length=data.get('context_length', 0),
            pricing=pricing,
            architecture=architecture,
            top_provider=data.get('top_provider'),
            per_request_limits=data.get('per_request_limits'),
            created=created,
            updated=updated
        )


@dataclass
class EndpointPricing:
    """Endpoint-specific pricing information"""
    prompt: float
    completion: float
    image: Optional[float] = None
    request: Optional[float] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EndpointPricing':
        """Create EndpointPricing from API response data"""
        return cls(
            prompt=float(data.get('prompt', 0)),
            completion=float(data.get('completion', 0)),
            image=float(data.get('image', 0)) if data.get('image') else None,
            request=float(data.get('request', 0)) if data.get('request') else None
        )


@dataclass
class Endpoint:
    """Model endpoint information"""
    name: str
    provider_name: str
    context_length: int = 0
    pricing: Optional[EndpointPricing] = None
    max_tokens: Optional[int] = None
    supported_parameters: Optional[List[str]] = None
    status: str = "active"
    uptime: Optional[float] = None
    latency: Optional[float] = None
    throughput: Optional[float] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Endpoint':
        """Create Endpoint from API response data"""
        # Parse pricing
        pricing = None
        if 'pricing' in data and data['pricing']:
            pricing = EndpointPricing.from_dict(data['pricing'])
        
        return cls(
            name=data.get('name', ''),
            provider_name=data.get('provider_name', ''),
            context_length=data.get('context_length', 0),
            pricing=pricing,
            max_tokens=data.get('max_tokens'),
            supported_parameters=data.get('supported_parameters', []),
            status=data.get('status', 'active'),
            uptime=data.get('uptime'),
            latency=data.get('latency'),
            throughput=data.get('throughput')
        )


@dataclass
class ModelEndpoints:
    """Complete model information with endpoints"""
    model: Model
    endpoints: List[Endpoint]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelEndpoints':
        """Create ModelEndpoints from API response data"""
        # Handle different API response formats
        if 'data' in data:
            # Response has 'data' wrapper
            model_data = data['data']
            model = Model.from_dict(model_data)
            
            endpoints = []
            if 'endpoints' in model_data:
                for endpoint_data in model_data['endpoints']:
                    endpoints.append(Endpoint.from_dict(endpoint_data))
        else:
            # Direct response format
            model = Model.from_dict(data)
            
            endpoints = []
            if 'endpoints' in data:
                for endpoint_data in data['endpoints']:
                    endpoints.append(Endpoint.from_dict(endpoint_data))
        
        return cls(
            model=model,
            endpoints=endpoints
        )


@dataclass
class OpenRouterStats:
    """OpenRouter usage statistics"""
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    models_used: Dict[str, int] = None
    providers_used: Dict[str, int] = None
    
    def __post_init__(self):
        if self.models_used is None:
            self.models_used = {}
        if self.providers_used is None:
            self.providers_used = {}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OpenRouterStats':
        """Create OpenRouterStats from API response data"""
        return cls(
            total_requests=data.get('total_requests', 0),
            total_tokens=data.get('total_tokens', 0),
            total_cost=float(data.get('total_cost', 0.0)),
            models_used=data.get('models_used', {}),
            providers_used=data.get('providers_used', {})
        )


@dataclass
class ModelFilter:
    """Filter criteria for model selection"""
    categories: Optional[List[str]] = None
    providers: Optional[List[str]] = None
    max_cost_per_token: Optional[float] = None
    min_context_length: Optional[int] = None
    modalities: Optional[List[str]] = None
    exclude_models: Optional[List[str]] = None
    
    def matches(self, model: Model) -> bool:
        """Check if a model matches the filter criteria"""
        # Check categories (if supported by the model)
        if self.categories:
            # This would need to be implemented based on model categorization
            pass
        
        # Check providers
        if self.providers and model.top_provider:
            provider_name = model.top_provider.get('name', '')
            if provider_name not in self.providers:
                return False
        
        # Check cost
        if self.max_cost_per_token and model.pricing:
            avg_cost = (model.pricing.prompt + model.pricing.completion) / 2
            if avg_cost > self.max_cost_per_token:
                return False
        
        # Check context length
        if self.min_context_length and model.context_length < self.min_context_length:
            return False
        
        # Check modalities
        if self.modalities and model.architecture:
            if model.architecture.modality not in self.modalities:
                return False
        
        # Check exclusions
        if self.exclude_models and model.id in self.exclude_models:
            return False
        
        return True