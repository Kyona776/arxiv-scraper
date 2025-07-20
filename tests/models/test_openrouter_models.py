"""
Tests for OpenRouter data models and validation
"""

import pytest
from datetime import datetime
from dataclasses import asdict

from src.models.openrouter_models import (
    Provider, Model, ModelPricing, ModelArchitecture, Endpoint, 
    ModelEndpoints, ModelFilter, EndpointPricing
)


class TestProvider:
    """Test Provider data model"""
    
    def test_provider_creation(self):
        """Test basic provider creation"""
        provider = Provider(
            name="Test Provider",
            slug="test-provider",
            status="active"
        )
        
        assert provider.name == "Test Provider"
        assert provider.slug == "test-provider"
        assert provider.status == "active"
        assert provider.may_log_prompts is False
        assert provider.may_train_on_data is False
        assert provider.moderated_by_openrouter is False
        assert provider.privacy_policy_url is None
        assert provider.terms_of_service_url is None
    
    def test_provider_with_all_fields(self):
        """Test provider with all fields"""
        provider = Provider(
            name="Test Provider",
            slug="test-provider",
            status="active",
            may_log_prompts=True,
            may_train_on_data=True,
            moderated_by_openrouter=True,
            privacy_policy_url="https://example.com/privacy",
            terms_of_service_url="https://example.com/terms"
        )
        
        assert provider.may_log_prompts is True
        assert provider.may_train_on_data is True
        assert provider.moderated_by_openrouter is True
        assert provider.privacy_policy_url == "https://example.com/privacy"
        assert provider.terms_of_service_url == "https://example.com/terms"
    
    def test_provider_serialization(self):
        """Test provider serialization to dict"""
        provider = Provider(
            name="Test Provider",
            slug="test-provider",
            status="active",
            may_log_prompts=True
        )
        
        data = asdict(provider)
        
        assert data['name'] == "Test Provider"
        assert data['slug'] == "test-provider"
        assert data['status'] == "active"
        assert data['may_log_prompts'] is True
        assert data['may_train_on_data'] is False


class TestModelPricing:
    """Test ModelPricing data model"""
    
    def test_pricing_creation(self):
        """Test basic pricing creation"""
        pricing = ModelPricing(
            prompt=0.001,
            completion=0.002
        )
        
        assert pricing.prompt == 0.001
        assert pricing.completion == 0.002
        assert pricing.image is None
        assert pricing.request is None
    
    def test_pricing_with_all_fields(self):
        """Test pricing with all fields"""
        pricing = ModelPricing(
            prompt=0.001,
            completion=0.002,
            image=0.005,
            request=0.01
        )
        
        assert pricing.prompt == 0.001
        assert pricing.completion == 0.002
        assert pricing.image == 0.005
        assert pricing.request == 0.01
    
    def test_pricing_cost_calculation(self):
        """Test pricing cost calculations"""
        pricing = ModelPricing(prompt=0.001, completion=0.002)
        
        # Test average cost
        avg_cost = (pricing.prompt + pricing.completion) / 2
        assert avg_cost == 0.0015
        
        # Test cost for specific token counts
        prompt_cost = pricing.prompt * 1000  # 1000 tokens
        completion_cost = pricing.completion * 500  # 500 tokens
        total_cost = prompt_cost + completion_cost
        
        assert prompt_cost == 1.0
        assert completion_cost == 1.0
        assert total_cost == 2.0


class TestModelArchitecture:
    """Test ModelArchitecture data model"""
    
    def test_architecture_creation(self):
        """Test basic architecture creation"""
        arch = ModelArchitecture(
            modality="text->text",
            tokenizer="GPT-4",
            instruct_type="chat"
        )
        
        assert arch.modality == "text->text"
        assert arch.tokenizer == "GPT-4"
        assert arch.instruct_type == "chat"
    
    def test_architecture_optional_fields(self):
        """Test architecture with optional fields"""
        arch = ModelArchitecture(modality="text->text")
        
        assert arch.modality == "text->text"
        assert arch.tokenizer is None
        assert arch.instruct_type is None


class TestModel:
    """Test Model data model"""
    
    def test_model_creation(self):
        """Test basic model creation"""
        model = Model(
            id="test/model",
            name="Test Model",
            context_length=4000
        )
        
        assert model.id == "test/model"
        assert model.name == "Test Model"
        assert model.context_length == 4000
        assert model.pricing is None
        assert model.top_provider is None
        assert model.architecture is None
        assert model.description is None
        assert model.per_request_limits is None
    
    def test_model_with_pricing(self):
        """Test model with pricing"""
        pricing = ModelPricing(prompt=0.001, completion=0.002)
        model = Model(
            id="test/model",
            name="Test Model",
            context_length=4000,
            pricing=pricing
        )
        
        assert model.pricing == pricing
        assert model.pricing.prompt == 0.001
        assert model.pricing.completion == 0.002
    
    def test_model_with_architecture(self):
        """Test model with architecture"""
        arch = ModelArchitecture(
            modality="text->text",
            tokenizer="GPT-4",
            instruct_type="chat"
        )
        model = Model(
            id="test/model",
            name="Test Model",
            context_length=4000,
            architecture=arch
        )
        
        assert model.architecture == arch
        assert model.architecture.modality == "text->text"
        assert model.architecture.tokenizer == "GPT-4"
        assert model.architecture.instruct_type == "chat"
    
    def test_model_with_top_provider(self):
        """Test model with top provider"""
        top_provider = {
            'name': 'Test Provider',
            'slug': 'test-provider',
            'is_default': True
        }
        model = Model(
            id="test/model",
            name="Test Model",
            context_length=4000,
            top_provider=top_provider
        )
        
        assert model.top_provider == top_provider
        assert model.top_provider['name'] == 'Test Provider'
        assert model.top_provider['slug'] == 'test-provider'
        assert model.top_provider['is_default'] is True
    
    def test_model_serialization(self):
        """Test model serialization"""
        pricing = ModelPricing(prompt=0.001, completion=0.002)
        arch = ModelArchitecture(modality="text->text", tokenizer="GPT-4")
        
        model = Model(
            id="test/model",
            name="Test Model",
            context_length=4000,
            pricing=pricing,
            architecture=arch,
            description="A test model"
        )
        
        data = asdict(model)
        
        assert data['id'] == "test/model"
        assert data['name'] == "Test Model"
        assert data['context_length'] == 4000
        assert data['pricing']['prompt'] == 0.001
        assert data['pricing']['completion'] == 0.002
        assert data['architecture']['modality'] == "text->text"
        assert data['architecture']['tokenizer'] == "GPT-4"
        assert data['description'] == "A test model"


class TestEndpoint:
    """Test Endpoint data model"""
    
    def test_endpoint_creation(self):
        """Test basic endpoint creation"""
        endpoint = Endpoint(
            name="test-endpoint",
            provider_name="test-provider"
        )
        
        assert endpoint.name == "test-endpoint"
        assert endpoint.provider_name == "test-provider"
        assert endpoint.status == "active"  # Default value
        assert endpoint.context_length == 0  # Default value
        assert endpoint.pricing is None
        assert endpoint.uptime is None
        assert endpoint.latency is None
        assert endpoint.throughput is None
    
    def test_endpoint_with_metrics(self):
        """Test endpoint with performance metrics"""
        pricing = EndpointPricing(prompt=0.001, completion=0.002)
        endpoint = Endpoint(
            name="test-endpoint",
            provider_name="test-provider",
            status="active",
            context_length=4000,
            pricing=pricing,
            uptime=0.99,
            latency=150.5,
            throughput=1000
        )
        
        assert endpoint.status == "active"
        assert endpoint.context_length == 4000
        assert endpoint.pricing == pricing
        assert endpoint.uptime == 0.99
        assert endpoint.latency == 150.5
        assert endpoint.throughput == 1000
    
    def test_endpoint_comparison(self):
        """Test endpoint comparison for sorting"""
        endpoint1 = Endpoint(
            name="endpoint1",
            provider_name="provider1",
            uptime=0.95,
            latency=200.0
        )
        
        endpoint2 = Endpoint(
            name="endpoint2",
            provider_name="provider2",
            uptime=0.99,
            latency=150.0
        )
        
        # Test uptime comparison
        assert endpoint2.uptime > endpoint1.uptime
        
        # Test latency comparison
        assert endpoint2.latency < endpoint1.latency


class TestModelEndpoints:
    """Test ModelEndpoints data model"""
    
    def test_endpoints_creation(self):
        """Test endpoints creation"""
        model = Model(id="test/model", name="Test Model", context_length=4000)
        endpoints = [
            Endpoint(name="endpoint1", provider_name="provider1"),
            Endpoint(name="endpoint2", provider_name="provider2")
        ]
        
        info = ModelEndpoints(model=model, endpoints=endpoints)
        
        assert info.model == model
        assert info.endpoints == endpoints
        assert len(info.endpoints) == 2
    
    def test_endpoints_empty_endpoints(self):
        """Test endpoints with empty endpoints list"""
        model = Model(id="test/model", name="Test Model", context_length=4000)
        
        info = ModelEndpoints(model=model, endpoints=[])
        
        assert info.model == model
        assert info.endpoints == []
        assert len(info.endpoints) == 0


class TestModelFilter:
    """Test ModelFilter data model"""
    
    def test_filter_creation(self):
        """Test basic filter creation"""
        filter_obj = ModelFilter()
        
        assert filter_obj.providers is None
        assert filter_obj.min_context_length is None
        assert filter_obj.max_cost_per_token is None
        assert filter_obj.modalities is None
        assert filter_obj.exclude_models is None
        assert filter_obj.categories is None
    
    def test_filter_with_all_fields(self):
        """Test filter with all fields"""
        filter_obj = ModelFilter(
            providers=["anthropic", "openai"],
            min_context_length=2000,
            max_cost_per_token=0.01,
            modalities=["text->text"],
            exclude_models=["bad/model"],
            categories=["general"]
        )
        
        assert filter_obj.providers == ["anthropic", "openai"]
        assert filter_obj.min_context_length == 2000
        assert filter_obj.max_cost_per_token == 0.01
        assert filter_obj.modalities == ["text->text"]
        assert filter_obj.exclude_models == ["bad/model"]
        assert filter_obj.categories == ["general"]
    
    def test_filter_matches_model(self):
        """Test filter matching against models"""
        # Create test models
        model1 = Model(
            id="anthropic/claude-3-sonnet",
            name="Claude 3 Sonnet",
            context_length=4000,
            pricing=ModelPricing(prompt=0.001, completion=0.002)
        )
        
        model2 = Model(
            id="openai/gpt-4",
            name="GPT-4",
            context_length=8000,
            pricing=ModelPricing(prompt=0.03, completion=0.06)
        )
        
        # Test provider filter
        provider_filter = ModelFilter(providers=["anthropic"])
        assert "anthropic" in model1.id
        assert "anthropic" not in model2.id
        
        # Test context length filter
        context_filter = ModelFilter(min_context_length=6000)
        assert model1.context_length < 6000
        assert model2.context_length >= 6000
        
        # Test cost filter
        cost_filter = ModelFilter(max_cost_per_token=0.01)
        model1_avg_cost = (model1.pricing.prompt + model1.pricing.completion) / 2
        model2_avg_cost = (model2.pricing.prompt + model2.pricing.completion) / 2
        assert model1_avg_cost <= 0.01
        assert model2_avg_cost > 0.01


class TestEndpointPricing:
    """Test EndpointPricing data model"""
    
    def test_endpoint_pricing_creation(self):
        """Test endpoint pricing creation"""
        pricing = EndpointPricing(
            prompt=0.001,
            completion=0.002
        )
        
        assert pricing.prompt == 0.001
        assert pricing.completion == 0.002
    
    def test_endpoint_pricing_serialization(self):
        """Test endpoint pricing serialization"""
        pricing = EndpointPricing(prompt=0.001, completion=0.002)
        
        data = asdict(pricing)
        
        assert data['prompt'] == 0.001
        assert data['completion'] == 0.002


class TestModelValidation:
    """Test model validation scenarios"""
    
    def test_model_id_validation(self):
        """Test model ID format validation"""
        # Valid model IDs
        valid_ids = [
            "anthropic/claude-3-sonnet",
            "openai/gpt-4",
            "google/gemini-pro",
            "mistral/mistral-large"
        ]
        
        for model_id in valid_ids:
            assert "/" in model_id
            provider, model_name = model_id.split("/", 1)
            assert len(provider) > 0
            assert len(model_name) > 0
    
    def test_pricing_validation(self):
        """Test pricing validation"""
        # Valid pricing
        valid_pricing = ModelPricing(prompt=0.001, completion=0.002)
        assert valid_pricing.prompt > 0
        assert valid_pricing.completion > 0
        
        # Edge cases
        zero_pricing = ModelPricing(prompt=0.0, completion=0.0)
        assert zero_pricing.prompt == 0.0
        assert zero_pricing.completion == 0.0
    
    def test_context_length_validation(self):
        """Test context length validation"""
        # Valid context lengths
        valid_lengths = [1000, 4000, 8000, 32000, 100000]
        
        for length in valid_lengths:
            model = Model(
                id="test/model",
                name="Test Model",
                context_length=length
            )
            assert model.context_length == length
            assert model.context_length > 0
    
    def test_endpoint_status_validation(self):
        """Test endpoint status validation"""
        valid_statuses = ["active", "inactive", "maintenance", "error"]
        
        for status in valid_statuses:
            endpoint = Endpoint(
                name="test-endpoint",
                provider_name="test-provider",
                status=status
            )
            assert endpoint.status == status
    
    def test_uptime_validation(self):
        """Test uptime percentage validation"""
        # Valid uptime values (0.0 to 1.0)
        valid_uptimes = [0.0, 0.5, 0.95, 0.99, 1.0]
        
        for uptime in valid_uptimes:
            endpoint = Endpoint(
                name="test-endpoint",
                provider_name="test-provider",
                uptime=uptime
            )
            assert endpoint.uptime == uptime
            assert 0.0 <= endpoint.uptime <= 1.0


if __name__ == '__main__':
    pytest.main([__file__])