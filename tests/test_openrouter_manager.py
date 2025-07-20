"""
Tests for OpenRouter Manager functionality
"""

import pytest
import os
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.openrouter_manager import OpenRouterManager, create_openrouter_manager
from src.models.openrouter_models import Provider, Model, ModelPricing, ModelFilter, Endpoint


class TestOpenRouterManager:
    """Test cases for OpenRouterManager"""
    
    def test_init_with_api_key(self):
        """Test initialization with API key"""
        config = {
            'api_key': 'test_key',
            'base_url': 'https://openrouter.ai/api/v1',
            'cache_enabled': True,
            'cache_ttl': 3600
        }
        manager = OpenRouterManager(config)
        
        assert manager.api_key == 'test_key'
        assert manager.base_url == 'https://openrouter.ai/api/v1'
        assert manager.cache_enabled is True
        assert manager.cache_ttl == 3600
    
    def test_init_with_env_var(self):
        """Test initialization with environment variable"""
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'env_key'}):
            config = {}
            manager = OpenRouterManager(config)
            
            assert manager.api_key == 'env_key'
    
    def test_is_available_with_api_key(self):
        """Test availability check with API key"""
        config = {'api_key': 'test_key'}
        manager = OpenRouterManager(config)
        
        assert manager.is_available() is True
    
    def test_is_available_without_api_key(self):
        """Test availability check without API key"""
        with patch.dict(os.environ, {}, clear=True):
            config = {}
            manager = OpenRouterManager(config)
            
            assert manager.is_available() is False
    
    def test_get_headers(self):
        """Test header generation"""
        config = {
            'api_key': 'test_key',
            'site_url': 'https://test.com',
            'site_name': 'Test App'
        }
        manager = OpenRouterManager(config)
        
        headers = manager._get_headers()
        
        assert headers['Authorization'] == 'Bearer test_key'
        assert headers['HTTP-Referer'] == 'https://test.com'
        assert headers['X-Title'] == 'Test App'
        assert headers['Content-Type'] == 'application/json'
    
    @patch('src.openrouter_manager.requests.request')
    def test_make_request_success(self, mock_request):
        """Test successful API request"""
        config = {'api_key': 'test_key'}
        manager = OpenRouterManager(config)
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'test_data'}
        mock_request.return_value = mock_response
        
        result = manager._make_request('/test')
        
        assert result == {'data': 'test_data'}
        mock_request.assert_called_once()
    
    @patch('src.openrouter_manager.requests.request')
    def test_make_request_rate_limit_retry(self, mock_request):
        """Test rate limit handling with retry"""
        config = {'api_key': 'test_key', 'max_retries': 2, 'retry_delay': 0.1}
        manager = OpenRouterManager(config)
        
        # First call: rate limit, second call: success
        rate_limit_response = Mock()
        rate_limit_response.status_code = 429
        
        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = {'data': 'success'}
        
        mock_request.side_effect = [rate_limit_response, success_response]
        
        result = manager._make_request('/test')
        
        assert result == {'data': 'success'}
        assert mock_request.call_count == 2
    
    @patch('src.openrouter_manager.requests.request')
    def test_make_request_failure(self, mock_request):
        """Test API request failure"""
        config = {'api_key': 'test_key', 'max_retries': 2}
        manager = OpenRouterManager(config)
        
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception('Server Error')
        mock_request.return_value = mock_response
        
        with pytest.raises(Exception):
            manager._make_request('/test')
    
    def test_cache_functionality(self):
        """Test caching functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'api_key': 'test_key',
                'cache_enabled': True,
                'cache_ttl': 3600,
                'cache_dir': temp_dir
            }
            manager = OpenRouterManager(config)
            
            # Mock fetch function
            mock_fetch = Mock(return_value={'data': 'test'})
            
            # First call - should call fetch function
            result1 = manager._get_cached_or_fetch('test_key', mock_fetch)
            assert result1 == {'data': 'test'}
            assert mock_fetch.call_count == 1
            
            # Second call - should use cache
            result2 = manager._get_cached_or_fetch('test_key', mock_fetch)
            assert result2 == {'data': 'test'}
            assert mock_fetch.call_count == 1  # Not called again
    
    def test_cache_disabled(self):
        """Test behavior when cache is disabled"""
        config = {
            'api_key': 'test_key',
            'cache_enabled': False
        }
        manager = OpenRouterManager(config)
        
        mock_fetch = Mock(return_value={'data': 'test'})
        
        # Each call should fetch fresh data
        result1 = manager._get_cached_or_fetch('test_key', mock_fetch)
        result2 = manager._get_cached_or_fetch('test_key', mock_fetch)
        
        assert result1 == {'data': 'test'}
        assert result2 == {'data': 'test'}
        assert mock_fetch.call_count == 2
    
    def test_clear_cache(self):
        """Test cache clearing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'api_key': 'test_key',
                'cache_enabled': True,
                'cache_dir': temp_dir
            }
            manager = OpenRouterManager(config)
            
            # Add some cache entries
            manager._cache['key1'] = Mock()
            manager._cache['key2'] = Mock()
            
            # Clear specific key
            manager.clear_cache('key1')
            assert 'key1' not in manager._cache
            assert 'key2' in manager._cache
            
            # Clear all cache
            manager.clear_cache()
            assert len(manager._cache) == 0
    
    @patch('src.openrouter_manager.OpenRouterManager._make_request')
    def test_get_providers(self, mock_request):
        """Test provider retrieval"""
        config = {'api_key': 'test_key', 'cache_enabled': False}
        manager = OpenRouterManager(config)
        
        mock_request.return_value = {
            'data': [
                {
                    'name': 'Test Provider',
                    'slug': 'test-provider',
                    'status': 'active',
                    'may_log_prompts': False,
                    'may_train_on_data': False
                }
            ]
        }
        
        providers = manager.get_providers()
        
        assert len(providers) == 1
        assert providers[0].name == 'Test Provider'
        assert providers[0].slug == 'test-provider'
        assert providers[0].status == 'active'
        mock_request.assert_called_with('/providers')
    
    @patch('src.openrouter_manager.OpenRouterManager._make_request')
    def test_get_models(self, mock_request):
        """Test model retrieval"""
        config = {'api_key': 'test_key', 'cache_enabled': False}
        manager = OpenRouterManager(config)
        
        mock_request.return_value = {
            'data': [
                {
                    'id': 'test/model',
                    'name': 'Test Model',
                    'context_length': 4000,
                    'pricing': {
                        'prompt': 0.001,
                        'completion': 0.002
                    }
                }
            ]
        }
        
        models = manager.get_models()
        
        assert len(models) == 1
        assert models[0].id == 'test/model'
        assert models[0].name == 'Test Model'
        assert models[0].context_length == 4000
        assert models[0].pricing.prompt == 0.001
        mock_request.assert_called_with('/models', params={})
    
    @patch('src.openrouter_manager.OpenRouterManager._make_request')
    def test_get_models_with_category(self, mock_request):
        """Test model retrieval with category filter"""
        config = {'api_key': 'test_key', 'cache_enabled': False}
        manager = OpenRouterManager(config)
        
        mock_request.return_value = {'data': []}
        
        manager.get_models(category='programming')
        
        mock_request.assert_called_with('/models', params={'category': 'programming'})
    
    @patch('src.openrouter_manager.OpenRouterManager.get_models')
    def test_get_model_by_id(self, mock_get_models):
        """Test getting model by ID"""
        config = {'api_key': 'test_key'}
        manager = OpenRouterManager(config)
        
        test_model = Model(
            id='test/model',
            name='Test Model',
            context_length=4000,
            pricing=ModelPricing(prompt=0.001, completion=0.002)
        )
        mock_get_models.return_value = [test_model]
        
        result = manager.get_model_by_id('test/model')
        
        assert result is not None
        assert result.id == 'test/model'
        assert result.name == 'Test Model'
        
        # Test non-existent model
        result = manager.get_model_by_id('nonexistent/model')
        assert result is None
    
    @patch('src.openrouter_manager.OpenRouterManager.get_model_by_id')
    def test_validate_model(self, mock_get_model):
        """Test model validation"""
        config = {'api_key': 'test_key'}
        manager = OpenRouterManager(config)
        
        # Test existing model
        mock_get_model.return_value = Mock()
        assert manager.validate_model('test/model') is True
        
        # Test non-existent model
        mock_get_model.return_value = None
        assert manager.validate_model('nonexistent/model') is False
    
    @patch('src.openrouter_manager.OpenRouterManager._make_request')
    def test_get_model_endpoints(self, mock_request):
        """Test endpoint retrieval for model"""
        config = {'api_key': 'test_key', 'cache_enabled': False}
        manager = OpenRouterManager(config)
        
        mock_request.return_value = {
            'id': 'test/model',
            'name': 'Test Model',
            'context_length': 4000,
            'endpoints': [
                {
                    'name': 'test-endpoint',
                    'provider_name': 'test-provider',
                    'status': 'active',
                    'context_length': 4000,
                    'pricing': {
                        'prompt': 0.001,
                        'completion': 0.002
                    }
                }
            ]
        }
        
        result = manager.get_model_endpoints('test/model')
        
        assert result is not None
        assert result.model.id == 'test/model'
        assert len(result.endpoints) == 1
        assert result.endpoints[0].name == 'test-endpoint'
        mock_request.assert_called_with('/models/test/model/endpoints')
    
    @patch('src.openrouter_manager.OpenRouterManager.get_model_endpoints')
    def test_get_best_endpoint(self, mock_get_endpoints):
        """Test getting best endpoint"""
        config = {'api_key': 'test_key'}
        manager = OpenRouterManager(config)
        
        # Mock endpoints with different costs
        endpoint1 = Endpoint(
            name='expensive',
            provider_name='provider1',
            pricing=Mock(prompt=0.01, completion=0.02)
        )
        endpoint2 = Endpoint(
            name='cheap',
            provider_name='provider2',
            pricing=Mock(prompt=0.001, completion=0.002)
        )
        
        mock_endpoints = Mock()
        mock_endpoints.endpoints = [endpoint1, endpoint2]
        mock_get_endpoints.return_value = mock_endpoints
        
        result = manager.get_best_endpoint('test/model', 'cost')
        
        assert result is not None
        assert result.name == 'cheap'
    
    @patch('src.openrouter_manager.OpenRouterManager.get_models')
    def test_filter_models(self, mock_get_models):
        """Test model filtering"""
        config = {'api_key': 'test_key'}
        manager = OpenRouterManager(config)
        
        # Create test models
        model1 = Model(
            id='test/model1',
            name='Test Model 1',
            context_length=4000,
            pricing=ModelPricing(prompt=0.001, completion=0.002)
        )
        model2 = Model(
            id='test/model2',
            name='Test Model 2',
            context_length=8000,
            pricing=ModelPricing(prompt=0.01, completion=0.02)
        )
        
        mock_get_models.return_value = [model1, model2]
        
        # Filter by context length
        filter_criteria = ModelFilter(min_context_length=6000)
        result = manager.filter_models(filter_criteria)
        
        assert len(result) == 1
        assert result[0].id == 'test/model2'
    
    @patch('src.openrouter_manager.OpenRouterManager.get_models')
    def test_get_recommendations(self, mock_get_models):
        """Test getting model recommendations"""
        config = {'api_key': 'test_key'}
        manager = OpenRouterManager(config)
        
        # Create test models
        model1 = Model(
            id='test/model1',
            name='Test Model 1',
            context_length=4000,
            pricing=ModelPricing(prompt=0.001, completion=0.002)
        )
        model2 = Model(
            id='test/model2',
            name='Test Model 2',
            context_length=8000,
            pricing=ModelPricing(prompt=0.01, completion=0.02)
        )
        
        mock_get_models.return_value = [model1, model2]
        
        recommendations = manager.get_recommendations('general', budget=0.005)
        
        assert len(recommendations) == 1
        assert recommendations[0].id == 'test/model1'


class TestCreateOpenRouterManager:
    """Test create_openrouter_manager function"""
    
    def test_create_with_config(self):
        """Test creating manager with configuration"""
        config = {
            'openrouter': {
                'api_key': 'test_key',
                'base_url': 'https://openrouter.ai/api/v1'
            }
        }
        
        manager = create_openrouter_manager(config)
        
        assert isinstance(manager, OpenRouterManager)
        assert manager.api_key == 'test_key'
        assert manager.base_url == 'https://openrouter.ai/api/v1'
    
    def test_create_without_openrouter_config(self):
        """Test creating manager without openrouter config"""
        config = {}
        
        manager = create_openrouter_manager(config)
        
        assert isinstance(manager, OpenRouterManager)
        assert manager.base_url == 'https://openrouter.ai/api/v1'  # Default value


if __name__ == '__main__':
    pytest.main([__file__])