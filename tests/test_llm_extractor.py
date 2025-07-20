"""
Tests for OpenRouter LLM integration
"""

import pytest
import os
import json
from unittest.mock import Mock, patch, MagicMock
from src.llm_extractor import (
    OpenRouterProvider, 
    ExtractionItem, 
    ExtractionResult,
    LLMExtractor
)


class TestOpenRouterProvider:
    """Test cases for OpenRouterProvider"""
    
    def test_init_with_api_key(self):
        """Test initialization with API key"""
        config = {
            'api_key': 'test_key',
            'model': 'anthropic/claude-3-sonnet',
            'base_url': 'https://openrouter.ai/api/v1',
            'temperature': 0.1,
            'max_tokens': 2000
        }
        provider = OpenRouterProvider(config)
        
        assert provider.api_key == 'test_key'
        assert provider.model == 'anthropic/claude-3-sonnet'
        assert provider.base_url == 'https://openrouter.ai/api/v1'
        assert provider.temperature == 0.1
        assert provider.max_tokens == 2000
    
    def test_init_with_env_var(self):
        """Test initialization with environment variable"""
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'env_key'}):
            config = {}
            provider = OpenRouterProvider(config)
            
            assert provider.api_key == 'env_key'
    
    def test_is_available_with_api_key(self):
        """Test availability check with API key"""
        config = {'api_key': 'test_key'}
        provider = OpenRouterProvider(config)
        
        assert provider.is_available() is True
    
    def test_is_available_without_api_key(self):
        """Test availability check without API key"""
        config = {}
        provider = OpenRouterProvider(config)
        
        assert provider.is_available() is False
    
    @patch('src.llm_extractor.requests.post')
    def test_successful_api_call(self, mock_post):
        """Test successful OpenRouter API call"""
        config = {'api_key': 'test_key'}
        provider = OpenRouterProvider(config)
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [
                {
                    'message': {
                        'content': '{"手法の肝": "テスト手法", "制限事項": "テスト制限"}'
                    }
                }
            ],
            'usage': {
                'prompt_tokens': 100,
                'completion_tokens': 50,
                'total_tokens': 150
            }
        }
        mock_post.return_value = mock_response
        
        result = provider._call_openrouter_api("test prompt")
        
        assert 'choices' in result
        assert len(result['choices']) > 0
        assert 'usage' in result
        
        # Check headers
        call_args = mock_post.call_args
        headers = call_args[1]['headers']
        assert 'Authorization' in headers
        assert 'HTTP-Referer' in headers
        assert 'X-Title' in headers
        assert headers['Authorization'] == 'Bearer test_key'
    
    @patch('src.llm_extractor.requests.post')
    def test_rate_limit_handling(self, mock_post):
        """Test rate limit handling"""
        config = {'api_key': 'test_key'}
        provider = OpenRouterProvider(config)
        
        # Mock rate limit response
        mock_response = Mock()
        mock_response.status_code = 429
        mock_post.return_value = mock_response
        
        with pytest.raises(Exception) as exc_info:
            provider._call_openrouter_api("test prompt")
        
        assert "Rate limit exceeded" in str(exc_info.value)
    
    def test_create_extraction_prompt(self):
        """Test extraction prompt creation"""
        config = {'api_key': 'test_key'}
        provider = OpenRouterProvider(config)
        
        items = [
            ExtractionItem(name="手法の肝", description="核となる技術手法", max_length=500),
            ExtractionItem(name="制限事項", description="手法の限界", max_length=500)
        ]
        
        metadata = {
            'title': 'Test Paper',
            'authors': ['Author 1', 'Author 2'],
            'categories': ['cs.AI'],
            'published': '2023-01-01',
            'abs_url': 'https://arxiv.org/abs/2301.00001'
        }
        
        prompt = provider._create_extraction_prompt("test text", items, metadata)
        
        assert "手法の肝" in prompt
        assert "制限事項" in prompt
        assert "Test Paper" in prompt
        assert "Author 1" in prompt
        assert "JSON形式" in prompt
        assert "test text" in prompt
    
    def test_parse_successful_response(self):
        """Test parsing successful response"""
        config = {'api_key': 'test_key'}
        provider = OpenRouterProvider(config)
        
        items = [
            ExtractionItem(name="手法の肝", description="核となる技術手法", max_length=500),
            ExtractionItem(name="制限事項", description="手法の限界", max_length=500)
        ]
        
        response = {
            'choices': [
                {
                    'message': {
                        'content': json.dumps({
                            "手法の肝": "深層学習を用いた画像認識",
                            "制限事項": "大量のデータが必要"
                        })
                    }
                }
            ],
            'usage': {
                'prompt_tokens': 100,
                'completion_tokens': 50
            }
        }
        
        result = provider._parse_openrouter_response(response, items)
        
        assert result.success is True
        assert result.data["手法の肝"] == "深層学習を用いた画像認識"
        assert result.data["制限事項"] == "大量のデータが必要"
        assert result.confidence_scores["手法の肝"] > 0
        assert result.metadata['usage']['prompt_tokens'] == 100
    
    def test_parse_json_parse_error(self):
        """Test handling JSON parse errors"""
        config = {'api_key': 'test_key'}
        provider = OpenRouterProvider(config)
        
        items = [
            ExtractionItem(name="手法の肝", description="核となる技術手法", max_length=500)
        ]
        
        response = {
            'choices': [
                {
                    'message': {
                        'content': '手法の肝: 深層学習を用いた画像認識'  # Invalid JSON
                    }
                }
            ]
        }
        
        result = provider._parse_openrouter_response(response, items)
        
        # Should fallback to text extraction
        assert result.success is True
        assert "深層学習を用いた画像認識" in result.data["手法の肝"]
    
    def test_extract_from_text(self):
        """Test text extraction fallback"""
        config = {'api_key': 'test_key'}
        provider = OpenRouterProvider(config)
        
        items = [
            ExtractionItem(name="手法の肝", description="核となる技術手法", max_length=500),
            ExtractionItem(name="制限事項", description="手法の限界", max_length=500)
        ]
        
        text = '''
        手法の肝: 深層学習を用いた画像認識手法
        制限事項: 大量のデータが必要で計算コストが高い
        '''
        
        result = provider._extract_from_text(text, items)
        
        assert "深層学習を用いた画像認識手法" in result["手法の肝"]
        assert "大量のデータが必要で計算コストが高い" in result["制限事項"]
    
    def test_calculate_confidence(self):
        """Test confidence calculation"""
        config = {'api_key': 'test_key'}
        provider = OpenRouterProvider(config)
        
        # Test empty value
        assert provider._calculate_confidence("") == 0.0
        assert provider._calculate_confidence("N/A") == 0.0
        
        # Test short value
        confidence = provider._calculate_confidence("短い")
        assert 0.0 < confidence < 1.0
        
        # Test longer value
        confidence = provider._calculate_confidence("これは長いテキストで、具体的な手法について説明しています")
        assert confidence > 0.5
        
        # Test with keywords
        confidence = provider._calculate_confidence("この手法は深層学習を用いたmethod")
        assert confidence > 0.7
    
    @patch('src.llm_extractor.requests.post')
    def test_full_extraction_flow(self, mock_post):
        """Test complete extraction flow"""
        config = {
            'api_key': 'test_key',
            'model': 'anthropic/claude-3-sonnet',
            'max_retries': 2
        }
        provider = OpenRouterProvider(config)
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [
                {
                    'message': {
                        'content': json.dumps({
                            "手法の肝": "Transformer architecture",
                            "制限事項": "Requires large computational resources"
                        })
                    }
                }
            ],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 50}
        }
        mock_post.return_value = mock_response
        
        items = [
            ExtractionItem(name="手法の肝", description="核となる技術手法", max_length=500),
            ExtractionItem(name="制限事項", description="手法の限界", max_length=500)
        ]
        
        result = provider.extract_information("test paper text", items)
        
        assert result.success is True
        assert result.data["手法の肝"] == "Transformer architecture"
        assert result.data["制限事項"] == "Requires large computational resources"
        assert result.model_used == 'anthropic/claude-3-sonnet'
        assert result.processing_time > 0


class TestLLMExtractorWithOpenRouter:
    """Test LLMExtractor with OpenRouter integration"""
    
    def test_init_with_openrouter_config(self):
        """Test LLMExtractor initialization with OpenRouter config"""
        config = {
            'model': 'anthropic/claude-3-sonnet',
            'openrouter': {
                'api_key': 'test_key',
                'model': 'anthropic/claude-3-sonnet'
            }
        }
        
        extractor = LLMExtractor(config)
        
        assert 'anthropic/claude-3-sonnet' in extractor.providers
        assert extractor.primary_provider == 'anthropic/claude-3-sonnet'
    
    def test_available_models_includes_openrouter(self):
        """Test that available models include OpenRouter models"""
        config = {
            'openrouter': {
                'api_key': 'test_key',
                'model': 'anthropic/claude-3-sonnet'
            }
        }
        
        extractor = LLMExtractor(config)
        available_models = extractor.get_available_models()
        
        assert 'anthropic/claude-3-sonnet' in available_models
        assert 'openai/gpt-4' in available_models
        assert 'mistral/mistral-large' in available_models
    
    @patch('src.llm_extractor.requests.post')
    def test_extraction_with_openrouter_fallback(self, mock_post):
        """Test extraction with OpenRouter as fallback"""
        config = {
            'model': 'gpt-4',  # Primary model (not available)
            'fallback_models': ['anthropic/claude-3-sonnet'],
            'openrouter': {
                'api_key': 'test_key',
                'model': 'anthropic/claude-3-sonnet'
            }
        }
        
        extractor = LLMExtractor(config)
        
        # Mock successful OpenRouter response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [
                {
                    'message': {
                        'content': json.dumps({
                            "手法の肝": "Neural networks",
                            "制限事項": "Overfitting issues",
                            "対象ナレッジ": "Machine learning",
                            "URL": "https://arxiv.org/abs/2301.00001",
                            "タイトル": "Test Paper",
                            "出版年": "2023",
                            "研究分野": "AI",
                            "課題設定": "Image classification",
                            "論文の主張": "Improved accuracy"
                        })
                    }
                }
            ],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 50}
        }
        mock_post.return_value = mock_response
        
        result = extractor.extract_from_text("test paper text")
        
        assert result.success is True
        assert result.data["手法の肝"] == "Neural networks"
        assert "anthropic/claude-3-sonnet" in result.model_used


if __name__ == '__main__':
    pytest.main([__file__])