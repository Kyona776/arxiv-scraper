"""
Tests for OCR API integration
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from src.ocr_processor import MistralOCRAPIProcessor, MistralOCROpenRouterProcessor, OCRResult


class TestMistralOCRAPIProcessor:
    """Test cases for MistralOCRAPIProcessor"""
    
    def test_init_with_api_key(self):
        """Test initialization with API key"""
        config = {
            'api_key': 'test_key',
            'model': 'mistral-ocr-latest',
            'base_url': 'https://api.mistral.ai'
        }
        processor = MistralOCRAPIProcessor(config)
        
        assert processor.api_key == 'test_key'
        assert processor.model == 'mistral-ocr-latest'
        assert processor.base_url == 'https://api.mistral.ai'
    
    def test_init_with_env_var(self):
        """Test initialization with environment variable"""
        with patch.dict(os.environ, {'MISTRAL_API_KEY': 'env_key'}):
            config = {}
            processor = MistralOCRAPIProcessor(config)
            
            assert processor.api_key == 'env_key'
    
    def test_is_available_with_api_key(self):
        """Test availability check with API key"""
        config = {'api_key': 'test_key'}
        processor = MistralOCRAPIProcessor(config)
        
        assert processor.is_available() is True
    
    def test_is_available_without_api_key(self):
        """Test availability check without API key"""
        config = {}
        processor = MistralOCRAPIProcessor(config)
        
        assert processor.is_available() is False
    
    def test_encode_pdf_to_base64(self):
        """Test PDF to base64 encoding"""
        config = {'api_key': 'test_key'}
        processor = MistralOCRAPIProcessor(config)
        
        # Mock file data
        mock_pdf_data = b'PDF content'
        
        with patch('builtins.open', mock_open(read_data=mock_pdf_data)):
            result = processor._encode_pdf_to_base64('test.pdf')
            
            # Check that it returns base64 encoded string
            import base64
            expected = base64.b64encode(mock_pdf_data).decode('utf-8')
            assert result == expected
    
    @patch('src.ocr_processor.requests.post')
    def test_successful_api_call(self, mock_post):
        """Test successful API call"""
        config = {'api_key': 'test_key'}
        processor = MistralOCRAPIProcessor(config)
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'text': 'Extracted text', 'confidence': 0.95}
        mock_post.return_value = mock_response
        
        document = {'type': 'document_url', 'document_url': 'data:application/pdf;base64,test'}
        result = processor._call_mistral_ocr_api(document)
        
        assert result == {'text': 'Extracted text', 'confidence': 0.95}
        mock_post.assert_called_once()
    
    @patch('src.ocr_processor.requests.post')
    def test_rate_limit_retry(self, mock_post):
        """Test rate limit handling with retry"""
        config = {'api_key': 'test_key', 'retry_delay': 0.1}
        processor = MistralOCRAPIProcessor(config)
        
        # Mock rate limit response followed by success
        rate_limit_response = Mock()
        rate_limit_response.status_code = 429
        
        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = {'text': 'Success after retry'}
        
        mock_post.side_effect = [rate_limit_response, success_response]
        
        document = {'type': 'document_url', 'document_url': 'data:application/pdf;base64,test'}
        result = processor._call_mistral_ocr_api(document)
        
        assert result == {'text': 'Success after retry'}
        assert mock_post.call_count == 2
    
    @patch('src.ocr_processor.fitz.open')
    @patch('src.ocr_processor.os.path.getsize')
    def test_process_pdf_file_too_large(self, mock_getsize, mock_fitz):
        """Test handling of PDF file that's too large"""
        config = {'api_key': 'test_key'}
        processor = MistralOCRAPIProcessor(config)
        
        # Mock file size > 50MB
        mock_getsize.return_value = 60 * 1024 * 1024  # 60MB
        
        result = processor.process_pdf('test.pdf')
        
        assert result.text == ""
        assert result.confidence == 0.0
        assert "exceeds maximum limit" in result.errors[0]


class TestMistralOCROpenRouterProcessor:
    """Test cases for MistralOCROpenRouterProcessor"""
    
    def test_init_with_api_key(self):
        """Test initialization with API key"""
        config = {
            'api_key': 'test_key',
            'model': 'mistral/mistral-ocr-latest',
            'base_url': 'https://openrouter.ai/api/v1'
        }
        processor = MistralOCROpenRouterProcessor(config)
        
        assert processor.api_key == 'test_key'
        assert processor.model == 'mistral/mistral-ocr-latest'
        assert processor.base_url == 'https://openrouter.ai/api/v1'
    
    def test_init_with_env_var(self):
        """Test initialization with environment variable"""
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'env_key'}):
            config = {}
            processor = MistralOCROpenRouterProcessor(config)
            
            assert processor.api_key == 'env_key'
    
    def test_is_available_with_api_key(self):
        """Test availability check with API key"""
        config = {'api_key': 'test_key'}
        processor = MistralOCROpenRouterProcessor(config)
        
        assert processor.is_available() is True
    
    def test_is_available_without_api_key(self):
        """Test availability check without API key"""
        config = {}
        processor = MistralOCROpenRouterProcessor(config)
        
        assert processor.is_available() is False
    
    @patch('src.ocr_processor.requests.post')
    def test_successful_api_call(self, mock_post):
        """Test successful OpenRouter API call"""
        config = {'api_key': 'test_key'}
        processor = MistralOCROpenRouterProcessor(config)
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{'message': {'content': 'Extracted text from OpenRouter'}}]
        }
        mock_post.return_value = mock_response
        
        messages = [{'role': 'user', 'content': 'test'}]
        result = processor._call_openrouter_ocr_api(messages)
        
        assert result == {'choices': [{'message': {'content': 'Extracted text from OpenRouter'}}]}
        mock_post.assert_called_once()
        
        # Check headers
        call_args = mock_post.call_args
        headers = call_args[1]['headers']
        assert 'Authorization' in headers
        assert 'HTTP-Referer' in headers
        assert 'X-Title' in headers
    
    @patch('src.ocr_processor.fitz.open')
    @patch('src.ocr_processor.os.path.getsize')
    def test_process_pdf_file_too_large(self, mock_getsize, mock_fitz):
        """Test handling of PDF file that's too large"""
        config = {'api_key': 'test_key'}
        processor = MistralOCROpenRouterProcessor(config)
        
        # Mock file size > 50MB
        mock_getsize.return_value = 60 * 1024 * 1024  # 60MB
        
        result = processor.process_pdf('test.pdf')
        
        assert result.text == ""
        assert result.confidence == 0.0
        assert "exceeds maximum limit" in result.errors[0]


def mock_open(read_data=b''):
    """Helper function to mock file opening"""
    mock = MagicMock()
    mock.return_value.__enter__ = mock
    mock.return_value.__exit__ = Mock()
    mock.return_value.read = Mock(return_value=read_data)
    return mock


if __name__ == '__main__':
    pytest.main([__file__])