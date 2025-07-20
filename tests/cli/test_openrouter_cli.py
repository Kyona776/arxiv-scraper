"""
Tests for OpenRouter CLI commands
"""

import pytest
import json
import sys
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
from pathlib import Path

from src.cli.openrouter_cli import OpenRouterCLI
from src.models.openrouter_models import Provider, Model, ModelPricing, Endpoint, ModelEndpoints, EndpointPricing


class TestOpenRouterCLI:
    """Test cases for OpenRouter CLI"""
    
    @pytest.fixture
    def mock_manager(self):
        """Mock OpenRouter manager"""
        with patch('src.cli.openrouter_cli.create_openrouter_manager') as mock_create:
            mock_manager = Mock()
            mock_manager.is_available.return_value = True
            mock_create.return_value = mock_manager
            yield mock_manager
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration"""
        with patch('src.cli.openrouter_cli.OpenRouterCLI._load_config') as mock_load:
            mock_load.return_value = {
                'openrouter': {
                    'api_key': 'test_key',
                    'base_url': 'https://openrouter.ai/api/v1'
                }
            }
            yield mock_load
    
    def test_init_with_available_manager(self, mock_manager, mock_config):
        """Test CLI initialization with available manager"""
        cli = OpenRouterCLI()
        assert cli.manager == mock_manager
        assert cli.config is not None
    
    def test_init_without_api_key(self, mock_config):
        """Test CLI initialization without API key"""
        with patch('src.cli.openrouter_cli.create_openrouter_manager') as mock_create:
            mock_manager = Mock()
            mock_manager.is_available.return_value = False
            mock_create.return_value = mock_manager
            
            with pytest.raises(SystemExit):
                OpenRouterCLI()
    
    def test_load_config_with_existing_file(self):
        """Test loading config from existing file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / 'config'
            config_dir.mkdir()
            config_file = config_dir / 'config.yaml'
            
            config_content = """
openrouter:
  api_key: test_key
  base_url: https://openrouter.ai/api/v1
"""
            config_file.write_text(config_content)
            
            # Mock the config path
            with patch('src.cli.openrouter_cli.Path.__truediv__', return_value=config_file):
                with patch('src.cli.openrouter_cli.create_openrouter_manager') as mock_create:
                    mock_manager = Mock()
                    mock_manager.is_available.return_value = True
                    mock_create.return_value = mock_manager
                    
                    cli = OpenRouterCLI()
                    assert cli.config['openrouter']['api_key'] == 'test_key'
    
    def test_load_config_without_file(self, mock_manager):
        """Test loading default config when file doesn't exist"""
        with patch('src.cli.openrouter_cli.Path.exists', return_value=False):
            cli = OpenRouterCLI()
            assert cli.config['openrouter']['base_url'] == 'https://openrouter.ai/api/v1'
    
    def test_list_providers_json_output(self, mock_manager, mock_config):
        """Test listing providers with JSON output"""
        # Mock providers
        providers = [
            Provider(
                name="Test Provider",
                slug="test-provider",
                status="active",
                may_log_prompts=False,
                may_train_on_data=False,
                moderated_by_openrouter=True
            )
        ]
        mock_manager.get_providers.return_value = providers
        
        cli = OpenRouterCLI()
        args = Mock()
        args.json = True
        
        # Capture stdout
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            cli.list_providers(args)
        
        # Verify JSON output
        output = captured_output.getvalue()
        parsed = json.loads(output)
        
        assert len(parsed) == 1
        assert parsed[0]['name'] == 'Test Provider'
        assert parsed[0]['slug'] == 'test-provider'
        assert parsed[0]['status'] == 'active'
        assert parsed[0]['may_log_prompts'] is False
        assert parsed[0]['may_train_on_data'] is False
        assert parsed[0]['moderated_by_openrouter'] is True
    
    def test_list_providers_table_output(self, mock_manager, mock_config):
        """Test listing providers with table output"""
        providers = [
            Provider(
                name="Test Provider",
                slug="test-provider",
                status="active",
                may_log_prompts=False,
                may_train_on_data=False,
                moderated_by_openrouter=True
            )
        ]
        mock_manager.get_providers.return_value = providers
        
        cli = OpenRouterCLI()
        args = Mock()
        args.json = False
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            cli.list_providers(args)
        
        output = captured_output.getvalue()
        assert "Test Provider" in output
        assert "test-provider" in output
        assert "active" in output
        assert "Total providers: 1" in output
    
    def test_list_models_json_output(self, mock_manager, mock_config):
        """Test listing models with JSON output"""
        models = [
            Model(
                id="test/model",
                name="Test Model",
                context_length=4000,
                pricing=ModelPricing(prompt=0.001, completion=0.002),
                top_provider={'name': 'Test Provider'}
            )
        ]
        mock_manager.get_models.return_value = models
        
        cli = OpenRouterCLI()
        args = Mock()
        args.json = True
        args.category = None
        args.provider = None
        args.max_cost = None
        args.min_context = None
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            cli.list_models(args)
        
        output = captured_output.getvalue()
        parsed = json.loads(output)
        
        assert len(parsed) == 1
        assert parsed[0]['id'] == 'test/model'
        assert parsed[0]['name'] == 'Test Model'
        assert parsed[0]['context_length'] == 4000
        assert parsed[0]['pricing']['prompt'] == 0.001
        assert parsed[0]['pricing']['completion'] == 0.002
        assert parsed[0]['provider'] == 'Test Provider'
    
    def test_list_models_with_filters(self, mock_manager, mock_config):
        """Test listing models with filters"""
        models = [
            Model(
                id="test/model",
                name="Test Model",
                context_length=4000,
                pricing=ModelPricing(prompt=0.001, completion=0.002)
            )
        ]
        mock_manager.get_models.return_value = models
        mock_manager.filter_models.return_value = models
        
        cli = OpenRouterCLI()
        args = Mock()
        args.json = False
        args.category = None
        args.provider = "test"
        args.max_cost = 0.01
        args.min_context = 2000
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            cli.list_models(args)
        
        # Verify filter_models was called
        mock_manager.filter_models.assert_called_once()
        filter_arg = mock_manager.filter_models.call_args[0][0]
        assert filter_arg.providers == ["test"]
        assert filter_arg.max_cost_per_token == 0.01
        assert filter_arg.min_context_length == 2000
    
    def test_check_model_exists(self, mock_manager, mock_config):
        """Test checking existing model"""
        model_info = {
            'exists': True,
            'model': {
                'id': 'test/model',
                'name': 'Test Model',
                'context_length': 4000,
                'pricing': {'prompt': 0.001, 'completion': 0.002},
                'description': 'A test model'
            },
            'endpoints': [
                {'name': 'test-endpoint', 'provider_name': 'test-provider'}
            ]
        }
        mock_manager.get_model_info.return_value = model_info
        
        cli = OpenRouterCLI()
        args = Mock()
        args.model_id = 'test/model'
        args.json = False
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            cli.check_model(args)
        
        output = captured_output.getvalue()
        assert "Model: Test Model" in output
        assert "ID: test/model" in output
        assert "Context Length: 4,000" in output
        assert "Prompt Cost: $0.001000" in output
        assert "✓ Model is available" in output
    
    def test_check_model_not_exists(self, mock_manager, mock_config):
        """Test checking non-existent model"""
        model_info = {'exists': False}
        mock_manager.get_model_info.return_value = model_info
        
        cli = OpenRouterCLI()
        args = Mock()
        args.model_id = 'nonexistent/model'
        args.json = False
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            cli.check_model(args)
        
        output = captured_output.getvalue()
        assert "✗ Model 'nonexistent/model' not found" in output
    
    def test_model_info_detailed(self, mock_manager, mock_config):
        """Test getting detailed model information"""
        model_info = {
            'exists': True,
            'model': {
                'id': 'test/model',
                'name': 'Test Model',
                'context_length': 4000,
                'pricing': {'prompt': 0.001, 'completion': 0.002},
                'description': 'A test model for testing',
                'architecture': {
                    'modality': 'text->text',
                    'tokenizer': 'GPT-4',
                    'instruct_type': 'chat'
                }
            },
            'endpoints': [
                {
                    'name': 'test-endpoint',
                    'provider_name': 'test-provider',
                    'status': 'active',
                    'uptime': 0.99
                }
            ]
        }
        mock_manager.get_model_info.return_value = model_info
        
        cli = OpenRouterCLI()
        args = Mock()
        args.model_id = 'test/model'
        args.json = False
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            cli.model_info(args)
        
        output = captured_output.getvalue()
        assert "Model Information: Test Model" in output
        assert "ID: test/model" in output
        assert "Context Length: 4,000" in output
        assert "Pricing:" in output
        assert "Architecture:" in output
        assert "Modality: text->text" in output
        assert "Tokenizer: GPT-4" in output
        assert "Description:" in output
        assert "A test model for testing" in output
        assert "Endpoints (1):" in output
        assert "• test-endpoint" in output
        assert "Uptime: 99.0%" in output
    
    def test_check_endpoints(self, mock_manager, mock_config):
        """Test checking model endpoints"""
        model = Model(id='test/model', name='Test Model', context_length=4000)
        endpoints = [
            Endpoint(
                name='test-endpoint',
                provider_name='test-provider',
                status='active',
                context_length=4000,
                uptime=0.99,
                latency=150.0
            )
        ]
        endpoints_info = ModelEndpoints(model=model, endpoints=endpoints)
        mock_manager.get_model_endpoints.return_value = endpoints_info
        
        cli = OpenRouterCLI()
        args = Mock()
        args.model_id = 'test/model'
        args.json = False
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            cli.check_endpoints(args)
        
        output = captured_output.getvalue()
        assert "Endpoints for Test Model:" in output
        assert "test-endpoint" in output
        assert "test-provider" in output
        assert "active" in output
        assert "4,000" in output
        assert "99.0%" in output
        assert "150ms" in output
        assert "Total endpoints: 1" in output
    
    def test_recommend_models(self, mock_manager, mock_config):
        """Test getting model recommendations"""
        models = [
            Model(
                id="test/model1",
                name="Test Model 1",
                context_length=4000,
                pricing=ModelPricing(prompt=0.001, completion=0.002),
                top_provider={'name': 'Test Provider'}
            ),
            Model(
                id="test/model2",
                name="Test Model 2",
                context_length=8000,
                pricing=ModelPricing(prompt=0.005, completion=0.010),
                top_provider={'name': 'Test Provider'}
            )
        ]
        mock_manager.get_recommendations.return_value = models
        
        cli = OpenRouterCLI()
        args = Mock()
        args.task_type = 'coding'
        args.budget = 0.01
        args.json = False
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            cli.recommend_models(args)
        
        output = captured_output.getvalue()
        assert "Recommended models for coding tasks:" in output
        assert "test/model1" in output
        assert "test/model2" in output
        assert "Test Model 1" in output
        assert "Test Model 2" in output
        assert "Showing top 2 recommendations" in output
        
        # Verify manager was called with correct parameters
        mock_manager.get_recommendations.assert_called_once_with(
            task_type='coding',
            budget=0.01
        )
    
    def test_compare_models(self, mock_manager, mock_config):
        """Test comparing multiple models"""
        comparison = {
            'test/model1': {
                'exists': True,
                'name': 'Test Model 1',
                'context_length': 4000,
                'pricing': {'prompt': 0.001, 'completion': 0.002},
                'provider': 'Test Provider'
            },
            'test/model2': {
                'exists': True,
                'name': 'Test Model 2',
                'context_length': 8000,
                'pricing': {'prompt': 0.005, 'completion': 0.010},
                'provider': 'Test Provider'
            },
            'nonexistent/model': {
                'exists': False
            }
        }
        mock_manager.compare_models.return_value = comparison
        
        cli = OpenRouterCLI()
        args = Mock()
        args.model_ids = ['test/model1', 'test/model2', 'nonexistent/model']
        args.json = False
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            cli.compare_models(args)
        
        output = captured_output.getvalue()
        assert "Model Comparison:" in output
        assert "test/model1" in output
        assert "test/model2" in output
        assert "nonexistent/model" in output
        assert "Test Model 1" in output
        assert "Test Model 2" in output
        assert "NOT FOUND" in output
        assert "$0.001000" in output
        assert "$0.002000" in output
        
        # Verify manager was called with correct parameters
        mock_manager.compare_models.assert_called_once_with(
            ['test/model1', 'test/model2', 'nonexistent/model']
        )
    
    def test_clear_cache_all(self, mock_manager, mock_config):
        """Test clearing all cache"""
        cli = OpenRouterCLI()
        args = Mock()
        args.key = None
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            cli.clear_cache(args)
        
        output = captured_output.getvalue()
        assert "All cache cleared" in output
        
        # Verify manager was called correctly
        mock_manager.clear_cache.assert_called_once_with(None)
    
    def test_clear_cache_specific_key(self, mock_manager, mock_config):
        """Test clearing specific cache key"""
        cli = OpenRouterCLI()
        args = Mock()
        args.key = 'test_key'
        
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            cli.clear_cache(args)
        
        output = captured_output.getvalue()
        assert "Cache cleared for key: test_key" in output
        
        # Verify manager was called correctly
        mock_manager.clear_cache.assert_called_once_with('test_key')
    
    def test_error_handling(self, mock_manager, mock_config):
        """Test error handling in CLI commands"""
        mock_manager.get_providers.side_effect = Exception("API Error")
        
        cli = OpenRouterCLI()
        args = Mock()
        args.json = False
        
        with pytest.raises(SystemExit):
            cli.list_providers(args)


class TestOpenRouterCLIMain:
    """Test the main CLI entry point"""
    
    def test_main_no_command(self):
        """Test main function with no command"""
        with patch('sys.argv', ['openrouter_cli.py']):
            with patch('src.cli.openrouter_cli.OpenRouterCLI') as mock_cli_class:
                with patch('argparse.ArgumentParser.print_help') as mock_help:
                    from src.cli.openrouter_cli import main
                    main()
                    mock_help.assert_called_once()
    
    def test_main_with_command(self):
        """Test main function with command"""
        with patch('sys.argv', ['openrouter_cli.py', 'list-providers']):
            with patch('src.cli.openrouter_cli.OpenRouterCLI') as mock_cli_class:
                mock_cli = Mock()
                mock_cli_class.return_value = mock_cli
                
                from src.cli.openrouter_cli import main
                main()
                
                mock_cli_class.assert_called_once()
    
    def test_main_keyboard_interrupt(self):
        """Test main function with keyboard interrupt"""
        with patch('sys.argv', ['openrouter_cli.py', 'list-providers']):
            with patch('src.cli.openrouter_cli.OpenRouterCLI') as mock_cli_class:
                mock_cli_class.side_effect = KeyboardInterrupt()
                
                with pytest.raises(SystemExit):
                    from src.cli.openrouter_cli import main
                    main()
    
    def test_main_unexpected_error(self):
        """Test main function with unexpected error"""
        with patch('sys.argv', ['openrouter_cli.py', 'list-providers']):
            with patch('src.cli.openrouter_cli.OpenRouterCLI') as mock_cli_class:
                mock_cli_class.side_effect = Exception("Unexpected error")
                
                with pytest.raises(SystemExit):
                    from src.cli.openrouter_cli import main
                    main()


if __name__ == '__main__':
    pytest.main([__file__])