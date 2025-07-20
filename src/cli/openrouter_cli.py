#!/usr/bin/env python3
"""
OpenRouter CLI commands for managing models, providers, and endpoints
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional
from pathlib import Path
import yaml
from tabulate import tabulate
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from openrouter_manager import create_openrouter_manager, OpenRouterManager
from models.openrouter_models import ModelFilter


class OpenRouterCLI:
    """Command-line interface for OpenRouter management"""
    
    def __init__(self):
        self.config = self._load_config()
        self.manager = create_openrouter_manager(self.config)
        
        if not self.manager.is_available():
            logger.error("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")
            sys.exit(1)
    
    def _load_config(self) -> Dict:
        """Load configuration from config.yaml"""
        config_path = Path(__file__).parent.parent.parent / 'config' / 'config.yaml'
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config
        else:
            # Default configuration
            return {
                'openrouter': {
                    'base_url': 'https://openrouter.ai/api/v1',
                    'timeout': 30,
                    'max_retries': 3,
                    'cache_enabled': True,
                    'cache_ttl': 3600
                }
            }
    
    def list_providers(self, args):
        """List all available providers"""
        try:
            providers = self.manager.get_providers()
            
            if args.json:
                print(json.dumps([{
                    'name': p.name,
                    'slug': p.slug,
                    'status': p.status,
                    'may_log_prompts': p.may_log_prompts,
                    'may_train_on_data': p.may_train_on_data,
                    'moderated_by_openrouter': p.moderated_by_openrouter
                } for p in providers], indent=2))
            else:
                headers = ['Name', 'Slug', 'Status', 'Logs Prompts', 'Trains on Data', 'Moderated']
                table_data = []
                
                for provider in providers:
                    table_data.append([
                        provider.name,
                        provider.slug,
                        provider.status,
                        '✓' if provider.may_log_prompts else '✗',
                        '✓' if provider.may_train_on_data else '✗',
                        '✓' if provider.moderated_by_openrouter else '✗'
                    ])
                
                print(tabulate(table_data, headers=headers, tablefmt='grid'))
                print(f"\\nTotal providers: {len(providers)}")
        
        except Exception as e:
            logger.error(f"Failed to list providers: {e}")
            sys.exit(1)
    
    def list_models(self, args):
        """List all available models"""
        try:
            models = self.manager.get_models(category=args.category)
            
            # Apply filters
            if args.provider or args.max_cost or args.min_context:
                filter_criteria = ModelFilter(
                    providers=[args.provider] if args.provider else None,
                    max_cost_per_token=args.max_cost,
                    min_context_length=args.min_context
                )
                models = self.manager.filter_models(filter_criteria)
            
            if args.json:
                print(json.dumps([{
                    'id': m.id,
                    'name': m.name,
                    'context_length': m.context_length,
                    'pricing': {
                        'prompt': m.pricing.prompt,
                        'completion': m.pricing.completion
                    } if m.pricing else None,
                    'provider': m.top_provider.get('name') if m.top_provider else None
                } for m in models], indent=2))
            else:
                headers = ['ID', 'Name', 'Context Length', 'Prompt Cost', 'Completion Cost', 'Provider']
                table_data = []
                
                for model in models:
                    table_data.append([
                        model.id,
                        model.name[:50] + '...' if len(model.name) > 50 else model.name,
                        f"{model.context_length:,}" if model.context_length else 'N/A',
                        f"${model.pricing.prompt:.6f}" if model.pricing else 'N/A',
                        f"${model.pricing.completion:.6f}" if model.pricing else 'N/A',
                        model.top_provider.get('name') if model.top_provider else 'N/A'
                    ])
                
                print(tabulate(table_data, headers=headers, tablefmt='grid'))
                print(f"\\nTotal models: {len(models)}")
        
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            sys.exit(1)
    
    def check_model(self, args):
        """Check if a model is available"""
        try:
            model_info = self.manager.get_model_info(args.model_id)
            
            if args.json:
                print(json.dumps(model_info, indent=2))
            else:
                if model_info.get('exists'):
                    model = model_info['model']
                    print(f"Model: {model['name']}")
                    print(f"ID: {model['id']}")
                    print(f"Context Length: {model['context_length']:,}")
                    
                    if model.get('pricing'):
                        pricing = model['pricing']
                        print(f"Prompt Cost: ${pricing['prompt']:.6f} per token")
                        print(f"Completion Cost: ${pricing['completion']:.6f} per token")
                    
                    if model.get('description'):
                        print(f"Description: {model['description']}")
                    
                    if model_info.get('endpoints'):
                        print(f"\\nEndpoints: {len(model_info['endpoints'])}")
                        for endpoint in model_info['endpoints']:
                            print(f"  - {endpoint['name']} ({endpoint['provider_name']})")
                    
                    print(f"\\n✓ Model is available")
                else:
                    print(f"✗ Model '{args.model_id}' not found")
        
        except Exception as e:
            logger.error(f"Failed to check model: {e}")
            sys.exit(1)
    
    def model_info(self, args):
        """Get detailed model information"""
        try:
            model_info = self.manager.get_model_info(args.model_id)
            
            if args.json:
                print(json.dumps(model_info, indent=2))
            else:
                if model_info.get('exists'):
                    model = model_info['model']
                    
                    print(f"Model Information: {model['name']}")
                    print("=" * 50)
                    print(f"ID: {model['id']}")
                    print(f"Context Length: {model['context_length']:,}")
                    
                    if model.get('pricing'):
                        pricing = model['pricing']
                        print(f"\\nPricing:")
                        print(f"  Prompt: ${pricing['prompt']:.6f} per token")
                        print(f"  Completion: ${pricing['completion']:.6f} per token")
                        
                        # Calculate costs for different usage scenarios
                        prompt_1k = pricing['prompt'] * 1000
                        completion_1k = pricing['completion'] * 1000
                        print(f"  Per 1K tokens: ${prompt_1k:.4f} prompt, ${completion_1k:.4f} completion")
                    
                    if model.get('architecture'):
                        arch = model['architecture']
                        print(f"\\nArchitecture:")
                        print(f"  Modality: {arch.get('modality', 'N/A')}")
                        if arch.get('tokenizer'):
                            print(f"  Tokenizer: {arch['tokenizer']}")
                        if arch.get('instruct_type'):
                            print(f"  Instruction Type: {arch['instruct_type']}")
                    
                    if model.get('description'):
                        print(f"\\nDescription:")
                        print(f"  {model['description']}")
                    
                    if model_info.get('endpoints'):
                        print(f"\\nEndpoints ({len(model_info['endpoints'])}):")
                        for endpoint in model_info['endpoints']:
                            print(f"  • {endpoint['name']}")
                            print(f"    Provider: {endpoint['provider_name']}")
                            print(f"    Status: {endpoint['status']}")
                            if endpoint.get('uptime'):
                                print(f"    Uptime: {endpoint['uptime']:.1%}")
                else:
                    print(f"Model '{args.model_id}' not found")
        
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            sys.exit(1)
    
    def check_endpoints(self, args):
        """List endpoints for a model"""
        try:
            endpoints_info = self.manager.get_model_endpoints(args.model_id)
            
            if not endpoints_info:
                print(f"No endpoints found for model '{args.model_id}'")
                return
            
            if args.json:
                print(json.dumps({
                    'model': endpoints_info.model.__dict__,
                    'endpoints': [ep.__dict__ for ep in endpoints_info.endpoints]
                }, indent=2))
            else:
                print(f"Endpoints for {endpoints_info.model.name}:")
                print("=" * 50)
                
                headers = ['Name', 'Provider', 'Status', 'Context Length', 'Uptime', 'Latency']
                table_data = []
                
                for endpoint in endpoints_info.endpoints:
                    table_data.append([
                        endpoint.name,
                        endpoint.provider_name,
                        endpoint.status,
                        f"{endpoint.context_length:,}" if endpoint.context_length else 'N/A',
                        f"{endpoint.uptime:.1%}" if endpoint.uptime else 'N/A',
                        f"{endpoint.latency:.0f}ms" if endpoint.latency else 'N/A'
                    ])
                
                print(tabulate(table_data, headers=headers, tablefmt='grid'))
                print(f"\\nTotal endpoints: {len(endpoints_info.endpoints)}")
        
        except Exception as e:
            logger.error(f"Failed to check endpoints: {e}")
            sys.exit(1)
    
    def recommend_models(self, args):
        """Get model recommendations"""
        try:
            models = self.manager.get_recommendations(
                task_type=args.task_type,
                budget=args.budget
            )
            
            if args.json:
                print(json.dumps([{
                    'id': m.id,
                    'name': m.name,
                    'context_length': m.context_length,
                    'pricing': {
                        'prompt': m.pricing.prompt,
                        'completion': m.pricing.completion
                    } if m.pricing else None,
                    'provider': m.top_provider.get('name') if m.top_provider else None
                } for m in models], indent=2))
            else:
                print(f"Recommended models for {args.task_type} tasks:")
                print("=" * 50)
                
                headers = ['Rank', 'ID', 'Name', 'Context Length', 'Avg Cost', 'Provider']
                table_data = []
                
                for i, model in enumerate(models, 1):
                    avg_cost = 'N/A'
                    if model.pricing:
                        avg_cost = f"${(model.pricing.prompt + model.pricing.completion) / 2:.6f}"
                    
                    table_data.append([
                        i,
                        model.id,
                        model.name[:40] + '...' if len(model.name) > 40 else model.name,
                        f"{model.context_length:,}" if model.context_length else 'N/A',
                        avg_cost,
                        model.top_provider.get('name') if model.top_provider else 'N/A'
                    ])
                
                print(tabulate(table_data, headers=headers, tablefmt='grid'))
                print(f"\\nShowing top {len(models)} recommendations")
        
        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            sys.exit(1)
    
    def compare_models(self, args):
        """Compare multiple models"""
        try:
            comparison = self.manager.compare_models(args.model_ids)
            
            if args.json:
                print(json.dumps(comparison, indent=2))
            else:
                print("Model Comparison:")
                print("=" * 50)
                
                headers = ['Model ID', 'Name', 'Context Length', 'Prompt Cost', 'Completion Cost', 'Provider']
                table_data = []
                
                for model_id, info in comparison.items():
                    if info.get('exists'):
                        table_data.append([
                            model_id,
                            info['name'][:30] + '...' if len(info['name']) > 30 else info['name'],
                            f"{info['context_length']:,}" if info['context_length'] else 'N/A',
                            f"${info['pricing']['prompt']:.6f}" if info.get('pricing') else 'N/A',
                            f"${info['pricing']['completion']:.6f}" if info.get('pricing') else 'N/A',
                            info.get('provider', 'N/A')
                        ])
                    else:
                        table_data.append([
                            model_id,
                            'NOT FOUND',
                            'N/A',
                            'N/A',
                            'N/A',
                            'N/A'
                        ])
                
                print(tabulate(table_data, headers=headers, tablefmt='grid'))
        
        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            sys.exit(1)
    
    def clear_cache(self, args):
        """Clear OpenRouter cache"""
        try:
            self.manager.clear_cache(args.key)
            if args.key:
                print(f"Cache cleared for key: {args.key}")
            else:
                print("All cache cleared")
        
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            sys.exit(1)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='OpenRouter CLI Management Tool')
    parser.add_argument('--json', action='store_true', help='Output in JSON format')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List providers command
    providers_parser = subparsers.add_parser('list-providers', help='List available providers')
    providers_parser.set_defaults(func=lambda cli, args: cli.list_providers(args))
    
    # List models command
    models_parser = subparsers.add_parser('list-models', help='List available models')
    models_parser.add_argument('--category', help='Filter by category')
    models_parser.add_argument('--provider', help='Filter by provider')
    models_parser.add_argument('--max-cost', type=float, help='Maximum cost per token')
    models_parser.add_argument('--min-context', type=int, help='Minimum context length')
    models_parser.set_defaults(func=lambda cli, args: cli.list_models(args))
    
    # Check model command
    check_parser = subparsers.add_parser('check-model', help='Check if model is available')
    check_parser.add_argument('model_id', help='Model ID to check')
    check_parser.set_defaults(func=lambda cli, args: cli.check_model(args))
    
    # Model info command
    info_parser = subparsers.add_parser('model-info', help='Get detailed model information')
    info_parser.add_argument('model_id', help='Model ID to get info for')
    info_parser.set_defaults(func=lambda cli, args: cli.model_info(args))
    
    # Check endpoints command
    endpoints_parser = subparsers.add_parser('check-endpoints', help='List endpoints for a model')
    endpoints_parser.add_argument('model_id', help='Model ID to check endpoints for')
    endpoints_parser.set_defaults(func=lambda cli, args: cli.check_endpoints(args))
    
    # Recommend models command
    recommend_parser = subparsers.add_parser('recommend', help='Get model recommendations')
    recommend_parser.add_argument('--task-type', choices=['general', 'coding', 'conversation', 'analysis'], 
                                 default='general', help='Task type for recommendations')
    recommend_parser.add_argument('--budget', type=float, help='Maximum cost per token')
    recommend_parser.set_defaults(func=lambda cli, args: cli.recommend_models(args))
    
    # Compare models command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple models')
    compare_parser.add_argument('model_ids', nargs='+', help='Model IDs to compare')
    compare_parser.set_defaults(func=lambda cli, args: cli.compare_models(args))
    
    # Clear cache command
    cache_parser = subparsers.add_parser('clear-cache', help='Clear OpenRouter cache')
    cache_parser.add_argument('--key', help='Specific cache key to clear')
    cache_parser.set_defaults(func=lambda cli, args: cli.clear_cache(args))
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Set up logging
    if args.verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")
    
    try:
        cli = OpenRouterCLI()
        args.func(cli, args)
    except KeyboardInterrupt:
        print("\\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()