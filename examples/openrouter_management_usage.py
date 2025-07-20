#!/usr/bin/env python3
"""
Comprehensive OpenRouter Management Usage Examples

This example demonstrates the complete OpenRouter management functionality including:
- Provider management and checking
- Model discovery and validation
- Endpoint management
- Model comparison and recommendations
- CLI usage examples
"""

import os
import sys
import json
import asyncio
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from openrouter_manager import create_openrouter_manager, OpenRouterManager
from models.openrouter_models import ModelFilter, Provider, Model


def load_config():
    """Load configuration"""
    return {
        'openrouter': {
            'api_key': os.getenv('OPENROUTER_API_KEY'),
            'base_url': 'https://openrouter.ai/api/v1',
            'site_url': 'https://arxiv-scraper.local',
            'site_name': 'ArXiv Scraper Demo',
            'timeout': 30,
            'max_retries': 3,
            'cache_enabled': True,
            'cache_ttl': 3600
        }
    }


def example_provider_management():
    """Example: Provider management"""
    print("=== Provider Management Example ===")
    
    if not os.getenv('OPENROUTER_API_KEY'):
        print("OPENROUTER_API_KEY not set. Skipping provider management example.")
        return
    
    config = load_config()
    manager = create_openrouter_manager(config)
    
    if not manager.is_available():
        print("OpenRouter manager not available")
        return
    
    try:
        # Get all providers
        print("Fetching providers...")
        providers = manager.get_providers()
        
        print(f"Found {len(providers)} providers:")
        for provider in providers[:5]:  # Show first 5
            print(f"  • {provider.name} ({provider.slug})")
            print(f"    Status: {provider.status}")
            print(f"    May log prompts: {provider.may_log_prompts}")
            print(f"    May train on data: {provider.may_train_on_data}")
            print(f"    Moderated: {provider.moderated_by_openrouter}")
            print()
        
        # Check specific provider
        anthropic_provider = manager.get_provider_by_slug('anthropic')
        if anthropic_provider:
            print(f"Anthropic provider details:")
            print(f"  Name: {anthropic_provider.name}")
            print(f"  Privacy policy: {anthropic_provider.privacy_policy_url}")
            print(f"  Terms of service: {anthropic_provider.terms_of_service_url}")
        
        # Check provider status
        status = manager.check_provider_status('anthropic')
        print(f"Anthropic status: {status}")
        
    except Exception as e:
        print(f"Error in provider management: {e}")


def example_model_discovery():
    """Example: Model discovery and validation"""
    print("\\n=== Model Discovery Example ===")
    
    if not os.getenv('OPENROUTER_API_KEY'):
        print("OPENROUTER_API_KEY not set. Skipping model discovery example.")
        return
    
    config = load_config()
    manager = create_openrouter_manager(config)
    
    if not manager.is_available():
        print("OpenRouter manager not available")
        return
    
    try:
        # Get all models
        print("Fetching models...")
        models = manager.get_models()
        
        print(f"Found {len(models)} models")
        
        # Show models by provider
        anthropic_models = manager.get_models_by_provider('anthropic')
        print(f"\\nAnthropic models ({len(anthropic_models)}):")
        for model in anthropic_models[:3]:  # Show first 3
            print(f"  • {model.name} ({model.id})")
            print(f"    Context length: {model.context_length:,}")
            if model.pricing:
                print(f"    Pricing: ${model.pricing.prompt:.6f} / ${model.pricing.completion:.6f}")
            print()
        
        # Validate specific models
        test_models = [
            'anthropic/claude-3-sonnet',
            'openai/gpt-4',
            'nonexistent/model'
        ]
        
        print("Model validation:")
        for model_id in test_models:
            is_valid = manager.validate_model(model_id)
            status = "✓" if is_valid else "✗"
            print(f"  {status} {model_id}")
        
        # Get model info
        model_info = manager.get_model_info('anthropic/claude-3-sonnet')
        if model_info.get('exists'):
            model = model_info['model']
            print(f"\\nClaude 3 Sonnet details:")
            print(f"  Context length: {model['context_length']:,}")
            if model.get('pricing'):
                pricing = model['pricing']
                print(f"  Pricing: ${pricing['prompt']:.6f} prompt, ${pricing['completion']:.6f} completion")
            print(f"  Endpoints: {len(model_info.get('endpoints', []))}")
        
    except Exception as e:
        print(f"Error in model discovery: {e}")


def example_model_filtering():
    """Example: Model filtering and recommendations"""
    print("\\n=== Model Filtering Example ===")
    
    if not os.getenv('OPENROUTER_API_KEY'):
        print("OPENROUTER_API_KEY not set. Skipping model filtering example.")
        return
    
    config = load_config()
    manager = create_openrouter_manager(config)
    
    if not manager.is_available():
        print("OpenRouter manager not available")
        return
    
    try:
        # Filter models by criteria
        print("Filtering models...")
        
        # High-context models
        high_context_filter = ModelFilter(min_context_length=100000)
        high_context_models = manager.filter_models(high_context_filter)
        print(f"High-context models (>100K tokens): {len(high_context_models)}")
        for model in high_context_models[:3]:
            print(f"  • {model.name}: {model.context_length:,} tokens")
        
        # Budget-friendly models
        budget_filter = ModelFilter(max_cost_per_token=0.000005)
        budget_models = manager.filter_models(budget_filter)
        print(f"\\nBudget-friendly models (<$0.000005/token): {len(budget_models)}")
        for model in budget_models[:3]:
            if model.pricing:
                avg_cost = (model.pricing.prompt + model.pricing.completion) / 2
                print(f"  • {model.name}: ${avg_cost:.6f} avg/token")
        
        # Get recommendations
        print("\\nModel recommendations:")
        
        # For coding tasks
        coding_models = manager.get_recommendations('coding', budget=0.00001)
        print(f"Best for coding ({len(coding_models)}):")
        for model in coding_models[:3]:
            print(f"  • {model.name}")
        
        # For conversation
        conversation_models = manager.get_recommendations('conversation', budget=0.000005)
        print(f"\\nBest for conversation ({len(conversation_models)}):")
        for model in conversation_models[:3]:
            print(f"  • {model.name}")
        
        # For analysis (high-context tasks)
        analysis_models = manager.get_recommendations('analysis')
        print(f"\\nBest for analysis ({len(analysis_models)}):")
        for model in analysis_models[:3]:
            print(f"  • {model.name}")
        
    except Exception as e:
        print(f"Error in model filtering: {e}")


def example_endpoint_management():
    """Example: Endpoint management"""
    print("\\n=== Endpoint Management Example ===")
    
    if not os.getenv('OPENROUTER_API_KEY'):
        print("OPENROUTER_API_KEY not set. Skipping endpoint management example.")
        return
    
    config = load_config()
    manager = create_openrouter_manager(config)
    
    if not manager.is_available():
        print("OpenRouter manager not available")
        return
    
    try:
        # Get endpoints for a model
        model_id = 'anthropic/claude-3-sonnet'
        print(f"Getting endpoints for {model_id}...")
        
        endpoints_info = manager.get_model_endpoints(model_id)
        if endpoints_info:
            print(f"Found {len(endpoints_info.endpoints)} endpoints:")
            for endpoint in endpoints_info.endpoints:
                print(f"  • {endpoint.name} ({endpoint.provider_name})")
                print(f"    Status: {endpoint.status}")
                print(f"    Context length: {endpoint.context_length:,}")
                if endpoint.uptime:
                    print(f"    Uptime: {endpoint.uptime:.1%}")
                if endpoint.latency:
                    print(f"    Latency: {endpoint.latency:.0f}ms")
                print()
            
            # Get best endpoint by different criteria
            best_cost = manager.get_best_endpoint(model_id, 'cost')
            best_latency = manager.get_best_endpoint(model_id, 'latency')
            best_uptime = manager.get_best_endpoint(model_id, 'uptime')
            
            print("Best endpoints:")
            if best_cost:
                print(f"  Cost: {best_cost.name} ({best_cost.provider_name})")
            if best_latency:
                print(f"  Latency: {best_latency.name} ({best_latency.provider_name})")
            if best_uptime:
                print(f"  Uptime: {best_uptime.name} ({best_uptime.provider_name})")
        
        # Check endpoint health
        if endpoints_info and endpoints_info.endpoints:
            first_endpoint = endpoints_info.endpoints[0]
            health = manager.check_endpoint_health(model_id, first_endpoint.name)
            print(f"\\nEndpoint health for {first_endpoint.name}:")
            print(f"  Status: {health.get('status', 'unknown')}")
            print(f"  Uptime: {health.get('uptime', 'N/A')}")
            print(f"  Latency: {health.get('latency', 'N/A')}")
        
    except Exception as e:
        print(f"Error in endpoint management: {e}")


def example_model_comparison():
    """Example: Model comparison"""
    print("\\n=== Model Comparison Example ===")
    
    if not os.getenv('OPENROUTER_API_KEY'):
        print("OPENROUTER_API_KEY not set. Skipping model comparison example.")
        return
    
    config = load_config()
    manager = create_openrouter_manager(config)
    
    if not manager.is_available():
        print("OpenRouter manager not available")
        return
    
    try:
        # Compare multiple models
        models_to_compare = [
            'anthropic/claude-3-opus',
            'anthropic/claude-3-sonnet',
            'anthropic/claude-3-haiku',
            'openai/gpt-4',
            'openai/gpt-3.5-turbo'
        ]
        
        print("Comparing models...")
        comparison = manager.compare_models(models_to_compare)
        
        print(f"{'Model':<30} {'Context':<10} {'Prompt Cost':<12} {'Completion Cost':<15} {'Provider':<15}")
        print("-" * 90)
        
        for model_id, info in comparison.items():
            if info.get('exists'):
                context = f"{info['context_length']:,}" if info['context_length'] else 'N/A'
                prompt_cost = f"${info['pricing']['prompt']:.6f}" if info.get('pricing') else 'N/A'
                completion_cost = f"${info['pricing']['completion']:.6f}" if info.get('pricing') else 'N/A'
                provider = info.get('provider', 'N/A')
                
                print(f"{model_id:<30} {context:<10} {prompt_cost:<12} {completion_cost:<15} {provider:<15}")
            else:
                print(f"{model_id:<30} {'NOT FOUND':<10} {'N/A':<12} {'N/A':<15} {'N/A':<15}")
        
        # Calculate cost for a typical paper processing task
        print("\\nCost estimation for processing one paper (8K input, 500 output tokens):")
        input_tokens = 8000
        output_tokens = 500
        
        for model_id, info in comparison.items():
            if info.get('exists') and info.get('pricing'):
                pricing = info['pricing']
                input_cost = (input_tokens / 1000) * pricing['prompt']
                output_cost = (output_tokens / 1000) * pricing['completion']
                total_cost = input_cost + output_cost
                
                print(f"  {model_id}: ${total_cost:.6f}")
        
    except Exception as e:
        print(f"Error in model comparison: {e}")


def example_cache_management():
    """Example: Cache management"""
    print("\\n=== Cache Management Example ===")
    
    if not os.getenv('OPENROUTER_API_KEY'):
        print("OPENROUTER_API_KEY not set. Skipping cache management example.")
        return
    
    config = load_config()
    manager = create_openrouter_manager(config)
    
    if not manager.is_available():
        print("OpenRouter manager not available")
        return
    
    try:
        # Enable caching and make some requests
        print("Making requests with caching enabled...")
        
        # First request - should fetch from API
        start_time = time.time()
        models1 = manager.get_models()
        time1 = time.time() - start_time
        print(f"First request: {time1:.2f}s, found {len(models1)} models")
        
        # Second request - should use cache
        start_time = time.time()
        models2 = manager.get_models()
        time2 = time.time() - start_time
        print(f"Second request: {time2:.2f}s, found {len(models2)} models")
        
        if time2 < time1:
            print("✓ Cache working - second request was faster!")
        
        # Clear cache
        print("\\nClearing cache...")
        manager.clear_cache()
        
        # Third request - should fetch from API again
        start_time = time.time()
        models3 = manager.get_models()
        time3 = time.time() - start_time
        print(f"Third request (after cache clear): {time3:.2f}s, found {len(models3)} models")
        
    except Exception as e:
        print(f"Error in cache management: {e}")


def example_cli_usage():
    """Example: CLI usage examples"""
    print("\\n=== CLI Usage Examples ===")
    
    print("The OpenRouter CLI provides these commands:")
    print()
    
    cli_examples = [
        ("List providers", "python src/cli/openrouter_cli.py list-providers"),
        ("List models", "python src/cli/openrouter_cli.py list-models"),
        ("Filter models by provider", "python src/cli/openrouter_cli.py list-models --provider anthropic"),
        ("Filter models by cost", "python src/cli/openrouter_cli.py list-models --max-cost 0.00001"),
        ("Check model availability", "python src/cli/openrouter_cli.py check-model anthropic/claude-3-sonnet"),
        ("Get model details", "python src/cli/openrouter_cli.py model-info anthropic/claude-3-sonnet"),
        ("Check model endpoints", "python src/cli/openrouter_cli.py check-endpoints anthropic/claude-3-sonnet"),
        ("Get recommendations", "python src/cli/openrouter_cli.py recommend --task-type coding"),
        ("Compare models", "python src/cli/openrouter_cli.py compare anthropic/claude-3-sonnet openai/gpt-4"),
        ("Clear cache", "python src/cli/openrouter_cli.py clear-cache"),
        ("JSON output", "python src/cli/openrouter_cli.py list-models --json"),
        ("Verbose output", "python src/cli/openrouter_cli.py list-models --verbose")
    ]
    
    for description, command in cli_examples:
        print(f"# {description}")
        print(f"{command}")
        print()
    
    print("Example workflow:")
    print("1. List available providers and models")
    print("2. Filter models by your requirements")
    print("3. Check model details and endpoints")
    print("4. Compare your shortlisted models")
    print("5. Get recommendations for your specific use case")


def main():
    """Main function to run all examples"""
    print("OpenRouter Management Usage Examples")
    print("=" * 50)
    
    # Check environment
    print("Environment check:")
    print(f"OPENROUTER_API_KEY: {'Set' if os.getenv('OPENROUTER_API_KEY') else 'Not set'}")
    print()
    
    # Run examples
    example_provider_management()
    example_model_discovery()
    example_model_filtering()
    example_endpoint_management()
    example_model_comparison()
    example_cache_management()
    example_cli_usage()
    
    print("=" * 50)
    print("Examples completed!")
    print()
    print("To get started:")
    print("1. Set your OPENROUTER_API_KEY environment variable")
    print("2. Run: python src/cli/openrouter_cli.py list-models")
    print("3. Explore the available models and providers")
    print("4. Update your config.yaml with preferred models")
    print("5. Use the arxiv-scraper with OpenRouter models")


if __name__ == "__main__":
    import time
    main()