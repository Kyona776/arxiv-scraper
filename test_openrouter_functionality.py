#!/usr/bin/env python3
"""
Test script to demonstrate OpenRouter functionality
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from openrouter_manager import create_openrouter_manager


def test_openrouter_functionality():
    """Test basic OpenRouter functionality"""
    print("Testing OpenRouter functionality...")
    
    # Create manager with configuration
    config = {
        'openrouter': {
            'api_key': os.getenv('OPENROUTER_API_KEY'),
            'base_url': 'https://openrouter.ai/api/v1',
            'cache_enabled': True,
            'cache_ttl': 300
        }
    }
    
    manager = create_openrouter_manager(config)
    
    # Test availability
    print(f"Manager available: {manager.is_available()}")
    
    if not manager.is_available():
        print("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")
        return
    
    try:
        # Test basic functionality
        print("\n1. Testing provider listing...")
        providers = manager.get_providers()
        print(f"Found {len(providers)} providers")
        
        # Show first few providers
        for provider in providers[:3]:
            print(f"  • {provider.name} ({provider.slug}) - Status: {provider.status}")
        
        print("\n2. Testing model listing...")
        models = manager.get_models()
        print(f"Found {len(models)} models")
        
        # Show first few models
        for model in models[:3]:
            print(f"  • {model.name} ({model.id}) - Context: {model.context_length:,}")
        
        print("\n3. Testing model validation...")
        test_models = [
            'anthropic/claude-3-sonnet',
            'openai/gpt-4',
            'nonexistent/model'
        ]
        
        for model_id in test_models:
            is_valid = manager.validate_model(model_id)
            status = "✓" if is_valid else "✗"
            print(f"  {status} {model_id}")
        
        print("\n4. Testing model filtering...")
        from models.openrouter_models import ModelFilter
        
        # High-context models
        high_context_filter = ModelFilter(min_context_length=50000)
        high_context_models = manager.filter_models(high_context_filter)
        print(f"High-context models (>50K): {len(high_context_models)}")
        
        # Budget models
        budget_filter = ModelFilter(max_cost_per_token=0.000005)
        budget_models = manager.filter_models(budget_filter)
        print(f"Budget models (<$0.000005/token): {len(budget_models)}")
        
        print("\n5. Testing recommendations...")
        coding_recs = manager.get_recommendations('coding', budget=0.00001)
        print(f"Coding recommendations: {len(coding_recs)}")
        
        for model in coding_recs[:3]:
            print(f"  • {model.name}")
        
        print("\n✓ All tests completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_openrouter_functionality()