#!/usr/bin/env python3
"""
Example usage of OpenRouter LLM integration

This example demonstrates how to use OpenRouter API for LLM-based information extraction
in the arxiv-scraper system.
"""

import os
import sys
import yaml
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from llm_extractor import OpenRouterProvider, ExtractionItem, create_llm_extractor


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def example_openrouter_provider():
    """Example using OpenRouter provider directly"""
    print("=== OpenRouter Provider Example ===")
    
    # Check if API key is available
    if not os.getenv('OPENROUTER_API_KEY'):
        print("OPENROUTER_API_KEY environment variable not set. Skipping OpenRouter example.")
        return
    
    # Configuration for OpenRouter
    config = {
        'api_key': os.getenv('OPENROUTER_API_KEY'),
        'model': 'anthropic/claude-3-sonnet',
        'base_url': 'https://openrouter.ai/api/v1',
        'temperature': 0.1,
        'max_tokens': 2000,
        'timeout': 120
    }
    
    provider = OpenRouterProvider(config)
    
    if not provider.is_available():
        print("OpenRouter provider is not available")
        return
    
    print(f"Provider available: {provider.is_available()}")
    print(f"Model: {provider.model}")
    print(f"Base URL: {provider.base_url}")
    
    # Define extraction items
    items = [
        ExtractionItem(name="手法の肝", description="論文の核となる技術手法・アプローチ", max_length=500),
        ExtractionItem(name="制限事項", description="手法の限界や制約条件", max_length=500),
        ExtractionItem(name="対象ナレッジ", description="扱う知識領域・データ種別", max_length=500),
        ExtractionItem(name="タイトル", description="論文タイトル", max_length=300),
        ExtractionItem(name="研究分野", description="分野分類", max_length=200),
    ]
    
    # Sample paper text
    sample_text = """
    This paper presents a novel approach to image classification using deep convolutional neural networks.
    We propose a new architecture that combines residual connections with attention mechanisms to improve
    accuracy on ImageNet dataset. The method achieves state-of-the-art results with 95.2% top-1 accuracy.
    However, the approach requires significant computational resources and large amounts of training data.
    The technique is particularly effective for computer vision tasks but may not generalize well to other domains.
    """
    
    # Paper metadata
    metadata = {
        'title': 'Deep Convolutional Networks for Image Classification',
        'authors': ['John Smith', 'Jane Doe'],
        'categories': ['cs.CV', 'cs.AI'],
        'published': '2023-01-15',
        'abs_url': 'https://arxiv.org/abs/2301.00001'
    }
    
    print("\\nProcessing sample paper...")
    result = provider.extract_information(sample_text, items, metadata)
    
    print(f"Success: {result.success}")
    print(f"Processing time: {result.processing_time:.2f} seconds")
    print(f"Model used: {result.model_used}")
    
    if result.success:
        print("\\nExtracted information:")
        for item_name, value in result.data.items():
            confidence = result.confidence_scores.get(item_name, 0.0)
            print(f"  {item_name}: {value} (confidence: {confidence:.2f})")
    
    if result.errors:
        print(f"\\nErrors: {result.errors}")


def example_different_models():
    """Example using different OpenRouter models"""
    print("\\n=== Different OpenRouter Models Example ===")
    
    if not os.getenv('OPENROUTER_API_KEY'):
        print("OPENROUTER_API_KEY environment variable not set. Skipping model comparison.")
        return
    
    # Test different models
    models_to_test = [
        'anthropic/claude-3-sonnet',
        'openai/gpt-4',
        'mistral/mistral-large',
        'google/gemini-pro'
    ]
    
    base_config = {
        'api_key': os.getenv('OPENROUTER_API_KEY'),
        'base_url': 'https://openrouter.ai/api/v1',
        'temperature': 0.1,
        'max_tokens': 1000,
        'timeout': 60
    }
    
    items = [
        ExtractionItem(name="手法の肝", description="論文の核となる技術手法", max_length=300),
        ExtractionItem(name="制限事項", description="手法の限界", max_length=300),
    ]
    
    sample_text = """
    We introduce a transformer-based architecture for natural language processing tasks.
    The model uses self-attention mechanisms and achieves superior performance on GLUE benchmark.
    The main limitation is the quadratic complexity with respect to sequence length.
    """
    
    for model in models_to_test:
        print(f"\\nTesting model: {model}")
        
        config = base_config.copy()
        config['model'] = model
        
        provider = OpenRouterProvider(config)
        
        if not provider.is_available():
            print(f"  {model} not available")
            continue
        
        try:
            result = provider.extract_information(sample_text, items)
            print(f"  Success: {result.success}")
            print(f"  Processing time: {result.processing_time:.2f}s")
            
            if result.success:
                print(f"  手法の肝: {result.data.get('手法の肝', 'N/A')[:100]}...")
                print(f"  制限事項: {result.data.get('制限事項', 'N/A')[:100]}...")
            
        except Exception as e:
            print(f"  Error: {str(e)}")


def example_llm_extractor_with_openrouter():
    """Example using LLM extractor with OpenRouter"""
    print("\\n=== LLM Extractor with OpenRouter Example ===")
    
    # Load configuration
    config = load_config()
    
    # Configure to use OpenRouter
    config['llm']['model'] = 'anthropic/claude-3-sonnet'
    config['llm']['fallback_models'] = ['openai/gpt-4', 'mistral/mistral-large']
    
    # Create LLM extractor
    extractor = create_llm_extractor(config)
    
    print(f"Available models: {extractor.get_available_models()}")
    
    if not extractor.get_available_models():
        print("No LLM models available. Please set API keys.")
        return
    
    # Sample paper text
    sample_text = """
    Abstract: This paper presents a novel deep learning framework for medical image analysis.
    Our approach combines convolutional neural networks with graph neural networks to analyze
    medical scans more effectively. We evaluate our method on chest X-ray classification tasks
    and achieve 98.5% accuracy, outperforming existing methods by 3.2%. The framework is
    designed to be interpretable and provides explanations for its predictions.
    
    1. Introduction
    Medical image analysis is a critical component of modern healthcare...
    
    2. Related Work
    Previous approaches to medical image analysis have focused on...
    
    3. Methodology
    Our framework consists of three main components: a CNN feature extractor,
    a graph neural network for spatial reasoning, and an attention mechanism...
    
    4. Experiments
    We conducted experiments on three datasets: ChestX-ray14, CheXpert, and MIMIC-CXR...
    
    5. Results
    Our method achieves state-of-the-art performance across all datasets...
    
    6. Limitations
    The main limitation of our approach is the computational complexity,
    requiring significant GPU resources for training and inference...
    
    7. Conclusion
    We have presented a novel framework that combines CNNs and GNNs for medical image analysis...
    """
    
    # Paper metadata
    metadata = {
        'title': 'Deep Learning Framework for Medical Image Analysis',
        'authors': ['Dr. Alice Johnson', 'Dr. Bob Smith'],
        'categories': ['cs.CV', 'cs.AI', 'eess.IV'],
        'published': '2023-03-15',
        'abs_url': 'https://arxiv.org/abs/2303.00001'
    }
    
    print("\\nProcessing medical paper...")
    result = extractor.extract_from_text(sample_text, metadata)
    
    print(f"Success: {result.success}")
    print(f"Processing time: {result.processing_time:.2f} seconds")
    print(f"Model used: {result.model_used}")
    
    if result.success:
        print("\\nExtracted information:")
        for item_name, value in result.data.items():
            confidence = result.confidence_scores.get(item_name, 0.0)
            print(f"  {item_name}: {value[:150]}... (confidence: {confidence:.2f})")
    
    if result.errors:
        print(f"\\nErrors: {result.errors}")


def example_pricing_comparison():
    """Example showing cost estimation for different models"""
    print("\\n=== Model Pricing Comparison Example ===")
    
    # Approximate pricing information (as of 2024)
    model_pricing = {
        'anthropic/claude-3-opus': {'input': 15.0, 'output': 75.0},  # per 1M tokens
        'anthropic/claude-3-sonnet': {'input': 3.0, 'output': 15.0},
        'anthropic/claude-3-haiku': {'input': 0.25, 'output': 1.25},
        'openai/gpt-4': {'input': 30.0, 'output': 60.0},
        'openai/gpt-3.5-turbo': {'input': 0.5, 'output': 1.5},
        'mistral/mistral-large': {'input': 8.0, 'output': 24.0},
        'mistral/mistral-medium': {'input': 2.7, 'output': 8.1},
        'google/gemini-pro': {'input': 0.5, 'output': 1.5},
    }
    
    # Estimate tokens for typical paper processing
    estimated_input_tokens = 8000  # ~10k chars of paper text
    estimated_output_tokens = 500  # ~9 extracted items
    
    print("Cost estimation for processing one paper:")
    print(f"Estimated input tokens: {estimated_input_tokens:,}")
    print(f"Estimated output tokens: {estimated_output_tokens:,}")
    print()
    
    for model, pricing in model_pricing.items():
        input_cost = (estimated_input_tokens / 1_000_000) * pricing['input']
        output_cost = (estimated_output_tokens / 1_000_000) * pricing['output']
        total_cost = input_cost + output_cost
        
        print(f"{model}:")
        print(f"  Input cost: ${input_cost:.6f}")
        print(f"  Output cost: ${output_cost:.6f}")
        print(f"  Total cost: ${total_cost:.6f}")
        print(f"  Cost per 1000 papers: ${total_cost * 1000:.2f}")
        print()


def main():
    """Main function to run all examples"""
    print("OpenRouter LLM Integration Examples")
    print("=" * 50)
    
    # Print environment variable status
    print("Environment Variables:")
    print(f"OPENROUTER_API_KEY: {'Set' if os.getenv('OPENROUTER_API_KEY') else 'Not set'}")
    print(f"OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")
    print(f"ANTHROPIC_API_KEY: {'Set' if os.getenv('ANTHROPIC_API_KEY') else 'Not set'}")
    print()
    
    # Run examples
    example_openrouter_provider()
    example_different_models()
    example_llm_extractor_with_openrouter()
    example_pricing_comparison()
    
    print("=" * 50)
    print("Examples completed!")
    print("\\nTo use OpenRouter in your configuration:")
    print("1. Set OPENROUTER_API_KEY environment variable")
    print("2. Update config.yaml to use OpenRouter models:")
    print("   llm:")
    print("     model: 'anthropic/claude-3-sonnet'")
    print("     fallback_models: ['openai/gpt-4', 'mistral/mistral-large']")
    print("3. The system will automatically use OpenRouter API for these models")


if __name__ == "__main__":
    main()