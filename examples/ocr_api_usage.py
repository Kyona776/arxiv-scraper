#!/usr/bin/env python3
"""
Example usage of Mistral OCR API processors

This example demonstrates how to use both direct Mistral API and OpenRouter API
for OCR processing with the arxiv-scraper system.
"""

import os
import sys
import yaml
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ocr_processor import MistralOCRAPIProcessor, MistralOCROpenRouterProcessor, create_ocr_manager


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def example_direct_mistral_api():
    """Example using direct Mistral API"""
    print("=== Direct Mistral API Example ===")
    
    # Check if API key is available
    if not os.getenv('MISTRAL_API_KEY'):
        print("MISTRAL_API_KEY environment variable not set. Skipping direct API example.")
        return
    
    # Configuration for direct Mistral API
    config = {
        'api_key': os.getenv('MISTRAL_API_KEY'),
        'model': 'mistral-ocr-latest',
        'base_url': 'https://api.mistral.ai',
        'timeout': 120,
        'max_retries': 3
    }
    
    processor = MistralOCRAPIProcessor(config)
    
    if not processor.is_available():
        print("Mistral API processor is not available")
        return
    
    print(f"Processor available: {processor.is_available()}")
    print(f"Model: {processor.model}")
    print(f"Base URL: {processor.base_url}")
    
    # Example with a sample PDF (replace with actual PDF path)
    sample_pdf = "sample_paper.pdf"
    
    if os.path.exists(sample_pdf):
        print(f"Processing {sample_pdf}...")
        result = processor.process_pdf(sample_pdf)
        
        print(f"Processing time: {result.processing_time:.2f} seconds")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Page count: {result.page_count}")
        print(f"Text length: {len(result.text)} characters")
        print(f"Model used: {result.model_used}")
        
        if result.errors:
            print(f"Errors: {result.errors}")
        
        # Show first 500 characters of extracted text
        if result.text:
            print(f"First 500 chars: {result.text[:500]}...")
    else:
        print(f"Sample PDF {sample_pdf} not found. Please provide a valid PDF path.")


def example_openrouter_api():
    """Example using OpenRouter API"""
    print("\\n=== OpenRouter API Example ===")
    
    # Check if API key is available
    if not os.getenv('OPENROUTER_API_KEY'):
        print("OPENROUTER_API_KEY environment variable not set. Skipping OpenRouter example.")
        return
    
    # Configuration for OpenRouter API
    config = {
        'api_key': os.getenv('OPENROUTER_API_KEY'),
        'model': 'mistral/mistral-ocr-latest',
        'base_url': 'https://openrouter.ai/api/v1',
        'site_url': 'https://arxiv-scraper.local',
        'site_name': 'ArXiv Scraper',
        'timeout': 120,
        'max_retries': 3
    }
    
    processor = MistralOCROpenRouterProcessor(config)
    
    if not processor.is_available():
        print("OpenRouter API processor is not available")
        return
    
    print(f"Processor available: {processor.is_available()}")
    print(f"Model: {processor.model}")
    print(f"Base URL: {processor.base_url}")
    
    # Example with a sample PDF (replace with actual PDF path)
    sample_pdf = "sample_paper.pdf"
    
    if os.path.exists(sample_pdf):
        print(f"Processing {sample_pdf}...")
        result = processor.process_pdf(sample_pdf)
        
        print(f"Processing time: {result.processing_time:.2f} seconds")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Page count: {result.page_count}")
        print(f"Text length: {len(result.text)} characters")
        print(f"Model used: {result.model_used}")
        
        if result.errors:
            print(f"Errors: {result.errors}")
        
        # Show first 500 characters of extracted text
        if result.text:
            print(f"First 500 chars: {result.text[:500]}...")
    else:
        print(f"Sample PDF {sample_pdf} not found. Please provide a valid PDF path.")


def example_ocr_manager_with_api():
    """Example using OCR Manager with API processors"""
    print("\\n=== OCR Manager with API Example ===")
    
    # Load configuration
    config = load_config()
    
    # Set primary model to use API
    config['ocr']['model'] = 'mistral_ocr_api'
    config['ocr']['fallback_models'] = ['mistral_ocr_openrouter', 'nougat', 'unstructured']
    
    # Create OCR manager
    ocr_manager = create_ocr_manager(config)
    
    print(f"Available models: {ocr_manager.get_available_models()}")
    
    # Example with a sample PDF (replace with actual PDF path)
    sample_pdf = "sample_paper.pdf"
    
    if os.path.exists(sample_pdf):
        print(f"Processing {sample_pdf} with OCR Manager...")
        result = ocr_manager.process_pdf(sample_pdf)
        
        print(f"Processing time: {result.processing_time:.2f} seconds")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Page count: {result.page_count}")
        print(f"Text length: {len(result.text)} characters")
        print(f"Model used: {result.model_used}")
        
        if result.errors:
            print(f"Errors: {result.errors}")
        
        # Show first 500 characters of extracted text
        if result.text:
            print(f"First 500 chars: {result.text[:500]}...")
    else:
        print(f"Sample PDF {sample_pdf} not found. Please provide a valid PDF path.")


def example_image_processing():
    """Example of image processing with API"""
    print("\\n=== Image Processing Example ===")
    
    # Check if API key is available
    if not os.getenv('MISTRAL_API_KEY'):
        print("MISTRAL_API_KEY environment variable not set. Skipping image example.")
        return
    
    config = {
        'api_key': os.getenv('MISTRAL_API_KEY'),
        'model': 'mistral-ocr-latest',
        'base_url': 'https://api.mistral.ai'
    }
    
    processor = MistralOCRAPIProcessor(config)
    
    # Example with a sample image (replace with actual image path)
    sample_image = "sample_document.png"
    
    if os.path.exists(sample_image):
        print(f"Processing {sample_image}...")
        result = processor.process_image(sample_image)
        
        print(f"Text length: {len(result)} characters")
        if result:
            print(f"Extracted text: {result[:500]}...")
        else:
            print("No text extracted")
    else:
        print(f"Sample image {sample_image} not found. Please provide a valid image path.")


def main():
    """Main function to run all examples"""
    print("Mistral OCR API Usage Examples")
    print("=" * 50)
    
    # Print environment variable status
    print("Environment Variables:")
    print(f"MISTRAL_API_KEY: {'Set' if os.getenv('MISTRAL_API_KEY') else 'Not set'}")
    print(f"OPENROUTER_API_KEY: {'Set' if os.getenv('OPENROUTER_API_KEY') else 'Not set'}")
    print()
    
    # Run examples
    example_direct_mistral_api()
    example_openrouter_api()
    example_ocr_manager_with_api()
    example_image_processing()
    
    print("\\n" + "=" * 50)
    print("Examples completed!")
    print("\\nTo use these processors in your code:")
    print("1. Set the appropriate environment variables (MISTRAL_API_KEY, OPENROUTER_API_KEY)")
    print("2. Update your config.yaml to use 'mistral_ocr_api' or 'mistral_ocr_openrouter'")
    print("3. Use the OCR manager as usual - it will handle the API calls automatically")


if __name__ == "__main__":
    main()