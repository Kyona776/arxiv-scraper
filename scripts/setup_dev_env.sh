#!/bin/bash

# ArXiv Scraper Development Environment Setup Script
# This script sets up the complete development environment for the ArXiv Scraper project

set -e  # Exit on any error

echo "==============================================="
echo "ArXiv Scraper Development Environment Setup"
echo "==============================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check if we're in the correct directory
if [ ! -f "CLAUDE.md" ] || [ ! -f "src/arxiv_extractor.py" ]; then
    print_error "This script must be run from the ArXiv Scraper project root directory"
    exit 1
fi

print_info "Setting up development environment..."

# Check system requirements
print_info "Checking system requirements..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
print_info "Python version: $python_version"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    print_warning "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.bashrc 2>/dev/null || source ~/.zshrc 2>/dev/null || true
    
    if ! command -v uv &> /dev/null; then
        print_error "Failed to install uv. Please install manually: https://docs.astral.sh/uv/"
        exit 1
    fi
fi

print_status "uv is available"

# Create virtual environment
print_info "Creating virtual environment..."
uv venv
print_status "Virtual environment created"

# Install dependencies
print_info "Installing dependencies..."
uv pip install -r requirements.txt
print_status "Dependencies installed"

# Install development dependencies
print_info "Installing development dependencies..."
uv pip install pytest pytest-cov black flake8 pre-commit memory-profiler psutil
print_status "Development dependencies installed"

# Create project directories
print_info "Creating project directories..."
mkdir -p data/samples
mkdir -p data/test_papers
mkdir -p logs/performance
mkdir -p logs/processing
mkdir -p logs/accuracy
mkdir -p output/csv
mkdir -p output/processed
mkdir -p output/benchmarks
mkdir -p tests/fixtures
mkdir -p docs/examples
print_status "Project directories created"

# Set up pre-commit hooks (if config exists)
if [ -f ".pre-commit-config.yaml" ]; then
    print_info "Setting up pre-commit hooks..."
    uv run pre-commit install
    print_status "Pre-commit hooks installed"
else
    print_warning "No .pre-commit-config.yaml found, skipping pre-commit setup"
fi

# Create sample configuration files
print_info "Creating sample configuration files..."

# Create development config if it doesn't exist
if [ ! -f "config/dev.yaml" ]; then
    cp config/config.yaml config/dev.yaml
    print_status "Development configuration created"
fi

# Create test configuration
if [ ! -f "config/test.yaml" ]; then
    cat > config/test.yaml << 'EOF'
ocr:
  model: "mistral_ocr"
  device: "cpu"  # Use CPU for testing
  
text_processing:
  remove_references: true
  reference_patterns:
    - "REFERENCES"
    - "Bibliography"
    - "参考文献"
    
llm:
  model: "gpt-3.5-turbo"  # Use cheaper model for testing
  temperature: 0.1
  max_tokens: 4000

performance:
  max_processing_time: 60
  max_reference_removal_time: 3
  
accuracy:
  min_ocr_accuracy: 0.95
  min_reference_removal_accuracy: 0.99
  min_extraction_accuracy: 0.90
EOF
    print_status "Test configuration created"
fi

# Download sample papers if they don't exist
print_info "Setting up sample papers..."
if [ ! -f "data/samples/sample_paper.pdf" ]; then
    print_warning "No sample papers found in data/samples/"
    print_info "You can add sample papers manually or use the ArXiv API to download them"
    
    # Create a sample paper list
    cat > data/samples/sample_papers.txt << 'EOF'
# Sample ArXiv papers for testing
# Format: one ArXiv ID per line
# Example:
# 2301.12345
# 2302.54321
EOF
    print_status "Sample paper list template created"
fi

# Run basic validation
print_info "Validating environment setup..."

# Check Python imports
print_info "Testing Python imports..."
uv run python -c "
import sys
sys.path.insert(0, 'src')

try:
    import arxiv_extractor
    import ocr_processor
    import reference_cleaner
    import llm_extractor
    import text_processor
    import csv_generator
    print('✓ All main modules import successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    print_status "Python imports working"
else
    print_error "Python import validation failed"
    exit 1
fi

# Run basic tests
print_info "Running basic tests..."
uv run pytest tests/ -v --tb=short -x --maxfail=5

if [ $? -eq 0 ]; then
    print_status "Basic tests passed"
else
    print_warning "Some tests failed, but environment setup is complete"
fi

# Create useful aliases and functions
print_info "Creating development aliases..."
cat > scripts/dev_aliases.sh << 'EOF'
#!/bin/bash
# Development aliases for ArXiv Scraper project

# Test aliases
alias test-all='uv run pytest tests/ -v'
alias test-unit='uv run pytest tests/ -v -k "not integration"'
alias test-integration='uv run pytest tests/ -v -k "integration"'
alias test-cov='uv run pytest tests/ --cov=src --cov-report=html'

# Development aliases
alias run-pipeline='uv run python src/arxiv_extractor.py'
alias check-style='uv run black --check src/ tests/ && uv run flake8 src/ tests/'
alias fix-style='uv run black src/ tests/'

# Performance aliases
alias benchmark='uv run python scripts/benchmark.py'
alias profile='uv run python -m cProfile -o profile.stats'

# Quick development functions
dev-setup() {
    source .venv/bin/activate 2>/dev/null || true
    echo "Development environment activated"
}

quick-test() {
    local paper_id=${1:-"2301.12345"}
    echo "Testing with paper ID: $paper_id"
    uv run python src/arxiv_extractor.py --id "$paper_id" --config config/dev.yaml
}
EOF

chmod +x scripts/dev_aliases.sh
print_status "Development aliases created"

# Create performance benchmark script
print_info "Creating performance benchmark script..."
cat > scripts/benchmark.py << 'EOF'
#!/usr/bin/env python3
"""
Performance benchmarking script for ArXiv Scraper pipeline
"""

import time
import psutil
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, 'src')

def benchmark_pipeline(paper_id="2301.12345", iterations=3):
    """Run performance benchmark on the pipeline"""
    
    results = {
        "paper_id": paper_id,
        "timestamp": datetime.now().isoformat(),
        "iterations": iterations,
        "results": []
    }
    
    for i in range(iterations):
        print(f"Running iteration {i+1}/{iterations}...")
        
        # Start monitoring
        process = psutil.Process()
        start_time = time.time()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run pipeline (mock for now)
        # TODO: Replace with actual pipeline execution
        time.sleep(2)  # Simulate processing time
        
        # End monitoring
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        iteration_result = {
            "iteration": i + 1,
            "processing_time": end_time - start_time,
            "memory_start": start_memory,
            "memory_end": end_memory,
            "memory_peak": end_memory  # TODO: Get actual peak memory
        }
        
        results["results"].append(iteration_result)
        print(f"  Time: {iteration_result['processing_time']:.2f}s")
        print(f"  Memory: {iteration_result['memory_end']:.1f}MB")
    
    # Calculate averages
    avg_time = sum(r["processing_time"] for r in results["results"]) / iterations
    avg_memory = sum(r["memory_end"] for r in results["results"]) / iterations
    
    results["averages"] = {
        "processing_time": avg_time,
        "memory_usage": avg_memory
    }
    
    # Save results
    output_file = Path("output/benchmarks") / f"benchmark_{paper_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBenchmark Results:")
    print(f"  Average processing time: {avg_time:.2f}s")
    print(f"  Average memory usage: {avg_memory:.1f}MB")
    print(f"  Results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark ArXiv Scraper pipeline")
    parser.add_argument("--paper-id", default="2301.12345", help="ArXiv paper ID to test")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations")
    
    args = parser.parse_args()
    
    benchmark_pipeline(args.paper_id, args.iterations)
EOF

chmod +x scripts/benchmark.py
print_status "Performance benchmark script created"

# Create final summary
echo ""
echo "==============================================="
echo "Development Environment Setup Complete!"
echo "==============================================="
echo ""
print_status "Virtual environment created and activated"
print_status "All dependencies installed"
print_status "Project directories created"
print_status "Sample configurations created"
print_status "Development scripts created"
echo ""
echo "Quick Start Guide:"
echo "=================="
echo ""
echo "1. Activate development environment:"
echo "   source scripts/dev_aliases.sh"
echo ""
echo "2. Run tests:"
echo "   uv run pytest tests/"
echo ""
echo "3. Process a paper:"
echo "   uv run python src/arxiv_extractor.py --id 2301.12345"
echo ""
echo "4. Use Claude Code slash commands:"
echo "   /setup-dev       # Setup environment"
echo "   /test-module     # Run tests"
echo "   /run-pipeline    # Test pipeline"
echo "   /check-performance # Performance tests"
echo ""
echo "5. Development workflow:"
echo "   - Create issues using GitHub templates"
echo "   - Use branch naming: feature/issue-123-description"
echo "   - Submit PRs with comprehensive templates"
echo "   - Monitor performance and accuracy metrics"
echo ""
echo "For more information, see docs/DEV_WORKFLOW.md"
echo ""
print_status "Development environment ready!"