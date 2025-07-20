# Setup Development Environment

Set up the development environment for the ArXiv Scraper project.

## Usage
```
/setup-dev
```

## What it does
- Creates and activates virtual environment
- Installs all dependencies
- Sets up pre-commit hooks
- Creates necessary directories
- Validates environment setup

---

Setting up development environment...

```bash
echo "Setting up ArXiv Scraper development environment..."
echo "=================================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install uv first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
uv venv
echo "✓ Virtual environment created"

# Install dependencies
echo "Installing dependencies..."
uv pip install -r requirements.txt
echo "✓ Dependencies installed"

# Install development dependencies
echo "Installing development dependencies..."
uv pip install pytest pytest-cov black flake8 pre-commit
echo "✓ Development dependencies installed"

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/samples
mkdir -p logs/performance
mkdir -p logs/processing
mkdir -p output/csv
mkdir -p output/processed
echo "✓ Project directories created"

# Set up pre-commit hooks (if .pre-commit-config.yaml exists)
if [ -f ".pre-commit-config.yaml" ]; then
    echo "Setting up pre-commit hooks..."
    uv run pre-commit install
    echo "✓ Pre-commit hooks installed"
fi

# Validate environment
echo "Validating environment setup..."
echo "Python version:"
uv run python --version

echo "Testing imports..."
uv run python -c "
try:
    import src.arxiv_extractor
    import src.ocr_processor
    import src.reference_cleaner
    import src.llm_extractor
    print('✓ All main modules import successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    exit(1)
"

echo "Running basic tests..."
uv run pytest tests/ -v --tb=short -x

echo ""
echo "Development environment setup complete!"
echo "======================================="
echo ""
echo "Quick start:"
echo "  uv run python src/arxiv_extractor.py --id 2301.12345"
echo ""
echo "Run tests:"
echo "  /test-module"
echo ""
echo "Check performance:"
echo "  /check-performance"
echo ""
echo "Create issue:"
echo "  /create-issue bug 'Issue description'"
```