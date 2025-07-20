# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an agentic LLM pipeline system for automatically extracting 9 specific items from arXiv papers and outputting them in CSV format. The system integrates OCR for high-precision text extraction and includes automatic reference section removal.

## Core Architecture

### Pipeline Flow
```
Input → PDF取得 → OCR処理 → Reference削除 → テキスト後処理 → LLM解析 → 検証 → CSV出力
```

### Main Components

#### 1. Paper Acquisition Module (`arxiv_api.py`)
- arXiv API integration for metadata and PDF retrieval
- Support for arXiv IDs, URLs, and direct PDF files
- Batch processing capabilities

#### 2. OCR Processing Module (`ocr_processor.py`)
- Multiple OCR model support: Mistral OCR (local), Mistral OCR API (direct), Mistral OCR OpenRouter, Nougat, Unstructured, Surya
- **API Integration**: Direct Mistral API and OpenRouter API support for cloud-based OCR
- Mathematical formula recognition with LaTeX conversion
- Figure/table caption extraction
- Layout preservation and section structure analysis

#### 3. Reference Removal Module (`reference_cleaner.py`) - **Critical Component**
- **Detection patterns**: "References", "Bibliography", "参考文献", numbered lists, author patterns
- **Algorithm**: Section boundary detection → Pattern matching → Structure analysis → Safe deletion
- **Quality assurance**: Prevents over-deletion while ensuring complete reference removal

#### 4. Text Post-processing Module (`text_processor.py`)
- OCR error correction
- Post-reference-removal quality verification
- Section boundary detection
- Structured data generation

#### 5. LLM Analysis Module (`llm_extractor.py`)
- **Multi-provider support**: OpenAI, Anthropic, and OpenRouter API integration
- **OpenRouter models**: Access to Claude, GPT-4, Mistral, Gemini, and more via single API
- Optimized prompts for each of the 9 extraction items
- Structured output generation with JSON formatting
- Quality evaluation and confidence scoring

#### 6. Output Management Module (`csv_generator.py`)
- CSV generation with UTF-8 encoding
- Error handling and validation
- Logging

## Target Extraction Items

The system extracts these 9 items from each paper:
1. **手法の肝** (Core methodology)
2. **制限事項** (Limitations)  
3. **対象ナレッジ** (Target knowledge domain)
4. **URL** (arXiv URL)
5. **タイトル** (Title)
6. **出版年** (Publication year)
7. **研究分野** (Research field)
8. **課題設定** (Problem setting)
9. **論文の主張** (Main claims)

## Development Commands

### Setup
```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

# Alternative: Direct installation without manual venv activation
uv sync

# Set up environment variables for API access
cp .env.example .env
# Edit .env with your actual API keys
```

### Single Paper Processing
```bash
# Using uv run (recommended - automatically manages environment)
uv run arxiv_extractor.py --id 2301.12345
uv run arxiv_extractor.py --pdf paper.pdf --remove-refs

# Traditional method (requires activated venv)
python arxiv_extractor.py --id 2301.12345
python arxiv_extractor.py --pdf paper.pdf --remove-refs
```

### Batch Processing
```bash
# Using uv run
uv run arxiv_extractor.py --batch paper_list.txt --config config.yaml

# Traditional method
python arxiv_extractor.py --batch paper_list.txt --config config.yaml
```

### Testing
```bash
# Run all tests using uv (recommended)
uv run pytest tests/

# Test specific modules
uv run pytest tests/test_reference_cleaner.py
uv run pytest tests/test_ocr_processor.py

# Test with coverage
uv run pytest --cov=src tests/

# Traditional method (requires activated venv)
pytest tests/
pytest tests/test_reference_cleaner.py
pytest --cov=src tests/
```

### Configuration

Main configuration in `config.yaml`:
```yaml
ocr:
  model: "mistral_ocr_api"  # Options: mistral_ocr, mistral_ocr_api, mistral_ocr_openrouter, nougat, unstructured, surya
  device: "cuda"
  
  # Direct Mistral API configuration
  mistral_ocr_api:
    api_key: null  # Set via MISTRAL_API_KEY environment variable
    base_url: "https://api.mistral.ai"
    model: "mistral-ocr-latest"
    timeout: 120
    max_retries: 3
  
  # OpenRouter API configuration
  mistral_ocr_openrouter:
    api_key: null  # Set via OPENROUTER_API_KEY environment variable
    base_url: "https://openrouter.ai/api/v1"
    model: "mistral/mistral-ocr-latest"
    timeout: 120
    max_retries: 3
  
text_processing:
  remove_references: true
  reference_patterns:
    - "REFERENCES"
    - "Bibliography" 
    - "参考文献"
    
llm:
  model: "anthropic/claude-3-sonnet"  # OpenRouter model
  fallback_models:
    - "openai/gpt-4"
    - "mistral/mistral-large"
  
  openrouter:
    api_key: null  # Set via OPENROUTER_API_KEY
    model: "anthropic/claude-3-sonnet"
    timeout: 120
    max_retries: 3
  
  openai:
    api_key: null  # Set via OPENAI_API_KEY
    
  anthropic:
    api_key: null  # Set via ANTHROPIC_API_KEY
```

### API Keys Configuration

Create a `.env` file from the example template:
```bash
cp .env.example .env
```

Set your API keys in the `.env` file:
```bash
# OpenRouter API (recommended - access to multiple models)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Direct Mistral API
MISTRAL_API_KEY=your_mistral_api_key_here

# Direct LLM APIs
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### OpenRouter Model Support

The system now supports OpenRouter API for accessing multiple LLM models through a single API:

#### Available Models via OpenRouter:
- **Anthropic**: `anthropic/claude-3-opus`, `anthropic/claude-3-sonnet`, `anthropic/claude-3-haiku`
- **OpenAI**: `openai/gpt-4`, `openai/gpt-3.5-turbo`
- **Google**: `google/gemini-pro`
- **Mistral**: `mistral/mistral-large`, `mistral/mistral-medium`
- **Meta**: `meta/llama-3-70b`
- **Cohere**: `cohere/command-r-plus`

#### Benefits of OpenRouter:
- **Single API key** for multiple model providers
- **Automatic failover** between models
- **Cost optimization** through competitive pricing
- **No need** for multiple API keys from different providers
- **Real-time model discovery** and validation
- **Endpoint health monitoring** and selection
- **Provider policy checking** for compliance

### OpenRouter Management Features

#### Model Discovery and Validation
```bash
# List all available models
python src/cli/openrouter_cli.py list-models

# Check if a model exists
python src/cli/openrouter_cli.py check-model anthropic/claude-3-sonnet

# Get detailed model information
python src/cli/openrouter_cli.py model-info anthropic/claude-3-sonnet

# Get model recommendations for specific tasks
python src/cli/openrouter_cli.py recommend --task-type coding --budget 0.00001
```

#### Provider Management
```bash
# List all providers
python src/cli/openrouter_cli.py list-providers

# Check provider policies (logging, training, moderation)
python src/cli/openrouter_cli.py list-providers --json
```

#### Model Filtering and Comparison
```bash
# Filter models by provider
python src/cli/openrouter_cli.py list-models --provider anthropic

# Filter by cost and context length
python src/cli/openrouter_cli.py list-models --max-cost 0.00001 --min-context 32000

# Compare multiple models
python src/cli/openrouter_cli.py compare anthropic/claude-3-sonnet openai/gpt-4 google/gemini-pro
```

#### Endpoint Management
```bash
# Check endpoints for a model
python src/cli/openrouter_cli.py check-endpoints anthropic/claude-3-sonnet

# View endpoint health and performance metrics
python src/cli/openrouter_cli.py check-endpoints anthropic/claude-3-sonnet --json
```

#### Configuration-Based Model Selection
```yaml
llm:
  openrouter:
    model_preferences:
      task_types:
        coding: ["anthropic/claude-3-opus", "openai/gpt-4"]
        conversation: ["anthropic/claude-3-sonnet", "anthropic/claude-3-haiku"]
        analysis: ["anthropic/claude-3-opus", "google/gemini-pro"]
    
    provider_preferences:
      preferred: ["anthropic", "openai", "google"]
      privacy_requirements:
        no_logging: true
        no_training: true
    
    model_filters:
      min_context_length: 8000
      max_cost_per_token: 0.00005
```

## Reference Removal Implementation Details

### Critical Patterns for Detection
- **Section headers**: Various formats of "References", "Bibliography"
- **Citation formats**: IEEE [1], APA (2023), Nature 1., arXiv patterns
- **Structural markers**: DOI, URL, author name patterns

### Safety Measures
- Preserve in-text citations in main content
- Log all deletions for verification
- Quality checks post-removal
- Fallback mechanisms for edge cases

## Performance Requirements

- Processing time: <60 seconds per paper (including OCR + reference removal)
- Reference removal: <3 seconds per paper
- OCR accuracy: >95%
- Reference removal accuracy: >99%
- Item extraction accuracy: >90%

## Error Handling Priorities

1. **Reference removal failures**: Fallback to manual pattern detection
2. **OCR failures**: Multiple model fallback chain
3. **LLM extraction errors**: Retry with modified prompts
4. **Memory issues**: Batch size reduction and streaming processing

## Code Organization

- `src/`: Main source code
- `tests/`: Test suites with focus on reference removal accuracy
- `config/`: Configuration files and templates
- `data/`: Sample papers and test datasets
- `output/`: Generated CSV files and processing logs

## Development Focus Areas

1. **Reference removal precision**: This is the most critical differentiator
2. **OCR integration**: Multiple model support with fallback chains
3. **Batch processing efficiency**: Memory optimization for large datasets
4. **Output validation**: Ensuring all 9 items are consistently extracted

## Development Workflow Integration

This project uses a comprehensive development workflow that integrates GitHub Projects, Issues, Pull Requests, and Claude Code slash commands. For complete workflow details, see `docs/DEV_WORKFLOW.md`.

### GitHub Integration

- **Project Board**: GitHub Project with Kanban-style workflow
- **Issue Templates**: Specialized templates for different issue types
- **PR Templates**: Comprehensive pull request templates with checklists
- **Automation**: Automated project management and status updates

### Claude Code Slash Commands

Custom slash commands are available in `.claude/commands/` for streamlined development:

#### `/test-module [module_name]`
Run tests for specific modules or all tests
```bash
/test-module                  # Run all tests
/test-module reference_cleaner # Test reference removal
/test-module ocr_processor    # Test OCR processing
```

#### `/check-performance [test_type]`
Run performance benchmarks
```bash
/check-performance           # Run all performance tests
/check-performance pipeline  # Test full pipeline
/check-performance ocr       # Test OCR performance
```

#### `/create-issue [type] [title]`
Create GitHub issues with proper templates
```bash
/create-issue bug "OCR accuracy issue"
/create-issue feature "Add new OCR model"
/create-issue performance "Slow processing"
```

#### `/setup-dev`
Set up the development environment
```bash
/setup-dev  # Complete environment setup
```

#### `/run-pipeline [paper_id]`
Execute full pipeline test
```bash
/run-pipeline 2301.12345              # Test with ArXiv ID
/run-pipeline data/sample_paper.pdf   # Test with local PDF
```

#### `/check-refs [file_path]`
Test reference removal accuracy
```bash
/check-refs data/sample_paper.pdf  # Test reference removal
```

### Development Scripts

Comprehensive automation scripts in `scripts/` directory:

- **`setup_dev_env.sh`**: Complete development environment setup
- **`run_tests.sh`**: Advanced test runner with coverage and reporting
- **`check_performance.sh`**: Performance monitoring and benchmarking
- **`create_release.sh`**: Automated release preparation and deployment

### Quick Development Commands

```bash
# Environment setup
/setup-dev

# Development workflow
/test-module all
/check-performance
/run-pipeline 2301.12345

# Issue management
/create-issue bug "Description"
/create-issue feature "New feature request"

# Performance monitoring
/check-refs data/sample.pdf
```

### Best Practices

1. **Always create issues** before starting development work
2. **Use appropriate issue templates** for consistent reporting
3. **Test thoroughly** with `/test-module` before committing
4. **Monitor performance** with `/check-performance` for critical changes
5. **Use descriptive commit messages** with issue references
6. **Follow PR template** for comprehensive code reviews

### Workflow Documentation

- **Main Guide**: `docs/DEV_WORKFLOW.md` - Complete workflow documentation
- **Project Setup**: `docs/project_setup.md` - GitHub Project configuration
- **Issue Templates**: `.github/ISSUE_TEMPLATE/` - Specialized issue templates
- **PR Template**: `.github/pull_request_template.md` - Pull request template

This integrated workflow ensures consistent development practices, high code quality, and efficient project management while maintaining the focus on the critical reference removal accuracy that differentiates this project.