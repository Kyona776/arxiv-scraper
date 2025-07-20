# Test Module Command

Test specific modules or run all tests for the ArXiv Scraper project.

## Usage
```
/test-module [module_name]
```

## Arguments
- `module_name` (optional): Name of the module to test (default: all)

## Available Modules
- `all` - Run all tests
- `reference_cleaner` - Test reference removal module
- `ocr_processor` - Test OCR processing module
- `llm_extractor` - Test LLM extraction module
- `arxiv_api` - Test ArXiv API module
- `text_processor` - Test text processing module
- `csv_generator` - Test CSV generation module

## Examples
```
/test-module
/test-module reference_cleaner
/test-module ocr_processor
```

---

Running tests for: $ARGUMENTS

```bash
# Determine which tests to run
if [ -z "$ARGUMENTS" ] || [ "$ARGUMENTS" = "all" ]; then
    echo "Running all tests..."
    uv run pytest tests/ -v --cov=src --cov-report=term-missing
elif [ "$ARGUMENTS" = "reference_cleaner" ]; then
    echo "Running reference cleaner tests..."
    uv run pytest tests/test_reference_cleaner.py -v
elif [ "$ARGUMENTS" = "ocr_processor" ]; then
    echo "Running OCR processor tests..."
    uv run pytest tests/test_ocr_processor.py -v
elif [ "$ARGUMENTS" = "llm_extractor" ]; then
    echo "Running LLM extractor tests..."
    uv run pytest tests/test_llm_extractor.py -v
elif [ "$ARGUMENTS" = "arxiv_api" ]; then
    echo "Running ArXiv API tests..."
    uv run pytest tests/test_arxiv_api.py -v
elif [ "$ARGUMENTS" = "text_processor" ]; then
    echo "Running text processor tests..."
    uv run pytest tests/test_text_processor.py -v
elif [ "$ARGUMENTS" = "csv_generator" ]; then
    echo "Running CSV generator tests..."
    uv run pytest tests/test_csv_generator.py -v
else
    echo "Unknown module: $ARGUMENTS"
    echo "Available modules: all, reference_cleaner, ocr_processor, llm_extractor, arxiv_api, text_processor, csv_generator"
    exit 1
fi
```