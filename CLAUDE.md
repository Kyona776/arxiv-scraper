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
- Multiple OCR model support: Mistral OCR, Nougat, Unstructured, Surya
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
- Optimized prompts for each of the 9 extraction items
- Structured output generation
- Quality evaluation

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
  model: "mistral_ocr"
  device: "cuda"
  
text_processing:
  remove_references: true
  reference_patterns:
    - "REFERENCES"
    - "Bibliography" 
    - "参考文献"
    
llm:
  model: "gpt-4"
  temperature: 0.1
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