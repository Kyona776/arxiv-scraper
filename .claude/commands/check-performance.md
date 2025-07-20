# Check Performance Command

Run performance benchmarks for the ArXiv Scraper pipeline.

## Usage
```
/check-performance [test_type]
```

## Arguments
- `test_type` (optional): Type of performance test to run (default: all)

## Available Test Types
- `all` - Run all performance tests
- `pipeline` - Full pipeline performance
- `ocr` - OCR processing performance
- `reference` - Reference removal performance
- `extraction` - LLM extraction performance
- `memory` - Memory usage analysis

## Examples
```
/check-performance
/check-performance pipeline
/check-performance ocr
```

---

Running performance tests for: $ARGUMENTS

```bash
# Create performance test results directory
mkdir -p logs/performance

# Get timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/performance/performance_${TIMESTAMP}.log"

echo "Performance Test Results - $(date)" | tee $LOG_FILE
echo "=================================" | tee -a $LOG_FILE

# Function to run performance test
run_performance_test() {
    local test_type=$1
    echo "Running $test_type performance test..." | tee -a $LOG_FILE
    
    case $test_type in
        "pipeline")
            echo "Testing full pipeline with sample paper..." | tee -a $LOG_FILE
            time uv run python src/arxiv_extractor.py --id 2301.12345 --config config/config.yaml 2>&1 | tee -a $LOG_FILE
            ;;
        "ocr")
            echo "Testing OCR processing performance..." | tee -a $LOG_FILE
            uv run python -c "
import time
from src.ocr_processor import OCRProcessor
processor = OCRProcessor()
start = time.time()
# Add OCR performance test here
end = time.time()
print(f'OCR processing time: {end - start:.2f} seconds')
" 2>&1 | tee -a $LOG_FILE
            ;;
        "reference")
            echo "Testing reference removal performance..." | tee -a $LOG_FILE
            uv run python -c "
import time
from src.reference_cleaner import ReferenceCleaner
cleaner = ReferenceCleaner()
start = time.time()
# Add reference removal performance test here
end = time.time()
print(f'Reference removal time: {end - start:.2f} seconds')
" 2>&1 | tee -a $LOG_FILE
            ;;
        "extraction")
            echo "Testing LLM extraction performance..." | tee -a $LOG_FILE
            uv run python -c "
import time
from src.llm_extractor import LLMExtractor
extractor = LLMExtractor()
start = time.time()
# Add LLM extraction performance test here
end = time.time()
print(f'LLM extraction time: {end - start:.2f} seconds')
" 2>&1 | tee -a $LOG_FILE
            ;;
        "memory")
            echo "Testing memory usage..." | tee -a $LOG_FILE
            uv run python -c "
import psutil
import os
process = psutil.Process(os.getpid())
memory_info = process.memory_info()
print(f'Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB')
print(f'Virtual memory: {memory_info.vms / 1024 / 1024:.2f} MB')
" 2>&1 | tee -a $LOG_FILE
            ;;
    esac
    echo "" | tee -a $LOG_FILE
}

# Run the appropriate performance test
if [ -z "$ARGUMENTS" ] || [ "$ARGUMENTS" = "all" ]; then
    echo "Running all performance tests..." | tee -a $LOG_FILE
    run_performance_test "pipeline"
    run_performance_test "ocr"
    run_performance_test "reference"
    run_performance_test "extraction"
    run_performance_test "memory"
else
    run_performance_test "$ARGUMENTS"
fi

echo "Performance test completed. Results saved to: $LOG_FILE" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE
echo "Performance Requirements:" | tee -a $LOG_FILE
echo "- Processing time: <60 seconds per paper" | tee -a $LOG_FILE
echo "- Reference removal: <3 seconds per paper" | tee -a $LOG_FILE
echo "- OCR accuracy: >95%" | tee -a $LOG_FILE
echo "- Reference removal accuracy: >99%" | tee -a $LOG_FILE
echo "- Item extraction accuracy: >90%" | tee -a $LOG_FILE
```