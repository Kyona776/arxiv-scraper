# Check References Command

Test reference removal accuracy on a specific paper or document.

## Usage
```
/check-refs [file_path]
```

## Arguments
- `file_path`: Path to PDF file to test reference removal

## Examples
```
/check-refs data/sample_paper.pdf
/check-refs /path/to/paper.pdf
```

---

Testing reference removal for: $ARGUMENTS

```bash
# Validate arguments
if [ -z "$ARGUMENTS" ]; then
    echo "Usage: /check-refs [file_path]"
    echo "Example: /check-refs data/sample_paper.pdf"
    exit 1
fi

FILE_PATH="$ARGUMENTS"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/processing/ref_check_${TIMESTAMP}.log"

# Create logs directory if it doesn't exist
mkdir -p logs/processing

# Check if file exists
if [ ! -f "$FILE_PATH" ]; then
    echo "Error: File not found: $FILE_PATH"
    exit 1
fi

echo "Reference Removal Test - $(date)" | tee $LOG_FILE
echo "=================================" | tee -a $LOG_FILE
echo "File: $FILE_PATH" | tee -a $LOG_FILE
echo "File size: $(ls -lh "$FILE_PATH" | awk '{print $5}')" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Record start time
START_TIME=$(date +%s)
echo "Starting reference removal test at $(date)" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Run reference removal test
echo "Testing reference removal..." | tee -a $LOG_FILE
echo "============================" | tee -a $LOG_FILE

# Create a Python script to test reference removal
uv run python -c "
import sys
import time
from pathlib import Path
from src.reference_cleaner import ReferenceCleaner
from src.ocr_processor import OCRProcessor

# Initialize processors
ocr_processor = OCRProcessor()
ref_cleaner = ReferenceCleaner()

file_path = '$FILE_PATH'
print(f'Processing file: {file_path}')

try:
    # Step 1: Extract text from PDF
    print('Step 1: Extracting text from PDF...')
    start_ocr = time.time()
    text = ocr_processor.process_pdf(file_path)
    end_ocr = time.time()
    print(f'OCR processing time: {end_ocr - start_ocr:.2f} seconds')
    print(f'Extracted text length: {len(text)} characters')
    
    # Step 2: Analyze references before removal
    print('\\nStep 2: Analyzing text structure...')
    lines = text.split('\\n')
    total_lines = len(lines)
    print(f'Total lines: {total_lines}')
    
    # Look for common reference patterns
    ref_patterns = ['References', 'Bibliography', 'REFERENCES', 'BIBLIOGRAPHY', '参考文献']
    ref_section_found = False
    ref_start_line = -1
    
    for i, line in enumerate(lines):
        for pattern in ref_patterns:
            if pattern in line and len(line.strip()) < 50:  # Likely a header
                ref_section_found = True
                ref_start_line = i
                print(f'Found reference section at line {i}: \"{line.strip()}\"')
                break
        if ref_section_found:
            break
    
    if not ref_section_found:
        print('No clear reference section header found')
    
    # Step 3: Remove references
    print('\\nStep 3: Removing references...')
    start_ref = time.time()
    cleaned_text = ref_cleaner.remove_references(text)
    end_ref = time.time()
    ref_removal_time = end_ref - start_ref
    print(f'Reference removal time: {ref_removal_time:.2f} seconds')
    
    # Step 4: Analyze results
    print('\\nStep 4: Analyzing results...')
    cleaned_lines = cleaned_text.split('\\n')
    cleaned_total_lines = len(cleaned_lines)
    removed_lines = total_lines - cleaned_total_lines
    removal_percentage = (removed_lines / total_lines) * 100
    
    print(f'Lines before removal: {total_lines}')
    print(f'Lines after removal: {cleaned_total_lines}')
    print(f'Lines removed: {removed_lines}')
    print(f'Removal percentage: {removal_percentage:.1f}%')
    
    # Step 5: Quality checks
    print('\\nStep 5: Quality assessment...')
    
    # Check if references were actually removed
    remaining_refs = 0
    for pattern in ref_patterns:
        if pattern in cleaned_text:
            remaining_refs += cleaned_text.count(pattern)
    
    if remaining_refs > 0:
        print(f'⚠️  Warning: {remaining_refs} reference headers still found in cleaned text')
    else:
        print('✓ No reference headers found in cleaned text')
    
    # Check for common reference patterns
    common_ref_indicators = ['doi:', 'arxiv:', 'www.', 'http://', 'https://']
    remaining_indicators = sum(cleaned_text.lower().count(indicator) for indicator in common_ref_indicators)
    original_indicators = sum(text.lower().count(indicator) for indicator in common_ref_indicators)
    
    if remaining_indicators > 0:
        reduction = ((original_indicators - remaining_indicators) / original_indicators) * 100
        print(f'Reference indicators reduced by {reduction:.1f}% ({original_indicators} → {remaining_indicators})')
    
    # Performance assessment
    print('\\nPerformance Assessment:')
    print('======================')
    
    if ref_removal_time < 3.0:
        print(f'✓ Reference removal time: {ref_removal_time:.2f}s (requirement: <3s)')
    else:
        print(f'✗ Reference removal time: {ref_removal_time:.2f}s (requirement: <3s)')
    
    if removal_percentage > 5 and removal_percentage < 30:
        print(f'✓ Removal percentage: {removal_percentage:.1f}% (reasonable range: 5-30%)')
    elif removal_percentage <= 5:
        print(f'⚠️  Removal percentage: {removal_percentage:.1f}% (may not have removed enough)')
    else:
        print(f'⚠️  Removal percentage: {removal_percentage:.1f}% (may have removed too much)')
    
    # Save samples for manual inspection
    output_dir = Path('output/processed')
    output_dir.mkdir(exist_ok=True)
    
    # Save before/after samples
    with open(output_dir / f'before_ref_removal_{TIMESTAMP}.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    
    with open(output_dir / f'after_ref_removal_{TIMESTAMP}.txt', 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    print(f'\\nSample files saved to: {output_dir}')
    print(f'  - before_ref_removal_{TIMESTAMP}.txt')
    print(f'  - after_ref_removal_{TIMESTAMP}.txt')
    
except Exception as e:
    print(f'Error during reference removal test: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" 2>&1 | tee -a $LOG_FILE

# Record end time and calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "" | tee -a $LOG_FILE
echo "Reference removal test completed at $(date)" | tee -a $LOG_FILE
echo "Total test time: ${DURATION} seconds" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

echo "Test completed. Review the output above and check the sample files." | tee -a $LOG_FILE
echo "Full log saved to: $LOG_FILE" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE
echo "Manual inspection recommended:" | tee -a $LOG_FILE
echo "1. Check before/after text files in output/processed/" | tee -a $LOG_FILE
echo "2. Verify that references were properly removed" | tee -a $LOG_FILE
echo "3. Ensure main content was preserved" | tee -a $LOG_FILE
```