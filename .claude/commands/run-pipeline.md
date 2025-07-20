# Run Pipeline Command

Execute the full ArXiv Scraper pipeline with a specific paper for testing.

## Usage
```
/run-pipeline [paper_id_or_path]
```

## Arguments
- `paper_id_or_path`: ArXiv ID (e.g., 2301.12345) or path to PDF file

## Examples
```
/run-pipeline 2301.12345
/run-pipeline data/sample_paper.pdf
/run-pipeline https://arxiv.org/abs/2301.12345
```

---

Running ArXiv Scraper pipeline with: $ARGUMENTS

```bash
# Validate arguments
if [ -z "$ARGUMENTS" ]; then
    echo "Usage: /run-pipeline [paper_id_or_path]"
    echo "Examples:"
    echo "  /run-pipeline 2301.12345"
    echo "  /run-pipeline data/sample_paper.pdf"
    echo "  /run-pipeline https://arxiv.org/abs/2301.12345"
    exit 1
fi

INPUT="$ARGUMENTS"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/processing/pipeline_${TIMESTAMP}.log"

# Create logs directory if it doesn't exist
mkdir -p logs/processing

echo "ArXiv Scraper Pipeline Test - $(date)" | tee $LOG_FILE
echo "====================================" | tee -a $LOG_FILE
echo "Input: $INPUT" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Determine input type and set appropriate flags
if [[ "$INPUT" =~ ^[0-9]{4}\.[0-9]{4,5}$ ]]; then
    # ArXiv ID format
    echo "Processing ArXiv ID: $INPUT" | tee -a $LOG_FILE
    CMD="uv run python src/arxiv_extractor.py --id $INPUT --remove-refs --config config/config.yaml"
elif [[ "$INPUT" =~ ^https?://arxiv\.org/abs/ ]]; then
    # ArXiv URL format
    echo "Processing ArXiv URL: $INPUT" | tee -a $LOG_FILE
    CMD="uv run python src/arxiv_extractor.py --url $INPUT --remove-refs --config config/config.yaml"
elif [[ -f "$INPUT" ]]; then
    # Local PDF file
    echo "Processing local PDF: $INPUT" | tee -a $LOG_FILE
    CMD="uv run python src/arxiv_extractor.py --pdf $INPUT --remove-refs --config config/config.yaml"
else
    echo "Error: Invalid input format or file not found: $INPUT" | tee -a $LOG_FILE
    echo "Expected formats:" | tee -a $LOG_FILE
    echo "  ArXiv ID: 2301.12345" | tee -a $LOG_FILE
    echo "  ArXiv URL: https://arxiv.org/abs/2301.12345" | tee -a $LOG_FILE
    echo "  Local PDF: path/to/paper.pdf" | tee -a $LOG_FILE
    exit 1
fi

# Record start time
START_TIME=$(date +%s)
echo "Starting pipeline at $(date)" | tee -a $LOG_FILE
echo "Command: $CMD" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Run the pipeline
echo "Running pipeline..." | tee -a $LOG_FILE
echo "===================" | tee -a $LOG_FILE

# Execute the command and capture output
eval $CMD 2>&1 | tee -a $LOG_FILE
PIPELINE_EXIT_CODE=$?

# Record end time and calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "" | tee -a $LOG_FILE
echo "Pipeline completed at $(date)" | tee -a $LOG_FILE
echo "Total processing time: ${DURATION} seconds" | tee -a $LOG_FILE
echo "Exit code: $PIPELINE_EXIT_CODE" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Performance assessment
echo "Performance Assessment:" | tee -a $LOG_FILE
echo "======================" | tee -a $LOG_FILE

if [ $DURATION -lt 60 ]; then
    echo "✓ Processing time: ${DURATION}s (requirement: <60s)" | tee -a $LOG_FILE
else
    echo "✗ Processing time: ${DURATION}s (requirement: <60s)" | tee -a $LOG_FILE
fi

# Check if output files were created
if [ -d "output/csv" ]; then
    CSV_FILES=$(find output/csv -name "*.csv" -newer logs/processing/pipeline_${TIMESTAMP}.log 2>/dev/null)
    if [ -n "$CSV_FILES" ]; then
        echo "✓ CSV output files created:" | tee -a $LOG_FILE
        echo "$CSV_FILES" | tee -a $LOG_FILE
    else
        echo "✗ No CSV output files found" | tee -a $LOG_FILE
    fi
fi

# Final status
echo "" | tee -a $LOG_FILE
if [ $PIPELINE_EXIT_CODE -eq 0 ]; then
    echo "✓ Pipeline completed successfully!" | tee -a $LOG_FILE
else
    echo "✗ Pipeline failed with exit code: $PIPELINE_EXIT_CODE" | tee -a $LOG_FILE
fi

echo "" | tee -a $LOG_FILE
echo "Full log saved to: $LOG_FILE" | tee -a $LOG_FILE
echo "Output files (if any) saved to: output/" | tee -a $LOG_FILE
```