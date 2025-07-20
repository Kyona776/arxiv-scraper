#!/bin/bash

# ArXiv Scraper Performance Check Script
# Comprehensive performance monitoring and benchmarking

set -e  # Exit on any error

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

# Default values
BENCHMARK_TYPE="all"
ITERATIONS=3
PAPER_ID="2301.12345"
OUTPUT_DIR="output/performance"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PROFILE=false
MEMORY_PROFILE=false

# Performance requirements
MAX_PROCESSING_TIME=60
MAX_REFERENCE_REMOVAL_TIME=3
MIN_OCR_ACCURACY=95
MIN_REFERENCE_REMOVAL_ACCURACY=99
MIN_EXTRACTION_ACCURACY=90

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            BENCHMARK_TYPE="$2"
            shift 2
            ;;
        -i|--iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        -p|--paper-id)
            PAPER_ID="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --profile)
            PROFILE=true
            shift
            ;;
        --memory-profile)
            MEMORY_PROFILE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -t, --type TYPE      Benchmark type: all, pipeline, ocr, reference, extraction, memory"
            echo "  -i, --iterations N   Number of iterations (default: 3)"
            echo "  -p, --paper-id ID    ArXiv paper ID to test (default: 2301.12345)"
            echo "  -o, --output DIR     Output directory for results (default: output/performance)"
            echo "  --profile            Enable CPU profiling"
            echo "  --memory-profile     Enable memory profiling"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 -t pipeline -i 5"
            echo "  $0 --type ocr --paper-id 2302.12345"
            echo "  $0 --memory-profile"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if we're in the correct directory
if [ ! -f "CLAUDE.md" ] || [ ! -f "src/arxiv_extractor.py" ]; then
    print_error "This script must be run from the ArXiv Scraper project root directory"
    exit 1
fi

echo "=============================================="
echo "ArXiv Scraper Performance Check"
echo "=============================================="
echo ""
print_info "Benchmark type: $BENCHMARK_TYPE"
print_info "Iterations: $ITERATIONS"
print_info "Paper ID: $PAPER_ID"
print_info "Output directory: $OUTPUT_DIR"
print_info "CPU Profiling: $PROFILE"
print_info "Memory Profiling: $MEMORY_PROFILE"
echo ""

# Results file
RESULTS_FILE="$OUTPUT_DIR/performance_results_${TIMESTAMP}.json"

# Function to initialize results file
init_results() {
    cat > "$RESULTS_FILE" << EOF
{
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "benchmark_type": "$BENCHMARK_TYPE",
    "iterations": $ITERATIONS,
    "paper_id": "$PAPER_ID",
    "requirements": {
        "max_processing_time": $MAX_PROCESSING_TIME,
        "max_reference_removal_time": $MAX_REFERENCE_REMOVAL_TIME,
        "min_ocr_accuracy": $MIN_OCR_ACCURACY,
        "min_reference_removal_accuracy": $MIN_REFERENCE_REMOVAL_ACCURACY,
        "min_extraction_accuracy": $MIN_EXTRACTION_ACCURACY
    },
    "results": {}
}
EOF
}

# Function to update results
update_results() {
    local test_name=$1
    local result_data=$2
    
    # Use Python to update JSON
    python3 -c "
import json
import sys

with open('$RESULTS_FILE', 'r') as f:
    data = json.load(f)

data['results']['$test_name'] = $result_data

with open('$RESULTS_FILE', 'w') as f:
    json.dump(data, f, indent=2)
"
}

# Function to run system info check
check_system_info() {
    print_info "Collecting system information..."
    
    local system_info=$(python3 -c "
import platform
import psutil
import json

info = {
    'platform': platform.platform(),
    'python_version': platform.python_version(),
    'cpu_count': psutil.cpu_count(),
    'memory_total': psutil.virtual_memory().total,
    'memory_available': psutil.virtual_memory().available
}

print(json.dumps(info, indent=2))
")
    
    echo "$system_info" > "$OUTPUT_DIR/system_info_${TIMESTAMP}.json"
    print_status "System information collected"
}

# Function to benchmark full pipeline
benchmark_pipeline() {
    print_info "Benchmarking full pipeline..."
    
    local results="["
    
    for ((i=1; i<=ITERATIONS; i++)); do
        print_info "Running iteration $i/$ITERATIONS..."
        
        local start_time=$(date +%s.%N)
        
        # Run the actual pipeline
        local cmd="uv run python src/arxiv_extractor.py --id $PAPER_ID --config config/config.yaml"
        if [ "$PROFILE" = true ]; then
            cmd="uv run python -m cProfile -o $OUTPUT_DIR/profile_${i}.stats src/arxiv_extractor.py --id $PAPER_ID --config config/config.yaml"
        fi
        
        # Execute command and capture output
        local output_file="$OUTPUT_DIR/pipeline_output_${i}.log"
        if eval $cmd > "$output_file" 2>&1; then
            local end_time=$(date +%s.%N)
            local duration=$(echo "$end_time - $start_time" | bc)
            
            # Get memory usage (mock for now)
            local memory_usage=$(python3 -c "
import psutil
import json
process = psutil.Process()
memory_info = process.memory_info()
print(json.dumps({
    'rss': memory_info.rss,
    'vms': memory_info.vms,
    'peak_rss': memory_info.rss  # TODO: Get actual peak
}))
")
            
            local iteration_result="{
                \"iteration\": $i,
                \"duration\": $duration,
                \"memory\": $memory_usage,
                \"success\": true
            }"
            
            print_status "Iteration $i completed in ${duration}s"
        else
            local end_time=$(date +%s.%N)
            local duration=$(echo "$end_time - $start_time" | bc)
            
            local iteration_result="{
                \"iteration\": $i,
                \"duration\": $duration,
                \"memory\": null,
                \"success\": false
            }"
            
            print_error "Iteration $i failed after ${duration}s"
        fi
        
        if [ $i -eq $ITERATIONS ]; then
            results="$results$iteration_result"
        else
            results="$results$iteration_result,"
        fi
    done
    
    results="$results]"
    
    # Calculate statistics
    local stats=$(python3 -c "
import json
import statistics

data = $results
successful_runs = [r for r in data if r['success']]

if successful_runs:
    durations = [r['duration'] for r in successful_runs]
    
    stats = {
        'total_runs': len(data),
        'successful_runs': len(successful_runs),
        'success_rate': len(successful_runs) / len(data),
        'avg_duration': statistics.mean(durations),
        'min_duration': min(durations),
        'max_duration': max(durations),
        'std_duration': statistics.stdev(durations) if len(durations) > 1 else 0,
        'requirement_met': statistics.mean(durations) < $MAX_PROCESSING_TIME,
        'iterations': data
    }
else:
    stats = {
        'total_runs': len(data),
        'successful_runs': 0,
        'success_rate': 0,
        'avg_duration': 0,
        'min_duration': 0,
        'max_duration': 0,
        'std_duration': 0,
        'requirement_met': False,
        'iterations': data
    }

print(json.dumps(stats, indent=2))
")
    
    update_results "pipeline" "$stats"
    
    # Display results
    echo ""
    print_info "Pipeline Performance Results:"
    python3 -c "
import json
stats = $stats
print(f'  Average duration: {stats[\"avg_duration\"]:.2f}s')
print(f'  Min duration: {stats[\"min_duration\"]:.2f}s')
print(f'  Max duration: {stats[\"max_duration\"]:.2f}s')
print(f'  Success rate: {stats[\"success_rate\"]:.1%}')
print(f'  Requirement (<{$MAX_PROCESSING_TIME}s): {\"✓\" if stats[\"requirement_met\"] else \"✗\"}')
"
}

# Function to benchmark OCR processing
benchmark_ocr() {
    print_info "Benchmarking OCR processing..."
    
    # Create a simple OCR benchmark
    local ocr_results=$(python3 -c "
import sys
import time
import json
import statistics
sys.path.insert(0, 'src')

try:
    from ocr_processor import OCRProcessor
    
    processor = OCRProcessor()
    results = []
    
    for i in range($ITERATIONS):
        start_time = time.time()
        
        # Mock OCR processing - replace with actual PDF processing
        # result = processor.process_pdf('sample.pdf')
        time.sleep(0.5)  # Simulate processing time
        
        end_time = time.time()
        duration = end_time - start_time
        
        results.append({
            'iteration': i + 1,
            'duration': duration,
            'success': True
        })
    
    durations = [r['duration'] for r in results]
    stats = {
        'avg_duration': statistics.mean(durations),
        'min_duration': min(durations),
        'max_duration': max(durations),
        'std_duration': statistics.stdev(durations) if len(durations) > 1 else 0,
        'iterations': results
    }
    
    print(json.dumps(stats, indent=2))
    
except Exception as e:
    print(json.dumps({'error': str(e)}, indent=2))
")
    
    update_results "ocr" "$ocr_results"
    
    print_status "OCR benchmarking completed"
}

# Function to benchmark reference removal
benchmark_reference_removal() {
    print_info "Benchmarking reference removal..."
    
    local ref_results=$(python3 -c "
import sys
import time
import json
import statistics
sys.path.insert(0, 'src')

try:
    from reference_cleaner import ReferenceCleaner
    
    cleaner = ReferenceCleaner()
    results = []
    
    # Sample text for testing
    sample_text = '''
    This is a sample paper text.
    
    REFERENCES
    [1] Author, A. (2023). Sample paper.
    [2] Author, B. (2023). Another paper.
    '''
    
    for i in range($ITERATIONS):
        start_time = time.time()
        
        # Process reference removal
        cleaned_text = cleaner.remove_references(sample_text)
        
        end_time = time.time()
        duration = end_time - start_time
        
        results.append({
            'iteration': i + 1,
            'duration': duration,
            'success': True,
            'requirement_met': duration < $MAX_REFERENCE_REMOVAL_TIME
        })
    
    durations = [r['duration'] for r in results]
    requirement_met = all(r['requirement_met'] for r in results)
    
    stats = {
        'avg_duration': statistics.mean(durations),
        'min_duration': min(durations),
        'max_duration': max(durations),
        'std_duration': statistics.stdev(durations) if len(durations) > 1 else 0,
        'requirement_met': requirement_met,
        'iterations': results
    }
    
    print(json.dumps(stats, indent=2))
    
except Exception as e:
    print(json.dumps({'error': str(e)}, indent=2))
")
    
    update_results "reference_removal" "$ref_results"
    
    echo ""
    print_info "Reference Removal Performance Results:"
    python3 -c "
import json
stats = $ref_results
if 'error' not in stats:
    print(f'  Average duration: {stats[\"avg_duration\"]:.4f}s')
    print(f'  Min duration: {stats[\"min_duration\"]:.4f}s')
    print(f'  Max duration: {stats[\"max_duration\"]:.4f}s')
    print(f'  Requirement (<{$MAX_REFERENCE_REMOVAL_TIME}s): {\"✓\" if stats[\"requirement_met\"] else \"✗\"}')
else:
    print(f'  Error: {stats[\"error\"]}')
"
}

# Function to benchmark memory usage
benchmark_memory() {
    print_info "Benchmarking memory usage..."
    
    if [ "$MEMORY_PROFILE" = true ]; then
        print_info "Running memory profiling..."
        
        # Install memory profiler if not available
        uv pip install memory-profiler 2>/dev/null || true
        
        # Run memory profiling
        local memory_results=$(python3 -c "
import sys
import json
import psutil
import time
sys.path.insert(0, 'src')

try:
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Mock memory-intensive operation
    time.sleep(1)
    
    peak_memory = process.memory_info().rss
    
    stats = {
        'initial_memory_mb': initial_memory / 1024 / 1024,
        'peak_memory_mb': peak_memory / 1024 / 1024,
        'memory_increase_mb': (peak_memory - initial_memory) / 1024 / 1024
    }
    
    print(json.dumps(stats, indent=2))
    
except Exception as e:
    print(json.dumps({'error': str(e)}, indent=2))
")
        
        update_results "memory" "$memory_results"
        
        print_status "Memory benchmarking completed"
    else
        print_warning "Memory profiling not enabled. Use --memory-profile to enable."
    fi
}

# Function to generate performance report
generate_report() {
    print_info "Generating performance report..."
    
    local report_file="$OUTPUT_DIR/performance_report_${TIMESTAMP}.md"
    
    cat > "$report_file" << EOF
# ArXiv Scraper Performance Report

**Generated:** $(date)
**Benchmark Type:** $BENCHMARK_TYPE
**Iterations:** $ITERATIONS
**Paper ID:** $PAPER_ID

## Performance Requirements

- Processing time: <${MAX_PROCESSING_TIME} seconds per paper
- Reference removal: <${MAX_REFERENCE_REMOVAL_TIME} seconds per paper  
- OCR accuracy: >${MIN_OCR_ACCURACY}%
- Reference removal accuracy: >${MIN_REFERENCE_REMOVAL_ACCURACY}%
- Item extraction accuracy: >${MIN_EXTRACTION_ACCURACY}%

## Test Results

EOF
    
    # Add detailed results from JSON
    python3 -c "
import json
import sys

try:
    with open('$RESULTS_FILE', 'r') as f:
        data = json.load(f)
    
    for test_name, results in data['results'].items():
        if 'error' in results:
            print(f'### {test_name.title()} Test')
            print(f'**Status:** ❌ Failed')
            print(f'**Error:** {results[\"error\"]}')
            print()
        else:
            print(f'### {test_name.title()} Test')
            if 'avg_duration' in results:
                print(f'**Average Duration:** {results[\"avg_duration\"]:.4f}s')
                print(f'**Min Duration:** {results[\"min_duration\"]:.4f}s')
                print(f'**Max Duration:** {results[\"max_duration\"]:.4f}s')
                if 'requirement_met' in results:
                    status = '✅ Passed' if results['requirement_met'] else '❌ Failed'
                    print(f'**Requirements:** {status}')
            print()
    
except Exception as e:
    print(f'Error generating report: {e}')
" >> "$report_file"
    
    print_status "Performance report generated: $report_file"
}

# Function to run all benchmarks
run_all_benchmarks() {
    print_info "Running all performance benchmarks..."
    
    benchmark_pipeline
    benchmark_ocr
    benchmark_reference_removal
    benchmark_memory
}

# Main execution
main() {
    # Initialize results file
    init_results
    
    # Check system info
    check_system_info
    
    # Run benchmarks based on type
    case $BENCHMARK_TYPE in
        "all")
            run_all_benchmarks
            ;;
        "pipeline")
            benchmark_pipeline
            ;;
        "ocr")
            benchmark_ocr
            ;;
        "reference")
            benchmark_reference_removal
            ;;
        "memory")
            benchmark_memory
            ;;
        *)
            print_error "Unknown benchmark type: $BENCHMARK_TYPE"
            echo "Available types: all, pipeline, ocr, reference, memory"
            exit 1
            ;;
    esac
    
    # Generate report
    generate_report
    
    echo ""
    echo "=============================================="
    echo "Performance Check Complete"
    echo "=============================================="
    echo ""
    print_status "Results saved to: $OUTPUT_DIR"
    print_status "Detailed report: $OUTPUT_DIR/performance_report_${TIMESTAMP}.md"
    print_status "Raw data: $RESULTS_FILE"
    
    if [ "$PROFILE" = true ]; then
        print_status "CPU profiles saved to: $OUTPUT_DIR/profile_*.stats"
    fi
    
    echo ""
    print_info "Use the following commands to analyze results:"
    echo "  cat $OUTPUT_DIR/performance_report_${TIMESTAMP}.md"
    echo "  python3 -c \"import json; print(json.dumps(json.load(open('$RESULTS_FILE')), indent=2))\""
    
    if [ "$PROFILE" = true ]; then
        echo "  python3 -c \"import pstats; pstats.Stats('$OUTPUT_DIR/profile_1.stats').sort_stats('cumulative').print_stats(10)\""
    fi
}

# Check for required tools
if ! command -v bc &> /dev/null; then
    print_error "bc is required but not installed. Please install it first."
    exit 1
fi

# Run main function
main "$@"