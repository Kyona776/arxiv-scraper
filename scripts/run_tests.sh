#!/bin/bash

# ArXiv Scraper Test Runner Script
# Comprehensive testing script with multiple test types and reporting

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
TEST_TYPE="all"
VERBOSE=false
COVERAGE=false
PARALLEL=false
PROFILE=false
OUTPUT_DIR="output/test_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            TEST_TYPE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        --profile)
            PROFILE=true
            shift
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -t, --type TYPE      Test type: all, unit, integration, performance, accuracy"
            echo "  -v, --verbose        Verbose output"
            echo "  -c, --coverage       Generate coverage report"
            echo "  -p, --parallel       Run tests in parallel"
            echo "  --profile            Enable profiling"
            echo "  -o, --output DIR     Output directory for results"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 -t unit -v -c"
            echo "  $0 --type integration --verbose"
            echo "  $0 --coverage --parallel"
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
echo "ArXiv Scraper Test Runner"
echo "=============================================="
echo ""
print_info "Test type: $TEST_TYPE"
print_info "Verbose: $VERBOSE"
print_info "Coverage: $COVERAGE"
print_info "Parallel: $PARALLEL"
print_info "Profile: $PROFILE"
print_info "Output directory: $OUTPUT_DIR"
echo ""

# Build pytest command
PYTEST_CMD="uv run pytest"

# Add verbosity
if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

# Add coverage
if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=src --cov-report=html:$OUTPUT_DIR/coverage_html --cov-report=term-missing --cov-report=json:$OUTPUT_DIR/coverage.json"
fi

# Add parallel execution
if [ "$PARALLEL" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -n auto"
fi

# Add profiling
if [ "$PROFILE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --profile"
fi

# Function to run specific test type
run_tests() {
    local test_type=$1
    local test_pattern=""
    local test_description=""
    
    case $test_type in
        "unit")
            test_pattern="tests/test_*.py -k 'not integration and not performance and not accuracy'"
            test_description="Unit Tests"
            ;;
        "integration")
            test_pattern="tests/test_*.py -k 'integration'"
            test_description="Integration Tests"
            ;;
        "performance")
            test_pattern="tests/test_*.py -k 'performance'"
            test_description="Performance Tests"
            ;;
        "accuracy")
            test_pattern="tests/test_*.py -k 'accuracy'"
            test_description="Accuracy Tests"
            ;;
        "reference")
            test_pattern="tests/test_reference_cleaner.py"
            test_description="Reference Removal Tests"
            ;;
        "ocr")
            test_pattern="tests/test_ocr_processor.py"
            test_description="OCR Processing Tests"
            ;;
        "llm")
            test_pattern="tests/test_llm_extractor.py"
            test_description="LLM Extraction Tests"
            ;;
        "api")
            test_pattern="tests/test_arxiv_api.py"
            test_description="ArXiv API Tests"
            ;;
        "all")
            test_pattern="tests/"
            test_description="All Tests"
            ;;
        *)
            print_error "Unknown test type: $test_type"
            echo "Available types: all, unit, integration, performance, accuracy, reference, ocr, llm, api"
            exit 1
            ;;
    esac
    
    print_info "Running $test_description..."
    echo "Command: $PYTEST_CMD $test_pattern"
    echo ""
    
    # Run the tests
    local start_time=$(date +%s)
    
    # Create test report file
    local report_file="$OUTPUT_DIR/test_report_${test_type}_${TIMESTAMP}.json"
    local junit_file="$OUTPUT_DIR/junit_${test_type}_${TIMESTAMP}.xml"
    
    # Add JUnit output
    local full_cmd="$PYTEST_CMD $test_pattern --junitxml=$junit_file --json-report --json-report-file=$report_file"
    
    if eval $full_cmd; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        print_status "$test_description completed successfully in ${duration}s"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        print_error "$test_description failed after ${duration}s"
        return 1
    fi
}

# Function to generate summary report
generate_summary() {
    local success=$1
    local test_type=$2
    
    echo ""
    echo "=============================================="
    echo "Test Summary"
    echo "=============================================="
    echo ""
    
    if [ $success -eq 0 ]; then
        print_status "All tests passed!"
    else
        print_error "Some tests failed!"
    fi
    
    echo ""
    print_info "Test Results Location: $OUTPUT_DIR"
    
    # List generated files
    if [ -d "$OUTPUT_DIR" ]; then
        echo ""
        print_info "Generated files:"
        ls -la "$OUTPUT_DIR"/ | grep -E "\.(json|xml|html)$" | while read -r line; do
            echo "  $line"
        done
    fi
    
    # Coverage summary
    if [ "$COVERAGE" = true ] && [ -f "$OUTPUT_DIR/coverage.json" ]; then
        echo ""
        print_info "Coverage Summary:"
        python3 -c "
import json
with open('$OUTPUT_DIR/coverage.json', 'r') as f:
    data = json.load(f)
    print(f'  Total Coverage: {data[\"totals\"][\"percent_covered\"]:.1f}%')
    print(f'  Lines Covered: {data[\"totals\"][\"covered_lines\"]}/{data[\"totals\"][\"num_statements\"]}')
    print(f'  Missing Lines: {data[\"totals\"][\"missing_lines\"]}')
"
    fi
    
    # Performance summary for performance tests
    if [ "$test_type" = "performance" ] || [ "$test_type" = "all" ]; then
        echo ""
        print_info "Performance Requirements Check:"
        echo "  Processing time: <60 seconds per paper"
        echo "  Reference removal: <3 seconds per paper"
        echo "  OCR accuracy: >95%"
        echo "  Reference removal accuracy: >99%"
        echo "  Item extraction accuracy: >90%"
    fi
    
    echo ""
    print_info "Timestamp: $TIMESTAMP"
    
    if [ $success -eq 0 ]; then
        echo ""
        print_status "Ready for development!"
    else
        echo ""
        print_warning "Please fix failing tests before continuing development"
    fi
}

# Function to run pre-test checks
run_pre_checks() {
    print_info "Running pre-test checks..."
    
    # Check virtual environment
    if [ ! -d ".venv" ]; then
        print_warning "Virtual environment not found. Run setup_dev_env.sh first."
    fi
    
    # Check dependencies
    if ! uv pip list | grep -q pytest; then
        print_error "pytest not installed. Run setup_dev_env.sh first."
        exit 1
    fi
    
    # Check source files
    if [ ! -f "src/arxiv_extractor.py" ]; then
        print_error "Source files not found. Check project structure."
        exit 1
    fi
    
    # Check test files
    if [ ! -d "tests" ]; then
        print_error "Tests directory not found."
        exit 1
    fi
    
    print_status "Pre-test checks passed"
}

# Function to cleanup after tests
cleanup() {
    print_info "Cleaning up temporary files..."
    
    # Remove Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    
    # Remove test artifacts
    rm -rf .pytest_cache 2>/dev/null || true
    rm -f profile.stats 2>/dev/null || true
    
    print_status "Cleanup completed"
}

# Main execution
main() {
    # Setup trap for cleanup
    trap cleanup EXIT
    
    # Run pre-checks
    run_pre_checks
    
    # Run tests based on type
    local exit_code=0
    
    case $TEST_TYPE in
        "all")
            print_info "Running all test types..."
            run_tests "unit" || exit_code=1
            run_tests "integration" || exit_code=1
            run_tests "performance" || exit_code=1
            run_tests "accuracy" || exit_code=1
            ;;
        *)
            run_tests "$TEST_TYPE" || exit_code=1
            ;;
    esac
    
    # Generate summary
    generate_summary $exit_code "$TEST_TYPE"
    
    # Exit with appropriate code
    exit $exit_code
}

# Run main function
main "$@"