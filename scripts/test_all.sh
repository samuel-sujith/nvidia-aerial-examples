#!/bin/bash

# NVIDIA Aerial Framework Examples - Test Runner Script
# Runs all available tests with comprehensive reporting

set -e

# Configuration
BUILD_DIR=${BUILD_DIR:-build}
TEST_OUTPUT_DIR=${TEST_OUTPUT_DIR:-test_results}
VERBOSE=${VERBOSE:-false}
PARALLEL_TESTS=${PARALLEL_TESTS:-true}
JOBS=${JOBS:-$(nproc)}

# Test categories
UNIT_TESTS=true
INTEGRATION_TESTS=true
PERFORMANCE_TESTS=true
GPU_TESTS=true

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run NVIDIA Aerial Framework Example Tests

OPTIONS:
    -h, --help              Show this help message
    -v, --verbose           Verbose test output
    -s, --serial            Run tests serially (not in parallel)
    -j, --jobs NUM          Number of parallel test jobs (default: $(nproc))
    --unit-only             Run only unit tests
    --integration-only      Run only integration tests  
    --performance-only      Run only performance tests
    --no-gpu                Skip GPU-dependent tests
    --build-first           Build before running tests
    --clean-results         Clean previous test results

ENVIRONMENT VARIABLES:
    BUILD_DIR               Build directory (default: build)
    TEST_OUTPUT_DIR         Test output directory (default: test_results)
    VERBOSE                 Enable verbose output (true/false)
    PARALLEL_TESTS          Enable parallel test execution (true/false)
    JOBS                    Number of parallel jobs

EXAMPLES:
    $0                      # Run all tests
    $0 --verbose            # Run with verbose output
    $0 --unit-only -v       # Run only unit tests with verbose output
    $0 --build-first        # Build and then test
    $0 --performance-only   # Run only performance benchmarks
EOF
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
BUILD_FIRST=false
CLEAN_RESULTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_usage
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -s|--serial)
            PARALLEL_TESTS=false
            shift
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        --unit-only)
            INTEGRATION_TESTS=false
            PERFORMANCE_TESTS=false
            shift
            ;;
        --integration-only)
            UNIT_TESTS=false
            PERFORMANCE_TESTS=false
            shift
            ;;
        --performance-only)
            UNIT_TESTS=false
            INTEGRATION_TESTS=false
            shift
            ;;
        --no-gpu)
            GPU_TESTS=false
            shift
            ;;
        --build-first)
            BUILD_FIRST=true
            shift
            ;;
        --clean-results)
            CLEAN_RESULTS=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validate environment
log_info "NVIDIA Aerial Examples Test Runner"
log_info "Build directory: $BUILD_DIR"
log_info "Test output directory: $TEST_OUTPUT_DIR"
log_info "Parallel jobs: $JOBS"

# Check if build directory exists
if [[ ! -d "$BUILD_DIR" ]]; then
    log_error "Build directory $BUILD_DIR does not exist"
    if [[ "$BUILD_FIRST" == true ]]; then
        log_info "Building first..."
        ./scripts/build.sh
    else
        log_error "Run with --build-first to build automatically"
        exit 1
    fi
fi

# Build if requested
if [[ "$BUILD_FIRST" == true ]]; then
    log_info "Building examples..."
    ./scripts/build.sh
fi

# Clean results if requested
if [[ "$CLEAN_RESULTS" == true ]]; then
    log_info "Cleaning previous test results..."
    rm -rf "$TEST_OUTPUT_DIR"
fi

# Create test output directory
mkdir -p "$TEST_OUTPUT_DIR"

# Check GPU availability
check_gpu() {
    if [[ "$GPU_TESTS" == true ]]; then
        if ! command -v nvidia-smi &> /dev/null; then
            log_warning "nvidia-smi not found, skipping GPU tests"
            return 1
        fi
        
        if ! nvidia-smi &> /dev/null; then
            log_warning "No NVIDIA GPU detected, skipping GPU tests"
            return 1
        fi
        
        log_info "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
        return 0
    fi
    return 1
}

# Test execution function
run_test_category() {
    local category=$1
    local test_pattern=$2
    local description=$3
    
    log_info "Running $description..."
    
    local test_args=()
    if [[ "$VERBOSE" == true ]]; then
        test_args+=(--verbose)
    fi
    
    if [[ "$PARALLEL_TESTS" == true ]]; then
        test_args+=(-j "$JOBS")
    fi
    
    test_args+=(--output-on-failure)
    test_args+=(--test-dir "$BUILD_DIR")
    
    if [[ -n "$test_pattern" ]]; then
        test_args+=(-R "$test_pattern")
    fi
    
    local output_file="$TEST_OUTPUT_DIR/${category}_tests.xml"
    
    # Run tests with XML output for detailed reporting
    if ctest "${test_args[@]}" --output-junit "$output_file"; then
        log_success "$description completed successfully"
        return 0
    else
        log_error "$description failed"
        return 1
    fi
}

# Change to build directory
cd "$BUILD_DIR"

# Test execution tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_CATEGORIES=()

# Run unit tests
if [[ "$UNIT_TESTS" == true ]]; then
    if run_test_category "unit" ".*unit.*" "Unit Tests"; then
        ((PASSED_TESTS++))
    else
        FAILED_CATEGORIES+=("Unit Tests")
    fi
    ((TOTAL_TESTS++))
fi

# Run integration tests
if [[ "$INTEGRATION_TESTS" == true ]]; then
    if run_test_category "integration" ".*integration.*" "Integration Tests"; then
        ((PASSED_TESTS++))
    else
        FAILED_CATEGORIES+=("Integration Tests")
    fi
    ((TOTAL_TESTS++))
fi

# Run GPU tests (if available)
GPU_AVAILABLE=false
if check_gpu; then
    GPU_AVAILABLE=true
    
    if run_test_category "gpu" ".*gpu.*" "GPU Tests"; then
        ((PASSED_TESTS++))
    else
        FAILED_CATEGORIES+=("GPU Tests")
    fi
    ((TOTAL_TESTS++))
fi

# Run performance tests
if [[ "$PERFORMANCE_TESTS" == true ]]; then
    log_info "Running Performance Benchmarks..."
    
    # Channel estimation benchmark
    if [[ -f "./channel_estimation_example" ]]; then
        log_info "Running channel estimation benchmark..."
        ./channel_estimation_example --benchmark > "$TEST_OUTPUT_DIR/channel_estimation_benchmark.txt" 2>&1
        if [[ $? -eq 0 ]]; then
            log_success "Channel estimation benchmark completed"
        else
            log_warning "Channel estimation benchmark failed"
        fi
    fi
    
    # Modulation benchmark
    if [[ -f "./modulation_mapping/modulation_example" ]]; then
        log_info "Running modulation benchmark..."
        ./modulation_mapping/modulation_example --benchmark > "$TEST_OUTPUT_DIR/modulation_benchmark.txt" 2>&1
        if [[ $? -eq 0 ]]; then
            log_success "Modulation benchmark completed"
        else
            log_warning "Modulation benchmark failed"
        fi
    fi
    
    # Comprehensive modulation benchmark
    if [[ -f "./modulation_mapping/modulation_mapping_example" ]]; then
        log_info "Running comprehensive modulation benchmark..."
        ./modulation_mapping/modulation_mapping_example --benchmark > "$TEST_OUTPUT_DIR/modulation_comprehensive_benchmark.txt" 2>&1
        if [[ $? -eq 0 ]]; then
            log_success "Comprehensive modulation benchmark completed"
        else
            log_warning "Comprehensive modulation benchmark failed"
        fi
    fi
    
    # FFT benchmark
    if [[ -f "./fft_processing/fft_example" ]]; then
        log_info "Running FFT benchmark..."
        ./fft_processing/fft_example --benchmark > "$TEST_OUTPUT_DIR/fft_benchmark.txt" 2>&1
        if [[ $? -eq 0 ]]; then
            log_success "FFT benchmark completed"
        else
            log_warning "FFT benchmark failed"
        fi
    fi
    
    # Comprehensive FFT benchmark
    if [[ -f "./fft_processing/fft_processing_example" ]]; then
        log_info "Running comprehensive FFT benchmark..."
        ./fft_processing/fft_processing_example --benchmark > "$TEST_OUTPUT_DIR/fft_comprehensive_benchmark.txt" 2>&1
        if [[ $? -eq 0 ]]; then
            log_success "Comprehensive FFT benchmark completed"
        else
            log_warning "Comprehensive FFT benchmark failed"
        fi
    fi
    
    # MIMO benchmark
    if [[ -f "./mimo_detection/mimo_example" ]]; then
        log_info "Running MIMO benchmark..."
        ./mimo_detection/mimo_example --benchmark > "$TEST_OUTPUT_DIR/mimo_benchmark.txt" 2>&1
        if [[ $? -eq 0 ]]; then
            log_success "MIMO benchmark completed"
        else
            log_warning "MIMO benchmark failed"
        fi
    fi
    
    # Comprehensive MIMO benchmark
    if [[ -f "./mimo_detection/mimo_detection_example" ]]; then
        log_info "Running comprehensive MIMO benchmark..."
        ./mimo_detection/mimo_detection_example --benchmark > "$TEST_OUTPUT_DIR/mimo_comprehensive_benchmark.txt" 2>&1
        if [[ $? -eq 0 ]]; then
            log_success "Comprehensive MIMO benchmark completed"
        else
            log_warning "Comprehensive MIMO benchmark failed"
        fi
    fi
    
    if run_test_category "performance" ".*perf.*|.*benchmark.*" "Performance Tests"; then
        ((PASSED_TESTS++))
    else
        FAILED_CATEGORIES+=("Performance Tests")
    fi
    ((TOTAL_TESTS++))
fi

# Generate summary report
cd ..

echo ""
echo "==============================================="
echo "             TEST SUMMARY REPORT"
echo "==============================================="
echo "Total test categories: $TOTAL_TESTS"
echo "Passed categories: $PASSED_TESTS"
echo "Failed categories: $((TOTAL_TESTS - PASSED_TESTS))"

if [[ "$GPU_AVAILABLE" == true ]]; then
    log_success "GPU tests were executed"
else
    log_warning "GPU tests were skipped (no GPU available)"
fi

if [[ ${#FAILED_CATEGORIES[@]} -gt 0 ]]; then
    echo ""
    log_error "Failed test categories:"
    for category in "${FAILED_CATEGORIES[@]}"; do
        echo "  - $category"
    done
    echo ""
fi

# Generate detailed HTML report if available
if command -v python3 &> /dev/null; then
    log_info "Generating detailed HTML report..."
    
    cat > "$TEST_OUTPUT_DIR/generate_report.py" << 'EOF'
import os
import glob
import xml.etree.ElementTree as ET
from datetime import datetime

def parse_junit_xml(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        tests = []
        for testcase in root.findall('.//testcase'):
            test = {
                'name': testcase.get('name', 'Unknown'),
                'classname': testcase.get('classname', 'Unknown'),
                'time': float(testcase.get('time', 0)),
                'status': 'PASSED'
            }
            
            if testcase.find('failure') is not None:
                test['status'] = 'FAILED'
                test['failure'] = testcase.find('failure').text
            elif testcase.find('error') is not None:
                test['status'] = 'ERROR'
                test['error'] = testcase.find('error').text
            elif testcase.find('skipped') is not None:
                test['status'] = 'SKIPPED'
                
            tests.append(test)
            
        return tests
    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")
        return []

def generate_html_report():
    xml_files = glob.glob("*.xml")
    all_tests = []
    
    for xml_file in xml_files:
        tests = parse_junit_xml(xml_file)
        category = xml_file.replace('_tests.xml', '')
        for test in tests:
            test['category'] = category
            all_tests.append(test)
    
    # Generate HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>NVIDIA Aerial Examples Test Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background: #333; color: white; padding: 20px; border-radius: 5px; }}
            .summary {{ margin: 20px 0; padding: 20px; background: #f5f5f5; border-radius: 5px; }}
            .test-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            .test-table th, .test-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .test-table th {{ background-color: #f2f2f2; }}
            .passed {{ background-color: #d4edda; }}
            .failed {{ background-color: #f8d7da; }}
            .skipped {{ background-color: #fff3cd; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>NVIDIA Aerial Examples Test Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="summary">
            <h2>Summary</h2>
            <p><strong>Total Tests:</strong> {len(all_tests)}</p>
            <p><strong>Passed:</strong> {len([t for t in all_tests if t['status'] == 'PASSED'])}</p>
            <p><strong>Failed:</strong> {len([t for t in all_tests if t['status'] == 'FAILED'])}</p>
            <p><strong>Skipped:</strong> {len([t for t in all_tests if t['status'] == 'SKIPPED'])}</p>
        </div>
        
        <h2>Test Details</h2>
        <table class="test-table">
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Test Name</th>
                    <th>Status</th>
                    <th>Time (s)</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for test in sorted(all_tests, key=lambda x: (x['category'], x['name'])):
        status_class = test['status'].lower()
        html += f"""
                <tr class="{status_class}">
                    <td>{test['category']}</td>
                    <td>{test['name']}</td>
                    <td>{test['status']}</td>
                    <td>{test['time']:.3f}</td>
                </tr>
        """
    
    html += """
            </tbody>
        </table>
    </body>
    </html>
    """
    
    with open('test_report.html', 'w') as f:
        f.write(html)
    
    print("HTML report generated: test_report.html")

if __name__ == "__main__":
    os.chdir('test_results')
    generate_html_report()
EOF
    
    python3 "$TEST_OUTPUT_DIR/generate_report.py"
    log_success "HTML report generated: $TEST_OUTPUT_DIR/test_report.html"
fi

echo ""
if [[ $PASSED_TESTS -eq $TOTAL_TESTS ]]; then
    log_success "All test categories passed!"
    exit 0
else
    log_error "Some test categories failed"
    exit 1
fi