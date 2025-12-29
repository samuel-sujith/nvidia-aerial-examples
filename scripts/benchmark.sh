#!/bin/bash

# NVIDIA Aerial Framework Examples - Benchmark Runner
# Comprehensive performance benchmarking and profiling script

set -e

# Configuration
BUILD_DIR=${BUILD_DIR:-build}
BENCHMARK_OUTPUT_DIR=${BENCHMARK_OUTPUT_DIR:-benchmark_results}
PROFILE_MODE=${PROFILE_MODE:-false}
WARMUP_ITERATIONS=${WARMUP_ITERATIONS:-10}
BENCHMARK_ITERATIONS=${BENCHMARK_ITERATIONS:-100}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run NVIDIA Aerial Framework Performance Benchmarks

OPTIONS:
    -h, --help                  Show this help message
    -p, --profile               Enable detailed profiling with nvprof/nsys
    -w, --warmup NUM            Number of warmup iterations (default: 10)
    -i, --iterations NUM        Number of benchmark iterations (default: 100)
    --channel-est               Run only channel estimation benchmarks
    --modulation                Run only modulation benchmarks  
    --fft                       Run only FFT benchmarks
    --mimo                      Run only MIMO benchmarks
    --all                       Run all benchmarks (default)
    --small                     Run small problem sizes only
    --large                     Run large problem sizes only
    --compare-implementations   Compare multiple algorithm implementations
    --save-csv                  Save detailed results to CSV files
    --generate-plots            Generate performance plots (requires Python)

ENVIRONMENT VARIABLES:
    BUILD_DIR                   Build directory (default: build)
    BENCHMARK_OUTPUT_DIR        Output directory (default: benchmark_results)
    PROFILE_MODE                Enable profiling (true/false)
    WARMUP_ITERATIONS           Warmup iterations
    BENCHMARK_ITERATIONS        Benchmark iterations

EXAMPLES:
    $0                          # Run all benchmarks
    $0 --profile               # Run with profiling enabled
    $0 --channel-est --large   # Run only channel estimation, large sizes
    $0 --compare-implementations --save-csv  # Compare algorithms, save CSV
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
RUN_CHANNEL_EST=false
RUN_MODULATION=false
RUN_FFT=false
RUN_MIMO=false
RUN_ALL=true
PROBLEM_SIZE="all"
COMPARE_IMPLEMENTATIONS=false
SAVE_CSV=false
GENERATE_PLOTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_usage
            exit 0
            ;;
        -p|--profile)
            PROFILE_MODE=true
            shift
            ;;
        -w|--warmup)
            WARMUP_ITERATIONS="$2"
            shift 2
            ;;
        -i|--iterations)
            BENCHMARK_ITERATIONS="$2"
            shift 2
            ;;
        --channel-est)
            RUN_CHANNEL_EST=true
            RUN_ALL=false
            shift
            ;;
        --modulation)
            RUN_MODULATION=true
            RUN_ALL=false
            shift
            ;;
        --fft)
            RUN_FFT=true
            RUN_ALL=false
            shift
            ;;
        --mimo)
            RUN_MIMO=true
            RUN_ALL=false
            shift
            ;;
        --all)
            RUN_ALL=true
            shift
            ;;
        --small)
            PROBLEM_SIZE="small"
            shift
            ;;
        --large)
            PROBLEM_SIZE="large"
            shift
            ;;
        --compare-implementations)
            COMPARE_IMPLEMENTATIONS=true
            shift
            ;;
        --save-csv)
            SAVE_CSV=true
            shift
            ;;
        --generate-plots)
            GENERATE_PLOTS=true
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
log_info "NVIDIA Aerial Framework Benchmark Runner"
log_info "Build directory: $BUILD_DIR"
log_info "Output directory: $BENCHMARK_OUTPUT_DIR"
log_info "Warmup iterations: $WARMUP_ITERATIONS"
log_info "Benchmark iterations: $BENCHMARK_ITERATIONS"
log_info "Profile mode: $PROFILE_MODE"

# Check if build directory exists
if [[ ! -d "$BUILD_DIR" ]]; then
    log_error "Build directory $BUILD_DIR does not exist"
    log_error "Run './scripts/build.sh' first"
    exit 1
fi

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found, GPU benchmarks require NVIDIA GPU"
    exit 1
fi

if ! nvidia-smi &> /dev/null; then
    log_error "No NVIDIA GPU detected"
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | head -1)
log_info "GPU: $GPU_INFO"

# Create output directory
mkdir -p "$BENCHMARK_OUTPUT_DIR"
cd "$BUILD_DIR"

# Profiling setup
setup_profiling() {
    if [[ "$PROFILE_MODE" == true ]]; then
        # Check for profiling tools
        if command -v nsys &> /dev/null; then
            PROFILER="nsys"
            PROFILE_CMD="nsys profile --output=$BENCHMARK_OUTPUT_DIR/profile_%p.nsys-rep"
            log_info "Using Nsight Systems for profiling"
        elif command -v nvprof &> /dev/null; then
            PROFILER="nvprof"
            PROFILE_CMD="nvprof --output-profile $BENCHMARK_OUTPUT_DIR/profile_%p.nvvp"
            log_warning "Using legacy nvprof (consider upgrading to Nsight Systems)"
        else
            log_warning "No profiler found, disabling profiling"
            PROFILE_MODE=false
        fi
    fi
}

# Run single benchmark
run_benchmark() {
    local executable=$1
    local description=$2
    local args=$3
    local output_prefix=$4
    
    if [[ ! -f "./$executable" ]]; then
        log_warning "$executable not found, skipping $description"
        return 1
    fi
    
    log_info "Running $description..."
    
    local cmd="./$executable $args --warmup=$WARMUP_ITERATIONS --iterations=$BENCHMARK_ITERATIONS"
    local output_file="$BENCHMARK_OUTPUT_DIR/${output_prefix}_benchmark.txt"
    
    if [[ "$PROFILE_MODE" == true ]]; then
        $PROFILE_CMD $cmd > "$output_file" 2>&1
    else
        $cmd > "$output_file" 2>&1
    fi
    
    if [[ $? -eq 0 ]]; then
        log_success "$description completed"
        
        # Extract key metrics
        if grep -q "Throughput:" "$output_file"; then
            local throughput=$(grep "Throughput:" "$output_file" | tail -1 | awk '{print $2}')
            log_info "  Throughput: $throughput ops/sec"
        fi
        
        if grep -q "Latency:" "$output_file"; then
            local latency=$(grep "Latency:" "$output_file" | tail -1 | awk '{print $2}')
            log_info "  Latency: $latency μs"
        fi
        
        return 0
    else
        log_error "$description failed"
        return 1
    fi
}

# Channel estimation benchmarks
run_channel_estimation_benchmarks() {
    log_info "=== Channel Estimation Benchmarks ==="
    
    local sizes=(
        "64 4"      # Small: 64 subcarriers, 4 antennas
        "1024 4"    # Medium: 1024 subcarriers, 4 antennas  
        "4096 8"    # Large: 4096 subcarriers, 8 antennas
        "8192 16"   # Extra large: 8192 subcarriers, 16 antennas
    )
    
    for size in "${sizes[@]}"; do
        local subcarriers=$(echo $size | awk '{print $1}')
        local antennas=$(echo $size | awk '{print $2}')
        
        # Skip based on problem size filter
        if [[ "$PROBLEM_SIZE" == "small" && $subcarriers -gt 1024 ]]; then
            continue
        elif [[ "$PROBLEM_SIZE" == "large" && $subcarriers -lt 1024 ]]; then
            continue
        fi
        
        local args="--subcarriers=$subcarriers --antennas=$antennas"
        local output_prefix="channel_est_${subcarriers}_${antennas}"
        
        run_benchmark "channel_estimation_example" \
                     "Channel Estimation ($subcarriers subcarriers, $antennas antennas)" \
                     "$args" \
                     "$output_prefix"
        
        # Test different algorithms if comparison mode enabled
        if [[ "$COMPARE_IMPLEMENTATIONS" == true ]]; then
            for algo in ls mmse linear_interpolation; do
                local algo_args="$args --algorithm=$algo"
                local algo_output_prefix="${output_prefix}_${algo}"
                
                run_benchmark "channel_estimation_example" \
                             "Channel Estimation - $algo ($subcarriers subcarriers)" \
                             "$algo_args" \
                             "$algo_output_prefix"
            done
        fi
    done
}

# Modulation benchmarks
run_modulation_benchmarks() {
    log_info "=== Modulation Benchmarks ==="
    
    local configs=(
        "qpsk 10000"      # QPSK, 10K symbols
        "16qam 10000"     # 16-QAM, 10K symbols
        "64qam 10000"     # 64-QAM, 10K symbols
        "256qam 10000"    # 256-QAM, 10K symbols
        "qpsk 100000"     # QPSK, 100K symbols (large)
        "16qam 100000"    # 16-QAM, 100K symbols (large)
    )
    
    for config in "${configs[@]}"; do
        local modulation=$(echo $config | awk '{print $1}')
        local symbols=$(echo $config | awk '{print $2}')
        
        # Skip based on problem size filter
        if [[ "$PROBLEM_SIZE" == "small" && $symbols -gt 50000 ]]; then
            continue
        elif [[ "$PROBLEM_SIZE" == "large" && $symbols -lt 50000 ]]; then
            continue
        fi
        
        local args="--modulation=$modulation --symbols=$symbols"
        local output_prefix="modulation_${modulation}_${symbols}"
        
        run_benchmark "modulation_mapping/modulation_example" \
                     "Modulation ($modulation, $symbols symbols)" \
                     "$args" \
                     "$output_prefix"
        
        # Also run comprehensive example if available
        if [[ -f "./modulation_mapping/modulation_mapping_example" ]]; then
            run_benchmark "modulation_mapping/modulation_mapping_example" \
                         "Modulation Comprehensive ($modulation, $symbols symbols)" \
                         "$args" \
                         "${output_prefix}_comprehensive"
        fi
    done
}

# FFT benchmarks
run_fft_benchmarks() {
    log_info "=== FFT Benchmarks ==="
    
    local sizes=(128 256 512 1024 2048 4096 8192 16384)
    
    for size in "${sizes[@]}"; do
        # Skip based on problem size filter
        if [[ "$PROBLEM_SIZE" == "small" && $size -gt 2048 ]]; then
            continue
        elif [[ "$PROBLEM_SIZE" == "large" && $size -lt 2048 ]]; then
            continue
        fi
        
        local args="--fft-size=$size --batch-size=64"
        local output_prefix="fft_${size}"
        
        run_benchmark "fft_processing/fft_example" \
                     "FFT (size $size, batch 64)" \
                     "$args" \
                     "$output_prefix"
        
        # Also run comprehensive example if available
        if [[ -f "./fft_processing/fft_processing_example" ]]; then
            run_benchmark "fft_processing/fft_processing_example" \
                         "FFT Comprehensive (size $size, batch 64)" \
                         "$args" \
                         "${output_prefix}_comprehensive"
        fi
        
        # Test different batch sizes for larger FFTs
        if [[ $size -ge 1024 && "$COMPARE_IMPLEMENTATIONS" == true ]]; then
            for batch in 32 128 256; do
                local batch_args="--fft-size=$size --batch-size=$batch"
                local batch_output_prefix="fft_${size}_batch${batch}"
                
                run_benchmark "fft_processing/fft_example" \
                             "FFT (size $size, batch $batch)" \
                             "$batch_args" \
                             "$batch_output_prefix"
            done
        fi
    done
}

# MIMO benchmarks
run_mimo_benchmarks() {
    log_info "=== MIMO Detection Benchmarks ==="
    
    local configs=(
        "2 2 qpsk"      # 2x2 MIMO, QPSK
        "4 4 qpsk"      # 4x4 MIMO, QPSK
        "4 4 16qam"     # 4x4 MIMO, 16-QAM
        "8 8 qpsk"      # 8x8 MIMO, QPSK (large)
        "8 8 16qam"     # 8x8 MIMO, 16-QAM (large)
    )
    
    for config in "${configs[@]}"; do
        local tx_antennas=$(echo $config | awk '{print $1}')
        local rx_antennas=$(echo $config | awk '{print $2}')
        local modulation=$(echo $config | awk '{print $3}')
        
        # Skip based on problem size filter
        if [[ "$PROBLEM_SIZE" == "small" && $tx_antennas -gt 4 ]]; then
            continue
        elif [[ "$PROBLEM_SIZE" == "large" && $tx_antennas -lt 8 ]]; then
            continue
        fi
        
        local args="--tx-antennas=$tx_antennas --rx-antennas=$rx_antennas --modulation=$modulation"
        local output_prefix="mimo_${tx_antennas}x${rx_antennas}_${modulation}"
        
        run_benchmark "mimo_detection/mimo_example" \
                     "MIMO Detection (${tx_antennas}x${rx_antennas}, $modulation)" \
                     "$args" \
                     "$output_prefix"
        
        # Also run comprehensive example if available
        if [[ -f "./mimo_detection/mimo_detection_example" ]]; then
            run_benchmark "mimo_detection/mimo_detection_example" \
                         "MIMO Detection Comprehensive (${tx_antennas}x${rx_antennas}, $modulation)" \
                         "$args" \
                         "${output_prefix}_comprehensive"
        fi
        
        # Test different detection algorithms if comparison mode enabled
        if [[ "$COMPARE_IMPLEMENTATIONS" == true ]]; then
            for detector in zf mmse ml_approx; do
                local detector_args="$args --detector=$detector"
                local detector_output_prefix="${output_prefix}_${detector}"
                
                run_benchmark "mimo_detection/mimo_example" \
                             "MIMO Detection - $detector (${tx_antennas}x${rx_antennas})" \
                             "$detector_args" \
                             "$detector_output_prefix"
            done
        fi
    done
}

# Setup profiling if enabled
setup_profiling

# Determine which benchmarks to run
if [[ "$RUN_ALL" == true ]]; then
    RUN_CHANNEL_EST=true
    RUN_MODULATION=true
    RUN_FFT=true
    RUN_MIMO=true
fi

# Run benchmarks
BENCHMARK_START_TIME=$(date +%s)

if [[ "$RUN_CHANNEL_EST" == true ]]; then
    run_channel_estimation_benchmarks
fi

if [[ "$RUN_MODULATION" == true ]]; then
    run_modulation_benchmarks
fi

if [[ "$RUN_FFT" == true ]]; then
    run_fft_benchmarks
fi

if [[ "$RUN_MIMO" == true ]]; then
    run_mimo_benchmarks
fi

BENCHMARK_END_TIME=$(date +%s)
TOTAL_TIME=$((BENCHMARK_END_TIME - BENCHMARK_START_TIME))

# Generate summary report
cd ..
log_info "Generating benchmark summary..."

cat > "$BENCHMARK_OUTPUT_DIR/benchmark_summary.txt" << EOF
NVIDIA Aerial Framework Examples - Benchmark Summary
====================================================

Benchmark Configuration:
- GPU: $GPU_INFO
- Warmup iterations: $WARMUP_ITERATIONS  
- Benchmark iterations: $BENCHMARK_ITERATIONS
- Profile mode: $PROFILE_MODE
- Problem size filter: $PROBLEM_SIZE
- Total execution time: ${TOTAL_TIME}s

Results:
EOF

# Aggregate results from all benchmark files
for result_file in "$BENCHMARK_OUTPUT_DIR"/*_benchmark.txt; do
    if [[ -f "$result_file" ]]; then
        filename=$(basename "$result_file" _benchmark.txt)
        echo "" >> "$BENCHMARK_OUTPUT_DIR/benchmark_summary.txt"
        echo "=== $filename ===" >> "$BENCHMARK_OUTPUT_DIR/benchmark_summary.txt"
        
        # Extract key metrics
        if grep -q "Throughput:" "$result_file"; then
            grep "Throughput:" "$result_file" | tail -1 >> "$BENCHMARK_OUTPUT_DIR/benchmark_summary.txt"
        fi
        if grep -q "Latency:" "$result_file"; then
            grep "Latency:" "$result_file" | tail -1 >> "$BENCHMARK_OUTPUT_DIR/benchmark_summary.txt"
        fi
        if grep -q "Bandwidth:" "$result_file"; then
            grep "Bandwidth:" "$result_file" | tail -1 >> "$BENCHMARK_OUTPUT_DIR/benchmark_summary.txt"
        fi
    fi
done

log_success "Benchmark summary saved to: $BENCHMARK_OUTPUT_DIR/benchmark_summary.txt"

# Generate CSV files if requested
if [[ "$SAVE_CSV" == true ]]; then
    log_info "Generating CSV files..."
    
    # This would require parsing the detailed output files
    # Implementation depends on the actual output format of the examples
    python3 << EOF
import os
import re
import csv
import glob

def parse_benchmark_file(filename):
    results = []
    with open(filename, 'r') as f:
        content = f.read()
        
        # Extract metrics using regex patterns
        throughput_match = re.search(r'Throughput:\s+([\d.]+)', content)
        latency_match = re.search(r'Latency:\s+([\d.]+)', content)
        
        if throughput_match or latency_match:
            result = {
                'benchmark': os.path.basename(filename).replace('_benchmark.txt', ''),
                'throughput': float(throughput_match.group(1)) if throughput_match else None,
                'latency': float(latency_match.group(1)) if latency_match else None
            }
            results.append(result)
    
    return results

# Parse all benchmark files
os.chdir('$BENCHMARK_OUTPUT_DIR')
all_results = []

for filename in glob.glob('*_benchmark.txt'):
    results = parse_benchmark_file(filename)
    all_results.extend(results)

# Write CSV
if all_results:
    with open('benchmark_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['benchmark', 'throughput', 'latency']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)
    
    print("CSV file generated: benchmark_results.csv")
EOF

    log_success "CSV files generated in: $BENCHMARK_OUTPUT_DIR/"
fi

# Generate plots if requested
if [[ "$GENERATE_PLOTS" == true ]]; then
    if command -v python3 &> /dev/null; then
        log_info "Generating performance plots..."
        
        # This would require matplotlib and implementation of plotting logic
        python3 << 'EOF'
try:
    import matplotlib.pyplot as plt
    import csv
    import numpy as np
    
    # Read benchmark results
    with open('benchmark_results.csv', 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    # Create throughput plot
    benchmarks = [d['benchmark'] for d in data if d['throughput']]
    throughputs = [float(d['throughput']) for d in data if d['throughput']]
    
    if benchmarks and throughputs:
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(benchmarks)), throughputs)
        plt.xlabel('Benchmark')
        plt.ylabel('Throughput (ops/sec)')
        plt.title('NVIDIA Aerial Framework - Throughput Comparison')
        plt.xticks(range(len(benchmarks)), benchmarks, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('throughput_comparison.png', dpi=300, bbox_inches='tight')
        print("Throughput plot saved: throughput_comparison.png")
    
except ImportError:
    print("matplotlib not available, skipping plot generation")
except Exception as e:
    print(f"Error generating plots: {e}")
EOF
    else
        log_warning "Python3 not available, skipping plot generation"
    fi
fi

echo ""
echo "==============================================="
echo "         BENCHMARK EXECUTION COMPLETE"
echo "==============================================="
log_success "All benchmarks completed in ${TOTAL_TIME} seconds"
log_info "Results saved in: $BENCHMARK_OUTPUT_DIR/"

if [[ "$PROFILE_MODE" == true ]]; then
    log_info "Profiling data available in: $BENCHMARK_OUTPUT_DIR/"
    if [[ "$PROFILER" == "nsys" ]]; then
        log_info "View with: nsys-ui $BENCHMARK_OUTPUT_DIR/*.nsys-rep"
    elif [[ "$PROFILER" == "nvprof" ]]; then
        log_info "View with: nvvp $BENCHMARK_OUTPUT_DIR/*.nvvp"
    fi
fi

echo ""
log_info "View summary: cat $BENCHMARK_OUTPUT_DIR/benchmark_summary.txt"