#!/bin/bash

# CUDA Nexus Benchmark Runner Script

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           CUDA NEXUS COMPREHENSIVE BENCHMARK SUITE                 ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if benchmarks are built
if [ ! -d "build" ]; then
    echo -e "${RED}Error: build directory not found${NC}"
    echo "Please build the project first:"
    echo "  mkdir build && cd build"
    echo "  cmake .. -DBUILD_BENCHMARKS=ON"
    echo "  make"
    exit 1
fi

cd build

if [ ! -f "benchmarks/benchmark_gemm" ]; then
    echo -e "${RED}Error: Benchmarks not built${NC}"
    echo "Please rebuild with: cmake .. -DBUILD_BENCHMARKS=ON && make"
    exit 1
fi

# Parse arguments
RUN_ALL=true
RUN_GEMM=false
RUN_ATTENTION=false
RUN_REDUCTION=false
OUTPUT_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gemm)
            RUN_ALL=false
            RUN_GEMM=true
            shift
            ;;
        --attention)
            RUN_ALL=false
            RUN_ATTENTION=true
            shift
            ;;
        --reduction)
            RUN_ALL=false
            RUN_REDUCTION=true
            shift
            ;;
        --output|-o)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gemm         Run only GEMM benchmarks"
            echo "  --attention    Run only Attention benchmarks"
            echo "  --reduction    Run only Reduction benchmarks"
            echo "  --output FILE  Save output to file"
            echo "  --help         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Run all benchmarks"
            echo "  $0 --gemm             # Run only GEMM"
            echo "  $0 --output results.txt  # Save to file"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

if [ "$RUN_ALL" = true ]; then
    RUN_GEMM=true
    RUN_ATTENTION=true
    RUN_REDUCTION=true
fi

# Function to run benchmark and capture output
run_benchmark() {
    local name=$1
    local executable=$2
    
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Running: $name${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    if [ -n "$OUTPUT_FILE" ]; then
        echo "Running: $name" >> "$OUTPUT_FILE"
        echo "Time: $(date)" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        $executable 2>&1 | tee -a "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        echo "----------------------------------------" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
    else
        $executable
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $name completed successfully${NC}"
    else
        echo -e "${RED}✗ $name failed${NC}"
    fi
}

# Initialize output file
if [ -n "$OUTPUT_FILE" ]; then
    echo "CUDA Nexus Benchmark Results" > "$OUTPUT_FILE"
    echo "Generated: $(date)" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "System Information:" >> "$OUTPUT_FILE"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader >> "$OUTPUT_FILE" 2>/dev/null || echo "nvidia-smi not available" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "========================================" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
fi

# Run benchmarks
if [ "$RUN_GEMM" = true ]; then
    run_benchmark "GEMM Benchmark" "./benchmarks/benchmark_gemm"
fi

if [ "$RUN_ATTENTION" = true ]; then
    run_benchmark "Attention Benchmark" "./benchmarks/benchmark_attention"
fi

if [ "$RUN_REDUCTION" = true ]; then
    run_benchmark "Reduction Benchmark" "./benchmarks/benchmark_reduction"
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    ALL BENCHMARKS COMPLETED                        ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

if [ -n "$OUTPUT_FILE" ]; then
    echo -e "${YELLOW}Results saved to: $OUTPUT_FILE${NC}"
fi

# Generate summary
echo "Summary:"
echo "--------"
if [ "$RUN_GEMM" = true ]; then
    echo "✓ GEMM benchmarks completed"
fi
if [ "$RUN_ATTENTION" = true ]; then
    echo "✓ Attention benchmarks completed"
fi
if [ "$RUN_REDUCTION" = true ]; then
    echo "✓ Reduction benchmarks completed"
fi

cd ..
