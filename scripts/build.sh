#!/bin/bash

# CUDA Nexus Build Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}CUDA Nexus Build Script${NC}"
echo "================================"

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}Error: CUDA toolkit not found!${NC}"
    echo "Please install CUDA toolkit and ensure nvcc is in your PATH"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
echo -e "CUDA Version: ${GREEN}${CUDA_VERSION}${NC}"

# Detect GPU architecture
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    echo -e "Detected GPU: ${GREEN}${GPU_NAME}${NC}"
    
    # Set CUDA architectures based on GPU
    if [[ $GPU_NAME == *"4090"* ]] || [[ $GPU_NAME == *"4080"* ]]; then
        CUDA_ARCH="89"
    elif [[ $GPU_NAME == *"3090"* ]] || [[ $GPU_NAME == *"3080"* ]]; then
        CUDA_ARCH="86"
    elif [[ $GPU_NAME == *"A100"* ]]; then
        CUDA_ARCH="80"
    elif [[ $GPU_NAME == *"H100"* ]]; then
        CUDA_ARCH="90"
    else
        CUDA_ARCH="80;86;89"  # Build for multiple architectures
    fi
else
    echo -e "${YELLOW}Warning: nvidia-smi not found, building for multiple architectures${NC}"
    CUDA_ARCH="80;86;89"
fi

echo "Target CUDA Architecture(s): $CUDA_ARCH"

# Parse arguments
BUILD_TYPE="Release"
BUILD_TESTS="ON"
BUILD_EXAMPLES="ON"
BUILD_BENCHMARKS="ON"
BUILD_DIR="build"

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --no-tests)
            BUILD_TESTS="OFF"
            shift
            ;;
        --no-examples)
            BUILD_EXAMPLES="OFF"
            shift
            ;;
        --no-benchmarks)
            BUILD_BENCHMARKS="OFF"
            shift
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --debug              Build in debug mode"
            echo "  --no-tests           Don't build tests"
            echo "  --no-examples        Don't build examples"
            echo "  --no-benchmarks      Don't build benchmarks"
            echo "  --build-dir DIR      Use DIR as build directory (default: build)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo ""
echo "Build Configuration:"
echo "  Build Type: $BUILD_TYPE"
echo "  Build Tests: $BUILD_TESTS"
echo "  Build Examples: $BUILD_EXAMPLES"
echo "  Build Benchmarks: $BUILD_BENCHMARKS"
echo "  Build Directory: $BUILD_DIR"
echo ""

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo -e "${GREEN}Configuring with CMake...${NC}"
cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DBUILD_TESTS="$BUILD_TESTS" \
    -DBUILD_EXAMPLES="$BUILD_EXAMPLES" \
    -DBUILD_BENCHMARKS="$BUILD_BENCHMARKS" \
    -DENABLE_TENSOR_CORES=ON

# Build
echo -e "${GREEN}Building...${NC}"
NUM_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
make -j"$NUM_CORES"

echo ""
echo -e "${GREEN}Build completed successfully!${NC}"

# Run tests if built
if [ "$BUILD_TESTS" = "ON" ]; then
    echo ""
    echo -e "${GREEN}Running tests...${NC}"
    if ctest --output-on-failure; then
        echo -e "${GREEN}All tests passed!${NC}"
    else
        echo -e "${RED}Some tests failed!${NC}"
        exit 1
    fi
fi

echo ""
echo "================================================"
echo -e "${GREEN}CUDA Nexus build complete!${NC}"
echo ""
echo "To install:"
echo "  sudo make install"
echo ""
echo "To run examples:"
echo "  ./examples/gemm_example"
echo "  ./examples/attention_example"
echo "================================================"
