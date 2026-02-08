# Third Party Dependencies

This directory manages external dependencies for CUDA Nexus.

## Automatic Download (Recommended)

The CMake configuration will automatically download required dependencies using FetchContent:

```bash
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON
```

## Dependencies

### Required (automatically downloaded)
- None - CUDA Nexus has no required external dependencies!

### Optional (automatically downloaded when enabled)

#### Google Test (for testing)
- **Enabled by**: `-DBUILD_TESTS=ON` (default)
- **Version**: v1.14.0
- **Purpose**: Unit testing framework
- **License**: BSD-3-Clause

#### Google Benchmark (for micro-benchmarks)
- **Enabled by**: `-DBUILD_BENCHMARKS_GBENCH=ON` (default: OFF)
- **Version**: v1.8.3
- **Purpose**: Micro-benchmarking framework
- **License**: Apache-2.0
- **Note**: Different from our macro benchmarks in benchmarks/

#### nlohmann/json (for configuration)
- **Enabled by**: `-DUSE_JSON=ON` (default: OFF)
- **Version**: v3.11.3
- **Purpose**: JSON parsing for config files
- **License**: MIT

#### spdlog (for logging)
- **Enabled by**: `-DUSE_SPDLOG=ON` (default: OFF)
- **Version**: v1.12.0
- **Purpose**: Fast C++ logging library
- **License**: MIT

## System Dependencies

These must be installed on your system:

### Required
- **CUDA Toolkit** (12.0+)
  - Ubuntu: `sudo apt install nvidia-cuda-toolkit`
  - Download: https://developer.nvidia.com/cuda-downloads

- **CMake** (3.18+)
  - Ubuntu: `sudo apt install cmake`
  - Download: https://cmake.org/download/

### Optional (for benchmarks)
- **cuBLAS**: Comes with CUDA Toolkit
- **cuDNN** (for comparison): Download from NVIDIA
- **Thrust**: Comes with CUDA Toolkit

## Manual Installation

If you prefer to install dependencies manually:

### Ubuntu/Debian
```bash
# Google Test
sudo apt install libgtest-dev

# Google Benchmark
sudo apt install libbenchmark-dev
```

### From Source
```bash
# Google Test
git clone https://github.com/google/googletest.git
cd googletest
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
sudo make install
```

Then configure CMake to use system libraries:
```bash
cmake .. -DFETCHCONTENT_FULLY_DISCONNECTED=ON
```

## Vendoring (Not Recommended)

If you want to vendor dependencies (include them in the repo):

1. Download dependencies:
```bash
cd third_party
git clone https://github.com/google/googletest.git
git clone https://github.com/google/benchmark.git
```

2. Update CMakeLists.txt to use `add_subdirectory()` instead of FetchContent

## License Compliance

All third-party dependencies use permissive licenses (BSD, MIT, Apache-2.0) compatible with CUDA Nexus's MIT license.

### License Summary
- **CUDA Nexus**: MIT
- **Google Test**: BSD-3-Clause
- **Google Benchmark**: Apache-2.0
- **nlohmann/json**: MIT
- **spdlog**: MIT

## Updating Dependencies

To update dependency versions, edit `third_party/CMakeLists.txt` and change the `GIT_TAG`:

```cmake
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.0  # Update this
    GIT_SHALLOW    TRUE
)
```

## Troubleshooting

### FetchContent fails to download
- Check internet connection
- Try: `cmake .. -DFETCHCONTENT_UPDATES_DISCONNECTED=OFF`
- Or use system packages: `cmake .. -DFETCHCONTENT_FULLY_DISCONNECTED=ON`

### Version conflicts
- Clear CMake cache: `rm -rf CMakeCache.txt CMakeFiles/`
- Rebuild: `cmake .. && make clean && make`

### Offline builds
- Pre-download dependencies with internet
- Use `cmake .. -DFETCHCONTENT_FULLY_DISCONNECTED=ON` for subsequent builds
