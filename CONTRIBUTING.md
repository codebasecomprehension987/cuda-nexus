# Contributing to CUDA Nexus

Thank you for your interest in contributing to CUDA Nexus! This document provides guidelines for contributing to the project.

## Code of Conduct

Be respectful and professional in all interactions. We want to maintain a welcoming environment for all contributors.

## How to Contribute

### Reporting Bugs

- Use the GitHub issue tracker
- Include: GPU model, CUDA version, driver version, OS
- Provide a minimal reproducible example
- Include error messages and stack traces

### Suggesting Enhancements

- Open an issue describing the enhancement
- Explain the use case and benefits
- Consider backward compatibility

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-kernel`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Update documentation
7. Commit with clear messages
8. Push to your fork
9. Open a pull request

## Development Setup

```bash
# Clone the repository
git clone https://github.com/codebasecomprehension987/cuda-nexus.git
cd cuda-nexus

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
      -DBUILD_TESTS=ON \
      -DBUILD_EXAMPLES=ON \
      ..

# Build
make -j$(nproc)

# Run tests
ctest --output-on-failure
```

## Coding Standards

### C++ Style

- Use C++17 features
- Follow Google C++ Style Guide
- Use meaningful variable names
- Add comments for complex logic

### CUDA Style

- Optimize for coalesced memory access
- Use shared memory effectively
- Avoid bank conflicts
- Document kernel launch configurations
- Profile before and after optimizations

### Naming Conventions

- Files: `snake_case.cu`, `snake_case.cuh`, `snake_case.h`
- Classes: `PascalCase`
- Functions: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Namespaces: `snake_case`

### Documentation

- Add docstrings to all public functions
- Include complexity analysis for kernels
- Document memory requirements
- Add usage examples

Example:
```cpp
/**
 * @brief Performs matrix multiplication using Tensor Cores
 * 
 * Computes C = alpha * A @ B + beta * C
 * 
 * @param A Input matrix A (M x K)
 * @param B Input matrix B (K x N)
 * @param C Output matrix C (M x N)
 * @param config GEMM configuration
 * @param stream CUDA stream for async execution
 * 
 * @note Requires compute capability >= 7.0 for Tensor Cores
 * @complexity O(M * N * K) FLOPs
 */
void gemm_tensor_core(...);
```

### Testing

- Write unit tests for all new kernels
- Include edge cases (empty matrices, large sizes)
- Test on multiple GPU architectures if possible
- Benchmark performance and compare to baselines

### Performance Guidelines

- Profile with Nsight Compute
- Target >80% of theoretical peak performance
- Document achieved TFLOPS/bandwidth
- Compare against cuBLAS/cuDNN when applicable

## Commit Messages

Use conventional commits format:

```
feat: add fused GELU activation kernel
fix: correct shared memory bank conflicts in reduction
docs: update GEMM API documentation
perf: optimize attention kernel memory access pattern
test: add unit tests for layer normalization
```

## Review Process

1. Automated CI checks must pass
2. At least one maintainer review required
3. Address all review comments
4. Squash commits before merge

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

