<div align="center">

# Contributing to CUDA Nexus

**Thank you for taking the time to contribute.**  
Every bug report, kernel optimization, and documentation fix makes CUDA Nexus better for the entire GPU computing community.

</div>

---

## 📋 Table of Contents

- [Code of Conduct](#-code-of-conduct)
- [Ways to Contribute](#-ways-to-contribute)
- [Development Setup](#-development-setup)
- [Coding Standards](#-coding-standards)
- [Documenting Your Kernel](#-documenting-your-kernel)
- [Testing Requirements](#-testing-requirements)
- [Performance Guidelines](#-performance-guidelines)
- [Commit Messages](#-commit-messages)
- [Pull Request Process](#-pull-request-process)

---

## 🤝 Code of Conduct

Be professional, constructive, and respectful in all interactions — in issues, PRs, and reviews. Harassment of any kind will not be tolerated.

---

## 💡 Ways to Contribute

### Reporting a Bug

Open a [GitHub Issue](https://github.com/codebasecomprehension987/cuda-nexus/issues) and include:

- GPU model and compute capability
- CUDA Toolkit version (`nvcc --version`)
- Driver version (`nvidia-smi`)
- Operating system
- A **minimal reproducible example** — the smallest code that triggers the bug
- Full error output and any stack traces

### Suggesting an Enhancement

Open an issue describing:

- The use case and the problem it solves
- The proposed API or behavior
- Any backward compatibility implications

### Submitting a Pull Request

See the [Pull Request Process](#-pull-request-process) section below.

---

## 🛠️ Development Setup

```bash
# 1. Fork and clone
git clone https://github.com/codebasecomprehension987/cuda-nexus.git
cd cuda-nexus

# 2. Create a feature branch
git checkout -b feature/your-kernel-name

# 3. Configure with tests and examples enabled
mkdir build && cd build
cmake -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
      -DBUILD_TESTS=ON                          \
      -DBUILD_EXAMPLES=ON                       \
      ..

# 4. Build
make -j$(nproc)

# 5. Run the test suite
ctest --output-on-failure
```

---

## 📐 Coding Standards

### C++ Style

- Target **C++17** — use modern features where they improve clarity
- Follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- Write self-documenting code; add inline comments only for non-obvious logic

### CUDA Style

- Optimize for **coalesced global memory access** — uncoalesced loads are the #1 performance killer
- Use shared memory deliberately — avoid bank conflicts (pad when necessary)
- Document your **kernel launch configuration** and the reasoning behind block/grid dimensions
- **Profile before and after** every optimization — never guess

### Naming Conventions

| Construct | Convention | Example |
|-----------|------------|---------|
| Files | `snake_case` | `fused_attention.cu` |
| Classes / Structs | `PascalCase` | `GEMMConfig` |
| Functions | `snake_case` | `gemm_persistent` |
| Constants / Macros | `UPPER_SNAKE_CASE` | `MAX_BLOCK_SIZE` |
| Namespaces | `snake_case` | `cuda_nexus::kernels` |

---

## 📝 Documenting Your Kernel

Every public kernel function must include a docstring in the following format:

```cpp
/**
 * @brief One-line summary of what the kernel does
 *
 * Longer description if needed — explain the algorithm,
 * key optimizations, and any important caveats.
 *
 * @param A     Input matrix A [M x K] — device pointer
 * @param B     Input matrix B [K x N] — device pointer
 * @param C     Output matrix C [M x N] — device pointer
 * @param config  GEMM configuration (precision, layout, alpha/beta)
 * @param stream  CUDA stream for async execution (default: 0)
 *
 * @note Requires compute capability >= 8.0 for BF16 Tensor Cores
 * @complexity O(M × N × K) FLOPs
 */
void gemm_tensor_core(
    const void* A,
    const void* B,
    void* C,
    const GEMMConfig& config,
    cudaStream_t stream = 0
);
```

---

## 🧪 Testing Requirements

- Write **unit tests** for all new kernels using GoogleTest
- Cover edge cases: minimum sizes, power-of-two vs non-power-of-two dimensions, empty inputs
- Test across at least the architectures you have access to — note any architecture-specific behavior
- Benchmark your kernel and include the numbers in your PR description

```bash
# Run specific test
./tests/test_gemm

# Run full suite
ctest --output-on-failure
```

---

## 📊 Performance Guidelines

- Profile with **Nsight Compute** (`ncu`) before submitting
- Target **>80% of theoretical peak** performance for compute-bound kernels
- For memory-bound kernels, target **>80% of peak memory bandwidth**
- Document achieved TFLOPS or GB/s in your PR
- Compare against **cuBLAS / cuDNN** as a baseline where applicable
- If your kernel is slower on some configurations, explain why and when it wins

---

## 💬 Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add fused SiLU activation kernel
fix: resolve shared memory bank conflict in block reduction
perf: improve attention memory access pattern — 1.3× speedup on A100
docs: add complexity analysis to LayerNorm docstring
test: add edge-case tests for strided batched GEMM
refactor: unify precision dispatch across all GEMM variants
```

Keep the subject line under 72 characters. Add a body if the change needs explanation.

---

## 🔁 Pull Request Process

1. **Fork** the repository and create your branch from `main`
2. **Make your changes** following the coding and documentation standards above
3. **Add or update tests** — all new kernels need test coverage
4. **Run the full test suite** locally and confirm it passes
5. **Benchmark** your change and include numbers in the PR description
6. **Open a pull request** against `main` with a clear title and description

### PR Checklist

Before requesting review, confirm:

- [ ] Code follows the naming and style conventions
- [ ] All public functions have docstrings
- [ ] New tests are included and passing
- [ ] Benchmark numbers are included in the PR description
- [ ] No regressions in existing benchmarks
- [ ] Commits follow the Conventional Commits format

### Review Process

1. Automated CI checks must pass
2. At least one maintainer review is required
3. Address all review comments before merge
4. Squash commits before the final merge

---

## 📄 License

By contributing to CUDA Nexus, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

<div align="center">

Every optimization counts. Happy hacking. 🚀

</div>
