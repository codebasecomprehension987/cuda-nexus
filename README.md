<div align="center">

<br/>

```
 ██████╗██╗   ██╗██████╗  █████╗     ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗
██╔════╝██║   ██║██╔══██╗██╔══██╗    ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝
██║     ██║   ██║██║  ██║███████║    ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗
██║     ██║   ██║██║  ██║██╔══██║    ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║
╚██████╗╚██████╔╝██████╔╝██║  ██║    ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║
 ╚═════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝    ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝
```

**High-performance CUDA kernel library for modern NVIDIA GPU architectures**

<br/>

[![License: MIT](https://img.shields.io/badge/License-MIT-00d4aa.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.0+-76b900.svg?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599c.svg?style=for-the-badge&logo=cplusplus)](https://isocpp.org/)
[![Architecture](https://img.shields.io/badge/GPU-Ampere%20%7C%20Ada%20%7C%20Hopper-ff6b35.svg?style=for-the-badge)](https://developer.nvidia.com)

<br/>

> *Pushing NVIDIA GPUs to their absolute limits — through Tensor Cores, persistent kernels, warp-specialized execution, and research-backed fusion strategies.*

<br/>

</div>

---

## 🧭 What is CUDA Nexus?

CUDA Nexus is a **production-grade CUDA kernel library** engineered for developers who refuse to leave performance on the table. It provides hand-tuned GPU primitives covering the full spectrum of deep learning and HPC workloads — from GEMM and attention to reductions and memory operations — with hardware-aware optimizations for Ampere, Ada Lovelace, and Hopper architectures.

Unlike general-purpose libraries (cuBLAS, cuDNN), CUDA Nexus exposes **fine-grained control** over execution policies, memory layouts, and precision modes. Whether you're building a custom ML framework, a research inference engine, or a high-performance compute pipeline, CUDA Nexus gives you the low-level leverage you need.

<br/>

## ⚡ Performance at a Glance

> Benchmarked on **NVIDIA RTX 4090 (Ada Lovelace)**

| Kernel | CUDA Nexus | cuBLAS / Baseline | Speedup |
|--------|-----------|-------------------|---------|
| GEMM (FP16, 4096²) | **412 TFLOPS** | 385 TFLOPS | **1.07×** |
| Fused Multi-Head Attention | **8.2 ms** | 11.4 ms | **1.39×** |
| Layer Normalization | **1.8 ms** | 2.3 ms | **1.28×** |

<br/>

## 🗂️ Repository Layout

```
cuda-nexus/
│
├── 📁 include/                     # Public API — include this in your project
│   ├── cuda_nexus.h                # ← Single-header entry point
│   ├── tensor.h                    # Tensor descriptor & layout utilities
│   ├── kernels/
│   │   ├── gemm.cuh                # Matrix multiply (FP32/FP16/BF16, Tensor Cores)
│   │   ├── attention.cuh           # Flash Attention, GQA, KV-cache variants
│   │   ├── reduction.cuh           # Warp, block, segmented reductions
│   │   ├── normalization.cuh       # LayerNorm, RMSNorm, GroupNorm
│   │   ├── activation.cuh          # GELU, SiLU, Swish, fused bias activations
│   │   ├── convolution.cuh         # Grouped convolutions, im2col
│   │   └── memory_ops.cuh          # Vectorized copy, gather/scatter, prefix-sum
│   └── utils/
│       ├── memory_pool.h           # GPU memory pool with async ops
│       ├── profiler.h              # Nsight-compatible markers & metrics
│       └── async_ops.h             # Async pipeline management
│
├── 📁 kernels/                     # Kernel implementations (.cu source)
├── 📁 src/                         # C++ utility implementations
├── 📁 examples/                    # Runnable examples (GEMM, Attention)
├── 📁 benchmarks/                  # Performance benchmarking suite
├── 📁 tests/                       # Unit + integration tests (GoogleTest)
├── 📁 docs/                        # API reference
├── 📁 scripts/                     # Build & benchmark automation
└── 📁 third_party/                 # External dependencies
```

<br/>

## 🛠️ Building from Source

### Prerequisites

| Requirement | Version |
|-------------|---------|
| CUDA Toolkit | 12.0+ |
| CMake | 3.18+ |
| GCC / Clang | C++17 support |
| NVIDIA GPU | Compute Capability 8.0+ (Ampere or newer) |

### Quick Build

```bash
git clone https://github.com/codebasecomprehension987/cuda-nexus.git
cd cuda-nexus

mkdir build && cd build

# Build for common modern architectures
cmake -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" ..

make -j$(nproc)
```

Or use the provided build script:

```bash
bash scripts/build.sh
```

### Build Options

```bash
# Build with benchmarks
cmake -DCUDA_NEXUS_BUILD_BENCHMARKS=ON ..

# Build with tests (requires GoogleTest)
cmake -DCUDA_NEXUS_BUILD_TESTS=ON ..

# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

<br/>

## 🚀 Quick Start

### 1. Include the library

```cpp
#include "cuda_nexus.h"
using namespace cuda_nexus;
```

### 2. Run your first GEMM

```cpp
// Configure a 1024×1024 FP16 matrix multiply with Tensor Cores
kernels::GEMMConfig config;
config.M = 1024; config.N = 1024; config.K = 1024;
config.precision        = Precision::FP16;
config.use_tensor_cores = true;
config.alpha = 1.0f;
config.beta  = 0.0f;

// Launch on an async stream
kernels::gemm(d_A, d_B, d_C, config, stream);
```

### 3. Fused multi-head attention

```cpp
kernels::AttentionConfig attn;
attn.batch_size  = 4;
attn.num_heads   = 16;
attn.seq_length  = 2048;
attn.head_dim    = 64;
attn.scale       = 1.0f / sqrtf(64.0f);
attn.causal      = true;          // autoregressive mask
attn.precision   = Precision::FP16;

kernels::fused_multi_head_attention(d_Q, d_K, d_V, d_out, attn, stream);
```

### 4. GPU memory pool

```cpp
// Avoid cudaMalloc/cudaFree in hot paths
utils::MemoryPool pool(512 * 1024 * 1024);  // 512 MB initial

void* buf = pool.allocate(my_size, stream);
// ... kernel calls ...
pool.free(buf);
```

See [`examples/`](examples/) for complete, runnable programs.

<br/>

## 🔬 Kernel Reference

### GEMM — `include/kernels/gemm.cuh`

All variants support FP32, FP16, BF16, INT8, and mixed precision with configurable row/column-major layouts.

| Function | Description |
|----------|-------------|
| `gemm(...)` | Standard `C = α·A@B + β·C` |
| `gemm_batched(...)` | Batched GEMM over pointer arrays |
| `gemm_strided_batched(...)` | Batched GEMM with fixed stride offsets |
| `gemm_fused_activation(...)` | GEMM + bias + activation in a single kernel pass |
| `gemm_wmma_fp16(...)` | Direct WMMA Tensor Core GEMM (FP16) |
| `gemm_persistent(...)` | Persistent-thread GEMM for minimum launch overhead |

### Attention — `include/kernels/attention.cuh`

Flash Attention-inspired tiled implementation with O(N) HBM complexity.

| Function | Description |
|----------|-------------|
| `fused_multi_head_attention(...)` | Standard MHA: `softmax(QKᵀ/√d)·V` |
| `masked_attention(...)` | Attention with arbitrary boolean mask |
| `attention_with_kv_cache(...)` | Decode-phase attention against a KV cache |
| `grouped_query_attention(...)` | GQA for models like LLaMA-2/3 |
| `attention_backward(...)` | Backward pass — computes grad Q, K, V |

### Reductions — `include/kernels/reduction.cuh`

| Variant | Description |
|---------|-------------|
| Warp-shuffle | Register-only, zero shared memory |
| Block-wide | Shared memory with bank-conflict avoidance |
| Segmented | Independent reductions per segment |
| Multi-dimensional | Arbitrary-axis reduction |

### Normalization — `include/kernels/normalization.cuh`

LayerNorm, RMSNorm, and GroupNorm — all with fused affine transform (γ, β) in a single kernel pass.

### Activations — `include/kernels/activation.cuh`

Fused GELU, SiLU, Swish — including fused-bias variants that eliminate an extra memory round-trip.

### Memory Operations — `include/kernels/memory_ops.cuh`

128-bit vectorized copy, in-place transpose, gather/scatter primitives, and inclusive/exclusive prefix-sum.

<br/>

## 🎛️ Precision & Execution Modes

```cpp
// Precision
Precision::FP32    // Full precision
Precision::FP16    // Half precision — Tensor Core eligible
Precision::BF16    // Brain float — better dynamic range than FP16
Precision::INT8    // Quantized inference
Precision::MIXED   // FP16 compute, FP32 accumulate

// Execution policy
ExecutionPolicy::DEFAULT             // Standard grid launch
ExecutionPolicy::PERSISTENT          // Work-queue kernels, minimum re-launch cost
ExecutionPolicy::COOPERATIVE         // Multi-block synchronization
ExecutionPolicy::DYNAMIC_PARALLELISM // Child kernel launches from device
```

<br/>

## 📊 Running Benchmarks

```bash
cd build
bash ../scripts/run_benchmarks.sh

# Or individually
./benchmarks/benchmark_gemm
./benchmarks/benchmark_attention
./benchmarks/benchmark_reduction
```

Output reports: operation, size, execution time (ms), throughput (GFLOPS / GB/s), and speedup vs baseline.

<br/>

## 🧪 Running Tests

```bash
cd build
ctest --output-on-failure

# Or directly
./tests/test_gemm
```

Tests require GoogleTest — if not found, CMake will warn and skip test targets.

<br/>

## 🔑 What Makes CUDA Nexus Different

**Wavefront-aware scheduling** — Kernel launch configs are tuned per SM count, saturating your specific GPU's compute grid rather than targeting a generic occupancy formula.

**Dynamic kernel fusion** — GEMM + bias + activation that naively costs 3 memory round-trips is collapsed to 1 at runtime.

**Persistent kernels** — Long-running kernels with internal work queues eliminate repeated launch overhead on latency-sensitive workloads.

**Warp specialization** — Different warp groups within a single thread block handle distinct roles (loading vs. computing), improving pipeline utilization.

**Adaptive precision** — Kernels can switch between FP32, FP16, and INT8 at runtime based on workload characteristics and detected GPU capability.

<br/>

## 📐 Architecture Support

| GPU Architecture | Compute Capability | Status |
|------------------|--------------------|--------|
| Ampere (A100, A30, RTX 3000) | 8.0, 8.6 | ✅ Full support |
| Ada Lovelace (RTX 4000) | 8.9 | ✅ Full support |
| Hopper (H100) | 9.0 | ✅ Full support |
| Turing (RTX 2000, T4) | 7.5 | ⚠️ Partial — no BF16 |

<br/>

## 📚 Research Foundation

| Technique | Source |
|-----------|--------|
| Flash Attention — tiled O(N) HBM attention | Dao et al., 2022 |
| Persistent kernels — work-queue dispatch | NVIDIA GTC, 2022 |
| Warp-specialized programming | SC '21 |
| Async memory ops — `__pipeline_memcpy_async` | CUDA 11+ Toolkit |
| Cooperative Groups — multi-block sync | CUDA Programming Guide |

<br/>

## 🤝 Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full guide — coding standards, commit conventions, the review process, and how to set up your dev environment.

<br/>

## 📄 License

Released under the **MIT License**. See [`LICENSE`](LICENSE) for full terms.

---

<div align="center">

Built for engineers who care about every nanosecond.

</div>
