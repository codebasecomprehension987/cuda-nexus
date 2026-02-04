# CUDA Nexus API Documentation

## Overview

CUDA Nexus provides high-performance GPU computing primitives optimized for modern NVIDIA architectures. This document covers the main APIs and usage patterns.

## Table of Contents

1. [Initialization](#initialization)
2. [Matrix Operations](#matrix-operations)
3. [Attention Mechanisms](#attention-mechanisms)
4. [Reduction Operations](#reduction-operations)
5. [Normalization](#normalization)
6. [Activation Functions](#activation-functions)
7. [Memory Operations](#memory-operations)
8. [Convolution](#convolution)
9. [Utilities](#utilities)

---

## Initialization

### Device Properties

```cpp
#include "cuda_nexus.h"

auto props = cuda_nexus::DeviceProperties::get(0);
std::cout << "SM Count: " << props.sm_count << std::endl;
std::cout << "Tensor Cores Available: " << props.tensor_cores_available << std::endl;
```

---

## Matrix Operations

### Basic GEMM

Matrix multiplication: `C = alpha * A @ B + beta * C`

```cpp
#include "cuda_nexus.h"

// Configure GEMM
cuda_nexus::kernels::GEMMConfig config;
config.M = 1024;
config.N = 1024;
config.K = 1024;
config.precision = cuda_nexus::Precision::FP32;
config.alpha = 1.0f;
config.beta = 0.0f;

// Execute
cuda_nexus::kernels::gemm(d_A, d_B, d_C, config, stream);
```

### Tensor Core GEMM (FP16)

```cpp
cuda_nexus::kernels::GEMMConfig config;
config.M = 1024;
config.N = 1024;
config.K = 1024;
config.precision = cuda_nexus::Precision::FP16;
config.use_tensor_cores = true;

cuda_nexus::kernels::gemm(d_A_fp16, d_B_fp16, d_C_fp16, config, stream);
```

### Batched GEMM

```cpp
// Batched with separate arrays
const void* A_array[] = {d_A1, d_A2, d_A3};
const void* B_array[] = {d_B1, d_B2, d_B3};
void* C_array[] = {d_C1, d_C2, d_C3};

config.batch_size = 3;
cuda_nexus::kernels::gemm_batched(A_array, B_array, C_array, config, stream);

// Strided batched (contiguous memory)
cuda_nexus::kernels::gemm_strided_batched(
    d_A, d_B, d_C,
    M * K * sizeof(float),  // stride_A
    K * N * sizeof(float),  // stride_B
    M * N * sizeof(float),  // stride_C
    config, stream
);
```

### Fused GEMM + Activation

```cpp
cuda_nexus::kernels::gemm_fused_activation(
    d_A, d_B, d_C,
    d_bias,
    config,
    "gelu",  // Options: "relu", "gelu", "silu"
    stream
);
```

---

## Attention Mechanisms

### Multi-Head Attention

```cpp
cuda_nexus::kernels::AttentionConfig attn_config;
attn_config.batch_size = 8;
attn_config.num_heads = 12;
attn_config.seq_length = 512;
attn_config.head_dim = 64;
attn_config.scale = 1.0f / sqrt(64.0f);
attn_config.causal = false;  // Set true for causal masking
attn_config.precision = cuda_nexus::Precision::FP16;

cuda_nexus::kernels::fused_multi_head_attention(
    d_Q, d_K, d_V,
    d_output,
    attn_config,
    stream
);
```

### Grouped Query Attention (GQA)

```cpp
cuda_nexus::kernels::grouped_query_attention(
    d_Q,        // [batch, num_q_heads, seq_len, head_dim]
    d_K,        // [batch, num_kv_heads, seq_len, head_dim]
    d_V,        // [batch, num_kv_heads, seq_len, head_dim]
    d_output,
    num_q_heads,   // e.g., 32
    num_kv_heads,  // e.g., 8
    attn_config,
    stream
);
```

### Attention with KV Cache

```cpp
// For autoregressive decoding
cuda_nexus::kernels::attention_with_kv_cache(
    d_Q_new_token,  // [batch, num_heads, 1, head_dim]
    d_K_cache,      // [batch, num_heads, cache_len, head_dim]
    d_V_cache,      // [batch, num_heads, cache_len, head_dim]
    d_output,
    attn_config,
    stream
);
```

---

## Reduction Operations

### Sum Reduction

```cpp
// Warp-level reduction (fast for small sizes)
cuda_nexus::kernels::warp_reduce_sum(d_input, d_output, size, stream);

// Block-level reduction (better for larger sizes)
cuda_nexus::kernels::block_reduce_sum(d_input, d_output, size, stream);
```

### Multi-dimensional Reduction

```cpp
int shape[] = {32, 128, 256};  // 3D tensor
int reduce_dim = 1;  // Reduce along dimension 1

cuda_nexus::kernels::reduce(
    d_input,
    d_output,
    shape,
    3,  // ndim
    reduce_dim,
    cuda_nexus::kernels::ReductionOp::SUM,
    cuda_nexus::Precision::FP32,
    stream
);
```

### Segmented Reduction

```cpp
// Reduce within segments
cuda_nexus::kernels::segmented_reduce(
    d_input,
    d_segment_ids,  // Array mapping elements to segments
    d_output,
    num_segments,
    size,
    cuda_nexus::kernels::ReductionOp::SUM,
    cuda_nexus::Precision::FP32,
    stream
);
```

### Prefix Sum (Scan)

```cpp
// Inclusive scan
cuda_nexus::kernels::prefix_sum_inclusive(d_input, d_output, size, stream);

// Exclusive scan
cuda_nexus::kernels::prefix_sum_exclusive(d_input, d_output, size, stream);
```

### ArgMax/ArgMin

```cpp
int shape[] = {128, 256};
int reduce_dim = 1;

cuda_nexus::kernels::argmax(
    d_input,
    d_output_indices,
    shape,
    2,  // ndim
    reduce_dim,
    cuda_nexus::Precision::FP32,
    stream
);
```

### Top-K Selection

```cpp
cuda_nexus::kernels::topk(
    d_input,
    d_output_values,
    d_output_indices,
    size,
    k,
    true,  // largest=true for top-k, false for bottom-k
    cuda_nexus::Precision::FP32,
    stream
);
```

---

## Normalization

### Layer Normalization

```cpp
cuda_nexus::kernels::layer_norm(
    d_input,
    d_gamma,    // Scale parameter
    d_beta,     // Shift parameter
    d_output,
    batch_size,
    hidden_size,
    1e-5f,      // epsilon
    cuda_nexus::Precision::FP32,
    stream
);
```

### RMS Normalization

```cpp
// RMS norm doesn't subtract mean
cuda_nexus::kernels::rms_norm(
    d_input,
    d_gamma,
    d_output,
    batch_size,
    hidden_size,
    1e-5f,
    cuda_nexus::Precision::FP32,
    stream
);
```

### Batch Normalization

```cpp
// Inference mode
cuda_nexus::kernels::batch_norm_inference(
    d_input,
    d_gamma,
    d_beta,
    d_running_mean,
    d_running_var,
    d_output,
    batch_size,
    channels,
    spatial_size,
    1e-5f,
    cuda_nexus::Precision::FP32,
    stream
);
```

---

## Activation Functions

### ReLU and Variants

```cpp
// Standard ReLU
cuda_nexus::kernels::relu(d_input, d_output, size, precision, stream);

// Leaky ReLU
cuda_nexus::kernels::leaky_relu(d_input, d_output, size, 0.01f, precision, stream);
```

### GELU

```cpp
// Exact GELU
cuda_nexus::kernels::gelu(d_input, d_output, size, precision, stream);

// Approximate GELU (faster)
cuda_nexus::kernels::gelu_approx(d_input, d_output, size, precision, stream);
```

### SiLU/Swish

```cpp
cuda_nexus::kernels::silu(d_input, d_output, size, precision, stream);
```

### Softmax

```cpp
cuda_nexus::kernels::softmax(
    d_input,
    d_output,
    batch_size,
    num_classes,
    precision,
    stream
);
```

### Fused Activation + Dropout

```cpp
cuda_nexus::kernels::activation_dropout(
    d_input,
    d_output,
    d_dropout_mask,
    size,
    0.1f,           // dropout probability
    "gelu",         // activation type
    seed,           // random seed
    precision,
    stream
);
```

---

## Memory Operations

### Transpose

```cpp
// 2D transpose
cuda_nexus::kernels::transpose_2d(
    d_input,
    d_output,
    rows,
    cols,
    precision,
    stream
);
```

### Gather/Scatter

```cpp
// Gather
cuda_nexus::kernels::gather(
    d_input,
    d_indices,
    d_output,
    num_indices,
    inner_size,
    precision,
    stream
);

// Scatter
cuda_nexus::kernels::scatter(
    d_input,
    d_indices,
    d_output,
    num_indices,
    inner_size,
    precision,
    stream
);
```

### Concatenate/Split

```cpp
// Concatenate
const void* inputs[] = {d_input1, d_input2, d_input3};
int input_sizes[] = {size1, size2, size3};

cuda_nexus::kernels::concatenate(
    inputs,
    3,              // num_inputs
    d_output,
    input_sizes,
    concat_dim,
    outer_size,
    inner_size,
    precision,
    stream
);
```

---

## Convolution

### 2D Convolution

```cpp
cuda_nexus::kernels::ConvConfig conv_config;
conv_config.batch_size = 32;
conv_config.in_channels = 3;
conv_config.out_channels = 64;
conv_config.input_height = 224;
conv_config.input_width = 224;
conv_config.kernel_height = 3;
conv_config.kernel_width = 3;
conv_config.stride_h = 1;
conv_config.stride_w = 1;
conv_config.padding_h = 1;
conv_config.padding_w = 1;

cuda_nexus::kernels::conv2d(
    d_input,
    d_weight,
    d_bias,
    d_output,
    conv_config,
    stream
);
```

### Depthwise Convolution

```cpp
cuda_nexus::kernels::depthwise_conv2d(
    d_input, d_weight, d_bias, d_output, conv_config, stream
);
```

### Pooling

```cpp
// Max pooling
cuda_nexus::kernels::max_pool2d(
    d_input,
    d_output,
    d_indices,  // Optional, for unpooling
    batch_size, channels,
    input_h, input_w,
    kernel_h, kernel_w,
    stride_h, stride_w,
    padding_h, padding_w,
    precision,
    stream
);
```

---

## Utilities

### Memory Pool

```cpp
// Get default memory pool
auto& pool = cuda_nexus::utils::get_default_pool();

// Allocate from pool
void* ptr = pool.allocate(1024 * 1024 * sizeof(float), stream);

// Use memory...

// Return to pool
pool.free(ptr);

// Or use RAII wrapper
{
    cuda_nexus::utils::PooledMemory mem(pool, size, stream);
    void* ptr = mem.get();
    // Automatically freed when mem goes out of scope
}
```

### Profiling

```cpp
// Profile a kernel
{
    cuda_nexus::utils::ScopedTimer timer("my_kernel", stream);
    my_kernel<<<grid, block, 0, stream>>>(...);
}

// Get statistics
auto stats = cuda_nexus::utils::Profiler::instance().get_stats("my_kernel");
std::cout << "Average time: " << stats.avg_time_ms << " ms\n";

// Print all statistics
cuda_nexus::utils::Profiler::instance().print_summary();
```

### Async Operations

```cpp
// Get stream from pool
auto& stream_pool = cuda_nexus::utils::get_stream_pool();
cudaStream_t stream = stream_pool.get_stream();

// Launch async operations
kernel1<<<grid, block, 0, stream>>>(...);
kernel2<<<grid, block, 0, stream>>>(...);

// Return stream to pool
stream_pool.return_stream(stream);
```

### Graph Capture

```cpp
cuda_nexus::utils::GraphCapture graph;

// Capture operations
graph.begin_capture(stream);
kernel1<<<grid, block, 0, stream>>>(...);
kernel2<<<grid, block, 0, stream>>>(...);
graph.end_capture();

// Execute captured graph (much faster for repeated operations)
graph.execute(stream);
```

---

## Error Handling

All operations use the `CUDA_CHECK` macro internally. For custom error handling:

```cpp
cudaError_t err = cudaMemcpy(...);
if (err != cudaSuccess) {
    // Handle error
}
```

---

## Performance Tips

1. **Use Tensor Cores**: For FP16/BF16 matrix operations, enable Tensor Cores
2. **Batch Operations**: Use batched variants when processing multiple problems
3. **Memory Pool**: Reuse allocations through memory pool to avoid malloc overhead
4. **Graph Capture**: Capture repeated operation sequences as CUDA graphs
5. **Stream Parallelism**: Use multiple streams to overlap computation and memory transfers
6. **Profile**: Use built-in profiler or Nsight Compute to identify bottlenecks

---

## Examples

See the `examples/` directory for complete working examples:
- `gemm_example.cu` - Matrix multiplication
- `attention_example.cu` - Multi-head attention
