#ifndef CUDA_NEXUS_CONVOLUTION_CUH
#define CUDA_NEXUS_CONVOLUTION_CUH

#include <cuda_runtime.h>
#include "cuda_nexus.h"

namespace cuda_nexus {
namespace kernels {

// Convolution configuration
struct ConvConfig {
    int batch_size;
    int in_channels;
    int out_channels;
    int input_height;
    int input_width;
    int kernel_height;
    int kernel_width;
    int stride_h;
    int stride_w;
    int padding_h;
    int padding_w;
    int dilation_h = 1;
    int dilation_w = 1;
    int groups = 1;
    Precision precision = Precision::FP32;
};

// Standard 2D convolution
void conv2d(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    const ConvConfig& config,
    cudaStream_t stream = 0
);

// Depthwise convolution
void depthwise_conv2d(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    const ConvConfig& config,
    cudaStream_t stream = 0
);

// Grouped convolution
void grouped_conv2d(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    const ConvConfig& config,
    cudaStream_t stream = 0
);

// Transposed convolution (deconvolution)
void conv_transpose2d(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    const ConvConfig& config,
    cudaStream_t stream = 0
);

// Im2col transformation
void im2col(
    const void* input,
    void* col,
    int batch_size,
    int channels,
    int height,
    int width,
    int kernel_h,
    int kernel_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    Precision precision,
    cudaStream_t stream = 0
);

// Col2im transformation
void col2im(
    const void* col,
    void* input,
    int batch_size,
    int channels,
    int height,
    int width,
    int kernel_h,
    int kernel_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    Precision precision,
    cudaStream_t stream = 0
);

// Winograd convolution (for 3x3 kernels)
void winograd_conv2d_3x3(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    const ConvConfig& config,
    cudaStream_t stream = 0
);

// FFT-based convolution (for large kernels)
void fft_conv2d(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    const ConvConfig& config,
    cudaStream_t stream = 0
);

// Max pooling 2D
void max_pool2d(
    const void* input,
    void* output,
    int* indices,
    int batch_size,
    int channels,
    int input_h,
    int input_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    Precision precision,
    cudaStream_t stream = 0
);

// Average pooling 2D
void avg_pool2d(
    const void* input,
    void* output,
    int batch_size,
    int channels,
    int input_h,
    int input_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    bool count_include_pad,
    Precision precision,
    cudaStream_t stream = 0
);

// Adaptive average pooling
void adaptive_avg_pool2d(
    const void* input,
    void* output,
    int batch_size,
    int channels,
    int input_h,
    int input_w,
    int output_h,
    int output_w,
    Precision precision,
    cudaStream_t stream = 0
);

// Backward pass for convolution
void conv2d_backward_data(
    const void* grad_output,
    const void* weight,
    void* grad_input,
    const ConvConfig& config,
    cudaStream_t stream = 0
);

void conv2d_backward_weight(
    const void* grad_output,
    const void* input,
    void* grad_weight,
    void* grad_bias,
    const ConvConfig& config,
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace cuda_nexus

#endif // CUDA_NEXUS_CONVOLUTION_CUH
