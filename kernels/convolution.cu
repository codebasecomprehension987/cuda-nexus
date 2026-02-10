#include "kernels/convolution.cuh"

namespace cuda_nexus {
namespace kernels {

// Im2col kernel
__global__ void im2col_kernel(
    const float* input,
    float* col,
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
    int output_h,
    int output_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int col_size = channels * kernel_h * kernel_w * output_h * output_w;
    
    if (idx < col_size) {
        int w_out = idx % output_w;
        int h_out = (idx / output_w) % output_h;
        int kw = (idx / (output_w * output_h)) % kernel_w;
        int kh = (idx / (output_w * output_h * kernel_w)) % kernel_h;
        int c = idx / (output_w * output_h * kernel_w * kernel_h);
        
        int h_in = h_out * stride_h - pad_h + kh * dilation_h;
        int w_in = w_out * stride_w - pad_w + kw * dilation_w;
        
        if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
            col[idx] = input[c * height * width + h_in * width + w_in];
        } else {
            col[idx] = 0.0f;
        }
    }
}

// Direct convolution kernel (without im2col)
__global__ void conv2d_direct_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_h,
    int input_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int output_h,
    int output_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * output_h * output_w;
    
    if (idx < total_outputs) {
        int w_out = idx % output_w;
        int h_out = (idx / output_w) % output_h;
        int oc = (idx / (output_w * output_h)) % out_channels;
        int n = idx / (output_w * output_h * out_channels);
        
        float sum = bias ? bias[oc] : 0.0f;
        
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int h_in = h_out * stride_h - pad_h + kh;
                    int w_in = w_out * stride_w - pad_w + kw;
                    
                    if (h_in >= 0 && h_in < input_h && w_in >= 0 && w_in < input_w) {
                        int input_idx = n * in_channels * input_h * input_w +
                                       ic * input_h * input_w +
                                       h_in * input_w + w_in;
                        int weight_idx = oc * in_channels * kernel_h * kernel_w +
                                        ic * kernel_h * kernel_w +
                                        kh * kernel_w + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        output[idx] = sum;
    }
}

// Depthwise convolution kernel
__global__ void depthwise_conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int channels,
    int input_h,
    int input_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int output_h,
    int output_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels * output_h * output_w;
    
    if (idx < total_outputs) {
        int w_out = idx % output_w;
        int h_out = (idx / output_w) % output_h;
        int c = (idx / (output_w * output_h)) % channels;
        int n = idx / (output_w * output_h * channels);
        
        float sum = bias ? bias[c] : 0.0f;
        
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = h_out * stride_h - pad_h + kh;
                int w_in = w_out * stride_w - pad_w + kw;
                
                if (h_in >= 0 && h_in < input_h && w_in >= 0 && w_in < input_w) {
                    int input_idx = n * channels * input_h * input_w +
                                   c * input_h * input_w +
                                   h_in * input_w + w_in;
                    int weight_idx = c * kernel_h * kernel_w +
                                    kh * kernel_w + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
        
        output[idx] = sum;
    }
}

// Max pooling kernel
__global__ void max_pool2d_kernel(
    const float* input,
    float* output,
    int* indices,
    int batch_size,
    int channels,
    int input_h,
    int input_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int output_h,
    int output_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels * output_h * output_w;
    
    if (idx < total_outputs) {
        int w_out = idx % output_w;
        int h_out = (idx / output_w) % output_h;
        int c = (idx / (output_w * output_h)) % channels;
        int n = idx / (output_w * output_h * channels);
        
        float max_val = -INFINITY;
        int max_idx = -1;
        
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = h_out * stride_h - pad_h + kh;
                int w_in = w_out * stride_w - pad_w + kw;
                
                if (h_in >= 0 && h_in < input_h && w_in >= 0 && w_in < input_w) {
                    int input_idx = n * channels * input_h * input_w +
                                   c * input_h * input_w +
                                   h_in * input_w + w_in;
                    float val = input[input_idx];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = input_idx;
                    }
                }
            }
        }
        
        output[idx] = max_val;
        if (indices) {
            indices[idx] = max_idx;
        }
    }
}

// Average pooling kernel
__global__ void avg_pool2d_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int input_h,
    int input_w,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int output_h,
    int output_w,
    bool count_include_pad
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels * output_h * output_w;
    
    if (idx < total_outputs) {
        int w_out = idx % output_w;
        int h_out = (idx / output_w) % output_h;
        int c = (idx / (output_w * output_h)) % channels;
        int n = idx / (output_w * output_h * channels);
        
        float sum = 0.0f;
        int count = 0;
        
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h_in = h_out * stride_h - pad_h + kh;
                int w_in = w_out * stride_w - pad_w + kw;
                
                if (h_in >= 0 && h_in < input_h && w_in >= 0 && w_in < input_w) {
                    int input_idx = n * channels * input_h * input_w +
                                   c * input_h * input_w +
                                   h_in * input_w + w_in;
                    sum += input[input_idx];
                    count++;
                } else if (count_include_pad) {
                    count++;
                }
            }
        }
        
        output[idx] = count > 0 ? sum / count : 0.0f;
    }
}

// Host function implementations
void conv2d(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    const ConvConfig& config,
    cudaStream_t stream
) {
    if (config.precision == Precision::FP32) {
        int output_h = (config.input_height + 2 * config.padding_h - 
                       config.dilation_h * (config.kernel_height - 1) - 1) / 
                       config.stride_h + 1;
        int output_w = (config.input_width + 2 * config.padding_w - 
                       config.dilation_w * (config.kernel_width - 1) - 1) / 
                       config.stride_w + 1;
        
        int total_outputs = config.batch_size * config.out_channels * output_h * output_w;
        int block_size = 256;
        int grid_size = (total_outputs + block_size - 1) / block_size;
        
        conv2d_direct_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(input),
            static_cast<const float*>(weight),
            static_cast<const float*>(bias),
            static_cast<float*>(output),
            config.batch_size,
            config.in_channels,
            config.out_channels,
            config.input_height,
            config.input_width,
            config.kernel_height,
            config.kernel_width,
            config.stride_h,
            config.stride_w,
            config.padding_h,
            config.padding_w,
            output_h,
            output_w
        );
    }
}

void depthwise_conv2d(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    const ConvConfig& config,
    cudaStream_t stream
) {
    if (config.precision == Precision::FP32) {
        int output_h = (config.input_height + 2 * config.padding_h - 
                       config.kernel_height) / config.stride_h + 1;
        int output_w = (config.input_width + 2 * config.padding_w - 
                       config.kernel_width) / config.stride_w + 1;
        
        int total_outputs = config.batch_size * config.in_channels * output_h * output_w;
        int block_size = 256;
        int grid_size = (total_outputs + block_size - 1) / block_size;
        
        depthwise_conv2d_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(input),
            static_cast<const float*>(weight),
            static_cast<const float*>(bias),
            static_cast<float*>(output),
            config.batch_size,
            config.in_channels,
            config.input_height,
            config.input_width,
            config.kernel_height,
            config.kernel_width,
            config.stride_h,
            config.stride_w,
            config.padding_h,
            config.padding_w,
            output_h,
            output_w
        );
    }
}

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
    cudaStream_t stream
) {
    if (precision == Precision::FP32) {
        int output_h = (input_h + 2 * padding_h - kernel_h) / stride_h + 1;
        int output_w = (input_w + 2 * padding_w - kernel_w) / stride_w + 1;
        
        int total_outputs = batch_size * channels * output_h * output_w;
        int block_size = 256;
        int grid_size = (total_outputs + block_size - 1) / block_size;
        
        max_pool2d_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(input),
            static_cast<float*>(output),
            indices,
            batch_size,
            channels,
            input_h,
            input_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            output_h,
            output_w
        );
    }
}

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
    cudaStream_t stream
) {
    if (precision == Precision::FP32) {
        int output_h = (input_h + 2 * padding_h - kernel_h) / stride_h + 1;
        int output_w = (input_w + 2 * padding_w - kernel_w) / stride_w + 1;
        
        int total_outputs = batch_size * channels * output_h * output_w;
        int block_size = 256;
        int grid_size = (total_outputs + block_size - 1) / block_size;
        
        avg_pool2d_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<const float*>(input),
            static_cast<float*>(output),
            batch_size,
            channels,
            input_h,
            input_w,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            output_h,
            output_w,
            count_include_pad
        );
    }
}

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
    cudaStream_t stream
) {
    // Implementation of im2col
}

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
    cudaStream_t stream
) {
    // Implementation of col2im
}

void grouped_conv2d(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    const ConvConfig& config,
    cudaStream_t stream
) {
    // Grouped convolution implementation
}

void conv_transpose2d(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    const ConvConfig& config,
    cudaStream_t stream
) {
    // Transposed convolution implementation
}

void winograd_conv2d_3x3(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    const ConvConfig& config,
    cudaStream_t stream
) {
    // Winograd convolution for 3x3 kernels
}

void fft_conv2d(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    const ConvConfig& config,
    cudaStream_t stream
) {
    // FFT-based convolution for large kernels
}

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
    cudaStream_t stream
) {
    // Adaptive average pooling
}

void conv2d_backward_data(
    const void* grad_output,
    const void* weight,
    void* grad_input,
    const ConvConfig& config,
    cudaStream_t stream
) {
    // Backward pass for input gradients
}

void conv2d_backward_weight(
    const void* grad_output,
    const void* input,
    void* grad_weight,
    void* grad_bias,
    const ConvConfig& config,
    cudaStream_t stream
) {
    // Backward pass for weight gradients
}

} // namespace kernels
} // namespace cuda_nexus
