#include "cuda_nexus.h"
#include <gtest/gtest.h>
#include <vector>
#include <random>

using namespace cuda_nexus;

class GEMMTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize random number generator
        gen.seed(42);
    }
    
    std::mt19937 gen;
};

TEST_F(GEMMTest, SmallMatrixMultiplication) {
    const int M = 64;
    const int N = 64;
    const int K = 64;
    
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    // Initialize matrices
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_C_expected(M * N, 0.0f);
    
    for (auto& val : h_A) val = dis(gen);
    for (auto& val : h_B) val = dis(gen);
    
    // Compute expected result on CPU
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += h_A[i * K + k] * h_B[k * N + j];
            }
            h_C_expected[i * N + j] = sum;
        }
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Configure and run GEMM
    kernels::GEMMConfig config;
    config.M = M;
    config.N = N;
    config.K = K;
    config.precision = Precision::FP32;
    config.alpha = 1.0f;
    config.beta = 0.0f;
    
    kernels::gemm(d_A, d_B, d_C, config, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify results
    const float epsilon = 1e-3f;
    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(h_C[i], h_C_expected[i], epsilon) 
            << "Mismatch at index " << i;
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

TEST_F(GEMMTest, LargeMatrixMultiplication) {
    const int M = 512;
    const int N = 512;
    const int K = 512;
    
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);
    
    for (auto& val : h_A) val = dis(gen);
    for (auto& val : h_B) val = dis(gen);
    
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    kernels::GEMMConfig config;
    config.M = M;
    config.N = N;
    config.K = K;
    config.precision = Precision::FP32;
    config.alpha = 1.0f;
    config.beta = 0.0f;
    
    kernels::gemm(d_A, d_B, d_C, config, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Spot check a few values
    const float epsilon = 1e-2f;
    for (int check = 0; check < 10; ++check) {
        int i = check * 50;
        int j = check * 50;
        
        float expected = 0.0f;
        for (int k = 0; k < K; ++k) {
            expected += h_A[i * K + k] * h_B[k * N + j];
        }
        
        EXPECT_NEAR(h_C[i * N + j], expected, epsilon);
    }
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

TEST_F(GEMMTest, AlphaBetaScaling) {
    const int M = 128;
    const int N = 128;
    const int K = 128;
    const float alpha = 2.0f;
    const float beta = 0.5f;
    
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);
    std::vector<float> h_C_initial(M * N);
    
    for (auto& val : h_A) val = dis(gen);
    for (auto& val : h_B) val = dis(gen);
    for (auto& val : h_C_initial) val = dis(gen);
    
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C_initial.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    kernels::GEMMConfig config;
    config.M = M;
    config.N = N;
    config.K = K;
    config.precision = Precision::FP32;
    config.alpha = alpha;
    config.beta = beta;
    
    kernels::gemm(d_A, d_B, d_C, config, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify: C = alpha * A * B + beta * C_initial
    const float epsilon = 1e-2f;
    for (int check = 0; check < 5; ++check) {
        int i = check * 25;
        int j = check * 25;
        
        float gemm_result = 0.0f;
        for (int k = 0; k < K; ++k) {
            gemm_result += h_A[i * K + k] * h_B[k * N + j];
        }
        
        float expected = alpha * gemm_result + beta * h_C_initial[i * N + j];
        EXPECT_NEAR(h_C[i * N + j], expected, epsilon);
    }
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
