cmake_minimum_required(VERSION 3.18)

# GEMM example
add_executable(gemm_example gemm_example.cu)
target_link_libraries(gemm_example cuda_nexus)
set_target_properties(gemm_example PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Attention example
add_executable(attention_example attention_example.cu)
target_link_libraries(attention_example cuda_nexus)
set_target_properties(attention_example PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Install examples
install(TARGETS gemm_example attention_example DESTINATION bin/examples)
