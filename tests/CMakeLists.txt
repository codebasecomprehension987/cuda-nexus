cmake_minimum_required(VERSION 3.18)

# Find Google Test (optional)
find_package(GTest)

if(GTest_FOUND)
    # GEMM tests
    add_executable(test_gemm test_gemm.cu)
    target_link_libraries(test_gemm cuda_nexus GTest::GTest GTest::Main)
    add_test(NAME test_gemm COMMAND test_gemm)
    
    # Attention tests
    add_executable(test_attention test_attention.cu)
    target_link_libraries(test_attention cuda_nexus GTest::GTest GTest::Main)
    add_test(NAME test_attention COMMAND test_attention)
    
    # Reduction tests
    add_executable(test_reduction test_reduction.cu)
    target_link_libraries(test_reduction cuda_nexus GTest::GTest GTest::Main)
    add_test(NAME test_reduction COMMAND test_reduction)
    
    # Memory operations tests
    add_executable(test_memory_ops test_memory_ops.cu)
    target_link_libraries(test_memory_ops cuda_nexus GTest::GTest GTest::Main)
    add_test(NAME test_memory_ops COMMAND test_memory_ops)
else()
    message(WARNING "Google Test not found. Tests will not be built.")
endif()
