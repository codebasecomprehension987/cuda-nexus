cmake_minimum_required(VERSION 3.18)

include(FetchContent)

# Google Test
if(BUILD_TESTS)
    message(STATUS "Fetching Google Test...")
    
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
        GIT_SHALLOW    TRUE
    )
    
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    
    FetchContent_MakeAvailable(googletest)
    
    message(STATUS "Google Test fetched successfully")
endif()

# Google Benchmark (optional)
if(BUILD_BENCHMARKS_GBENCH)
    message(STATUS "Fetching Google Benchmark...")
    
    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.8.3
        GIT_SHALLOW    TRUE
    )
    
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    
    FetchContent_MakeAvailable(googlebenchmark)
    
    message(STATUS "Google Benchmark fetched successfully")
endif()

# JSON library (optional, for configuration files)
option(USE_JSON "Use nlohmann/json for config files" OFF)
if(USE_JSON)
    message(STATUS "Fetching nlohmann/json...")
    
    FetchContent_Declare(
        json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        v3.11.3
        GIT_SHALLOW    TRUE
    )
    
    FetchContent_MakeAvailable(json)
    
    message(STATUS "nlohmann/json fetched successfully")
endif()

# spdlog (optional, for logging)
option(USE_SPDLOG "Use spdlog for logging" OFF)
if(USE_SPDLOG)
    message(STATUS "Fetching spdlog...")
    
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG        v1.12.0
        GIT_SHALLOW    TRUE
    )
    
    FetchContent_MakeAvailable(spdlog)
    
    message(STATUS "spdlog fetched successfully")
endif()
