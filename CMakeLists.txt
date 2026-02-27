cmake_minimum_required(VERSION 3.28)

project(FAYN
    VERSION 0.1.0
    DESCRIPTION "Forward-only neural network research framework"
    LANGUAGES CXX CUDA
)

# ----------------------------------------------------------------------------
# Standards
# ----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ----------------------------------------------------------------------------
# CUDA architecture: Blackwell GB202 (RTX 5090)
# Requires CUDA Toolkit >= 12.8
# ----------------------------------------------------------------------------
set(CMAKE_CUDA_ARCHITECTURES 120)

# Require CUDA 12.8+
find_package(CUDAToolkit 12.8 REQUIRED)
message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# ----------------------------------------------------------------------------
# Build type
# ----------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# ----------------------------------------------------------------------------
# Compiler flags (guarded by compiler ID to support both GCC and MSVC)
# ----------------------------------------------------------------------------
set(_gcc_like $<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>)

add_compile_options(
    # GCC / Clang C++ warnings
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:${_gcc_like}>>:-Wall>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:${_gcc_like}>>:-Wextra>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:${_gcc_like}>>:-Wpedantic>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:${_gcc_like}>>:-Wno-unused-parameter>
    # MSVC C++ warnings
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W3>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/permissive->
    # CUDA-specific flags (compiler-agnostic)
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# ----------------------------------------------------------------------------
# External dependencies via FetchContent (minimal policy)
# ----------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)

FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        10.2.1
    GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(nlohmann_json fmt)

# ----------------------------------------------------------------------------
# Subdirectories
# ----------------------------------------------------------------------------
add_subdirectory(src)
add_subdirectory(experiments)
add_subdirectory(tools)
