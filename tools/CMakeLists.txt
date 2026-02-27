add_executable(fayn_runner
    runner.cpp
)

target_include_directories(fayn_runner PRIVATE
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(fayn_runner PRIVATE
    fayn_exp_hebbian_mnist
    fayn_experiments
    fayn_ops
    fayn_topology
    fayn_io
    fayn_stats
    fayn_core
    fayn_cuda
    fmt::fmt
)

set_target_properties(fayn_runner PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME "fayn"
)

# ---------------------------------------------------------------------------
# Smoke test: exercises all major subsystems without external data.
# Run with: ./fayn_smoke (or fayn_smoke.exe on Windows)
# ---------------------------------------------------------------------------
add_executable(fayn_smoke
    smoke_test.cu
)

target_include_directories(fayn_smoke PRIVATE
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(fayn_smoke PRIVATE
    fayn_ops
    fayn_stats
    fayn_core
    fayn_cuda
    CUDA::cudart
)

set_target_properties(fayn_smoke PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
)
