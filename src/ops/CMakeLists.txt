add_library(fayn_ops STATIC
    activations.cu
    dense.cu
    hebbian.cu
    one_hot.cu
)

target_include_directories(fayn_ops PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

target_link_libraries(fayn_ops PUBLIC
    fayn_core
    fayn_stats
    fayn_cuda
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
)

set_target_properties(fayn_ops PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)
