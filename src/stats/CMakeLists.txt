add_library(fayn_stats STATIC
    event_bus.cpp
    activation_stats.cu
)

target_include_directories(fayn_stats PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

target_link_libraries(fayn_stats PUBLIC
    fayn_core
    CUDA::cudart
)

set_target_properties(fayn_stats PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)
