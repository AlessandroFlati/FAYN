add_library(fayn_cuda STATIC
    stream_pool.cpp
)

target_include_directories(fayn_cuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

target_link_libraries(fayn_cuda PUBLIC
    fayn_core
    CUDA::cudart
)

set_target_properties(fayn_cuda PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
