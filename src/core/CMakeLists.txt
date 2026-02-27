add_library(fayn_core STATIC
    tensor.cpp
    graph.cpp
)

target_include_directories(fayn_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

target_link_libraries(fayn_core PUBLIC
    CUDA::cudart
)

set_target_properties(fayn_core PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
