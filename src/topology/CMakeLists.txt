add_library(fayn_topology STATIC
    mutation_engine.cpp
)

target_include_directories(fayn_topology PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

target_link_libraries(fayn_topology PUBLIC
    fayn_core
    fayn_stats
)

set_target_properties(fayn_topology PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)
