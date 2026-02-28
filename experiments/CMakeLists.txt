add_library(fayn_experiments STATIC
    experiment.cpp
)

target_include_directories(fayn_experiments PUBLIC
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(fayn_experiments PUBLIC
    fayn_core
    fayn_stats
    fayn_topology
    fayn_io
)

set_target_properties(fayn_experiments PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

add_subdirectory(hebbian_mnist)
add_subdirectory(ensemble_mnist)
add_subdirectory(deep_elm)
