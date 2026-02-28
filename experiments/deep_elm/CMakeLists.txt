add_library(fayn_exp_deep_elm STATIC
    deep_elm.cpp
)

target_include_directories(fayn_exp_deep_elm PUBLIC
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(fayn_exp_deep_elm PUBLIC
    fayn_experiments
    fayn_ops
    fmt::fmt
)

set_target_properties(fayn_exp_deep_elm PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)
