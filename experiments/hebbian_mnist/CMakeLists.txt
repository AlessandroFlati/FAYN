add_library(fayn_exp_hebbian_mnist STATIC
    hebbian_mnist.cpp
)

target_include_directories(fayn_exp_hebbian_mnist PUBLIC
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(fayn_exp_hebbian_mnist PUBLIC
    fayn_experiments
    fayn_ops
    fmt::fmt
)

set_target_properties(fayn_exp_hebbian_mnist PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
