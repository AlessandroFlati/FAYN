add_library(fayn_exp_ensemble_mnist STATIC
    ensemble_mnist.cpp
)

target_include_directories(fayn_exp_ensemble_mnist PUBLIC
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(fayn_exp_ensemble_mnist PUBLIC
    fayn_experiments
    fayn_ops
    fmt::fmt
)

set_target_properties(fayn_exp_ensemble_mnist PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)
