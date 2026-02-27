add_library(fayn_io STATIC
    mnist_loader.cpp
    logger.cpp
)

target_include_directories(fayn_io PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

target_link_libraries(fayn_io PUBLIC
    fayn_core
    fayn_stats
    CUDA::cudart          # for cuda_bf16.h / cuda_fp16.h in mnist_loader
    nlohmann_json::nlohmann_json
)

set_target_properties(fayn_io PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)
