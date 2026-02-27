#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace fayn {

// ---------------------------------------------------------------------------
// Device tag
// ---------------------------------------------------------------------------
enum class Device : uint8_t {
    CPU  = 0,
    CUDA = 1,
};

inline const char* device_name(Device d) noexcept {
    return d == Device::CPU ? "cpu" : "cuda";
}

// ---------------------------------------------------------------------------
// CUDA error handling
// ---------------------------------------------------------------------------
struct CudaError : std::runtime_error {
    explicit CudaError(cudaError_t err, const char* file, int line)
        : std::runtime_error(
              std::string(cudaGetErrorString(err)) +
              " [" + file + ":" + std::to_string(line) + "]")
    {}
};

#define FAYN_CUDA_CHECK(expr)                                          \
    do {                                                               \
        cudaError_t _fayn_err = (expr);                                \
        if (_fayn_err != cudaSuccess)                                  \
            throw fayn::CudaError(_fayn_err, __FILE__, __LINE__);      \
    } while (0)

#define FAYN_CUBLAS_CHECK(expr)                                        \
    do {                                                               \
        cublasStatus_t _fayn_st = (expr);                              \
        if (_fayn_st != CUBLAS_STATUS_SUCCESS)                         \
            throw std::runtime_error(                                  \
                "cuBLAS error " + std::to_string(static_cast<int>(_fayn_st)) + \
                " [" + __FILE__ + ":" + std::to_string(__LINE__) + "]"); \
    } while (0)

} // namespace fayn
