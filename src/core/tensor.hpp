#pragma once

#include "device.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <stdexcept>

namespace fayn {

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------
enum class DType : uint8_t {
    Float32  = 0,
    Float16  = 1,
    BFloat16 = 2,
    Int32    = 3,
    Int64    = 4,
};

// Device-safe size lookup (no throw — returns 0 for unknown types).
__host__ __device__ inline size_t dtype_bytes(DType dt) noexcept {
    switch (dt) {
        case DType::Float32:  return 4;
        case DType::Float16:  return 2;
        case DType::BFloat16: return 2;
        case DType::Int32:    return 4;
        case DType::Int64:    return 8;
        default:              return 0;
    }
}

inline const char* dtype_name(DType dt) {
    switch (dt) {
        case DType::Float32:  return "float32";
        case DType::Float16:  return "float16";
        case DType::BFloat16: return "bfloat16";
        case DType::Int32:    return "int32";
        case DType::Int64:    return "int64";
    }
    throw std::invalid_argument("Unknown DType");
}

// ---------------------------------------------------------------------------
// Maximum number of dimensions supported by CUDA kernels via TensorView.
// ---------------------------------------------------------------------------
static constexpr int FAYN_MAX_DIMS = 8;

// ---------------------------------------------------------------------------
// TensorView: POD struct safe to pass into CUDA kernels.
// Shape and strides are measured in elements.
// ---------------------------------------------------------------------------
struct TensorView {
    void*  data                   = nullptr;
    size_t shape[FAYN_MAX_DIMS]   = {};
    size_t strides[FAYN_MAX_DIMS] = {};
    int    ndim                   = 0;
    DType  dtype                  = DType::Float32;

    __host__ __device__ size_t numel() const noexcept {
        size_t n = 1;
        for (int i = 0; i < ndim; ++i) n *= shape[i];
        return n;
    }

    // Linear index -> element pointer (no bounds check in device code).
    __host__ __device__ void* ptr_at_linear(size_t idx) const noexcept {
        // For a contiguous row-major tensor this is just data + idx * elem_bytes.
        // For strided tensors we compute the full nd offset.
        size_t offset = 0;
        for (int i = ndim - 1; i >= 0; --i) {
            size_t dim_idx = idx % shape[i];
            offset += dim_idx * strides[i];
            idx /= shape[i];
        }
        return static_cast<uint8_t*>(data) + offset * dtype_bytes(dtype);
    }
};

// ---------------------------------------------------------------------------
// Tensor: owning, RAII, host or device buffer.
// Non-copyable; use Tensor::view() for lightweight non-owning access.
// ---------------------------------------------------------------------------
struct Tensor {
    void*               data    = nullptr;
    std::vector<size_t> shape;
    std::vector<size_t> strides;   // in elements
    DType               dtype   = DType::BFloat16;
    Device              device  = Device::CPU;

    Tensor() = default;
    ~Tensor();

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;

    // Allocate a contiguous C-order (row-major) tensor.
    static Tensor make(std::vector<size_t> shape, DType dtype, Device device);

    // Total number of elements.
    size_t numel() const noexcept;

    // Size in bytes.
    size_t nbytes() const noexcept;

    // True if strides are contiguous C-order.
    bool is_contiguous() const noexcept;

    // Create a POD TensorView (safe to pass to CUDA kernels).
    TensorView view() const noexcept;

    // Return a new tensor on the target device (copies data if needed).
    Tensor to(Device target) const;

    // Return a contiguous copy (noop if already contiguous).
    Tensor contiguous() const;

private:
    void free_data();
};

// Compute C-order strides for a given shape.
std::vector<size_t> c_order_strides(const std::vector<size_t>& shape);

} // namespace fayn
