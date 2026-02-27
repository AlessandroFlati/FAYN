#include "tensor.hpp"

#include <cstring>
#include <stdexcept>
#include <numeric>

#include <cuda_runtime.h>

namespace fayn {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
std::vector<size_t> c_order_strides(const std::vector<size_t>& shape) {
    const int ndim = static_cast<int>(shape.size());
    std::vector<size_t> strides(ndim, 1);
    for (int i = ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

// ---------------------------------------------------------------------------
// Tensor static factory
// ---------------------------------------------------------------------------
Tensor Tensor::make(std::vector<size_t> shape, DType dtype, Device device) {
    Tensor t;
    t.shape   = shape;
    t.strides = c_order_strides(shape);
    t.dtype   = dtype;
    t.device  = device;

    if (dtype_bytes(dtype) == 0)
        throw std::invalid_argument("Tensor::make: unknown DType");
    const size_t bytes = t.nbytes();
    if (bytes == 0) {
        t.data = nullptr;
        return t;
    }

    if (device == Device::CPU) {
        t.data = ::operator new(bytes);
        std::memset(t.data, 0, bytes);
    } else {
        FAYN_CUDA_CHECK(cudaMalloc(&t.data, bytes));
        FAYN_CUDA_CHECK(cudaMemset(t.data, 0, bytes));
    }
    return t;
}

// ---------------------------------------------------------------------------
// Destructor / move
// ---------------------------------------------------------------------------
void Tensor::free_data() {
    if (!data) return;
    if (device == Device::CPU) {
        ::operator delete(data);
    } else {
        // Ignore CUDA error during destruction to avoid throwing in destructor.
        cudaFree(data);
    }
    data = nullptr;
}

Tensor::~Tensor() {
    if (owned) free_data();
}

Tensor::Tensor(Tensor&& other) noexcept
    : data(other.data)
    , shape(std::move(other.shape))
    , strides(std::move(other.strides))
    , dtype(other.dtype)
    , device(other.device)
    , owned(other.owned)
{
    other.data  = nullptr;
    other.owned = true;  // moved-from: no data to free, flag reset
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) return *this;
    if (owned) free_data();
    data    = other.data;
    shape   = std::move(other.shape);
    strides = std::move(other.strides);
    dtype   = other.dtype;
    device  = other.device;
    owned   = other.owned;
    other.data  = nullptr;
    other.owned = true;
    return *this;
}

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------
size_t Tensor::numel() const noexcept {
    if (shape.empty()) return 0;
    size_t n = 1;
    for (size_t s : shape) n *= s;
    return n;
}

size_t Tensor::nbytes() const noexcept {
    return numel() * dtype_bytes(dtype);
}

bool Tensor::is_contiguous() const noexcept {
    auto expected = c_order_strides(shape);
    return strides == expected;
}

TensorView Tensor::view() const noexcept {
    TensorView v;
    v.data  = data;
    v.ndim  = static_cast<int>(shape.size());
    v.dtype = dtype;
    for (int i = 0; i < v.ndim && i < FAYN_MAX_DIMS; ++i) {
        v.shape[i]   = shape[i];
        v.strides[i] = strides[i];
    }
    return v;
}

// ---------------------------------------------------------------------------
// Device transfer
// ---------------------------------------------------------------------------
Tensor Tensor::to(Device target) const {
    if (device == target) {
        // Return a fresh copy on the same device.
        Tensor dst = Tensor::make(shape, dtype, target);
        if (target == Device::CPU) {
            std::memcpy(dst.data, data, nbytes());
        } else {
            FAYN_CUDA_CHECK(cudaMemcpy(dst.data, data, nbytes(), cudaMemcpyDeviceToDevice));
        }
        return dst;
    }

    Tensor dst = Tensor::make(shape, dtype, target);
    if (target == Device::CUDA) {
        FAYN_CUDA_CHECK(cudaMemcpy(dst.data, data, nbytes(), cudaMemcpyHostToDevice));
    } else {
        FAYN_CUDA_CHECK(cudaMemcpy(dst.data, data, nbytes(), cudaMemcpyDeviceToHost));
    }
    return dst;
}

Tensor Tensor::borrow() const noexcept {
    Tensor t;
    t.data    = data;
    t.shape   = shape;
    t.strides = strides;
    t.dtype   = dtype;
    t.device  = device;
    t.owned   = false;
    return t;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) return to(device);
    // Allocate and scatter-gather via CPU for now.
    // TODO: implement a CUDA gather kernel for on-device contiguous copy.
    throw std::runtime_error("contiguous() for non-contiguous strided tensors not yet implemented");
}

} // namespace fayn
