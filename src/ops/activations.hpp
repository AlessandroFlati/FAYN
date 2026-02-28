#pragma once

#include "../core/tensor.hpp"
#include "../core/layer.hpp"

#include <cuda_runtime.h>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace fayn {

// ---------------------------------------------------------------------------
// Activation function type tag (used for dispatch and serialization).
// ---------------------------------------------------------------------------
enum class ActivationType : uint8_t {
    ReLU       = 0,
    LeakyReLU  = 1,
    GELU       = 2,
    SiLU       = 3,
    Tanh       = 4,
    Sigmoid    = 5,
    Custom     = 255,
};

inline const char* activation_name(ActivationType t) {
    switch (t) {
        case ActivationType::ReLU:      return "relu";
        case ActivationType::LeakyReLU: return "leaky_relu";
        case ActivationType::GELU:      return "gelu";
        case ActivationType::SiLU:      return "silu";
        case ActivationType::Tanh:      return "tanh";
        case ActivationType::Sigmoid:   return "sigmoid";
        case ActivationType::Custom:    return "custom";
    }
    throw std::invalid_argument("Unknown ActivationType");
}

// ---------------------------------------------------------------------------
// Device activation dispatch.
// All functions operate in-place on 'x' (BF16 or FP16 on device).
// 'stream' may be nullptr (uses default stream).
// ---------------------------------------------------------------------------
void apply_relu              (Tensor& x, cudaStream_t stream = nullptr);
void apply_leaky_relu        (Tensor& x, float alpha = 0.01f, cudaStream_t stream = nullptr);
void apply_leaky_relu_inverse(Tensor& x, float alpha = 0.01f, cudaStream_t stream = nullptr);
void apply_gelu              (Tensor& x, cudaStream_t stream = nullptr);
void apply_silu              (Tensor& x, cudaStream_t stream = nullptr);
void apply_tanh              (Tensor& x, cudaStream_t stream = nullptr);
void apply_sigmoid           (Tensor& x, cudaStream_t stream = nullptr);

// ---------------------------------------------------------------------------
// ADMM utility kernels (FP32, CUDA only).
// ---------------------------------------------------------------------------

// Fused Z-update for one hidden layer:
//   Z[i] = (rho*(A[i] - u[i]) + mu*leaky_relu_inv(T_target[i])) / (rho + mu)
// All tensors must be Float32 on CUDA with identical shape.
void admm_z_update(Tensor& Z, const Tensor& A, const Tensor& u,
                   const Tensor& T_target, float rho, float mu,
                   float leaky_alpha, cudaStream_t stream = nullptr);

// Dual update: u[i] += A[i] - Z[i]
// All tensors must be Float32 on CUDA with identical shape.
void admm_dual_update(Tensor& u, const Tensor& A, const Tensor& Z,
                      cudaStream_t stream = nullptr);

// ---------------------------------------------------------------------------
// Custom activation plugin.
// Users register a named kernel via ActivationRegistry.
// ---------------------------------------------------------------------------
using ActivationKernelFn = std::function<void(Tensor&, cudaStream_t)>;

class ActivationRegistry {
public:
    static ActivationRegistry& instance();

    void register_activation(const std::string& name, ActivationKernelFn fn);
    void apply(const std::string& name, Tensor& x, cudaStream_t stream = nullptr) const;
    bool has(const std::string& name) const;

private:
    std::unordered_map<std::string, ActivationKernelFn> registry_;
};

// ---------------------------------------------------------------------------
// ActivationLayer: a standalone layer wrapping an activation function.
// Can be inserted as a graph node.
// ---------------------------------------------------------------------------
// Factory functions — return a Layer-compatible shared_ptr usable as a graph node.
LayerPtr make_activation_layer(ActivationType type, float param = 0.01f);
LayerPtr make_custom_activation_layer(const std::string& name);

} // namespace fayn
