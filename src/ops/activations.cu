#include "activations.hpp"
#include "../core/layer.hpp"
#include "../stats/activation_stats.hpp"
#include "../stats/event_bus.hpp"
#include "../cuda/stream_pool.hpp"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <memory>

namespace fayn {

// ---------------------------------------------------------------------------
// Templated element-wise kernels
// ---------------------------------------------------------------------------

template<typename T>
__device__ __forceinline__ T relu_fn(T x) {
    return x > T(0) ? x : T(0);
}

template<typename T>
__device__ __forceinline__ T leaky_relu_fn(T x, float alpha) {
    return x > T(0) ? x : T(alpha) * x;
}

template<typename T>
__device__ __forceinline__ T leaky_relu_inv_fn(T x, float alpha) {
    // Inverse of leaky_relu: y<0 → y/alpha  (alpha must be != 0)
    return x >= T(0) ? x : x / T(alpha);
}

// GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
template<typename T>
__device__ __forceinline__ T gelu_fn(T x) {
    float xf = static_cast<float>(x);
    float t  = tanhf(0.7978845608f * (xf + 0.044715f * xf * xf * xf));
    return T(xf * 0.5f * (1.0f + t));
}

// SiLU: x * sigmoid(x)
template<typename T>
__device__ __forceinline__ T silu_fn(T x) {
    float xf = static_cast<float>(x);
    return T(xf / (1.0f + expf(-xf)));
}

template<typename T>
__device__ __forceinline__ T tanh_fn(T x) {
    return T(tanhf(static_cast<float>(x)));
}

template<typename T>
__device__ __forceinline__ T sigmoid_fn(T x) {
    return T(1.0f / (1.0f + expf(-static_cast<float>(x))));
}

// ---------------------------------------------------------------------------
// Generic in-place activation kernel
// ---------------------------------------------------------------------------
template<typename T, typename Fn>
__global__ void inplace_activation_kernel(T* data, size_t n, Fn fn) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = fn(data[idx]);
    }
}

template<typename T, typename Fn>
static void launch_activation(T* data, size_t n, cudaStream_t stream, Fn fn) {
    constexpr int BLOCK = 256;
    int grid = static_cast<int>((n + BLOCK - 1) / BLOCK);
    inplace_activation_kernel<<<grid, BLOCK, 0, stream>>>(data, n, fn);
    FAYN_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Dispatch helpers: select T based on tensor DType
// ---------------------------------------------------------------------------
template<typename Fn16, typename FnBf16, typename Fn32>
static void dispatch_by_dtype(Tensor& x, cudaStream_t stream,
                               Fn16 fn16, FnBf16 fnbf16, Fn32 fn32) {
    if (x.device != Device::CUDA)
        throw std::runtime_error("Activation kernels require a CUDA tensor");
    const size_t n = x.numel();
    switch (x.dtype) {
        case DType::Float16:
            launch_activation(static_cast<__half*>(x.data), n, stream, fn16);
            break;
        case DType::BFloat16:
            launch_activation(static_cast<__nv_bfloat16*>(x.data), n, stream, fnbf16);
            break;
        case DType::Float32:
            launch_activation(static_cast<float*>(x.data), n, stream, fn32);
            break;
        default:
            throw std::runtime_error("Activation: unsupported dtype");
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
void apply_relu(Tensor& x, cudaStream_t stream) {
    dispatch_by_dtype(x, stream,
        []__device__(__half v)           { return relu_fn(v); },
        []__device__(__nv_bfloat16 v)    { return relu_fn(v); },
        []__device__(float v)            { return relu_fn(v); });
}

void apply_leaky_relu(Tensor& x, float alpha, cudaStream_t stream) {
    dispatch_by_dtype(x, stream,
        [alpha]__device__(__half v)          { return leaky_relu_fn(v, alpha); },
        [alpha]__device__(__nv_bfloat16 v)   { return leaky_relu_fn(v, alpha); },
        [alpha]__device__(float v)           { return leaky_relu_fn(v, alpha); });
}

void apply_leaky_relu_inverse(Tensor& x, float alpha, cudaStream_t stream) {
    if (alpha == 0.f)
        throw std::invalid_argument("apply_leaky_relu_inverse: alpha must be non-zero");
    dispatch_by_dtype(x, stream,
        [alpha]__device__(__half v)          { return leaky_relu_inv_fn(v, alpha); },
        [alpha]__device__(__nv_bfloat16 v)   { return leaky_relu_inv_fn(v, alpha); },
        [alpha]__device__(float v)           { return leaky_relu_inv_fn(v, alpha); });
}

// ---------------------------------------------------------------------------
// ADMM FP32-only kernels
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// admm_z_update_kernel: element-wise exact minimizer of the ADMM Z-subproblem
//
// min_z  (rho/2)(z - c)^2  +  (mu/2)(LeakyReLU(z) - t)^2
//
// where c = A[i] - u[i]  (ADMM-corrected bottom-up pre-activation)
//       t = T_target[i]  (post-activation target from top-down propagation)
//
// Closed-form case analysis (LeakyReLU slope = leaky_alpha):
//   Case z >= 0: sigma(z) = z   → z1 = (rho*c + mu*t) / (rho+mu)
//   Case z <  0: sigma(z) = a*z → z2 = (rho*c + mu*a*t) / (rho + mu*a^2)
//
//   Decision:
//     z1 >= 0 → use z1  (the z>=0 branch minimizer is in the z>=0 region)
//     z1 <  0 → check z<0 branch:
//       z2 <  0 → use z2
//       z2 >= 0 → use 0  (boundary: both branch minimizers infeasible, f is V-shaped at 0)
//
// This avoids the 1/alpha amplification of the old linear blend.
// ---------------------------------------------------------------------------
__global__ void admm_z_update_kernel(
    float* __restrict__       Z,
    const float* __restrict__ A,
    const float* __restrict__ u,
    const float* __restrict__ T_target,
    float rho, float mu, float inv_rho_mu, float leaky_alpha,
    size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float c  = A[i] - u[i];
    const float t  = T_target[i];
    const float z1 = (rho * c + mu * t) * inv_rho_mu;
    if (z1 >= 0.f) {
        Z[i] = z1;
    } else {
        const float z2 = (rho * c + mu * leaky_alpha * t)
                       / (rho + mu * leaky_alpha * leaky_alpha);
        Z[i] = (z2 < 0.f) ? z2 : 0.f;
    }
}

__global__ void admm_dual_update_kernel(
    float* __restrict__       u,
    const float* __restrict__ A,
    const float* __restrict__ Z,
    size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u[i] += A[i] - Z[i];
}

void admm_z_update(Tensor& Z, const Tensor& A, const Tensor& u,
                   const Tensor& T_target, float rho, float mu,
                   float leaky_alpha, cudaStream_t stream)
{
    if (Z.dtype != DType::Float32 || A.dtype != DType::Float32 ||
        u.dtype != DType::Float32 || T_target.dtype != DType::Float32)
        throw std::runtime_error("admm_z_update: all tensors must be Float32");
    if (Z.device != Device::CUDA)
        throw std::runtime_error("admm_z_update: tensors must be on CUDA");
    const size_t n = Z.numel();
    constexpr int BLOCK = 256;
    int grid = static_cast<int>((n + BLOCK - 1) / BLOCK);
    float inv_total = 1.f / (rho + mu);
    admm_z_update_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<float*>(Z.data),
        static_cast<const float*>(A.data),
        static_cast<const float*>(u.data),
        static_cast<const float*>(T_target.data),
        rho, mu, inv_total, leaky_alpha, n);
    FAYN_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// admm_z_update_tanh_kernel: element-wise proximal ALS Z-update for tanh.
//
// min_z  (rho/2)(z - A[i])^2  +  (mu/2)(z - atanh(T_target[i]))^2
//
// Approximates the true tanh subproblem by working in pre-activation space:
// the residual (tanh(z) - T)^2 is replaced by (z - atanh(T))^2.
// Exact solution: z = (rho * A + mu * atanh(clamp(T, -1+eps, 1-eps))) / (rho + mu)
//
// atanh(0.9999) ≈ 4.6 — limits pre-activation target magnitude so tanh
// does not saturate excessively on the first few iterations.
// ---------------------------------------------------------------------------
__global__ void admm_z_update_tanh_kernel(
    float* __restrict__       Z,
    const float* __restrict__ A,
    const float* __restrict__ T_target,
    float rho, float mu, float inv_rho_mu,
    size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // eps=0.1: atanh(0.9) ≈ 1.47, keeping pre-activation Z bounded and
    // preventing tanh saturation that makes H_k near-constant (degenerate solve).
    constexpr float eps = 0.1f;
    float t = fminf(fmaxf(T_target[i], -1.f + eps), 1.f - eps);
    Z[i] = (rho * A[i] + mu * atanhf(t)) * inv_rho_mu;
}

void admm_z_update_tanh(Tensor& Z, const Tensor& A, const Tensor& T_target,
                         float rho, float mu, cudaStream_t stream)
{
    if (Z.dtype != DType::Float32 || A.dtype != DType::Float32 ||
        T_target.dtype != DType::Float32)
        throw std::runtime_error("admm_z_update_tanh: all tensors must be Float32");
    if (Z.device != Device::CUDA)
        throw std::runtime_error("admm_z_update_tanh: tensors must be on CUDA");
    const size_t n = Z.numel();
    constexpr int BLOCK = 256;
    int grid = static_cast<int>((n + BLOCK - 1) / BLOCK);
    float inv_total = 1.f / (rho + mu);
    admm_z_update_tanh_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<float*>(Z.data),
        static_cast<const float*>(A.data),
        static_cast<const float*>(T_target.data),
        rho, mu, inv_total, n);
    FAYN_CUDA_CHECK(cudaGetLastError());
}

void admm_dual_update(Tensor& u, const Tensor& A, const Tensor& Z,
                      cudaStream_t stream)
{
    if (u.dtype != DType::Float32 || A.dtype != DType::Float32 ||
        Z.dtype != DType::Float32)
        throw std::runtime_error("admm_dual_update: all tensors must be Float32");
    if (u.device != Device::CUDA)
        throw std::runtime_error("admm_dual_update: tensors must be on CUDA");
    const size_t n = u.numel();
    constexpr int BLOCK = 256;
    int grid = static_cast<int>((n + BLOCK - 1) / BLOCK);
    admm_dual_update_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<float*>(u.data),
        static_cast<const float*>(A.data),
        static_cast<const float*>(Z.data),
        n);
    FAYN_CUDA_CHECK(cudaGetLastError());
}

void apply_gelu(Tensor& x, cudaStream_t stream) {
    dispatch_by_dtype(x, stream,
        []__device__(__half v)           { return gelu_fn(v); },
        []__device__(__nv_bfloat16 v)    { return gelu_fn(v); },
        []__device__(float v)            { return gelu_fn(v); });
}

void apply_silu(Tensor& x, cudaStream_t stream) {
    dispatch_by_dtype(x, stream,
        []__device__(__half v)           { return silu_fn(v); },
        []__device__(__nv_bfloat16 v)    { return silu_fn(v); },
        []__device__(float v)            { return silu_fn(v); });
}

void apply_tanh(Tensor& x, cudaStream_t stream) {
    dispatch_by_dtype(x, stream,
        []__device__(__half v)           { return tanh_fn(v); },
        []__device__(__nv_bfloat16 v)    { return tanh_fn(v); },
        []__device__(float v)            { return tanh_fn(v); });
}

void apply_sigmoid(Tensor& x, cudaStream_t stream) {
    dispatch_by_dtype(x, stream,
        []__device__(__half v)           { return sigmoid_fn(v); },
        []__device__(__nv_bfloat16 v)    { return sigmoid_fn(v); },
        []__device__(float v)            { return sigmoid_fn(v); });
}

// ---------------------------------------------------------------------------
// ActivationRegistry
// ---------------------------------------------------------------------------
ActivationRegistry& ActivationRegistry::instance() {
    static ActivationRegistry reg;
    return reg;
}

void ActivationRegistry::register_activation(const std::string& name, ActivationKernelFn fn) {
    if (registry_.count(name))
        throw std::runtime_error("ActivationRegistry: '" + name + "' already registered");
    registry_[name] = std::move(fn);
}

void ActivationRegistry::apply(const std::string& name, Tensor& x, cudaStream_t stream) const {
    auto it = registry_.find(name);
    if (it == registry_.end())
        throw std::runtime_error("ActivationRegistry: unknown activation '" + name + "'");
    it->second(x, stream);
}

bool ActivationRegistry::has(const std::string& name) const {
    return registry_.count(name) > 0;
}

// ---------------------------------------------------------------------------
// ActivationLayer: graph node wrapping an activation function.
// ---------------------------------------------------------------------------
class ActivationLayer : public Layer {
public:
    explicit ActivationLayer(ActivationType type, float param = 0.01f)
        : type_(type), param_(param)
    {
        name_ = std::string(activation_name(type));
    }

    explicit ActivationLayer(std::string custom_name)
        : type_(ActivationType::Custom), custom_name_(std::move(custom_name))
    {
        name_ = "custom_" + custom_name_;
    }

    Tensor forward(const Tensor& x) override {
        // Activation is in-place; we work on a copy to avoid modifying input.
        Tensor out = x.to(x.device);
        StreamPool::Guard guard;
        switch (type_) {
            case ActivationType::ReLU:      apply_relu(out, guard.stream());       break;
            case ActivationType::LeakyReLU: apply_leaky_relu(out, param_, guard.stream()); break;
            case ActivationType::GELU:      apply_gelu(out, guard.stream());       break;
            case ActivationType::SiLU:      apply_silu(out, guard.stream());       break;
            case ActivationType::Tanh:      apply_tanh(out, guard.stream());       break;
            case ActivationType::Sigmoid:   apply_sigmoid(out, guard.stream());    break;
            case ActivationType::Custom:
                ActivationRegistry::instance().apply(custom_name_, out, guard.stream());
                break;
        }
        // Stats kernel synchronises the stream internally.
        // When stats are disabled, still sync the stream so subsequent
        // layers can safely read this layer's output.
        if (compute_stats_) {
            StatsSnapshot snap = layer_stats_.compute_and_snapshot(id_, next_step(), out, guard.stream());

            ActivationEvent ev;
            ev.layer_id = snap.layer_id;
            ev.step     = snap.step;
            ev.stats    = std::move(snap);
            EventBus::instance().emit(ev);
        } else {
            FAYN_CUDA_CHECK(cudaStreamSynchronize(guard.stream()));
        }

        return out;
    }

    void set_compute_stats(bool v) noexcept override { compute_stats_ = v; }

    std::string name()        const override { return name_; }
    size_t      output_size() const override { return 0; }  // passthrough
    size_t      num_params()  const override { return 0; }

private:
    ActivationType type_;
    float          param_ = 0.01f;
    std::string    custom_name_;
    std::string    name_;
    LayerStats     layer_stats_;   // lazy-init on first forward()
    bool           compute_stats_ = true;
};

// ---------------------------------------------------------------------------
// Factory functions (public API)
// ---------------------------------------------------------------------------
LayerPtr make_activation_layer(ActivationType type, float param) {
    return std::make_shared<ActivationLayer>(type, param);
}

LayerPtr make_custom_activation_layer(const std::string& name) {
    return std::make_shared<ActivationLayer>(name);
}

} // namespace fayn
