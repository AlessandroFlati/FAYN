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
