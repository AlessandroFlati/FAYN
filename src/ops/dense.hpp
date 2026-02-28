#pragma once

#include "../core/layer.hpp"
#include "../core/tensor.hpp"
#include "../stats/activation_stats.hpp"

#include <cublas_v2.h>
#include <memory>
#include <string>

namespace fayn {

// ---------------------------------------------------------------------------
// DenseLayer: fully-connected y = x W^T + b
//
// Weights:  [out_features, in_features]  BFloat16
// Bias:     [out_features]               Float32
// Input:    [batch_size,   in_features]  BFloat16
// Output:   [batch_size,   out_features] BFloat16
//
// Matrix multiply uses cuBLASLt with CUBLAS_COMPUTE_32F for precision.
// Bias add is a custom CUDA kernel.
// ---------------------------------------------------------------------------
class DenseLayer : public Layer {
public:
    // init_scale: multiplier on the Kaiming uniform range for the weight
    // initialisation of this layer.  Default (1.0) is standard Kaiming.
    // Set > 1 (e.g. sqrt(fan_in/2) ≈ 19.8 for fan_in=784) to approximate
    // the Uniform(-1,1) initialisation typical in ELM literature.
    DenseLayer(size_t in_features, size_t out_features,
               bool use_bias = true, float init_scale = 1.0f);
    ~DenseLayer() override;

    Tensor forward(const Tensor& x) override;

    std::string name()        const override { return name_; }
    size_t      output_size() const override { return out_features_; }
    size_t      num_params()  const override;

    // Access weight/bias tensors (e.g. for initialization or inspection).
    const Tensor& weights() const { return weights_; }
    const Tensor& bias()    const { return bias_; }
    Tensor&       weights()       { return weights_; }
    Tensor&       bias()          { return bias_; }

    // Optional FP32 weight tensor for full-precision forward pass and updates.
    // When enabled, forward() uses a FP32×FP32→BF16 GEMM path, eliminating
    // weight-quantisation error. HebbianUpdater accumulates deltas directly
    // into the FP32 tensor; elm_fit() writes W* here without BF16 rounding.
    // Call after construction but before the first forward() call.
    void          enable_fp32_weights();
    bool          has_fp32_weights()   const noexcept { return weights_fp32_.data != nullptr; }
    Tensor&       weights_fp32()             { return weights_fp32_; }
    const Tensor& weights_fp32()       const { return weights_fp32_; }

    size_t in_features()  const noexcept { return in_features_; }
    size_t out_features() const noexcept { return out_features_; }

    // Optional activation caching (for Hebbian and perturbation updates).
    // Disabled by default; call set_cache_activations(true) before the run loop.
    void          set_cache_activations(bool v) { cache_activations_ = v; }
    bool          cache_activations()  const noexcept { return cache_activations_; }
    const Tensor& last_input()         const { return last_input_; }
    const Tensor& last_output()        const { return last_output_; }

    // Target activations for supervised Hebbian.
    // Set to a one-hot BF16 tensor [batch, out_features] before emitting
    // the RewardEvent; HebbianUpdater::SupervisedHebbian uses this as the
    // post-synaptic signal instead of the actual forward-pass output.
    void          set_target_activations(Tensor t) noexcept { target_activations_ = std::move(t); }
    bool          has_target_activations() const noexcept   { return target_activations_.data != nullptr; }
    const Tensor& target_activations()     const            { return target_activations_; }

    // Access per-layer stats (EMA state, dead ratio, etc.).
    const LayerStats& stats() const { return layer_stats_; }

    // Disable per-forward-pass stats collection (reduces D2H copies + stream
    // syncs for ensemble members that don't need live monitoring).
    void set_compute_stats(bool v) noexcept override { compute_stats_ = v; }

    DenseLayer(const DenseLayer&) = delete;
    DenseLayer& operator=(const DenseLayer&) = delete;

private:
    void init_weights();

    size_t      in_features_;
    size_t      out_features_;
    bool        use_bias_;
    float       init_scale_;
    std::string name_;

    Tensor weights_;       // BF16, device
    Tensor weights_fp32_;  // FP32, device (optional; set via enable_fp32_weights())
    Tensor bias_;          // FP32, device

    cublasHandle_t cublas_handle_ = nullptr;

    LayerStats layer_stats_;
    bool       cache_activations_ = false;
    bool       compute_stats_     = true;
    Tensor     last_input_;
    Tensor     last_output_;
    Tensor     target_activations_;  // [batch, out_features] BF16, device (optional)
};

// ---------------------------------------------------------------------------
// Kernel: add bias vector to each row of a 2D matrix (BF16 + FP32 bias).
// Called from dense.cu.
// ---------------------------------------------------------------------------
void add_bias_bf16(
    __nv_bfloat16* output,    // [batch, out]
    const float*   bias,      // [out]
    size_t         batch_size,
    size_t         out_features,
    cudaStream_t   stream);

// Reset the global Kaiming seed counter used by DenseLayer::init_weights().
// Call before building an experiment to ensure reproducible or comparable
// weight initialisations across separate process invocations.
void reset_kaiming_seed(uint64_t val = 42);

} // namespace fayn
