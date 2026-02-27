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
    DenseLayer(size_t in_features, size_t out_features, bool use_bias = true);
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

    DenseLayer(const DenseLayer&) = delete;
    DenseLayer& operator=(const DenseLayer&) = delete;

private:
    void init_weights();

    size_t      in_features_;
    size_t      out_features_;
    bool        use_bias_;
    std::string name_;

    Tensor weights_;   // BF16, device
    Tensor bias_;      // FP32, device

    cublasHandle_t cublas_handle_ = nullptr;

    LayerStats layer_stats_;
    bool       cache_activations_ = false;
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

} // namespace fayn
