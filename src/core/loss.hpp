#pragma once

#include "tensor.hpp"
#include "device.hpp"

#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace fayn {

// ---------------------------------------------------------------------------
// LossFn: a callable that takes (output, target) tensors and returns a scalar.
//
// Convention used throughout FAYN:
//   output — [batch, C] BFloat16, on CUDA device (network output logits)
//   target — [batch]   Int32,    on CUDA device (class indices, 0-based)
//
// All standard implementations below copy tensors to host internally.
// ---------------------------------------------------------------------------
using LossFn = std::function<float(const Tensor& output, const Tensor& target)>;

// ---------------------------------------------------------------------------
// cross_entropy: numerically-stable softmax cross-entropy.
// Returns the mean negative log-likelihood over the batch.
// ---------------------------------------------------------------------------
inline float cross_entropy(const Tensor& output, const Tensor& target) {
    if (output.dtype  != DType::BFloat16) throw std::runtime_error("cross_entropy: output must be BFloat16");
    if (output.device != Device::CUDA)    throw std::runtime_error("cross_entropy: output must be on CUDA");
    if (target.dtype  != DType::Int32)    throw std::runtime_error("cross_entropy: target must be Int32");
    if (target.device != Device::CUDA)    throw std::runtime_error("cross_entropy: target must be on CUDA");
    if (output.shape.size() != 2)         throw std::runtime_error("cross_entropy: output must be 2D [batch, C]");

    const size_t batch = output.shape[0];
    const size_t C     = output.shape[1];

    std::vector<__nv_bfloat16> out_h(batch * C);
    std::vector<int32_t>       lbl_h(batch);

    FAYN_CUDA_CHECK(cudaMemcpy(out_h.data(), output.data,
                               batch * C * sizeof(__nv_bfloat16),
                               cudaMemcpyDeviceToHost));
    FAYN_CUDA_CHECK(cudaMemcpy(lbl_h.data(), target.data,
                               batch * sizeof(int32_t),
                               cudaMemcpyDeviceToHost));

    float total = 0.f;
    for (size_t i = 0; i < batch; ++i) {
        // Numerically stable: subtract max before exp.
        float max_v = static_cast<float>(out_h[i * C]);
        for (size_t j = 1; j < C; ++j) {
            float v = static_cast<float>(out_h[i * C + j]);
            if (v > max_v) max_v = v;
        }
        float sum_exp = 0.f;
        for (size_t j = 0; j < C; ++j)
            sum_exp += std::exp(static_cast<float>(out_h[i * C + j]) - max_v);

        const int32_t lbl      = lbl_h[i];
        const float   logit    = static_cast<float>(out_h[i * C + lbl]) - max_v;
        total += -(logit - std::log(sum_exp));
    }
    return total / static_cast<float>(batch);
}

// ---------------------------------------------------------------------------
// accuracy: top-1 accuracy (argmax match).
// Returns fraction of correctly classified samples in [0, 1].
// ---------------------------------------------------------------------------
inline float accuracy(const Tensor& output, const Tensor& target) {
    if (output.dtype  != DType::BFloat16) throw std::runtime_error("accuracy: output must be BFloat16");
    if (output.device != Device::CUDA)    throw std::runtime_error("accuracy: output must be on CUDA");
    if (target.dtype  != DType::Int32)    throw std::runtime_error("accuracy: target must be Int32");
    if (target.device != Device::CUDA)    throw std::runtime_error("accuracy: target must be on CUDA");
    if (output.shape.size() != 2)         throw std::runtime_error("accuracy: output must be 2D [batch, C]");

    const size_t batch = output.shape[0];
    const size_t C     = output.shape[1];

    std::vector<__nv_bfloat16> out_h(batch * C);
    std::vector<int32_t>       lbl_h(batch);

    FAYN_CUDA_CHECK(cudaMemcpy(out_h.data(), output.data,
                               batch * C * sizeof(__nv_bfloat16),
                               cudaMemcpyDeviceToHost));
    FAYN_CUDA_CHECK(cudaMemcpy(lbl_h.data(), target.data,
                               batch * sizeof(int32_t),
                               cudaMemcpyDeviceToHost));

    int correct = 0;
    for (size_t i = 0; i < batch; ++i) {
        size_t pred = 0;
        float  best = static_cast<float>(out_h[i * C]);
        for (size_t j = 1; j < C; ++j) {
            float v = static_cast<float>(out_h[i * C + j]);
            if (v > best) { best = v; pred = j; }
        }
        if (static_cast<int32_t>(pred) == lbl_h[i]) ++correct;
    }
    return static_cast<float>(correct) / static_cast<float>(batch);
}

// ---------------------------------------------------------------------------
// mse: mean squared error with one-hot targets.
// output logits vs. one-hot-encoded class labels.
// ---------------------------------------------------------------------------
inline float mse(const Tensor& output, const Tensor& target) {
    if (output.dtype  != DType::BFloat16) throw std::runtime_error("mse: output must be BFloat16");
    if (output.device != Device::CUDA)    throw std::runtime_error("mse: output must be on CUDA");
    if (target.dtype  != DType::Int32)    throw std::runtime_error("mse: target must be Int32");
    if (target.device != Device::CUDA)    throw std::runtime_error("mse: target must be on CUDA");
    if (output.shape.size() != 2)         throw std::runtime_error("mse: output must be 2D [batch, C]");

    const size_t batch = output.shape[0];
    const size_t C     = output.shape[1];

    std::vector<__nv_bfloat16> out_h(batch * C);
    std::vector<int32_t>       lbl_h(batch);

    FAYN_CUDA_CHECK(cudaMemcpy(out_h.data(), output.data,
                               batch * C * sizeof(__nv_bfloat16),
                               cudaMemcpyDeviceToHost));
    FAYN_CUDA_CHECK(cudaMemcpy(lbl_h.data(), target.data,
                               batch * sizeof(int32_t),
                               cudaMemcpyDeviceToHost));

    float total = 0.f;
    for (size_t i = 0; i < batch; ++i) {
        for (size_t j = 0; j < C; ++j) {
            float pred   = static_cast<float>(out_h[i * C + j]);
            float target_v = (static_cast<size_t>(lbl_h[i]) == j) ? 1.f : 0.f;
            float diff   = pred - target_v;
            total += diff * diff;
        }
    }
    return total / static_cast<float>(batch * C);
}

} // namespace fayn
