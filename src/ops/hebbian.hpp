#pragma once

#include "../core/tensor.hpp"

#include <cuda_runtime.h>

namespace fayn {

// ---------------------------------------------------------------------------
// hebbian_update: in-place Hebbian weight update.
//
// Implements:  W += (lr / batch) * post^T @ pre
//
//   W:    [out, in]    BF16, device  (updated in place)
//   pre:  [batch, in]  BF16, device  (pre-synaptic activations)
//   post: [batch, out] BF16, device  (post-synaptic activations)
//
// The outer product is computed in BF16 via cuBLAS, then a fused
// scale-and-cast-add kernel updates W in BF16.
//
// stream: CUDA stream to use. If nullptr, uses the default stream.
// ---------------------------------------------------------------------------
void hebbian_update(
    Tensor&       weights,
    const Tensor& pre,
    const Tensor& post,
    float         lr,
    cudaStream_t  stream = nullptr);

// ---------------------------------------------------------------------------
// normalize_weights_rows: divide each row of W by its L2 norm.
//
// Prevents weight explosion under repeated Hebbian updates.
// A minimum norm floor (eps) prevents division by zero for dead rows.
//
// W: [out, in] BF16, device (updated in place).
// ---------------------------------------------------------------------------
void normalize_weights_rows(
    Tensor&      weights,
    float        eps    = 1e-8f,
    cudaStream_t stream = nullptr);

// ---------------------------------------------------------------------------
// weight_decay_weights: W ← W * (1 - decay), element-wise.
//
// Pre-step L2 regularisation. Applied before the Hebbian update so that
// steady state satisfies  decay·W* = (lr/N)·H^T T,  i.e.
//   W* ∝ H^T T   (centroid direction, magnitude lr/(N·decay)).
//
// Use instead of normalize_weights_rows when you want soft magnitude control
// rather than a hard unit-sphere projection.
//
// W: [out, in] BF16, device (updated in place).
// ---------------------------------------------------------------------------
void weight_decay_weights(
    Tensor&      weights,
    float        decay,
    cudaStream_t stream = nullptr);

} // namespace fayn
