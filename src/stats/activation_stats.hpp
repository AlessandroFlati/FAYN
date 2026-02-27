#pragma once

#include "../core/tensor.hpp"
#include "ema.hpp"
#include "events.hpp"

#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

namespace fayn {

// ---------------------------------------------------------------------------
// BatchStatsBuffers: pre-allocated device + host buffers for one stats pass.
// Reused across forward calls to avoid repeated allocation.
// ---------------------------------------------------------------------------
struct BatchStatsBuffers {
    // Device-side reduction outputs.
    Tensor neuron_mean;       // [neurons] FP32
    Tensor neuron_var;        // [neurons] FP32
    Tensor neuron_abs_mean;   // [neurons] FP32
    Tensor total_abs_mean;    // [1]       FP32
    Tensor dead_count;        // [1]       Int32

    // Host-side mirror (filled after cudaMemcpy).
    std::vector<float> h_neuron_mean;
    std::vector<float> h_neuron_var;
    std::vector<float> h_neuron_abs_mean;
    float    h_total_abs_mean = 0.f;
    uint32_t h_dead_count     = 0;

    // Allocate all buffers for num_neurons.
    void init(size_t num_neurons);

    // True once init() has been called.
    bool ready() const noexcept { return !h_neuron_mean.empty(); }
};

// ---------------------------------------------------------------------------
// compute_activation_stats
//
// Launches the per-neuron batch-reduction kernel on 'stream', synchronises
// the stream, then copies results to the host fields of 'out'.
//
// activations: [batch_size, num_neurons], dtype = BF16 | FP16 | FP32, device.
// dead_threshold: neuron considered dead if mean(|activation|) < threshold.
// ---------------------------------------------------------------------------
void compute_activation_stats(
    const Tensor&     activations,
    BatchStatsBuffers& out,
    float             dead_threshold,
    cudaStream_t      stream);

// ---------------------------------------------------------------------------
// LayerStats: owns EMA state and device buffers for one layer.
//
// Call init() once (in the layer constructor or on first forward).
// Call compute_and_snapshot() after each forward pass to get a populated
// StatsSnapshot ready to emit as an ActivationEvent.
// ---------------------------------------------------------------------------
struct LayerStats {
    BatchStatsBuffers buffers;
    EmaVector         ema_mean;
    EmaVector         ema_var;
    EmaVector         ema_abs_mean;
    EmaScalar         ema_magnitude;
    float             dead_threshold = 1e-3f;

    // Allocate buffers and set EMA alpha.
    void init(size_t num_neurons, float alpha = 0.05f, float dead_thr = 1e-3f);

    // Run the stats kernel, update EMAs, return a populated StatsSnapshot.
    // stream should be the same stream used for the layer's forward kernel.
    StatsSnapshot compute_and_snapshot(
        int          layer_id,
        size_t       step,
        const Tensor& output,   // [batch, num_neurons] BF16/FP16/FP32
        cudaStream_t  stream);

    bool ready() const noexcept { return buffers.ready(); }
};

} // namespace fayn
