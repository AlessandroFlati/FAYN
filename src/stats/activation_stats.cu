#include "activation_stats.hpp"
#include "../core/device.hpp"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <cstring>

namespace fayn {

// ---------------------------------------------------------------------------
// BatchStatsBuffers::init
// ---------------------------------------------------------------------------
void BatchStatsBuffers::init(size_t num_neurons) {
    neuron_mean     = Tensor::make({num_neurons}, DType::Float32, Device::CUDA);
    neuron_var      = Tensor::make({num_neurons}, DType::Float32, Device::CUDA);
    neuron_abs_mean = Tensor::make({num_neurons}, DType::Float32, Device::CUDA);
    total_abs_mean  = Tensor::make({1},           DType::Float32, Device::CUDA);
    dead_count      = Tensor::make({1},           DType::Int32,   Device::CUDA);

    h_neuron_mean.assign(num_neurons, 0.f);
    h_neuron_var.assign(num_neurons,  0.f);
    h_neuron_abs_mean.assign(num_neurons, 0.f);
    h_total_abs_mean = 0.f;
    h_dead_count     = 0;
}

// ---------------------------------------------------------------------------
// CUDA reduction kernel: one block per neuron.
//
// Computes per-neuron signed mean, variance (E[x^2] - E[x]^2), and
// absolute mean over the batch dimension.  Atomically accumulates the
// global mean absolute magnitude and the dead-neuron count.
//
// Shared memory layout: [3 * blockDim.x] floats
//   [0 .. B-1]       = sum(x)     thread partial sums
//   [B .. 2B-1]      = sum(x^2)   thread partial sums
//   [2B .. 3B-1]     = sum(|x|)   thread partial sums
// ---------------------------------------------------------------------------
template<typename T>
__global__ void activation_stats_kernel(
    const T*  data,
    float*    neuron_mean,
    float*    neuron_var,
    float*    neuron_abs_mean,
    float*    total_abs_mean,  // atomic accumulate
    int*      dead_count,      // atomic accumulate
    size_t    batch_size,
    size_t    num_neurons,
    float     dead_threshold)
{
    const size_t j = blockIdx.x;
    if (j >= num_neurons) return;

    extern __shared__ float smem[];
    float* s_sum    = smem;
    float* s_sum_sq = smem + blockDim.x;
    float* s_abs    = smem + 2 * blockDim.x;

    float sum = 0.f, sum_sq = 0.f, abs_sum = 0.f;
    for (size_t b = threadIdx.x; b < batch_size; b += blockDim.x) {
        float v  = static_cast<float>(data[b * num_neurons + j]);
        sum     += v;
        sum_sq  += v * v;
        abs_sum += fabsf(v);
    }

    s_sum[threadIdx.x]    = sum;
    s_sum_sq[threadIdx.x] = sum_sq;
    s_abs[threadIdx.x]    = abs_sum;
    __syncthreads();

    // Tree reduction.
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x]    += s_sum[threadIdx.x + s];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + s];
            s_abs[threadIdx.x]    += s_abs[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float B     = static_cast<float>(batch_size);
        const float mean  = s_sum[0] / B;
        const float msq   = s_sum_sq[0] / B;
        const float abs_m = s_abs[0] / B;

        neuron_mean[j]     = mean;
        neuron_var[j]      = fmaxf(msq - mean * mean, 0.f);
        neuron_abs_mean[j] = abs_m;

        atomicAdd(total_abs_mean, abs_m / static_cast<float>(num_neurons));
        if (abs_m < dead_threshold) {
            atomicAdd(dead_count, 1);
        }
    }
}

// ---------------------------------------------------------------------------
// compute_activation_stats
// ---------------------------------------------------------------------------
void compute_activation_stats(
    const Tensor&      activations,
    BatchStatsBuffers& out,
    float              dead_threshold,
    cudaStream_t       stream)
{
    if (activations.device != Device::CUDA)
        throw std::runtime_error("compute_activation_stats: tensor must be on CUDA");
    if (activations.shape.size() != 2)
        throw std::runtime_error("compute_activation_stats: expected 2D tensor [batch, neurons]");

    const size_t batch     = activations.shape[0];
    const size_t neurons   = activations.shape[1];

    if (!out.ready())
        throw std::runtime_error("compute_activation_stats: BatchStatsBuffers not initialised");

    // Reset scalar accumulators.
    FAYN_CUDA_CHECK(cudaMemsetAsync(out.total_abs_mean.data, 0, sizeof(float),   stream));
    FAYN_CUDA_CHECK(cudaMemsetAsync(out.dead_count.data,     0, sizeof(int32_t), stream));

    const int block = static_cast<int>(std::min(batch, size_t{256}));
    const int grid  = static_cast<int>(neurons);
    const int smem  = 3 * block * static_cast<int>(sizeof(float));

    switch (activations.dtype) {
        case DType::BFloat16:
            activation_stats_kernel<__nv_bfloat16><<<grid, block, smem, stream>>>(
                static_cast<const __nv_bfloat16*>(activations.data),
                static_cast<float*>(out.neuron_mean.data),
                static_cast<float*>(out.neuron_var.data),
                static_cast<float*>(out.neuron_abs_mean.data),
                static_cast<float*>(out.total_abs_mean.data),
                static_cast<int*>(out.dead_count.data),
                batch, neurons, dead_threshold);
            FAYN_CUDA_CHECK(cudaGetLastError());
            break;
        case DType::Float16:
            activation_stats_kernel<__half><<<grid, block, smem, stream>>>(
                static_cast<const __half*>(activations.data),
                static_cast<float*>(out.neuron_mean.data),
                static_cast<float*>(out.neuron_var.data),
                static_cast<float*>(out.neuron_abs_mean.data),
                static_cast<float*>(out.total_abs_mean.data),
                static_cast<int*>(out.dead_count.data),
                batch, neurons, dead_threshold);
            FAYN_CUDA_CHECK(cudaGetLastError());
            break;
        case DType::Float32:
            activation_stats_kernel<float><<<grid, block, smem, stream>>>(
                static_cast<const float*>(activations.data),
                static_cast<float*>(out.neuron_mean.data),
                static_cast<float*>(out.neuron_var.data),
                static_cast<float*>(out.neuron_abs_mean.data),
                static_cast<float*>(out.total_abs_mean.data),
                static_cast<int*>(out.dead_count.data),
                batch, neurons, dead_threshold);
            FAYN_CUDA_CHECK(cudaGetLastError());
            break;
        default:
            throw std::runtime_error("compute_activation_stats: unsupported dtype");
    }

    // Sync and copy to host.
    FAYN_CUDA_CHECK(cudaStreamSynchronize(stream));

    FAYN_CUDA_CHECK(cudaMemcpy(out.h_neuron_mean.data(),
        out.neuron_mean.data, neurons * sizeof(float), cudaMemcpyDeviceToHost));
    FAYN_CUDA_CHECK(cudaMemcpy(out.h_neuron_var.data(),
        out.neuron_var.data, neurons * sizeof(float), cudaMemcpyDeviceToHost));
    FAYN_CUDA_CHECK(cudaMemcpy(out.h_neuron_abs_mean.data(),
        out.neuron_abs_mean.data, neurons * sizeof(float), cudaMemcpyDeviceToHost));
    FAYN_CUDA_CHECK(cudaMemcpy(&out.h_total_abs_mean,
        out.total_abs_mean.data, sizeof(float), cudaMemcpyDeviceToHost));
    {
        int32_t raw = 0;
        FAYN_CUDA_CHECK(cudaMemcpy(&raw,
            out.dead_count.data, sizeof(int32_t), cudaMemcpyDeviceToHost));
        out.h_dead_count = static_cast<uint32_t>(raw);
    }
}

// ---------------------------------------------------------------------------
// LayerStats
// ---------------------------------------------------------------------------
void LayerStats::init(size_t num_neurons, float alpha, float dead_thr) {
    dead_threshold = dead_thr;
    buffers.init(num_neurons);
    ema_mean.resize(num_neurons, alpha);
    ema_var.resize(num_neurons, alpha);
    ema_abs_mean.resize(num_neurons, alpha);
    ema_magnitude = EmaScalar(alpha);
}

StatsSnapshot LayerStats::compute_and_snapshot(
    int           layer_id,
    size_t        step,
    const Tensor& output,
    cudaStream_t  stream)
{
    // Lazy init on first call (for layers that don't know output_size at ctor time).
    if (!buffers.ready()) {
        if (output.shape.size() < 2)
            throw std::runtime_error("LayerStats: cannot init from tensor with ndim < 2");
        init(output.shape.back());
    }

    compute_activation_stats(output, buffers, dead_threshold, stream);

    // Update EMAs.
    ema_mean.update(buffers.h_neuron_mean.data(), buffers.h_neuron_mean.size());
    ema_var.update(buffers.h_neuron_var.data(), buffers.h_neuron_var.size());
    ema_abs_mean.update(buffers.h_neuron_abs_mean.data(), buffers.h_neuron_abs_mean.size());
    ema_magnitude.update(buffers.h_total_abs_mean);

    // Compute dead ratio from current EMA abs mean.
    size_t n_dead = 0;
    for (float v : ema_abs_mean.mean()) {
        if (v < dead_threshold) ++n_dead;
    }

    StatsSnapshot snap;
    snap.layer_id      = layer_id;
    snap.step          = step;
    snap.ema_mean      = ema_mean.mean();
    snap.ema_var       = ema_var.mean();
    snap.dead_ratio    = static_cast<float>(n_dead) / static_cast<float>(ema_abs_mean.size());
    snap.ema_magnitude = ema_magnitude.mean();
    return snap;
}

} // namespace fayn
