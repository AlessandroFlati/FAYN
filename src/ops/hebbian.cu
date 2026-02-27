#include "hebbian.hpp"
#include "../core/device.hpp"

#include <cuda_bf16.h>
#include <stdexcept>

namespace fayn {

// ---------------------------------------------------------------------------
// Fused outer-product + weight-update kernel.
//
// Each thread handles one (out_idx, in_idx) pair.  It accumulates the
// Hebbian correlation over the batch dimension and immediately applies
// the update to W, avoiding an intermediate delta buffer.
//
// W:    [out, in]   BF16, row-major, updated in place
// pre:  [batch, in] BF16, row-major
// post: [batch, out] BF16, row-major
// ---------------------------------------------------------------------------
__global__ void hebbian_fused_kernel(
    __nv_bfloat16*       W,
    const __nv_bfloat16* pre,
    const __nv_bfloat16* post,
    float                scale,
    size_t               batch,
    size_t               in_feat,
    size_t               out_feat)
{
    const size_t out_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t in_idx  = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_feat || in_idx >= in_feat) return;

    float delta = 0.f;
    for (size_t b = 0; b < batch; ++b)
        delta += static_cast<float>(pre[b * in_feat + in_idx]) *
                 static_cast<float>(post[b * out_feat + out_idx]);

    const size_t w = out_idx * in_feat + in_idx;
    W[w] = __float2bfloat16(static_cast<float>(W[w]) + scale * delta);
}

// ---------------------------------------------------------------------------
// Row-norm kernel: divide each row of W by its L2 norm.
// One block per row.
// ---------------------------------------------------------------------------
__global__ void row_normalize_kernel(
    __nv_bfloat16* W,
    size_t         out_features,
    size_t         in_features,
    float          eps)
{
    const size_t row = blockIdx.x;
    if (row >= out_features) return;

    extern __shared__ float smem[];

    // Accumulate squared elements for this row.
    float sum_sq = 0.f;
    for (size_t j = threadIdx.x; j < in_features; j += blockDim.x) {
        float v  = static_cast<float>(W[row * in_features + j]);
        sum_sq  += v * v;
    }
    smem[threadIdx.x] = sum_sq;
    __syncthreads();

    // Block reduce.
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) smem[0] = rsqrtf(fmaxf(smem[0], eps * eps));
    __syncthreads();

    const float inv_norm = smem[0];
    for (size_t j = threadIdx.x; j < in_features; j += blockDim.x) {
        float v = static_cast<float>(W[row * in_features + j]) * inv_norm;
        W[row * in_features + j] = __float2bfloat16(v);
    }
}

// ---------------------------------------------------------------------------
// hebbian_update
// ---------------------------------------------------------------------------
void hebbian_update(
    Tensor&       weights,
    const Tensor& pre,
    const Tensor& post,
    float         lr,
    cudaStream_t  stream)
{
    if (weights.device != Device::CUDA || pre.device != Device::CUDA || post.device != Device::CUDA)
        throw std::runtime_error("hebbian_update: all tensors must be on CUDA");
    if (weights.dtype != DType::BFloat16 || pre.dtype != DType::BFloat16 || post.dtype != DType::BFloat16)
        throw std::runtime_error("hebbian_update: all tensors must be BFloat16");
    if (pre.shape.size() != 2 || post.shape.size() != 2 || weights.shape.size() != 2)
        throw std::runtime_error("hebbian_update: all tensors must be 2D");

    const size_t batch      = pre.shape[0];
    const size_t in_feat    = pre.shape[1];
    const size_t out_feat   = post.shape[1];

    if (post.shape[0] != batch)
        throw std::runtime_error("hebbian_update: batch size mismatch");
    if (weights.shape[0] != out_feat || weights.shape[1] != in_feat)
        throw std::runtime_error("hebbian_update: weight shape mismatch");

    // Fused outer-product + weight update:
    //   W[g, f] += (lr / batch) * sum_b pre[b, f] * post[b, g]
    //
    // Implemented as a custom kernel to avoid cuBLAS (OP_N, OP_T) BF16 issues.
    const float scale = lr / static_cast<float>(batch);

    constexpr int TX = 16, TY = 16;
    dim3 block(TX, TY);
    dim3 grid(static_cast<unsigned>((in_feat  + TX - 1) / TX),
              static_cast<unsigned>((out_feat + TY - 1) / TY));

    hebbian_fused_kernel<<<grid, block, 0, stream>>>(
        static_cast<__nv_bfloat16*>(weights.data),
        static_cast<const __nv_bfloat16*>(pre.data),
        static_cast<const __nv_bfloat16*>(post.data),
        scale, batch, in_feat, out_feat);
    FAYN_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Weight-decay kernel: W[i] *= scale   (scale = 1 - decay)
// ---------------------------------------------------------------------------
__global__ void scale_bf16_kernel(
    __nv_bfloat16* W,
    float          scale,
    size_t         n)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        W[i] = __float2bfloat16(static_cast<float>(W[i]) * scale);
}

void weight_decay_weights(Tensor& weights, float decay, cudaStream_t stream) {
    if (weights.device != Device::CUDA)
        throw std::runtime_error("weight_decay_weights: tensor must be on CUDA");
    if (weights.dtype != DType::BFloat16)
        throw std::runtime_error("weight_decay_weights: tensor must be BFloat16");

    const size_t n     = weights.numel();
    constexpr int TX   = 256;
    const int     grid = static_cast<int>((n + TX - 1) / TX);

    scale_bf16_kernel<<<grid, TX, 0, stream>>>(
        static_cast<__nv_bfloat16*>(weights.data),
        1.0f - decay, n);
    FAYN_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// normalize_weights_rows
// ---------------------------------------------------------------------------
void normalize_weights_rows(Tensor& weights, float eps, cudaStream_t stream) {
    if (weights.device != Device::CUDA)
        throw std::runtime_error("normalize_weights_rows: tensor must be on CUDA");
    if (weights.dtype != DType::BFloat16)
        throw std::runtime_error("normalize_weights_rows: tensor must be BFloat16");
    if (weights.shape.size() != 2)
        throw std::runtime_error("normalize_weights_rows: expected 2D tensor");

    const size_t out = weights.shape[0];
    const size_t in  = weights.shape[1];

    const int block = static_cast<int>(std::min(in, size_t{256}));
    const int grid  = static_cast<int>(out);
    const int smem  = block * static_cast<int>(sizeof(float));

    row_normalize_kernel<<<grid, block, smem, stream>>>(
        static_cast<__nv_bfloat16*>(weights.data),
        out, in, eps);
    FAYN_CUDA_CHECK(cudaGetLastError());
}

} // namespace fayn
