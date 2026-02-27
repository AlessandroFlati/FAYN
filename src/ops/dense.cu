#include "dense.hpp"
#include "../stats/activation_stats.hpp"
#include "../stats/event_bus.hpp"
#include "../cuda/stream_pool.hpp"

#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <random>
#include <vector>

namespace fayn {

// ---------------------------------------------------------------------------
// Bias-add kernel: output[b, j] += bias[j]  (BF16 output, FP32 bias)
// ---------------------------------------------------------------------------
__global__ void bias_add_kernel(
    __nv_bfloat16* output,
    const float*   bias,
    size_t         batch_size,
    size_t         out_features)
{
    size_t b = blockIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size && j < out_features) {
        float val   = static_cast<float>(output[b * out_features + j]) + bias[j];
        output[b * out_features + j] = __float2bfloat16(val);
    }
}

void add_bias_bf16(
    __nv_bfloat16* output,
    const float*   bias,
    size_t         batch_size,
    size_t         out_features,
    cudaStream_t   stream)
{
    constexpr int TX = 128;
    dim3 block(TX);
    dim3 grid(static_cast<unsigned>((out_features + TX - 1) / TX),
              static_cast<unsigned>(batch_size));
    bias_add_kernel<<<grid, block, 0, stream>>>(output, bias, batch_size, out_features);
    FAYN_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Weight init: Kaiming uniform in FP32, then cast to BF16 on device.
// ---------------------------------------------------------------------------
static void kaiming_uniform_bf16(Tensor& w, size_t fan_in) {
    // w: [out, in] BF16 on device.
    const size_t n   = w.numel();
    const float  std = sqrtf(2.0f / static_cast<float>(fan_in));

    // Allocate FP32 on host, fill, then cast on host, upload.
    std::vector<float>            fp32(n);
    std::default_random_engine    rng(42);
    std::uniform_real_distribution<float> dist(-std, std);
    for (float& v : fp32) v = dist(rng);

    // Cast to BF16 host-side.
    std::vector<__nv_bfloat16> bf16(n);
    for (size_t i = 0; i < n; ++i) bf16[i] = __float2bfloat16(fp32[i]);

    FAYN_CUDA_CHECK(cudaMemcpy(w.data, bf16.data(),
                               n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
}

static void zeros_fp32(Tensor& b) {
    FAYN_CUDA_CHECK(cudaMemset(b.data, 0, b.nbytes()));
}

// ---------------------------------------------------------------------------
// DenseLayer
// ---------------------------------------------------------------------------
DenseLayer::DenseLayer(size_t in_features, size_t out_features, bool use_bias)
    : in_features_(in_features)
    , out_features_(out_features)
    , use_bias_(use_bias)
{
    FAYN_CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    init_weights();
}

DenseLayer::~DenseLayer() {
    if (cublas_handle_) cublasDestroy(cublas_handle_);
}

void DenseLayer::init_weights() {
    weights_ = Tensor::make({out_features_, in_features_}, DType::BFloat16, Device::CUDA);
    kaiming_uniform_bf16(weights_, in_features_);
    name_ = "dense";

    if (use_bias_) {
        bias_ = Tensor::make({out_features_}, DType::Float32, Device::CUDA);
        zeros_fp32(bias_);
    }

    layer_stats_.init(out_features_);
}

size_t DenseLayer::num_params() const {
    return out_features_ * in_features_ + (use_bias_ ? out_features_ : 0);
}

// ---------------------------------------------------------------------------
// Forward pass: y = x W^T + b
//
// cuBLASLt convention: by default matrices are column-major.
// For row-major storage, set CUBLASLT_ORDER_ROW on each layout descriptor.
//
// y = x * W^T
//   x: [batch, in]  row-major  -> A in GEMM, transa=N
//   W: [out, in]    row-major  -> B in GEMM, transb=T  (gives W^T=[in,out])
//   y: [batch, out] row-major  -> C in GEMM
//
// TODO: validate GEMM dimensions with a unit test (batch=1, small in/out).
// ---------------------------------------------------------------------------
Tensor DenseLayer::forward(const Tensor& x) {
    if (x.device != Device::CUDA)
        throw std::runtime_error("DenseLayer::forward: input must be on CUDA");
    if (x.dtype != DType::BFloat16)
        throw std::runtime_error("DenseLayer::forward: input must be BFloat16");
    if (x.shape.size() != 2)
        throw std::runtime_error("DenseLayer::forward: expected 2D input [batch, in]");

    const size_t batch = x.shape[0];
    const size_t in    = x.shape[1];
    if (in != in_features_)
        throw std::runtime_error("DenseLayer::forward: input feature mismatch");

    Tensor output = Tensor::make({batch, out_features_}, DType::BFloat16, Device::CUDA);

    StreamPool::Guard guard;
    cudaStream_t stream = guard.stream();
    FAYN_CUBLAS_CHECK(cublasSetStream(cublas_handle_, stream));

    // y[batch, out] = x[batch, in] @ W^T[in, out]   (row-major)
    //
    // Row-major <-> col-major duality:
    //   A row-major M[m,n] == col-major M^T[n,m] with leading dimension n.
    //
    // Equivalent col-major GEMM: y^T[out,batch] = W[out,in] @ x^T[in,batch]
    //   A = W.data → col-major [in, out] (ld=in)   TRANSA=T → W[out,in]
    //   B = x.data → col-major [in, batch] (ld=in) TRANSB=N → x^T[in,batch]
    //   C = y.data → col-major [out, batch] (ld=out)
    //
    float alpha = 1.0f;
    float beta  = 0.0f;
    FAYN_CUBLAS_CHECK(cublasGemmEx(
        cublas_handle_,
        CUBLAS_OP_T,                  // transa
        CUBLAS_OP_N,                  // transb
        static_cast<int>(out_features_),   // m
        static_cast<int>(batch),           // n
        static_cast<int>(in_features_),    // k
        &alpha,
        weights_.data, CUDA_R_16BF, static_cast<int>(in_features_),  // A, ld=in
        x.data,        CUDA_R_16BF, static_cast<int>(in_features_),  // B, ld=in
        &beta,
        output.data,   CUDA_R_16BF, static_cast<int>(out_features_), // C, ld=out
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));

    // Bias add.
    if (use_bias_) {
        add_bias_bf16(
            static_cast<__nv_bfloat16*>(output.data),
            static_cast<const float*>(bias_.data),
            batch, out_features_, stream);
    }

    // Cache activations if requested (for Hebbian / perturbation updates).
    // Use cudaMemcpyAsync on the same stream so the copy is ordered AFTER
    // the GEMM+bias (needed because StreamPool uses cudaStreamNonBlocking,
    // which does not synchronise with the null/default stream).
    if (cache_activations_) {
        last_input_ = Tensor::make(x.shape, x.dtype, Device::CUDA);
        FAYN_CUDA_CHECK(cudaMemcpyAsync(
            last_input_.data, x.data, x.nbytes(),
            cudaMemcpyDeviceToDevice, stream));

        last_output_ = Tensor::make(output.shape, output.dtype, Device::CUDA);
        FAYN_CUDA_CHECK(cudaMemcpyAsync(
            last_output_.data, output.data, output.nbytes(),
            cudaMemcpyDeviceToDevice, stream));
    }

    // Compute per-neuron stats, update EMAs, emit ActivationEvent.
    // Note: compute_and_snapshot() synchronises the stream internally.
    StatsSnapshot snap = layer_stats_.compute_and_snapshot(id_, next_step(), output, stream);

    ActivationEvent ev;
    ev.layer_id = snap.layer_id;
    ev.step     = snap.step;
    ev.stats    = std::move(snap);
    EventBus::instance().emit(ev);

    return output;
}

} // namespace fayn
