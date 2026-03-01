#include "conv_frontend.hpp"
#include "activations.hpp"

#include <cuda_bf16.h>
#include <cublas_v2.h>

#include <atomic>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace fayn {
namespace {

// MNIST image dimensions (fixed).
static constexpr int IMG_H = 28;
static constexpr int IMG_W = 28;

// K=5 geometry — used only by the learned-conv primitives (which don't support other K).
static constexpr int K5_OUT = IMG_H - 5 + 1;   // 24
static constexpr int K5_P   = K5_OUT * K5_OUT;  // 576
static constexpr int K5_KA  = 5 * 5;            // 25

// ---------------------------------------------------------------------------
// im2col_tmpl<KS>: extract KS×KS patches from BF16 images.
//
// x   [N, 784] BF16 (flat, row-major)
// col [N*P, KS*KS] FP32  (one row per spatial position, one col per patch pixel)
//
// Supported KS: 3, 5, 7.  Grid-stride loop: one thread per element of col.
// ---------------------------------------------------------------------------
template <int KS>
__global__ void im2col_tmpl(
    const __nv_bfloat16* __restrict__ x,
    float*               __restrict__ col,
    int N)
{
    constexpr int OUT  = IMG_H - KS + 1;  // IMG is square
    constexpr int P    = OUT * OUT;
    constexpr int KA   = KS * KS;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * P * KA;
    for (; idx < total; idx += (int)(blockDim.x * gridDim.x)) {
        const int k   = idx % KA;
        const int row = idx / KA;
        const int n   = row / P;
        const int hw  = row % P;
        const int oh  = hw / OUT;
        const int ow  = hw % OUT;
        const int kh  = k / KS;
        const int kw  = k % KS;
        const int ih  = oh + kh;
        const int iw  = ow + kw;
        col[row * KA + k] = __bfloat162float(x[n * IMG_H * IMG_W + ih * IMG_W + iw]);
    }
}

// Dispatch im2col for the given runtime kernel size.
// col must be pre-allocated to [N * (28-k+1)^2, k*k] FP32.
static void launch_im2col(const __nv_bfloat16* x, float* col, int N, int k_size) {
    const int OUT   = IMG_H - k_size + 1;
    const int total = N * OUT * OUT * k_size * k_size;
    const int blk   = 256;
    const int grd   = (total + blk - 1) / blk;
    switch (k_size) {
        case 3: im2col_tmpl<3><<<grd, blk>>>(x, col, N); break;
        case 5: im2col_tmpl<5><<<grd, blk>>>(x, col, N); break;
        case 7: im2col_tmpl<7><<<grd, blk>>>(x, col, N); break;
        default: throw std::invalid_argument(
            "ConvFrontend: unsupported kernel size (must be 3, 5, or 7)");
    }
}

// ---------------------------------------------------------------------------
// maxpool2x2_nhwc: 2×2 max-pool (stride=2) on data in NHWC layout.
//
// x   [N, H, W, C] FP32
// out [N, H/2, W/2, C] FP32
//
// Grid-stride loop: one thread per element of out.
// ---------------------------------------------------------------------------
__global__ void maxpool2x2_nhwc(
    const float* __restrict__ x,
    float*       __restrict__ out,
    int N, int H, int W, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_H = H / 2;
    const int out_W = W / 2;
    const int total = N * out_H * out_W * C;
    for (; idx < total; idx += (int)(blockDim.x * gridDim.x)) {
        const int c   = idx % C;
        int       tmp = idx / C;
        const int ow  = tmp % out_W;  tmp /= out_W;
        const int oh  = tmp % out_H;
        const int n   = tmp / out_H;
        const int ih  = oh * 2;
        const int iw  = ow * 2;
        const float v00 = x[((n * H + ih    ) * W + iw    ) * C + c];
        const float v01 = x[((n * H + ih    ) * W + iw + 1) * C + c];
        const float v10 = x[((n * H + ih + 1) * W + iw    ) * C + c];
        const float v11 = x[((n * H + ih + 1) * W + iw + 1) * C + c];
        out[idx] = fmaxf(fmaxf(v00, v01), fmaxf(v10, v11));
    }
}

// ---------------------------------------------------------------------------
// upsample_2x_nhwc: nearest-neighbour 2×2 upsample in NHWC layout.
// in  [N, H/2, W/2, C] FP32  →  out [N, H, W, C] FP32
// out[((n*H + oh)*W + ow)*C + c] = in[((n*H2 + oh/2)*W2 + ow/2)*C + c]
// ---------------------------------------------------------------------------
__global__ void upsample_2x_nhwc(
    const float* __restrict__ in,
    float*       __restrict__ out,
    int N, int H2, int W2, int C)   // H2 = H/2 = 12, W2 = W/2 = 12
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int H = H2 * 2;
    const int W = W2 * 2;
    const int total = N * H * W * C;
    for (; idx < total; idx += (int)(blockDim.x * gridDim.x)) {
        const int c   = idx % C;
        int       tmp = idx / C;
        const int ow  = tmp % W;  tmp /= W;
        const int oh  = tmp % H;
        const int n   = tmp / H;
        out[idx] = in[((n * H2 + oh / 2) * W2 + ow / 2) * C + c];
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// ConvFrontend implementation
// ---------------------------------------------------------------------------

ConvFrontend::ConvFrontend(int C_out, bool max_pool, int k)
    : C_out_(C_out)
    , max_pool_(max_pool)
    , k_size_(k)
    , W_(Tensor::make({(size_t)C_out, (size_t)(k * k)}, DType::Float32, Device::CUDA))
{
    if (k != 3 && k != 5 && k != 7)
        throw std::invalid_argument("ConvFrontend: k must be 3, 5, or 7");
    FAYN_CUBLAS_CHECK(cublasCreate(&cublas_));
    kaiming_init();
}

ConvFrontend::~ConvFrontend() {
    if (cublas_) cublasDestroy(cublas_);
}

// Global seed counter — each ConvFrontend instance gets a unique seed so that
// ensemble members receive different random filter initialisations.
static std::atomic<uint64_t> conv_kaiming_seed{42};

void ConvFrontend::kaiming_init() {
    // Kaiming uniform: U(-k, k) where k = 1/sqrt(fan_in), fan_in = C_in * kH * kW.
    const int   karea = k_size_ * k_size_;
    const float bound = 1.f / std::sqrt(static_cast<float>(karea));
    const uint64_t seed = conv_kaiming_seed.fetch_add(1, std::memory_order_relaxed);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-bound, bound);
    std::vector<float> w_host(static_cast<size_t>(C_out_) * karea);
    for (auto& v : w_host) v = dist(rng);
    FAYN_CUDA_CHECK(cudaMemcpy(W_.data, w_host.data(),
                               w_host.size() * sizeof(float),
                               cudaMemcpyHostToDevice));
}

Tensor ConvFrontend::forward(const Tensor& x) const {
    if (x.dtype  != DType::BFloat16)
        throw std::invalid_argument("ConvFrontend::forward: x must be BFloat16");
    if (x.device != Device::CUDA)
        throw std::invalid_argument("ConvFrontend::forward: x must be on CUDA");

    const int N    = static_cast<int>(x.shape[0]);
    const int OUT  = IMG_H - k_size_ + 1;  // e.g. 26 (K=3), 24 (K=5), 22 (K=7)
    const int P    = OUT * OUT;
    const int KA   = k_size_ * k_size_;

    // ---- Step 1: im2col → col [N*P, KA] FP32 ----
    Tensor col = Tensor::make({(size_t)(N * P), (size_t)KA}, DType::Float32, Device::CUDA);
    launch_im2col((const __nv_bfloat16*)x.data, (float*)col.data, N, k_size_);
    FAYN_CUDA_CHECK(cudaGetLastError());

    // ---- Step 2: GEMM — act [N*P, C_out] = col [N*P, KA] × W^T [KA, C_out] ----
    //
    // Row-major duality: act_rm [M, C_out] = col_rm [M, KA] × W_rm^T [C_out, KA]
    //   cublasSgemm(OP_T, OP_N, C_out, M, KA, 1, W_, KA, col, KA, 0, act, C_out)
    Tensor act = Tensor::make({(size_t)(N * P), (size_t)C_out_}, DType::Float32, Device::CUDA);
    {
        const float alpha = 1.f, beta = 0.f;
        FAYN_CUBLAS_CHECK(cublasSgemm(cublas_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            C_out_, N * P, KA,
            &alpha,
            (float*)W_.data,  KA,
            (float*)col.data, KA,
            &beta,
            (float*)act.data, C_out_));
    }

    // ---- Step 3: ReLU in-place ----
    apply_relu(act, /*stream=*/nullptr);

    if (!max_pool_) {
        Tensor out = Tensor::make({(size_t)N, (size_t)(P * C_out_)},
                                  DType::Float32, Device::CUDA);
        FAYN_CUDA_CHECK(cudaMemcpy(out.data, act.data,
                                   (size_t)N * P * C_out_ * sizeof(float),
                                   cudaMemcpyDeviceToDevice));
        return out;
    }

    // ---- Step 4: 2×2 max-pool ----
    // act NHWC layout [N, OUT, OUT, C_out].  Pool output: [N, OUT/2, OUT/2, C_out].
    const int pool_H = OUT / 2;
    const int pool_W = OUT / 2;
    Tensor out = Tensor::make({(size_t)N, (size_t)(pool_H * pool_W * C_out_)},
                              DType::Float32, Device::CUDA);
    {
        const int total   = N * pool_H * pool_W * C_out_;
        const int threads = 256;
        const int blocks  = (total + threads - 1) / threads;
        maxpool2x2_nhwc<<<blocks, threads>>>(
            (const float*)act.data, (float*)out.data, N, OUT, OUT, C_out_);
        FAYN_CUDA_CHECK(cudaGetLastError());
    }
    FAYN_CUDA_CHECK(cudaDeviceSynchronize());
    return out;
}

// ---------------------------------------------------------------------------
// Learned-conv update primitives
// ---------------------------------------------------------------------------

Tensor ConvFrontend::compute_im2col(const Tensor& x) const {
    if (k_size_ != 5)
        throw std::logic_error("ConvFrontend::compute_im2col: only supported for K=5");
    const int N = static_cast<int>(x.shape[0]);
    Tensor col = Tensor::make({(size_t)(N * K5_P), (size_t)K5_KA}, DType::Float32, Device::CUDA);
    launch_im2col((const __nv_bfloat16*)x.data, (float*)col.data, N, 5);
    FAYN_CUDA_CHECK(cudaGetLastError());
    FAYN_CUDA_CHECK(cudaDeviceSynchronize());
    return col;
}

Tensor ConvFrontend::upsample_pool_target(const Tensor& T0_b) const {
    if (k_size_ != 5)
        throw std::logic_error("ConvFrontend::upsample_pool_target: only supported for K=5");
    if (!max_pool_)
        throw std::logic_error("ConvFrontend::upsample_pool_target: max_pool must be true");
    const int bs     = static_cast<int>(T0_b.shape[0]);
    const int pool_H = K5_OUT / 2;  // 12
    const int pool_W = K5_OUT / 2;  // 12
    Tensor up = Tensor::make({(size_t)(bs * K5_P), (size_t)C_out_}, DType::Float32, Device::CUDA);
    {
        const int total   = bs * K5_OUT * K5_OUT * C_out_;
        const int threads = 256;
        const int blocks  = (total + threads - 1) / threads;
        upsample_2x_nhwc<<<blocks, threads>>>(
            (const float*)T0_b.data, (float*)up.data, bs, pool_H, pool_W, C_out_);
        FAYN_CUDA_CHECK(cudaGetLastError());
        FAYN_CUDA_CHECK(cudaDeviceSynchronize());
    }
    return up;
}

void ConvFrontend::reset_gram() {
    if (k_size_ != 5)
        throw std::logic_error("ConvFrontend::reset_gram: only supported for K=5");
    G_acc_ = Tensor::make({(size_t)K5_KA, (size_t)K5_KA}, DType::Float32, Device::CUDA);
    b_acc_ = Tensor::make({(size_t)K5_KA, (size_t)C_out_}, DType::Float32, Device::CUDA);
    FAYN_CUDA_CHECK(cudaMemset(G_acc_.data, 0, (size_t)K5_KA * K5_KA  * sizeof(float)));
    FAYN_CUDA_CHECK(cudaMemset(b_acc_.data, 0, (size_t)K5_KA * C_out_ * sizeof(float)));
}

void ConvFrontend::accumulate_gram(const Tensor& col, const Tensor& T0_up) {
    if (k_size_ != 5)
        throw std::logic_error("ConvFrontend::accumulate_gram: only supported for K=5");
    const int M   = static_cast<int>(col.shape[0]);
    const float one = 1.f;
    FAYN_CUBLAS_CHECK(cublasSgemm(cublas_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        K5_KA, K5_KA, M,
        &one,
        (float*)col.data,    K5_KA,
        (float*)col.data,    K5_KA,
        &one,
        (float*)G_acc_.data, K5_KA));
    FAYN_CUBLAS_CHECK(cublasSgemm(cublas_,
        CUBLAS_OP_N, CUBLAS_OP_T,
        K5_KA, C_out_, M,
        &one,
        (float*)col.data,    K5_KA,
        (float*)T0_up.data,  C_out_,
        &one,
        (float*)b_acc_.data, K5_KA));
}

void ConvFrontend::solve_gram(float lambda) {
    if (k_size_ != 5)
        throw std::logic_error("ConvFrontend::solve_gram: only supported for K=5");
    // Download G [K5_KA, K5_KA] and b [K5_KA, C_out] to CPU.
    const int K = K5_KA;
    std::vector<float> G_host(K * K);
    std::vector<float> b_host(K * C_out_);
    FAYN_CUDA_CHECK(cudaMemcpy(G_host.data(), G_acc_.data, K * K * sizeof(float), cudaMemcpyDeviceToHost));
    FAYN_CUDA_CHECK(cudaMemcpy(b_host.data(), b_acc_.data, K * C_out_ * sizeof(float), cudaMemcpyDeviceToHost));
    FAYN_CUDA_CHECK(cudaDeviceSynchronize());

    // Add lambda * I to diagonal.
    for (int i = 0; i < K; ++i) G_host[i * K + i] += lambda;

    // Gauss-Jordan elimination to solve G @ X = b in-place (G destroyed, b → X = W^T).
    // G [K, K] and b [K, C_out] both in row-major.
    for (int col_idx = 0; col_idx < K; ++col_idx) {
        // Partial pivoting.
        int pivot = col_idx;
        for (int row = col_idx + 1; row < K; ++row)
            if (std::abs(G_host[row * K + col_idx]) > std::abs(G_host[pivot * K + col_idx]))
                pivot = row;
        if (pivot != col_idx) {
            for (int c = 0; c < K;      ++c) std::swap(G_host[col_idx * K + c], G_host[pivot * K + c]);
            for (int c = 0; c < C_out_; ++c) std::swap(b_host[col_idx * C_out_ + c], b_host[pivot * C_out_ + c]);
        }
        // Scale pivot row.
        const float scale = G_host[col_idx * K + col_idx];
        for (int c = 0; c < K;      ++c) G_host[col_idx * K + c]       /= scale;
        for (int c = 0; c < C_out_; ++c) b_host[col_idx * C_out_ + c] /= scale;
        // Eliminate column.
        for (int row = 0; row < K; ++row) {
            if (row == col_idx) continue;
            const float f = G_host[row * K + col_idx];
            for (int c = 0; c < K;      ++c) G_host[row * K + c]       -= f * G_host[col_idx * K + c];
            for (int c = 0; c < C_out_; ++c) b_host[row * C_out_ + c] -= f * b_host[col_idx * C_out_ + c];
        }
    }
    // b_host is now W^T [K, C_out]. Transpose to W [C_out, K] and upload.
    std::vector<float> W_host(C_out_ * K);
    for (int c = 0; c < C_out_; ++c)
        for (int k = 0; k < K; ++k)
            W_host[c * K + k] = b_host[k * C_out_ + c];
    FAYN_CUDA_CHECK(cudaMemcpy(W_.data, W_host.data(), C_out_ * K * sizeof(float), cudaMemcpyHostToDevice));
}

} // namespace fayn
