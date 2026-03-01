#pragma once

#include "src/core/tensor.hpp"
#include <cublas_v2.h>

namespace fayn {

// ---------------------------------------------------------------------------
// ConvFrontend: 2-D convolution + ReLU + optional 2×2 max-pool.
//
// Supports arbitrary input channels (c_in) and image sizes (img_h × img_w).
// Default configuration matches MNIST: C_in=1, H=W=28, stride=1, no padding.
//
// Kernel size k ∈ {3, 5, 7} (default 5).  Output spatial size = H-k+1, W-k+1.
// After optional 2×2 max-pool (stride=2): (H-k+1)/2 × (W-k+1)/2.
//
//   MNIST (C_in=1, 28×28):
//     k=3 → after pool [N, C_out×169]; k=5 → [N, C_out×144]; k=7 → [N, C_out×121]
//   CIFAR-10 (C_in=3, 32×32):
//     k=3 → after pool [N, C_out×225]; k=5 → [N, C_out×196]; k=7 → [N, C_out×169]
//
// Filters are Kaiming-initialised (U(-b, b), b=1/sqrt(c_in×k²)) and frozen.
// Learned-conv update primitives (compute_im2col etc.) only support c_in=1, k=5.
// ---------------------------------------------------------------------------
class ConvFrontend {
public:
    // C_out:   number of convolutional filters.
    // max_pool: apply 2×2 max-pool after ReLU.
    // k:       kernel size (3, 5, or 7).
    // c_in:    number of input channels (1 for MNIST, 3 for CIFAR-10).
    // img_h:   input image height in pixels.
    // img_w:   input image width  in pixels.
    explicit ConvFrontend(int C_out, bool max_pool = true, int k = 5,
                          int c_in = 1, int img_h = 28, int img_w = 28);

    ~ConvFrontend();

    ConvFrontend(const ConvFrontend&)            = delete;
    ConvFrontend& operator=(const ConvFrontend&) = delete;

    // x: [N, 784] BF16 on CUDA.
    // Returns: [N, output_features()] FP32 on CUDA.
    Tensor forward(const Tensor& x) const;

    // Number of output features per sample.
    int output_features() const {
        const int OUT = img_h_ - k_size_ + 1;
        return max_pool_ ? C_out_ * (OUT/2) * (OUT/2) : C_out_ * OUT * OUT;
    }

    int c_in()  const { return c_in_;  }
    int img_h() const { return img_h_; }
    int img_w() const { return img_w_; }

    // Filter weights [C_out, c_in*k_size_*k_size_] FP32 on CUDA.
    Tensor&       weights()       { return W_; }
    const Tensor& weights() const { return W_; }

    // Reinitialise with Kaiming uniform: U(-k, k) where k = 1/sqrt(C_in*kH*kW) = 0.2.
    void kaiming_init();

    // ---------------------------------------------------------------------------
    // Primitives for the ELM conv-filter update (learned conv).
    //
    // The update minimises ||im2col(X) @ W^T - T_0_prepool||^2 + lambda * ||W||^2
    // where T_0_prepool [N*576, C_out] = 2×2 upsample of T_0 [N, C_out*144].
    //
    // Normal equations (accumulated incrementally over mini-batches):
    //   G  = im2col^T im2col + lambda * I    [25, 25]
    //   b  = im2col^T T_0_prepool            [25, C_out]
    //   W^T = G^{-1} b                       [25, C_out]  → W [C_out, 25]
    //
    // Usage:
    //   front.reset_gram();
    //   for each batch b:
    //     Tensor col   = front.compute_im2col(X_b);
    //     Tensor T0_up = front.upsample_pool_target(T0_b);
    //     front.accumulate_gram(col, T0_up);
    //   front.solve_gram(lambda);  // updates W_
    // ---------------------------------------------------------------------------

    // x: [bs, 784] BF16 → [bs*576, 25] FP32.
    Tensor compute_im2col(const Tensor& x) const;

    // T0_b: [bs, C_out*144] FP32 → [bs*576, C_out] FP32
    // (2×2 nearest-neighbour upsample to undo max-pool, then NHWC reshape).
    Tensor upsample_pool_target(const Tensor& T0_b) const;

    // Initialise G_acc_ = 0 and b_acc_ = 0 on GPU.
    void reset_gram();

    // G_acc_ += col^T @ col ;  b_acc_ += col^T @ T0_up.
    // col [M, 25], T0_up [M, C_out], both FP32, M = bs*576.
    void accumulate_gram(const Tensor& col, const Tensor& T0_up);

    // Solve (G_acc_ + lambda*I) W^T = b_acc_ on CPU (25×25 Gauss-Jordan),
    // then upload the result to W_.
    void solve_gram(float lambda);

private:
    int            C_out_;
    bool           max_pool_;
    int            k_size_;   // kernel size (3, 5, or 7)
    int            c_in_;     // input channels (1 for MNIST, 3 for CIFAR-10)
    int            img_h_;    // input image height
    int            img_w_;    // input image width
    Tensor         W_;        // [C_out, c_in*k_size_*k_size_] FP32 on CUDA
    cublasHandle_t cublas_ = nullptr;

    // Gram accumulators for learned-conv update (allocated on demand).
    Tensor G_acc_;   // [25, 25] FP32 on CUDA
    Tensor b_acc_;   // [25, C_out] FP32 on CUDA
};

} // namespace fayn
