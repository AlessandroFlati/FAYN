#pragma once

// ---------------------------------------------------------------------------
// ElmSolver: reusable FP32 solver for Extreme Learning Machine normal equations.
//
// Provides three GPU operations sharing a single cuBLAS handle:
//   solve()             -- W = (H^T H + λI)^{-1} H^T T   (ELM normal equations)
//   propagate_target()  -- H* = T (W W^T)^{-1} W          (target back-projection)
//   relu_forward()      -- H_out = ReLU(H @ W^T)           (FP32 hidden layer)
//
// cuBLAS convention note (row-major vs col-major):
//   Row-major X[M, N] stored in memory is read by cuBLAS as col-major X^T[N, M].
//   So H_rm[N, d] passed to cuBLAS looks like H^T_cm[d, N].
//   cublasSgemm(OP_N, OP_T, d, d, N, H, d, H, d, A, d)
//     computes  A_cm = H_cm * H_cm^T = H^T * H  ← exactly H^T H ✓
// ---------------------------------------------------------------------------

#include <cmath>
#include <stdexcept>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "src/core/device.hpp"
#include "src/core/tensor.hpp"
#include "src/ops/activations.hpp"

namespace fayn {

namespace detail {

// Gaussian elimination with partial pivoting.
// A: [n, n] row-major FP32 — modified in-place (LU-factored).
// b: [n, C] row-major FP32 — overwritten with solution on return.
// Throws if A is (near-)singular.
inline void gauss_solve_elm(std::vector<float>& A, std::vector<float>& b,
                             int n, int C) {
    for (int k = 0; k < n; ++k) {
        // Partial pivot.
        int pivot = k;
        for (int i = k + 1; i < n; ++i)
            if (std::abs(A[i * n + k]) > std::abs(A[pivot * n + k]))
                pivot = i;
        if (std::abs(A[pivot * n + k]) < 1e-8f)
            throw std::runtime_error(
                "ElmSolver: H^T H is singular (degenerate or all-zero features)");
        for (int j = 0; j < n; ++j) std::swap(A[k * n + j], A[pivot * n + j]);
        for (int j = 0; j < C; ++j) std::swap(b[k * C + j], b[pivot * C + j]);
        // Eliminate below diagonal.
        float inv = 1.f / A[k * n + k];
        for (int i = k + 1; i < n; ++i) {
            float f = A[i * n + k] * inv;
            for (int j = k; j < n; ++j) A[i * n + j] -= f * A[k * n + j];
            for (int j = 0; j < C; ++j)  b[i * C + j] -= f * b[k * C + j];
        }
    }
    // Back-substitution.
    for (int k = n - 1; k >= 0; --k) {
        float inv = 1.f / A[k * n + k];
        for (int j = 0; j < C; ++j) b[k * C + j] *= inv;
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < C; ++j)
                b[i * C + j] -= A[i * n + k] * b[k * C + j];
    }
}

} // namespace detail


class ElmSolver {
    cublasHandle_t cublas_ = nullptr;

public:
    ElmSolver()  { FAYN_CUBLAS_CHECK(cublasCreate(&cublas_)); }
    ~ElmSolver() { if (cublas_) cublasDestroy(cublas_); }

    ElmSolver(const ElmSolver&)            = delete;
    ElmSolver& operator=(const ElmSolver&) = delete;
    ElmSolver(ElmSolver&&)                 = delete;
    ElmSolver& operator=(ElmSolver&&)      = delete;

    // -----------------------------------------------------------------------
    // solve: W = (H^T H + λI)^{-1} H^T T
    //
    // H:      FP32 [N, d_in]  on device
    // T:      FP32 [N, d_out] on device
    // lambda: Tikhonov regularization (≥ 0)
    // Returns W: FP32 [d_out, d_in] on device  (DenseLayer weight format [out, in])
    // -----------------------------------------------------------------------
    Tensor solve(const Tensor& H, const Tensor& T, float lambda = 1e-4f) const {
        if (H.dtype != DType::Float32 || T.dtype != DType::Float32)
            throw std::invalid_argument("ElmSolver::solve: H and T must be Float32");

        const int N     = static_cast<int>(H.shape[0]);
        const int d_in  = static_cast<int>(H.shape[1]);
        const int d_out = static_cast<int>(T.shape[1]);
        if (N != static_cast<int>(T.shape[0]))
            throw std::invalid_argument("ElmSolver::solve: H and T batch size mismatch");

        // Normal equations on GPU.
        // A = H^T H [d_in, d_in], b = H^T T [d_in, d_out].
        // col-major trick: H_rm[N,d_in] = H^T_cm[d_in,N]
        //   A_cm = H_cm * H_cm^T = H^T * H  ← H^T H ✓
        //   b_cm = H_cm * T_cm^T = H^T * T  ← H^T T ✓  (T_rm[N,d_out]=T^T_cm[d_out,N])
        Tensor A_dev = Tensor::make({(size_t)d_in, (size_t)d_in},  DType::Float32, Device::CUDA);
        Tensor b_dev = Tensor::make({(size_t)d_in, (size_t)d_out}, DType::Float32, Device::CUDA);
        const float alpha = 1.f, beta = 0.f;

        FAYN_CUBLAS_CHECK(cublasSgemm(
            cublas_, CUBLAS_OP_N, CUBLAS_OP_T,
            d_in, d_in, N, &alpha,
            static_cast<const float*>(H.data), d_in,
            static_cast<const float*>(H.data), d_in,
            &beta, static_cast<float*>(A_dev.data), d_in));

        FAYN_CUBLAS_CHECK(cublasSgemm(
            cublas_, CUBLAS_OP_N, CUBLAS_OP_T,
            d_in, d_out, N, &alpha,
            static_cast<const float*>(H.data), d_in,
            static_cast<const float*>(T.data), d_out,
            &beta, static_cast<float*>(b_dev.data), d_in));

        FAYN_CUDA_CHECK(cudaDeviceSynchronize());

        // Copy A (symmetric → col-major == row-major flat) and b to host.
        std::vector<float> A_cm(static_cast<size_t>(d_in) * d_in);
        std::vector<float> b_cm(static_cast<size_t>(d_in) * d_out);
        FAYN_CUDA_CHECK(cudaMemcpy(A_cm.data(), A_dev.data,
                                   A_cm.size() * sizeof(float), cudaMemcpyDeviceToHost));
        FAYN_CUDA_CHECK(cudaMemcpy(b_cm.data(), b_dev.data,
                                   b_cm.size() * sizeof(float), cudaMemcpyDeviceToHost));

        // Add Tikhonov regularization to diagonal.
        for (int i = 0; i < d_in; ++i)
            A_cm[static_cast<size_t>(i) * d_in + i] += lambda;

        // Transpose b from col-major [d_in, d_out] to row-major [d_in, d_out]:
        //   b_rm[i * d_out + j] = b_cm[i + j * d_in]
        std::vector<float> b_rm(static_cast<size_t>(d_in) * d_out);
        for (int i = 0; i < d_in; ++i)
            for (int j = 0; j < d_out; ++j)
                b_rm[static_cast<size_t>(i) * d_out + j] =
                    b_cm[static_cast<size_t>(i) + static_cast<size_t>(j) * d_in];

        detail::gauss_solve_elm(A_cm, b_rm, d_in, d_out);

        // b_rm is now W_solve [d_in, d_out] row-major.
        // DenseLayer expects W [d_out, d_in] row-major: W[j, i] = W_solve[i, j].
        std::vector<float> W_host(static_cast<size_t>(d_out) * d_in);
        for (int i = 0; i < d_in; ++i)
            for (int j = 0; j < d_out; ++j)
                W_host[static_cast<size_t>(j) * d_in + i] =
                    b_rm[static_cast<size_t>(i) * d_out + j];

        Tensor W_dev = Tensor::make({(size_t)d_out, (size_t)d_in}, DType::Float32, Device::CUDA);
        FAYN_CUDA_CHECK(cudaMemcpy(W_dev.data, W_host.data(),
                                   W_host.size() * sizeof(float), cudaMemcpyHostToDevice));
        return W_dev;
    }

    // -----------------------------------------------------------------------
    // propagate_target: H_target = T @ (W W^T)^{-1} @ W
    //
    // This is the minimum-norm solution to  H_target W^T = T.
    // Equivalent to right-multiplying T by the right pseudoinverse of W^T.
    //
    // W:      FP32 [d_out, d_in] on device  (DenseLayer weight format)
    // T:      FP32 [N, d_out]    on device
    // Returns H_target: FP32 [N, d_in] on device
    // -----------------------------------------------------------------------
    Tensor propagate_target(const Tensor& W, const Tensor& T) const {
        if (W.dtype != DType::Float32 || T.dtype != DType::Float32)
            throw std::invalid_argument("ElmSolver::propagate_target: tensors must be Float32");

        const int N     = static_cast<int>(T.shape[0]);
        const int d_out = static_cast<int>(W.shape[0]);
        const int d_in  = static_cast<int>(W.shape[1]);
        if (d_out != static_cast<int>(T.shape[1]))
            throw std::invalid_argument(
                "ElmSolver::propagate_target: W.shape[0] must equal T.shape[1]");

        // Copy W [d_out, d_in] to host.
        std::vector<float> W_host(static_cast<size_t>(d_out) * d_in);
        FAYN_CUDA_CHECK(cudaMemcpy(W_host.data(), W.data,
                                   W_host.size() * sizeof(float), cudaMemcpyDeviceToHost));

        // G = W @ W^T [d_out, d_out] on CPU.
        // d_out = C = 10 — nested loops are negligible.
        std::vector<float> G(static_cast<size_t>(d_out) * d_out, 0.f);
        for (int i = 0; i < d_out; ++i)
            for (int j = 0; j < d_out; ++j)
                for (int k = 0; k < d_in; ++k)
                    G[static_cast<size_t>(i) * d_out + j] +=
                        W_host[static_cast<size_t>(i) * d_in + k] *
                        W_host[static_cast<size_t>(j) * d_in + k];

        // Invert G via Gaussian elimination: solve G X = I, X = G_inv.
        std::vector<float> I_mat(static_cast<size_t>(d_out) * d_out, 0.f);
        for (int i = 0; i < d_out; ++i)
            I_mat[static_cast<size_t>(i) * d_out + i] = 1.f;
        detail::gauss_solve_elm(G, I_mat, d_out, d_out);
        // I_mat is now G_inv [d_out, d_out] row-major.

        // M = G_inv @ W [d_out, d_in] on CPU.
        std::vector<float> M_host(static_cast<size_t>(d_out) * d_in, 0.f);
        for (int i = 0; i < d_out; ++i)
            for (int k = 0; k < d_out; ++k)
                for (int j = 0; j < d_in; ++j)
                    M_host[static_cast<size_t>(i) * d_in + j] +=
                        I_mat[static_cast<size_t>(i) * d_out + k] *
                        W_host[static_cast<size_t>(k) * d_in + j];

        // Upload M to device.
        Tensor M_dev = Tensor::make({(size_t)d_out, (size_t)d_in}, DType::Float32, Device::CUDA);
        FAYN_CUDA_CHECK(cudaMemcpy(M_dev.data, M_host.data(),
                                   M_host.size() * sizeof(float), cudaMemcpyHostToDevice));

        // H_target = T @ M [N, d_in] via cuBLAS.
        // col-major reading: M_rm[d_out, d_in] = M^T_cm[d_in, d_out]
        //                    T_rm[N, d_out]     = T^T_cm[d_out, N]
        // cublasSgemm(OP_N, OP_N, d_in, N, d_out, M, d_in, T, d_out, H_target, d_in)
        //   C_cm[d_in, N] = M^T @ T^T = (T @ M)^T → col-major stored = T @ M row-major ✓
        Tensor H_target = Tensor::make({(size_t)N, (size_t)d_in}, DType::Float32, Device::CUDA);
        const float alpha = 1.f, beta = 0.f;
        FAYN_CUBLAS_CHECK(cublasSgemm(
            cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
            d_in, N, d_out, &alpha,
            static_cast<const float*>(M_dev.data), d_in,
            static_cast<const float*>(T.data), d_out,
            &beta, static_cast<float*>(H_target.data), d_in));

        FAYN_CUDA_CHECK(cudaDeviceSynchronize());
        return H_target;
    }

    // -----------------------------------------------------------------------
    // relu_forward: H_out = ReLU(H @ W^T)  — all FP32, all on device.
    //
    // H:      FP32 [N, d_in]  on device
    // W:      FP32 [d_out, d_in] on device  (DenseLayer weight format [out, in])
    // Returns H_out: FP32 [N, d_out] on device
    // -----------------------------------------------------------------------
    Tensor relu_forward(const Tensor& H, const Tensor& W) const {
        if (H.dtype != DType::Float32 || W.dtype != DType::Float32)
            throw std::invalid_argument("ElmSolver::relu_forward: tensors must be Float32");

        const int N     = static_cast<int>(H.shape[0]);
        const int d_in  = static_cast<int>(H.shape[1]);
        const int d_out = static_cast<int>(W.shape[0]);
        if (d_in != static_cast<int>(W.shape[1]))
            throw std::invalid_argument("ElmSolver::relu_forward: H/W inner dimension mismatch");

        Tensor H_out = Tensor::make({(size_t)N, (size_t)d_out}, DType::Float32, Device::CUDA);
        const float alpha = 1.f, beta = 0.f;

        // H_out = H @ W^T [N, d_out]
        // col-major: W_rm[d_out, d_in] = W^T_cm[d_in, d_out]
        //            H_rm[N, d_in]     = H^T_cm[d_in, N]
        // cublasSgemm(OP_T, OP_N, d_out, N, d_in, W, d_in, H, d_in, H_out, d_out)
        //   C_cm[d_out, N] = W_cm^T @ H_cm = W @ H^T = H_out^T → stored col-major = H_out row-major ✓
        FAYN_CUBLAS_CHECK(cublasSgemm(
            cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
            d_out, N, d_in, &alpha,
            static_cast<const float*>(W.data), d_in,
            static_cast<const float*>(H.data), d_in,
            &beta, static_cast<float*>(H_out.data), d_out));

        FAYN_CUDA_CHECK(cudaStreamSynchronize(nullptr));

        // apply_relu dispatches on DType::Float32 — supported.
        apply_relu(H_out, /*stream=*/nullptr);
        return H_out;
    }
};

} // namespace fayn
