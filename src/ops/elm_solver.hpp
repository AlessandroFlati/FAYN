#pragma once

// ---------------------------------------------------------------------------
// ElmSolver: reusable FP32 solver for Extreme Learning Machine normal equations.
//
// Provides four GPU operations sharing cuBLAS + cuSOLVER handles:
//   solve()             -- W = (H^T H + λI)^{-1} H^T T   (ELM normal equations)
//   propagate_target()  -- H* = T (W W^T + λI)^{-1} W    (regularized Gram back-projection)
//   relu_forward()      -- H_out = ReLU(H @ W^T)           (FP32 hidden layer)
//   gradient_step()     -- W += (lr/N) * (H_tgt - H)^T @ H_pre  (in-place delta rule)
//
// GPU solve() uses LU factorization (Sgetrf/Sgetrs) — robust for ill-conditioned Gram.
// GPU propagate_target() uses regularized Gram + Cholesky for square W [d,d];
// falls back to CPU Gram path for non-square W [10,d] (readout).
// Regularization makes propagate_target() stable even when W is rank-deficient
// (e.g., W solved from sparse ReLU-clipped targets has rank ≤ n_classes).
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
#include <cusolverDn.h>

#include "src/core/device.hpp"
#include "src/core/tensor.hpp"
#include "src/ops/activations.hpp"

#define FAYN_CUSOLVER_CHECK(expr)                                                   \
    do {                                                                             \
        cusolverStatus_t __st = (expr);                                              \
        if (__st != CUSOLVER_STATUS_SUCCESS)                                         \
            throw std::runtime_error(                                                \
                std::string("cuSOLVER error ") + std::to_string((int)__st) +        \
                " at " __FILE__ ":" + std::to_string(__LINE__));                     \
    } while (0)

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
    cublasHandle_t     cublas_   = nullptr;
    cusolverDnHandle_t cusolver_ = nullptr;

public:
    ElmSolver() {
        FAYN_CUBLAS_CHECK(cublasCreate(&cublas_));
        FAYN_CUSOLVER_CHECK(cusolverDnCreate(&cusolver_));
    }
    ~ElmSolver() {
        if (cublas_)   cublasDestroy(cublas_);
        if (cusolver_) cusolverDnDestroy(cusolver_);
    }

    ElmSolver(const ElmSolver&)            = delete;
    ElmSolver& operator=(const ElmSolver&) = delete;
    ElmSolver(ElmSolver&&)                 = delete;
    ElmSolver& operator=(ElmSolver&&)      = delete;

    // -----------------------------------------------------------------------
    // solve: W = (H^T H + λI)^{-1} H^T T   — GPU LU, no CPU round-trip.
    //
    // H:      FP32 [N, d_in]  on device
    // T:      FP32 [N, d_out] on device
    // lambda: Tikhonov regularization (≥ 0)
    // Returns W: FP32 [d_out, d_in] on device  (DenseLayer weight format [out, in])
    //
    // Uses LU factorization (cusolverDnSgetrf + OP_N solve) rather than
    // Cholesky because H^T H can be numerically ill-conditioned even after
    // adding λI (when λ is small relative to the spectral norm of H^T H).
    // LU with partial pivoting matches the robustness of the old CPU Gaussian
    // elimination and handles all cases that Cholesky cannot.
    //
    // Math (col-major/row-major duality):
    //   A_cm[d_in,d_in] = H^T H + λI  (GPU GEMM + diagonal add)
    //   b_cm[d_in,d_out] = H^T T       (GPU GEMM)
    //   After cusolverDnSgetrs(OP_N): b_cm = A^{-1} b = W^T
    //   W^T col-major [d_in,d_out] ≡ W row-major [d_out,d_in] — same bytes. ✓
    // -----------------------------------------------------------------------
    Tensor solve(const Tensor& H, const Tensor& T, float lambda = 1e-4f) const {
        if (H.dtype != DType::Float32 || T.dtype != DType::Float32)
            throw std::invalid_argument("ElmSolver::solve: H and T must be Float32");

        const int N     = static_cast<int>(H.shape[0]);
        const int d_in  = static_cast<int>(H.shape[1]);
        const int d_out = static_cast<int>(T.shape[1]);
        if (N != static_cast<int>(T.shape[0]))
            throw std::invalid_argument("ElmSolver::solve: H and T batch size mismatch");

        const float alpha = 1.f, beta = 0.f;

        // A = H^T H [d_in, d_in] (col-major trick, see header comment).
        Tensor A = Tensor::make(
            {static_cast<size_t>(d_in), static_cast<size_t>(d_in)},
            DType::Float32, Device::CUDA);
        FAYN_CUBLAS_CHECK(cublasSgemm(
            cublas_, CUBLAS_OP_N, CUBLAS_OP_T,
            d_in, d_in, N, &alpha,
            static_cast<const float*>(H.data), d_in,
            static_cast<const float*>(H.data), d_in,
            &beta, static_cast<float*>(A.data), d_in));

        // b = H^T T [d_in, d_out].
        Tensor b = Tensor::make(
            {static_cast<size_t>(d_in), static_cast<size_t>(d_out)},
            DType::Float32, Device::CUDA);
        FAYN_CUBLAS_CHECK(cublasSgemm(
            cublas_, CUBLAS_OP_N, CUBLAS_OP_T,
            d_in, d_out, N, &alpha,
            static_cast<const float*>(H.data), d_in,
            static_cast<const float*>(T.data), d_out,
            &beta, static_cast<float*>(b.data), d_in));

        // Add λI to A diagonal on GPU.
        // Use a ones vector [d_in] with incx=1 (incx=0 is undefined per BLAS spec).
        // With incy=d_in+1 the strides land on A[0,0], A[1,1], …, A[d_in-1,d_in-1].
        {
            std::vector<float> ones_h(static_cast<size_t>(d_in), 1.f);
            Tensor ones_dev = Tensor::make(
                {static_cast<size_t>(d_in)}, DType::Float32, Device::CUDA);
            FAYN_CUDA_CHECK(cudaMemcpy(ones_dev.data, ones_h.data(),
                static_cast<size_t>(d_in) * sizeof(float), cudaMemcpyHostToDevice));
            FAYN_CUBLAS_CHECK(cublasSaxpy(
                cublas_, d_in, &lambda,
                static_cast<const float*>(ones_dev.data), 1,
                static_cast<float*>(A.data), d_in + 1));
        }

        FAYN_CUDA_CHECK(cudaDeviceSynchronize());

        // LU factorize A with partial pivoting (robust for ill-conditioned matrices).
        int* devIpiv = nullptr;
        int* devInfo = nullptr;
        FAYN_CUDA_CHECK(cudaMalloc(&devIpiv, static_cast<size_t>(d_in) * sizeof(int)));
        FAYN_CUDA_CHECK(cudaMalloc(&devInfo, sizeof(int)));

        int lwork = 0;
        FAYN_CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(
            cusolver_, d_in, d_in, static_cast<float*>(A.data), d_in, &lwork));
        Tensor workspace = Tensor::make(
            {static_cast<size_t>(std::max(lwork, 1))}, DType::Float32, Device::CUDA);
        FAYN_CUSOLVER_CHECK(cusolverDnSgetrf(
            cusolver_, d_in, d_in, static_cast<float*>(A.data), d_in,
            static_cast<float*>(workspace.data), devIpiv, devInfo));

        // Check LU factorization success.
        {
            int devInfo_h = 0;
            FAYN_CUDA_CHECK(cudaDeviceSynchronize());
            FAYN_CUDA_CHECK(cudaMemcpy(&devInfo_h, devInfo, sizeof(int),
                                       cudaMemcpyDeviceToHost));
            if (devInfo_h != 0) {
                cudaFree(devIpiv);
                cudaFree(devInfo);
                throw std::runtime_error(
                    "ElmSolver::solve: LU factorization failed (devInfo=" +
                    std::to_string(devInfo_h) +
                    "). H^T H + λI is singular.");
            }
        }

        // Solve A X = b in-place with OP_N.
        // b_cm[d_in,d_out] becomes W^T = A^{-1}(H^T T).
        // b col-major [d_in,d_out] ≡ W row-major [d_out,d_in] — same bytes. ✓
        FAYN_CUSOLVER_CHECK(cusolverDnSgetrs(
            cusolver_, CUBLAS_OP_N, d_in, d_out,
            static_cast<const float*>(A.data), d_in, devIpiv,
            static_cast<float*>(b.data), d_in, devInfo));

        FAYN_CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(devIpiv);
        cudaFree(devInfo);

        // Copy b → W (Tensor is not trivially re-shapeable).
        // b[d_in, d_out] col-major == W[d_out, d_in] row-major — same byte layout.
        Tensor W = Tensor::make(
            {static_cast<size_t>(d_out), static_cast<size_t>(d_in)},
            DType::Float32, Device::CUDA);
        FAYN_CUDA_CHECK(cudaMemcpy(W.data, b.data,
            static_cast<size_t>(d_out) * d_in * sizeof(float),
            cudaMemcpyDeviceToDevice));
        return W;
    }

    // -----------------------------------------------------------------------
    // propagate_target: H_target = T @ (W W^T + lambda_prop*I)^{-1} @ W
    //
    // For square W [d, d] (intermediate hidden layers): GPU regularized Gram + Cholesky.
    // For non-square W [d_out, d_in] with d_out ≠ d_in (readout [10, d]): CPU path.
    //
    // W:           FP32 [d_out, d_in] on device  (DenseLayer weight format)
    // T:           FP32 [N, d_out]    on device
    // lambda_prop: Tikhonov regularization for (W W^T + λI); must be > 0.
    //              Ensures Cholesky succeeds even for rank-deficient W
    //              (e.g., W solved from sparse ReLU-clipped targets has rank ≤ n_classes).
    //              Null-space contributions vanish in the final product because
    //              null(W W^T) = null(W^T), so W applied to null-dir gives 0.
    // Returns H_target: FP32 [N, d_in] on device
    //
    // GPU Gram math (all col-major via cuBLAS/cuSOLVER row-major duality):
    //   Step 1: G_cm = W_rm @ W_rm^T = W @ W^T  [cublasSgemm(OP_T,OP_N)]
    //   Step 2: G += lambda_prop * I             [cublasSaxpy diagonal]
    //   Step 3: Cholesky factorize G (SPD)       [cusolverDnSpotrf]
    //   Step 4: Solve G @ X = T^T in-place       [cusolverDnSpotrs on T_copy_rm as T^T_cm]
    //           → T_copy now holds G^{-1}T^T as col-major [d,N]
    //   Step 5: H_target_cm[d,N] = W^T_cm @ (G^{-1}T^T)_cm = W^T G^{-1} T^T = H_target^T
    //           [cublasSgemm(OP_N,OP_N)] → read as row-major [N,d] = H_target ✓
    // -----------------------------------------------------------------------
    Tensor propagate_target(const Tensor& W, const Tensor& T,
                            float lambda_prop = 1e-4f) const {
        if (W.dtype != DType::Float32 || T.dtype != DType::Float32)
            throw std::invalid_argument(
                "ElmSolver::propagate_target: tensors must be Float32");

        const int d_out = static_cast<int>(W.shape[0]);
        const int d_in  = static_cast<int>(W.shape[1]);
        const int N     = static_cast<int>(T.shape[0]);
        if (d_out != static_cast<int>(T.shape[1]))
            throw std::invalid_argument(
                "ElmSolver::propagate_target: W.shape[0] must equal T.shape[1]");

        if (d_out != d_in) {
            // Non-square W (readout [10, d]): CPU Gram path.
            return propagate_target_cpu(W, T);
        }

        // Square W [d, d]: GPU regularized Gram path.
        // H_target = T @ (W W^T + lambda_prop*I)^{-1} @ W   [N, d]
        const int d = d_out;
        const float alpha = 1.f, beta = 0.f;

        // Step 1: G = W @ W^T  [d, d]
        // cublasSgemm(OP_T, OP_N, d, d, d, W_rm, W_rm, G):
        //   G_cm = (W^T_cm)^T @ W^T_cm = W_rm @ W_rm^T = W @ W^T ✓
        Tensor G = Tensor::make(
            {static_cast<size_t>(d), static_cast<size_t>(d)}, DType::Float32, Device::CUDA);
        FAYN_CUBLAS_CHECK(cublasSgemm(
            cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
            d, d, d, &alpha,
            static_cast<const float*>(W.data), d,
            static_cast<const float*>(W.data), d,
            &beta, static_cast<float*>(G.data), d));

        // Step 2: G += lambda_prop * I  (ones vector with incy = d+1 hits diagonal)
        {
            std::vector<float> ones_h(static_cast<size_t>(d), 1.f);
            Tensor ones_dev = Tensor::make(
                {static_cast<size_t>(d)}, DType::Float32, Device::CUDA);
            FAYN_CUDA_CHECK(cudaMemcpy(ones_dev.data, ones_h.data(),
                static_cast<size_t>(d) * sizeof(float), cudaMemcpyHostToDevice));
            FAYN_CUBLAS_CHECK(cublasSaxpy(
                cublas_, d, &lambda_prop,
                static_cast<const float*>(ones_dev.data), 1,
                static_cast<float*>(G.data), d + 1));
        }

        FAYN_CUDA_CHECK(cudaDeviceSynchronize());

        // Step 3: Cholesky factorize G (SPD with lambda_prop > 0).
        int* devInfo = nullptr;
        FAYN_CUDA_CHECK(cudaMalloc(&devInfo, sizeof(int)));
        int lwork = 0;
        FAYN_CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(
            cusolver_, CUBLAS_FILL_MODE_LOWER, d,
            static_cast<float*>(G.data), d, &lwork));
        Tensor workspace = Tensor::make(
            {static_cast<size_t>(std::max(lwork, 1))}, DType::Float32, Device::CUDA);
        FAYN_CUSOLVER_CHECK(cusolverDnSpotrf(
            cusolver_, CUBLAS_FILL_MODE_LOWER, d,
            static_cast<float*>(G.data), d,
            static_cast<float*>(workspace.data), lwork, devInfo));

        {
            int devInfo_h = 0;
            FAYN_CUDA_CHECK(cudaDeviceSynchronize());
            FAYN_CUDA_CHECK(cudaMemcpy(&devInfo_h, devInfo, sizeof(int),
                                       cudaMemcpyDeviceToHost));
            if (devInfo_h != 0) {
                cudaFree(devInfo);
                throw std::runtime_error(
                    "ElmSolver::propagate_target: Cholesky failed (devInfo=" +
                    std::to_string(devInfo_h) +
                    "). W W^T + lambda_prop*I not positive definite.");
            }
        }

        // Step 4: Solve G @ X = T^T in-place.
        // T_copy_rm[N,d] passed as col-major [d,N] = T^T_cm.
        // After Spotrs: T_copy contains G^{-1} T^T (col-major [d,N]).
        Tensor T_copy = Tensor::make(
            {static_cast<size_t>(N), static_cast<size_t>(d)}, DType::Float32, Device::CUDA);
        FAYN_CUDA_CHECK(cudaMemcpy(T_copy.data, T.data,
            static_cast<size_t>(N) * d * sizeof(float), cudaMemcpyDeviceToDevice));
        FAYN_CUSOLVER_CHECK(cusolverDnSpotrs(
            cusolver_, CUBLAS_FILL_MODE_LOWER, d, N,
            static_cast<const float*>(G.data), d,
            static_cast<float*>(T_copy.data), d, devInfo));

        FAYN_CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(devInfo);

        // Step 5: H_target_cm[d,N] = W^T_cm @ (G^{-1}T^T)_cm = W^T G^{-1} T^T = H_target^T
        // cublasSgemm(OP_N, OP_N, d, N, d, W_rm, d, T_copy, d, H_target, d):
        //   C_cm[d,N] = W_cm[d,d] @ T_copy_cm[d,N] = W^T @ G^{-1}T^T = H_target^T ✓
        //   Read C as row-major [N,d] = H_target ✓
        Tensor H_target = Tensor::make(
            {static_cast<size_t>(N), static_cast<size_t>(d)}, DType::Float32, Device::CUDA);
        FAYN_CUBLAS_CHECK(cublasSgemm(
            cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
            d, N, d, &alpha,
            static_cast<const float*>(W.data), d,
            static_cast<const float*>(T_copy.data), d,
            &beta, static_cast<float*>(H_target.data), d));

        FAYN_CUDA_CHECK(cudaDeviceSynchronize());
        return H_target;
    }

    // -----------------------------------------------------------------------
    // linear_forward: H_out = H @ W^T  — all FP32, all on device, no activation.
    //
    // H:      FP32 [N, d_in]     on device
    // W:      FP32 [d_out, d_in] on device
    // Returns H_out: FP32 [N, d_out] on device
    // -----------------------------------------------------------------------
    Tensor linear_forward(const Tensor& H, const Tensor& W) const {
        if (H.dtype != DType::Float32 || W.dtype != DType::Float32)
            throw std::invalid_argument("ElmSolver::linear_forward: tensors must be Float32");

        const int N     = static_cast<int>(H.shape[0]);
        const int d_in  = static_cast<int>(H.shape[1]);
        const int d_out = static_cast<int>(W.shape[0]);
        if (d_in != static_cast<int>(W.shape[1]))
            throw std::invalid_argument("ElmSolver::linear_forward: H/W inner dimension mismatch");

        Tensor H_out = Tensor::make(
            {static_cast<size_t>(N), static_cast<size_t>(d_out)}, DType::Float32, Device::CUDA);
        const float alpha = 1.f, beta = 0.f;

        // H_out_cm[d_out,N] = W_cm[d_out,d_in] @ H_cm[d_in,N] = W @ H^T = H_out^T
        // Read H_out_cm col-major as H_out_rm[N,d_out] ✓
        FAYN_CUBLAS_CHECK(cublasSgemm(
            cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
            d_out, N, d_in, &alpha,
            static_cast<const float*>(W.data), d_in,
            static_cast<const float*>(H.data), d_in,
            &beta, static_cast<float*>(H_out.data), d_out));

        FAYN_CUDA_CHECK(cudaStreamSynchronize(nullptr));
        return H_out;
    }

    // -----------------------------------------------------------------------
    // leaky_relu_forward: H_out = LeakyReLU(H @ W^T)
    // -----------------------------------------------------------------------
    Tensor leaky_relu_forward(const Tensor& H, const Tensor& W, float alpha) const {
        Tensor H_out = linear_forward(H, W);
        apply_leaky_relu(H_out, alpha, /*stream=*/nullptr);
        return H_out;
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

        Tensor H_out = Tensor::make(
            {static_cast<size_t>(N), static_cast<size_t>(d_out)}, DType::Float32, Device::CUDA);
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

    // -----------------------------------------------------------------------
    // gradient_step: W += (lr/N) * (H_target - H_actual)^T @ H_pre  [in-place, FP32]
    //
    // H_pre:    FP32 [N, d_in]    device  — pre-synaptic features
    // H_actual: FP32 [N, d_out]   device  — current hidden output
    // H_target: FP32 [N, d_out]   device  — propagated target
    // W:        FP32 [d_out, d_in] device  — updated in-place
    // lr:       learning rate (divided by N internally)
    //
    // Computes:
    //   error  = H_target - H_actual           [cublasSgeam]
    //   ΔW     = error^T @ H_pre               [cublasSgemm]
    //   W     += (lr/N) * ΔW                   [cublasSaxpy]
    // -----------------------------------------------------------------------
    void gradient_step(Tensor& W, const Tensor& H_pre,
                       const Tensor& H_actual, const Tensor& H_target,
                       float lr) const {
        if (W.dtype != DType::Float32 || H_pre.dtype != DType::Float32 ||
            H_actual.dtype != DType::Float32 || H_target.dtype != DType::Float32)
            throw std::invalid_argument("ElmSolver::gradient_step: all tensors must be Float32");

        const int N     = static_cast<int>(H_pre.shape[0]);
        const int d_in  = static_cast<int>(H_pre.shape[1]);
        const int d_out = static_cast<int>(H_actual.shape[1]);

        const float one = 1.f, neg_one = -1.f, zero = 0.f;

        // error [N, d_out] = H_target - H_actual
        Tensor error = Tensor::make(
            {static_cast<size_t>(N), static_cast<size_t>(d_out)}, DType::Float32, Device::CUDA);
        FAYN_CUBLAS_CHECK(cublasSgeam(
            cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
            d_out, N,
            &one,     static_cast<const float*>(H_target.data), d_out,
            &neg_one, static_cast<const float*>(H_actual.data), d_out,
            static_cast<float*>(error.data), d_out));

        // dW_cm[d_in, d_out] = H_pre_cm[d_in, N] @ error_cm[d_out, N]^T
        //   = H_pre^T @ error  →  dW_rm[d_out, d_in] = error^T @ H_pre  ✓
        Tensor dW = Tensor::make(
            {static_cast<size_t>(d_out), static_cast<size_t>(d_in)}, DType::Float32, Device::CUDA);
        FAYN_CUBLAS_CHECK(cublasSgemm(
            cublas_, CUBLAS_OP_N, CUBLAS_OP_T,
            d_in, d_out, N, &one,
            static_cast<const float*>(H_pre.data),   d_in,
            static_cast<const float*>(error.data),   d_out,
            &zero, static_cast<float*>(dW.data), d_in));

        FAYN_CUDA_CHECK(cudaDeviceSynchronize());

        // W += (lr / N) * dW  — flat SAXPY over all d_out * d_in elements
        float scale = lr / static_cast<float>(N);
        FAYN_CUBLAS_CHECK(cublasSaxpy(
            cublas_, d_out * d_in, &scale,
            static_cast<const float*>(dW.data), 1,
            static_cast<float*>(W.data), 1));

        FAYN_CUDA_CHECK(cudaDeviceSynchronize());
    }

private:
    // -----------------------------------------------------------------------
    // propagate_target_cpu: CPU Gram path for non-square W [d_out, d_in].
    // Used for readout W [10, d] where d_out = 10 is tiny (loops are negligible).
    // -----------------------------------------------------------------------
    Tensor propagate_target_cpu(const Tensor& W, const Tensor& T) const {
        const int d_out = static_cast<int>(W.shape[0]);
        const int d_in  = static_cast<int>(W.shape[1]);
        const int N     = static_cast<int>(T.shape[0]);

        // Copy W [d_out, d_in] to host.
        std::vector<float> W_host(static_cast<size_t>(d_out) * d_in);
        FAYN_CUDA_CHECK(cudaMemcpy(W_host.data(), W.data,
                                   W_host.size() * sizeof(float), cudaMemcpyDeviceToHost));

        // G = W @ W^T [d_out, d_out] on CPU.
        // d_out = 10 — nested loops are negligible.
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
        Tensor M_dev = Tensor::make(
            {static_cast<size_t>(d_out), static_cast<size_t>(d_in)}, DType::Float32, Device::CUDA);
        FAYN_CUDA_CHECK(cudaMemcpy(M_dev.data, M_host.data(),
                                   M_host.size() * sizeof(float), cudaMemcpyHostToDevice));

        // H_target = T @ M [N, d_in] via cuBLAS.
        // col-major reading: M_rm[d_out, d_in] = M^T_cm[d_in, d_out]
        //                    T_rm[N, d_out]     = T^T_cm[d_out, N]
        // cublasSgemm(OP_N, OP_N, d_in, N, d_out, M, d_in, T, d_out, H_target, d_in)
        //   C_cm[d_in, N] = M^T @ T^T = (T @ M)^T → col-major stored = T @ M row-major ✓
        Tensor H_target = Tensor::make(
            {static_cast<size_t>(N), static_cast<size_t>(d_in)}, DType::Float32, Device::CUDA);
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
};

} // namespace fayn
