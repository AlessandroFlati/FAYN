#pragma once

#include "experiments/experiment.hpp"
#include "src/ops/dense.hpp"
#include "src/ops/elm_solver.hpp"
#include "src/io/mnist_loader.hpp"

#include <memory>
#include <string>
#include <vector>

namespace fayn {

// ---------------------------------------------------------------------------
// DeepELMExperiment: N-layer alternating ELM with target propagation.
//
// Architecture:
//   x[N,784] → W_0[784→d0, frozen] → ReLU → H_0[N,d0]
//            → W_1[d0→d, ELM]      → ReLU → H_1[N,d]
//            → ...                 → ReLU → ...
//            → W_n[d→d, ELM]       → ReLU → H_n[N,d]
//            → readout[d→10, ELM]  → Y[N,10]
//
// Algorithm (n_cycles iterations):
//   1. Precompute H_0 = ReLU(X W_0^T) once (W_0 frozen after construction).
//   2. Repeat:
//      a. Compute H_1,...,H_n forward (relu_forward)
//      b. Solve readout optimally for current H_n
//      c. Back-propagate targets: T_n = propagate_target(readout, T); apply_relu
//         Then for each layer k from n down to 1:
//           T_{k-1} = propagate_target(W_k, T_k); apply_relu
//           W_k = solve(H_{k-1}, T_k, λ)
//
// W_1..W_n and readout use enable_fp32_weights() — full-precision solve.
// solve() uses GPU Cholesky (cuSOLVER); propagate_target() uses GPU LU for
// square W [d,d] and CPU Gram for non-square readout [10,d].
// ---------------------------------------------------------------------------
class DeepELMExperiment : public Experiment {
public:
    DeepELMExperiment(const ExperimentConfig& cfg,
                      std::string data_path,
                      int   d0       = 256,
                      int   d        = 256,    // width of all ELM hidden layers
                      int   n_hidden = 1,      // # hidden ELM layers (excl. readout)
                      int   n_cycles = 5,
                      float lambda   = 1e-4f);

    void  setup()           override;
    float run_epoch(size_t) override;

private:
    std::string data_path_;
    int         d0_;
    int         d_;
    int         n_hidden_;
    int         n_cycles_;
    float       lambda_;

    std::shared_ptr<DenseLayer>              w0_;       // frozen projection [784 → d0]
    std::vector<std::shared_ptr<DenseLayer>> hidden_;   // n_hidden layers: [d0→d], [d→d]...
    std::shared_ptr<DenseLayer>              readout_;  // ELM readout [d → 10]
    ElmSolver                                solver_;

    Tensor  H0_dev_;    // FP32 [N_fit, d0] on device — precomputed once in setup()
    Tensor  T_dev_;     // FP32 [N_fit, 10] on device — one-hot labels
    size_t  N_fit_ = 0;

    void                precompute_h0_t();
    std::vector<Tensor> compute_hidden_activations() const;
    float               evaluate(DataSource& ds);
    void                write_fp32_weights(DenseLayer& layer, const Tensor& W_fp32);

    std::unique_ptr<MnistLoader> test_data_;
};

// ---------------------------------------------------------------------------
// HybridElmHebbianExperiment: N-layer hybrid — analytical readout each epoch,
// gradient steps on hidden layers.
//
// Per-epoch algorithm:
//   1. H_1,...,H_n = forward activations
//   2. readout = ELM(H_n, T, λ2)                [re-solved every epoch]
//   3. Back-propagate targets down through hidden layers:
//        T_k = propagate_target(W_{k+1}, T_{k+1}); apply_relu
//        W_k += (lr/N) * (T_k - H_k)^T @ H_{k-1}
//
// When elm_init_=true, epoch 0 first runs one deep-ELM cycle to warm-start
// all hidden layers before switching to the per-epoch ELM+gradient pattern.
// ---------------------------------------------------------------------------
class HybridElmHebbianExperiment : public Experiment {
public:
    HybridElmHebbianExperiment(const ExperimentConfig& cfg,
                               std::string data_path,
                               int   d0       = 256,
                               int   d        = 256,
                               int   n_hidden = 1,
                               float lambda2  = 1e-4f,
                               float lr_w     = 0.1f,
                               bool  elm_init = false);

    void  setup()           override;
    float run_epoch(size_t) override;

private:
    std::string data_path_;
    int         d0_, d_, n_hidden_;
    float       lambda2_, lr_w_;
    bool        elm_init_;

    std::shared_ptr<DenseLayer>              w0_;
    std::vector<std::shared_ptr<DenseLayer>> hidden_;
    std::shared_ptr<DenseLayer>              readout_;
    ElmSolver                                solver_;

    Tensor  H0_dev_;
    Tensor  T_dev_;
    size_t  N_fit_ = 0;

    void                precompute_h0_t();
    std::vector<Tensor> compute_hidden_activations() const;
    float               evaluate(DataSource& ds);
    void                write_fp32_weights(DenseLayer& layer, const Tensor& W_fp32);

    std::unique_ptr<MnistLoader> test_data_;
};

// ---------------------------------------------------------------------------
// AdmmElmExperiment: N-layer ELM trained via proximal ALS with LeakyReLU.
//
// Problem: treat pre-activations Z_k as free variables to decouple layers:
//
//   min_{W_k, Z_k}  ||H_n W_r^T - T||^2_F + lambda*sum||W_k||^2_F
//                 + (rho/2)*sum||Z_k - H_{k-1}W_k^T||^2_F   [linear constraint]
//                 + (mu/2)*sum||sigma(Z_k) - T_k||^2_F       [top-down target]
//
// where T_k is the Gram-propagated top-down target for H_k.
//
// Proximal ALS iteration (no dual variables — coordinate descent on the
// penalty objective, guaranteed non-increasing loss each step):
//
//   1. W_k  = solve(H_{k-1}, Z_k, lambda/rho)     [W maps H_prev → current Z_k]
//   2. W_r  = solve(H_n, T, lambda)               [ELM readout; H_n = sigma(Z_n)]
//   3. A_k  = H_{k-1} @ W_k^T                    [new linear pre-acts with updated W]
//   4. T_k  = Gram-propagate: T_n=prop(W_r,T); T_{k-1}=prop(W_k,T_k)  [top-down]
//   5. Z_k  = argmin_z (rho/2)(z-A_k)^2 + (mu/2)(sigma(z)-T_k)^2      [Z-update]
//             Closed-form element-wise case split (sigma = LeakyReLU, slope alpha):
//               c = A_k[i], t = T_k[i]
//               z1 = (rho*c + mu*t) / (rho+mu)            -- valid if z1 >= 0
//               z2 = (rho*c + mu*alpha*t) / (rho+mu*a^2)  -- valid if z2 < 0
//               else z = 0  (V-shaped; boundary is global min)
//
// Key properties vs. target propagation (DeepELMExperiment):
//   - Z_k BLENDS bottom-up (A_k from W) and top-down (T_k) information
//   - W is solved to match the current Z_k (not directly the propagated target)
//   - No dual variable accumulation → stable for non-convex problems
//   - Z step is (rho/(rho+mu)) A_k + (mu/(rho+mu)) sigma^{-1}(T_k) in the z>=0 case
//     → rho/mu ratio controls bottom-up vs top-down trade-off
//
// LeakyReLU (alpha > 0) is used throughout so sigma^{-1} is well-defined:
//   sigma^{-1}(y) = y if y>=0, y/alpha if y<0.  ReLU (alpha=0) is NOT invertible.
//   alpha=0.1 ensures the negative branch case-split stays within factor 10.
// ---------------------------------------------------------------------------
class AdmmElmExperiment : public Experiment {
public:
    AdmmElmExperiment(const ExperimentConfig& cfg,
                      std::string data_path,
                      int   d0          = 256,
                      int   d           = 256,
                      int   n_hidden    = 1,
                      int   n_admm      = 20,
                      float lambda      = 1e-4f,
                      float rho         = 1.f,
                      float mu          = 1.f,
                      float leaky_alpha = 0.1f);

    void  setup()           override;
    float run_epoch(size_t) override;

private:
    std::string data_path_;
    int   d0_, d_, n_hidden_, n_admm_;
    float lambda_, rho_, mu_, leaky_alpha_;

    std::shared_ptr<DenseLayer>              w0_;
    std::vector<std::shared_ptr<DenseLayer>> hidden_;
    std::shared_ptr<DenseLayer>              readout_;
    ElmSolver                                solver_;

    Tensor H0_dev_;
    Tensor T_dev_;
    size_t N_fit_ = 0;

    // Proximal ALS state
    std::vector<Tensor> Z_;   // consensus pre-activations [N_fit, d] per hidden layer
    Tensor              u_zero_;  // pre-allocated zero tensor [N_fit, d] — avoids per-iter alloc

    void                precompute_h0_t();
    std::vector<Tensor> get_hidden_activations() const;  // H_k = LeakyReLU(Z_k)
    float               evaluate(DataSource& ds);
    void                write_fp32_weights(DenseLayer& layer, const Tensor& W_fp32);

    std::unique_ptr<MnistLoader> test_data_;
};

} // namespace fayn
