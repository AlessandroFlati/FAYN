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

} // namespace fayn
