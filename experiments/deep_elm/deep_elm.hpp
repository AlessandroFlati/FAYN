#pragma once

#include "experiments/experiment.hpp"
#include "src/ops/dense.hpp"
#include "src/ops/elm_solver.hpp"
#include "src/io/mnist_loader.hpp"

#include <memory>
#include <string>

namespace fayn {

// ---------------------------------------------------------------------------
// DeepELMExperiment: alternating ELM with target propagation.
//
// Architecture:
//   x[N,784] → W_0[784→d0, frozen] → ReLU → H_0[N,d0]
//            → W_1[d0→d1,  ELM]    → ReLU → H_1[N,d1]
//            → W_2[d1→10,  ELM]    → Y[N,10]
//
// Algorithm (n_cycles iterations):
//   1. Precompute H_0 = ReLU(X W_0^T) once (W_0 frozen after construction).
//   2. Repeat:
//      a. H_1 = ReLU(H_0 W_1^T)               [ElmSolver::relu_forward]
//      b. W_2 = (H_1^T H_1 + λ2 I)^{-1} H_1^T T  [ElmSolver::solve]
//      c. H_1^* = T (W_2 W_2^T)^{-1} W_2      [ElmSolver::propagate_target]
//      d. W_1 = (H_0^T H_0 + λ1 I)^{-1} H_0^T H_1^*  [ElmSolver::solve]
//
// Convergence: each solve is the global optimum for that layer given the others
// fixed (coordinate descent on a non-convex objective) → monotone convergence.
//
// W_1 and W_2 use enable_fp32_weights() — full-precision forward and solve,
// no BF16 rounding of the analytical solution.
// ---------------------------------------------------------------------------
class DeepELMExperiment : public Experiment {
public:
    DeepELMExperiment(const ExperimentConfig& cfg,
                      std::string data_path,
                      int d0       = 256,
                      int d1       = 256,
                      int n_cycles = 5,
                      float lambda1 = 1e-4f,
                      float lambda2 = 1e-4f);

    void  setup()           override;
    float run_epoch(size_t) override;

private:
    std::string data_path_;
    int         d0_;
    int         d1_;
    int         n_cycles_;
    float       lambda1_;   // regularization for W_1 solve
    float       lambda2_;   // regularization for W_2 solve

    std::shared_ptr<DenseLayer> w0_;   // frozen projection [784 → d0]
    std::shared_ptr<DenseLayer> w1_;   // inner layer       [d0  → d1], FP32 weights
    std::shared_ptr<DenseLayer> w2_;   // readout           [d1  → 10], FP32 weights
    ElmSolver                   solver_;

    Tensor  H0_dev_;    // FP32 [N_fit, d0] on device — precomputed once in setup()
    Tensor  T_dev_;     // FP32 [N_fit, 10] on device — one-hot labels
    size_t  N_fit_ = 0; // number of training samples used (floor of 60k / batch_size)

    void   precompute_h0_t();
    Tensor compute_h1() const;
    float  evaluate();
    void   write_fp32_weights(DenseLayer& layer, const Tensor& W_fp32);
};

} // namespace fayn
