#include "registry.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Experiment registration.
// Include each experiment header and register it here. Static initializers
// in runner.cpp are guaranteed to run; registering here avoids the
// --whole-archive linker requirement for experiment static libraries.
// Add one block per experiment as it is created.
// ---------------------------------------------------------------------------
#include "experiments/hebbian_mnist/hebbian_mnist.hpp"
#include "experiments/ensemble_mnist/ensemble_mnist.hpp"
#include "experiments/deep_elm/deep_elm.hpp"

FAYN_REGISTER_EXPERIMENT("hebbian_mnist",  fayn::HebbianMnistExperiment,         hebbian_mnist)
FAYN_REGISTER_EXPERIMENT("ensemble_mnist", fayn::EnsembleHebbianMnistExperiment, ensemble_mnist)
FAYN_REGISTER_EXPERIMENT("elm_ensemble",   fayn::ELMEnsembleExperiment,          elm_ensemble)

// Scaled-init variants: d0 weights drawn from Uniform(-scale, +scale) where
// scale = sqrt(2/fan_in) * init_scale.  With init_scale ≈ sqrt(fan_in/2) = 19.8
// the range becomes Uniform(-1, 1), matching standard ELM initialisation.
// Pre-activation std ≈ 4.9 vs ≈ 0.27 for Kaiming — much more informative features.
namespace {
[[maybe_unused]] static const bool _fayn_reg_ensemble_mnist_scaled = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "ensemble_mnist_scaled",
        [](const fayn::ExperimentConfig& cfg) {
            constexpr float   kScale = 19.8f;  // sqrt(784/2) → Uniform(-1,1) range
            constexpr int64_t kSeed  = 42;     // reproducible d0 projections
            return std::make_unique<fayn::EnsembleHebbianMnistExperiment>(
                cfg, "data/mnist", /*lr=*/0.01f, /*K=*/10, /*norm_every=*/1,
                kScale, kSeed);
        });
    return true;
}();
[[maybe_unused]] static const bool _fayn_reg_elm_ensemble_scaled = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "elm_ensemble_scaled",
        [](const fayn::ExperimentConfig& cfg) {
            constexpr float   kScale = 19.8f;
            constexpr int64_t kSeed  = 42;     // same projections as ensemble_mnist_scaled
            return std::make_unique<fayn::ELMEnsembleExperiment>(
                cfg, "data/mnist", /*K=*/10, kScale, kSeed);
        });
    return true;
}();
// Feature-normalised Hebbian: L2-normalise each hidden vector before the
// outer product — makes H^T H ≈ (N/d)·I, pulling Hebbian toward ELM solution.
[[maybe_unused]] static const bool _fayn_reg_ensemble_mnist_normed = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "ensemble_mnist_normed",
        [](const fayn::ExperimentConfig& cfg) {
            constexpr float   kScale = 19.8f;
            constexpr int64_t kSeed  = 42;
            return std::make_unique<fayn::EnsembleHebbianMnistExperiment>(
                cfg, "data/mnist", /*lr=*/0.01f, /*K=*/10, /*norm_every=*/1,
                kScale, kSeed, /*normalize_pre=*/true);
        });
    return true;
}();
// Cosine-annealed LR: start at 0.1, decay to 0.001 over all training steps.
[[maybe_unused]] static const bool _fayn_reg_ensemble_mnist_lrs = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "ensemble_mnist_lrs",
        [](const fayn::ExperimentConfig& cfg) {
            constexpr float   kScale    = 19.8f;
            constexpr int64_t kSeed     = 42;
            constexpr float   kLrStart  = 0.1f;
            constexpr float   kLrFinal  = 0.001f;
            return std::make_unique<fayn::EnsembleHebbianMnistExperiment>(
                cfg, "data/mnist", /*lr=*/kLrStart, /*K=*/10, /*norm_every=*/1,
                kScale, kSeed, /*normalize_pre=*/false, kLrFinal);
        });
    return true;
}();
// Both: feature normalisation + cosine LR annealing.
[[maybe_unused]] static const bool _fayn_reg_ensemble_mnist_normed_lrs = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "ensemble_mnist_normed_lrs",
        [](const fayn::ExperimentConfig& cfg) {
            constexpr float   kScale    = 19.8f;
            constexpr int64_t kSeed     = 42;
            constexpr float   kLrStart  = 0.1f;
            constexpr float   kLrFinal  = 0.001f;
            return std::make_unique<fayn::EnsembleHebbianMnistExperiment>(
                cfg, "data/mnist", /*lr=*/kLrStart, /*K=*/10, /*norm_every=*/1,
                kScale, kSeed, /*normalize_pre=*/true, kLrFinal);
        });
    return true;
}();
// ---------------------------------------------------------------------------
// FAILED EXPERIMENTS — do not use.
// Weight-decay variants replacing row-norm with pre-step L2 decay.
// All plateau at ~55% due to BF16 precision ceiling: steady-state weight
// magnitude grows until Hebbian deltas (0.0025) fall below the BF16 step
// (0.016 at W*=2.5 for λ=1e-3). Weights freeze at a poor direction.
// For the λ that would keep W in the BF16-precise range (λ>0.028), memory
// horizon is <36 steps — too short to accumulate class prototypes.
// See CP-17 in PROGRESS.md.
// [[maybe_unused]] static const bool _fayn_reg_ensemble_mnist_wd1 = []() {
//     fayn::ExperimentRegistry::instance().register_experiment("ensemble_mnist_wd1e3", ...);
// [[maybe_unused]] static const bool _fayn_reg_ensemble_mnist_wd2 = []() {
//     fayn::ExperimentRegistry::instance().register_experiment("ensemble_mnist_wd1e4", ...);
// [[maybe_unused]] static const bool _fayn_reg_ensemble_mnist_wd3 = []() {
//     fayn::ExperimentRegistry::instance().register_experiment("ensemble_mnist_wd1e5", ...);
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Wide variants: hidden_dim = 2048 (8× more neurons than baseline).
// All other settings identical to the 256-neuron counterparts so results
// are directly comparable. ELM ceiling expected: ~93-95% (vs 86% at 256).
// ---------------------------------------------------------------------------
[[maybe_unused]] static const bool _fayn_reg_hebbian_mnist_2048 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "hebbian_mnist_2048",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::HebbianMnistExperiment>(
                cfg, "data/mnist", /*lr=*/0.01f, /*norm_every=*/1, /*hidden_dim=*/2048);
        });
    return true;
}();
[[maybe_unused]] static const bool _fayn_reg_ensemble_mnist_2048 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "ensemble_mnist_2048",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::EnsembleHebbianMnistExperiment>(
                cfg, "data/mnist", /*lr=*/0.01f, /*K=*/10, /*norm_every=*/1,
                /*scale=*/1.0f, /*seed=*/42LL,
                /*normalize_pre=*/false, /*lr_final=*/-1.f,
                /*row_normalize=*/true, /*weight_decay=*/0.f, /*hidden_dim=*/2048);
        });
    return true;
}();
[[maybe_unused]] static const bool _fayn_reg_ensemble_mnist_scaled_2048 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "ensemble_mnist_scaled_2048",
        [](const fayn::ExperimentConfig& cfg) {
            constexpr float   kScale = 19.8f;
            constexpr int64_t kSeed  = 42;
            return std::make_unique<fayn::EnsembleHebbianMnistExperiment>(
                cfg, "data/mnist", /*lr=*/0.01f, /*K=*/10, /*norm_every=*/1,
                kScale, kSeed,
                /*normalize_pre=*/false, /*lr_final=*/-1.f,
                /*row_normalize=*/true, /*weight_decay=*/0.f, /*hidden_dim=*/2048);
        });
    return true;
}();
[[maybe_unused]] static const bool _fayn_reg_elm_ensemble_2048 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "elm_ensemble_2048",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::ELMEnsembleExperiment>(
                cfg, "data/mnist", /*K=*/10, /*scale=*/1.0f, /*seed=*/42LL,
                /*hidden_dim=*/2048);
        });
    return true;
}();
[[maybe_unused]] static const bool _fayn_reg_elm_ensemble_scaled_2048 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "elm_ensemble_scaled_2048",
        [](const fayn::ExperimentConfig& cfg) {
            constexpr float   kScale = 19.8f;
            constexpr int64_t kSeed  = 42;
            return std::make_unique<fayn::ELMEnsembleExperiment>(
                cfg, "data/mnist", /*K=*/10, kScale, kSeed, /*hidden_dim=*/2048);
        });
    return true;
}();
// ---------------------------------------------------------------------------
// Delta-rule variants: post = (T − Ŷ) instead of post = T (SupervisedHebbian).
// This is the gradient of MSE and converges iteratively to the ELM solution
// (H^T H)^{-1} H^T T without a matrix solve. Self-stabilizing (no row-norm
// or weight-decay needed — updates → 0 as Ŷ → T).
// ---------------------------------------------------------------------------
[[maybe_unused]] static const bool _fayn_reg_ensemble_mnist_delta = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "ensemble_mnist_delta",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::EnsembleHebbianMnistExperiment>(
                cfg, "data/mnist", /*lr=*/0.01f, /*K=*/10, /*norm_every=*/1,
                /*scale=*/1.0f, /*seed=*/42LL,
                /*normalize_pre=*/false, /*lr_final=*/-1.f,
                /*row_normalize=*/false, /*weight_decay=*/0.f,
                /*hidden_dim=*/256, /*use_delta_rule=*/true);
        });
    return true;
}();
[[maybe_unused]] static const bool _fayn_reg_ensemble_mnist_delta_scaled = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "ensemble_mnist_delta_scaled",
        [](const fayn::ExperimentConfig& cfg) {
            constexpr float   kScale = 19.8f;
            constexpr int64_t kSeed  = 42;
            return std::make_unique<fayn::EnsembleHebbianMnistExperiment>(
                cfg, "data/mnist", /*lr=*/0.01f, /*K=*/10, /*norm_every=*/1,
                kScale, kSeed,
                /*normalize_pre=*/false, /*lr_final=*/-1.f,
                /*row_normalize=*/false, /*weight_decay=*/0.f,
                /*hidden_dim=*/256, /*use_delta_rule=*/true);
        });
    return true;
}();
[[maybe_unused]] static const bool _fayn_reg_ensemble_mnist_delta_2048 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "ensemble_mnist_delta_2048",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::EnsembleHebbianMnistExperiment>(
                cfg, "data/mnist", /*lr=*/0.01f, /*K=*/10, /*norm_every=*/1,
                /*scale=*/1.0f, /*seed=*/42LL,
                /*normalize_pre=*/false, /*lr_final=*/-1.f,
                /*row_normalize=*/false, /*weight_decay=*/0.f,
                /*hidden_dim=*/2048, /*use_delta_rule=*/true);
        });
    return true;
}();
[[maybe_unused]] static const bool _fayn_reg_ensemble_mnist_delta_scaled_2048 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "ensemble_mnist_delta_scaled_2048",
        [](const fayn::ExperimentConfig& cfg) {
            constexpr float   kScale = 19.8f;
            constexpr int64_t kSeed  = 42;
            return std::make_unique<fayn::EnsembleHebbianMnistExperiment>(
                cfg, "data/mnist", /*lr=*/0.01f, /*K=*/10, /*norm_every=*/1,
                kScale, kSeed,
                /*normalize_pre=*/false, /*lr_final=*/-1.f,
                /*row_normalize=*/false, /*weight_decay=*/0.f,
                /*hidden_dim=*/2048, /*use_delta_rule=*/true);
        });
    return true;
}();
// Delta-rule + normalize_pre: L2-normalise each hidden vector before outer
// product so that lambda_max(H^T H) is independent of init_scale.
// Fixes the LMS instability of the scaled-init delta-rule variants above.
[[maybe_unused]] static const bool _fayn_reg_ensemble_mnist_delta_scaled_normed = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "ensemble_mnist_delta_scaled_normed",
        [](const fayn::ExperimentConfig& cfg) {
            constexpr float   kScale = 19.8f;
            constexpr int64_t kSeed  = 42;
            return std::make_unique<fayn::EnsembleHebbianMnistExperiment>(
                cfg, "data/mnist", /*lr=*/0.01f, /*K=*/10, /*norm_every=*/1,
                kScale, kSeed,
                /*normalize_pre=*/true, /*lr_final=*/-1.f,
                /*row_normalize=*/false, /*weight_decay=*/0.f,
                /*hidden_dim=*/256, /*use_delta_rule=*/true);
        });
    return true;
}();
[[maybe_unused]] static const bool _fayn_reg_ensemble_mnist_delta_scaled_2048_normed = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "ensemble_mnist_delta_scaled_2048_normed",
        [](const fayn::ExperimentConfig& cfg) {
            constexpr float   kScale = 19.8f;
            constexpr int64_t kSeed  = 42;
            return std::make_unique<fayn::EnsembleHebbianMnistExperiment>(
                cfg, "data/mnist", /*lr=*/0.01f, /*K=*/10, /*norm_every=*/1,
                kScale, kSeed,
                /*normalize_pre=*/true, /*lr_final=*/-1.f,
                /*row_normalize=*/false, /*weight_decay=*/0.f,
                /*hidden_dim=*/2048, /*use_delta_rule=*/true);
        });
    return true;
}();
// ---------------------------------------------------------------------------
// CIW (Cluster-based Input Weight) variants.
// d0 weights are replaced by k-means centroids (100 mini-batch iterations,
// batch_sz=256) learned from the training images, then L2-normalised.
// Each ensemble member uses a distinct seed for centroid diversity.
// Delta rule on d1 (FP32 accumulation) for both 256-neuron and 2048-neuron.
// ELM baseline with CIW features replaces the Hebbian readout with the exact
// normal-equations solution — measures the full information content of CIW.
// ---------------------------------------------------------------------------
[[maybe_unused]] static const bool _fayn_reg_ensemble_mnist_delta_ciw = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "ensemble_mnist_delta_ciw",
        [](const fayn::ExperimentConfig& cfg) {
            // CIW features are all-positive with E[H_ij] ≈ 3.6.  The Gram matrix
            // H^T H / B has λ_max ≈ d * E[H²] ≈ 256 * 13 ≈ 3300.  Delta rule
            // convergence requires lr < 2/λ_max ≈ 6e-4.  lr=0.001 is above the
            // stability threshold (divergent); lr=2e-4 is safely below it.
            return std::make_unique<fayn::EnsembleHebbianMnistExperiment>(
                cfg, "data/mnist", /*lr=*/2e-4f, /*K=*/10, /*norm_every=*/1,
                /*scale=*/1.0f, /*seed=*/42LL,
                /*normalize_pre=*/false, /*lr_final=*/-1.f,
                /*row_normalize=*/false, /*weight_decay=*/0.f,
                /*hidden_dim=*/256, /*use_delta_rule=*/true,
                /*use_ciw=*/true);
        });
    return true;
}();
[[maybe_unused]] static const bool _fayn_reg_ensemble_mnist_delta_ciw_2048 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "ensemble_mnist_delta_ciw_2048",
        [](const fayn::ExperimentConfig& cfg) {
            // λ_max scales linearly with hidden_dim: 2048h needs 8× smaller lr
            // than 256h.  lr=2e-4/8=2.5e-5; use 2e-5 for a comfortable margin.
            return std::make_unique<fayn::EnsembleHebbianMnistExperiment>(
                cfg, "data/mnist", /*lr=*/2e-5f, /*K=*/10, /*norm_every=*/1,
                /*scale=*/1.0f, /*seed=*/42LL,
                /*normalize_pre=*/false, /*lr_final=*/-1.f,
                /*row_normalize=*/false, /*weight_decay=*/0.f,
                /*hidden_dim=*/2048, /*use_delta_rule=*/true,
                /*use_ciw=*/true);
        });
    return true;
}();
[[maybe_unused]] static const bool _fayn_reg_elm_ensemble_ciw = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "elm_ensemble_ciw",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::ELMEnsembleExperiment>(
                cfg, "data/mnist", /*K=*/10, /*scale=*/1.0f, /*seed=*/42LL,
                /*hidden_dim=*/256, /*use_ciw=*/true);
        });
    return true;
}();
[[maybe_unused]] static const bool _fayn_reg_elm_ensemble_ciw_2048 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "elm_ensemble_ciw_2048",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::ELMEnsembleExperiment>(
                cfg, "data/mnist", /*K=*/10, /*scale=*/1.0f, /*seed=*/42LL,
                /*hidden_dim=*/2048, /*use_ciw=*/true);
        });
    return true;
}();
// ---------------------------------------------------------------------------
// Deep alternating ELM: two analytically-solved layers with target propagation.
// W_0 frozen random; W_1 and W_2 solved alternately via normal equations.
// Convergence: each solve is the global optimum for that layer given the other
// fixed (coordinate descent) → monotone convergence to a local minimum.
// ---------------------------------------------------------------------------
[[maybe_unused]] static const bool _fayn_reg_deep_elm_256 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_256",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/256, /*d=*/256, /*n_hidden=*/1, /*n_cycles=*/20,
                /*lambda=*/1e-4f);
        });
    return true;
}();
[[maybe_unused]] static const bool _fayn_reg_deep_elm_800 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_800",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/800, /*d=*/800, /*n_hidden=*/1, /*n_cycles=*/20,
                /*lambda=*/1e-4f);
        });
    return true;
}();
// Wide random projection (8192) compressed into a 800-dim ELM hidden layer.
// Isolates the effect of random feature width from learned representation width.
[[maybe_unused]] static const bool _fayn_reg_deep_elm_8192_800 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_8192_800",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/8192, /*d=*/800, /*n_hidden=*/1, /*n_cycles=*/20,
                /*lambda=*/1e-4f);
        });
    return true;
}();
// Wide random projection (8192) into a 1600-dim ELM hidden layer — tests whether
// doubling learned width beyond 800 further closes the gap to backprop baseline.
[[maybe_unused]] static const bool _fayn_reg_deep_elm_8192_1600 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_8192_1600",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/8192, /*d=*/1600, /*n_hidden=*/1, /*n_cycles=*/20,
                /*lambda=*/1e-4f);
        });
    return true;
}();
[[maybe_unused]] static const bool _fayn_reg_deep_elm_32k_800 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_32k_800",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/32768, /*d=*/800, /*n_hidden=*/1, /*n_cycles=*/20,
                /*lambda=*/1e-4f);
        });
    return true;
}();
// ---------------------------------------------------------------------------
// ADMM-ELM experiments
// ---------------------------------------------------------------------------
// Baseline: d0=256, d=256, 1 hidden layer — directly comparable to deep_elm_256.
[[maybe_unused]] static const bool _fayn_reg_admm_elm_256 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "admm_elm_256",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::AdmmElmExperiment>(
                cfg, "data/mnist",
                /*d0=*/256, /*d=*/256, /*n_hidden=*/1,
                /*n_admm=*/20, /*lambda=*/1e-4f,
                /*rho=*/1.f, /*mu=*/1.f, /*leaky_alpha=*/0.1f);
        });
    return true;
}();
// Wide frozen projection sweep: d0=8192, d=800 — comparable to deep_elm_8192_800.
[[maybe_unused]] static const bool _fayn_reg_admm_elm_8192_800 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "admm_elm_8192_800",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::AdmmElmExperiment>(
                cfg, "data/mnist",
                /*d0=*/8192, /*d=*/800, /*n_hidden=*/1,
                /*n_admm=*/20, /*lambda=*/1e-4f,
                /*rho=*/1.f, /*mu=*/1.f, /*leaky_alpha=*/0.1f);
        });
    return true;
}();
[[maybe_unused]] static const bool _fayn_reg_deep_elm_2048 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_2048",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/2048, /*d=*/2048, /*n_hidden=*/1, /*n_cycles=*/5,
                /*lambda=*/1e-4f);
        });
    return true;
}();
// ---------------------------------------------------------------------------
// Hybrid ELM+gradient experiments: re-solve readout analytically each epoch
// and apply one full-batch gradient step on each hidden layer (delta-rule
// using propagated targets).
// ---------------------------------------------------------------------------
[[maybe_unused]] static const bool _fayn_reg_deep_elm_hebb_256 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_hebb_256",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::HybridElmHebbianExperiment>(
                cfg, "data/mnist", /*d0=*/256, /*d=*/256, /*n_hidden=*/1,
                /*lambda2=*/1e-4f, /*lr_w=*/0.1f, /*elm_init=*/false);
        });
    return true;
}();
[[maybe_unused]] static const bool _fayn_reg_deep_elm_init_hebb_256 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_init_hebb_256",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::HybridElmHebbianExperiment>(
                cfg, "data/mnist", /*d0=*/256, /*d=*/256, /*n_hidden=*/1,
                /*lambda2=*/1e-4f, /*lr_w=*/0.1f, /*elm_init=*/true);
        });
    return true;
}();
[[maybe_unused]] static const bool _fayn_reg_deep_elm_init_hebb_2048 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_init_hebb_2048",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::HybridElmHebbianExperiment>(
                cfg, "data/mnist", /*d0=*/2048, /*d=*/2048, /*n_hidden=*/1,
                /*lambda2=*/1e-4f, /*lr_w=*/0.1f, /*elm_init=*/true);
        });
    return true;
}();
// Lower lr for 2048-dim: 0.1 overshoots the ELM warm-start drastically.
[[maybe_unused]] static const bool _fayn_reg_deep_elm_init_hebb_2048_lr01 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_init_hebb_2048_lr01",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::HybridElmHebbianExperiment>(
                cfg, "data/mnist", /*d0=*/2048, /*d=*/2048, /*n_hidden=*/1,
                /*lambda2=*/1e-4f, /*lr_w=*/0.01f, /*elm_init=*/true);
        });
    return true;
}();
// ---------------------------------------------------------------------------
// N-layer experiments at d=4096: 2 hidden ELM layers + readout (3 trained
// layers total). cuSOLVER handles d=4096 solves without CPU round-trips.
// ---------------------------------------------------------------------------
// Pure alternating ELM: 5 cycles on epoch 0, then fixed network.
[[maybe_unused]] static const bool _fayn_reg_deep_elm_4096_L3 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_4096_L3",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/4096, /*d=*/4096, /*n_hidden=*/2, /*n_cycles=*/5,
                /*lambda=*/1e-4f);
        });
    return true;
}();
// ---------------------------------------------------------------------------
// Depth vs. activation analysis: 5-layer networks (4 ELM hidden + readout)
// at d=512, compared across activation function and algorithm.
//
// Architecture: 784 → 512 (frozen ReLU) → [512]×4 (trained) → 10 (readout)
//
// Activation comparison:
//   ReLU  (non-invertible): target propagation clips negative targets to 0,
//          introducing systematic error that compounds over 4 layers.
//   tanh  (bijective ℝ→(-1,1)): propagated targets used as-is (no clipping);
//          Z-update uses atanh blend — exact in pre-activation space.
//   LeakyReLU (bijective ℝ→ℝ, alpha=0.1): case-split Z-update (exact).
//
// Algorithm comparison:
//   target_prop: solve each layer to fit propagated targets (top-down only).
//   proximal_als: Z_k blends bottom-up (A_k) and top-down (T_k) each iter.
// ---------------------------------------------------------------------------
// Single-layer baselines (L1) — reference for depth contribution.
[[maybe_unused]] static const bool _fayn_reg_deep_elm_relu_512_L1 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_relu_512_L1",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/512, /*d=*/512, /*n_hidden=*/1, /*n_cycles=*/20,
                /*lambda=*/1e-4f, /*use_tanh=*/false);
        });
    return true;
}();
[[maybe_unused]] static const bool _fayn_reg_admm_elm_tanh_512_L1 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "admm_elm_tanh_512_L1",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::AdmmElmExperiment>(
                cfg, "data/mnist",
                /*d0=*/512, /*d=*/512, /*n_hidden=*/1,
                /*n_admm=*/20, /*lambda=*/1e-4f,
                /*rho=*/1.f, /*mu=*/1.f, /*leaky_alpha=*/0.1f,
                /*use_tanh=*/true);
        });
    return true;
}();
// 5-layer deep networks (n_hidden=4 ELM layers + readout).
[[maybe_unused]] static const bool _fayn_reg_deep_elm_relu_512_L5 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_relu_512_L5",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/512, /*d=*/512, /*n_hidden=*/4, /*n_cycles=*/20,
                /*lambda=*/1e-4f, /*use_tanh=*/false);
        });
    return true;
}();
[[maybe_unused]] static const bool _fayn_reg_deep_elm_tanh_512_L5 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_tanh_512_L5",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/512, /*d=*/512, /*n_hidden=*/4, /*n_cycles=*/20,
                /*lambda=*/1e-4f, /*use_tanh=*/true);
        });
    return true;
}();
[[maybe_unused]] static const bool _fayn_reg_admm_elm_leaky_512_L5 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "admm_elm_leaky_512_L5",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::AdmmElmExperiment>(
                cfg, "data/mnist",
                /*d0=*/512, /*d=*/512, /*n_hidden=*/4,
                /*n_admm=*/20, /*lambda=*/1e-4f,
                /*rho=*/1.f, /*mu=*/1.f, /*leaky_alpha=*/0.1f,
                /*use_tanh=*/false);
        });
    return true;
}();
[[maybe_unused]] static const bool _fayn_reg_admm_elm_tanh_512_L5 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "admm_elm_tanh_512_L5",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::AdmmElmExperiment>(
                cfg, "data/mnist",
                /*d0=*/512, /*d=*/512, /*n_hidden=*/4,
                /*n_admm=*/20, /*lambda=*/1e-4f,
                /*rho=*/1.f, /*mu=*/1.f, /*leaky_alpha=*/0.1f,
                /*use_tanh=*/true);
        });
    return true;
}();
// ---------------------------------------------------------------------------
// Width-1024 sweep: same 2×2 (algorithm × activation) matrix as 512, plus
// single-layer references. Wider layers give more capacity — tests whether
// depth benefit emerges once the bottleneck is relaxed.
// ---------------------------------------------------------------------------
[[maybe_unused]] static const bool _fayn_reg_deep_elm_relu_1024_L1 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_relu_1024_L1",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/1024, /*d=*/1024, /*n_hidden=*/1, /*n_cycles=*/20,
                /*lambda=*/1e-4f, /*use_tanh=*/false);
        });
    return true;
}();
[[maybe_unused]] static const bool _fayn_reg_admm_elm_tanh_1024_L1 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "admm_elm_tanh_1024_L1",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::AdmmElmExperiment>(
                cfg, "data/mnist",
                /*d0=*/1024, /*d=*/1024, /*n_hidden=*/1,
                /*n_admm=*/20, /*lambda=*/1e-4f,
                /*rho=*/1.f, /*mu=*/1.f, /*leaky_alpha=*/0.1f,
                /*use_tanh=*/true);
        });
    return true;
}();
[[maybe_unused]] static const bool _fayn_reg_deep_elm_relu_1024_L5 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_relu_1024_L5",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/1024, /*d=*/1024, /*n_hidden=*/4, /*n_cycles=*/20,
                /*lambda=*/1e-4f, /*use_tanh=*/false);
        });
    return true;
}();
[[maybe_unused]] static const bool _fayn_reg_deep_elm_tanh_1024_L5 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_tanh_1024_L5",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/1024, /*d=*/1024, /*n_hidden=*/4, /*n_cycles=*/20,
                /*lambda=*/1e-4f, /*use_tanh=*/true);
        });
    return true;
}();
[[maybe_unused]] static const bool _fayn_reg_admm_elm_leaky_1024_L5 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "admm_elm_leaky_1024_L5",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::AdmmElmExperiment>(
                cfg, "data/mnist",
                /*d0=*/1024, /*d=*/1024, /*n_hidden=*/4,
                /*n_admm=*/20, /*lambda=*/1e-4f,
                /*rho=*/1.f, /*mu=*/1.f, /*leaky_alpha=*/0.1f,
                /*use_tanh=*/false);
        });
    return true;
}();
[[maybe_unused]] static const bool _fayn_reg_admm_elm_tanh_1024_L5 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "admm_elm_tanh_1024_L5",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::AdmmElmExperiment>(
                cfg, "data/mnist",
                /*d0=*/1024, /*d=*/1024, /*n_hidden=*/4,
                /*n_admm=*/20, /*lambda=*/1e-4f,
                /*rho=*/1.f, /*mu=*/1.f, /*leaky_alpha=*/0.1f,
                /*use_tanh=*/true);
        });
    return true;
}();
// ELM warm-start on epoch 0, then gradient steps on both hidden layers.
[[maybe_unused]] static const bool _fayn_reg_deep_elm_init_hebb_4096_L3 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_init_hebb_4096_L3",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::HybridElmHebbianExperiment>(
                cfg, "data/mnist",
                /*d0=*/4096, /*d=*/4096, /*n_hidden=*/2,
                /*lambda2=*/1e-4f, /*lr_w=*/0.01f, /*elm_init=*/true);
        });
    return true;
}();
// ---------------------------------------------------------------------------
// Random Fourier Features (RFF): W_0 rows ~ N(0, rff_gamma * I_784),
// activation = cos(W_0 x + b), b ~ U[0, 2π].
//
// By Bochner's theorem (Rahimi & Recht, 2007), this approximates:
//   K_RBF(x,y) = exp(-rff_gamma * ||x-y||^2)
//
// rff_gamma=0.01: ||x||^2 ≈ 50-100 for digit images → Var[w·x] ≈ 0.5-1.0
//   → cos argument std ≈ 0.7-1.0 (good frequency spread across the dataset).
//
// L0 = pure kernel ridge regression (no ELM hidden layer), direct comparison
//      with kernel SVM (98.6%). L1/L3 add ELM depth on top of the RFF basis.
// ReLU L1 baseline (d=4096) added for fair single-layer comparison.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Training-time augmentation: each training image is accompanied by 4 pixel-
// shifted copies (left/right/up/down by 1 pixel).  The ELM fit sees a 5×
// larger H matrix (299520 rows instead of 59904) with the same labels.
// The Gram computation cost scales as O(N·d²) so it is 5× more expensive,
// but the solve (O(d³)) is unchanged.  Inference uses the original images.
//
// Expected gain over pure ELM: ~+0.3-0.5% from improved translation-invariance.
// ---------------------------------------------------------------------------
[[maybe_unused]] static const bool _fayn_reg_deep_elm_aug_4096_L3 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_aug_4096_L3",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/4096, /*d=*/4096, /*n_hidden=*/2, /*n_cycles=*/5,
                /*lambda=*/1e-4f, /*use_tanh=*/false, /*use_rff=*/false,
                /*rff_gamma=*/0.01f, /*augment=*/true);
        });
    return true;
}();
// RFF + augmentation: kernel features with 5× training coverage.
[[maybe_unused]] static const bool _fayn_reg_deep_elm_rff_aug_4096_L0 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_rff_aug_4096_L0",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/4096, /*d=*/4096, /*n_hidden=*/0, /*n_cycles=*/1,
                /*lambda=*/1e-4f, /*use_tanh=*/false, /*use_rff=*/true,
                /*rff_gamma=*/0.01f, /*augment=*/true);
        });
    return true;
}();
// ---------------------------------------------------------------------------
// Wider frozen projection: d0=8192 (2× more random features), ELM d=4096.
// Tests whether a richer random basis improves over d0=4096.
// First ELM layer compresses 8192 → 4096 (dimensionality reduction).
// H0^T H0 is [8192, 8192] → 256 MB, feasible.
// ---------------------------------------------------------------------------
[[maybe_unused]] static const bool _fayn_reg_deep_elm_8192_4096_L3 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_8192_4096_L3",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/8192, /*d=*/4096, /*n_hidden=*/2, /*n_cycles=*/5,
                /*lambda=*/1e-4f);
        });
    return true;
}();
// λ=1e-3: stronger regularisation to close the train/test gap seen at λ=1e-4.
[[maybe_unused]] static const bool _fayn_reg_deep_elm_8192_4096_L3_lam1e3 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_8192_4096_L3_lam1e3",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/8192, /*d=*/4096, /*n_hidden=*/2, /*n_cycles=*/5,
                /*lambda=*/1e-3f);
        });
    return true;
}();
// λ=1e-2: even stronger regularisation.
[[maybe_unused]] static const bool _fayn_reg_deep_elm_8192_4096_L3_lam1e2 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_8192_4096_L3_lam1e2",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/8192, /*d=*/4096, /*n_hidden=*/2, /*n_cycles=*/5,
                /*lambda=*/1e-2f);
        });
    return true;
}();
// Augmentation with properly scaled lambda: 5× more samples → λ × 5 to maintain
// the same effective regularisation per sample (H^T H eigenvalues scale as N).
[[maybe_unused]] static const bool _fayn_reg_deep_elm_aug_4096_L3_lam5 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_aug_4096_L3_lam5",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/4096, /*d=*/4096, /*n_hidden=*/2, /*n_cycles=*/5,
                /*lambda=*/5e-4f, /*use_tanh=*/false, /*use_rff=*/false,
                /*rff_gamma=*/0.01f, /*augment=*/true);
        });
    return true;
}();
// Deep ReLU L5 at d=4096: 4 ELM hidden layers + readout. Tests whether depth
// benefit continues past L3 (98.04%) at d=4096, before implementing augmentation.
[[maybe_unused]] static const bool _fayn_reg_deep_elm_relu_4096_L5 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_relu_4096_L5",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/4096, /*d=*/4096, /*n_hidden=*/4, /*n_cycles=*/5,
                /*lambda=*/1e-4f);
        });
    return true;
}();
// ReLU L1 at d=4096: baseline for single ELM hidden layer at this width.
[[maybe_unused]] static const bool _fayn_reg_deep_elm_relu_4096_L1 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_relu_4096_L1",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/4096, /*d=*/4096, /*n_hidden=*/1, /*n_cycles=*/5,
                /*lambda=*/1e-4f);
        });
    return true;
}();
// RFF L0: W_0 cos features → linear readout only (kernel ridge regression).
[[maybe_unused]] static const bool _fayn_reg_deep_elm_rff_4096_L0 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_rff_4096_L0",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/4096, /*d=*/4096, /*n_hidden=*/0, /*n_cycles=*/1,
                /*lambda=*/1e-4f, /*use_tanh=*/false, /*use_rff=*/true, /*rff_gamma=*/0.01f);
        });
    return true;
}();
[[maybe_unused]] static const bool _fayn_reg_deep_elm_rff_4096_L1 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_rff_4096_L1",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/4096, /*d=*/4096, /*n_hidden=*/1, /*n_cycles=*/5,
                /*lambda=*/1e-4f, /*use_tanh=*/false, /*use_rff=*/true, /*rff_gamma=*/0.01f);
        });
    return true;
}();
[[maybe_unused]] static const bool _fayn_reg_deep_elm_rff_4096_L3 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_rff_4096_L3",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/4096, /*d=*/4096, /*n_hidden=*/2, /*n_cycles=*/5,
                /*lambda=*/1e-4f, /*use_tanh=*/false, /*use_rff=*/true, /*rff_gamma=*/0.01f);
        });
    return true;
}();
// Convolutional front-end experiments.
// Replace W_0 [784→d0] with 5×5 conv (C_out filters) + ReLU + 2×2 max-pool.
// Output: C_out * 12 * 12 features. Kaiming-init frozen filters.
//
// deep_elm_conv32_L1: C_out=32, d0=4608, d=4096, 1 ELM hidden layer + readout.
[[maybe_unused]] static const bool _fayn_reg_deep_elm_conv32_L1 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_conv32_L1",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/4608, /*d=*/4096, /*n_hidden=*/1, /*n_cycles=*/5,
                /*lambda=*/1e-4f, /*use_tanh=*/false, /*use_rff=*/false,
                /*rff_gamma=*/0.01f, /*augment=*/false, /*use_tta=*/false,
                /*use_conv=*/true, /*conv_c_out=*/32);
        });
    return true;
}();
// deep_elm_conv32_L3: C_out=32, d0=4608, d=4096, 2 ELM hidden layers + readout.
[[maybe_unused]] static const bool _fayn_reg_deep_elm_conv32_L3 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_conv32_L3",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/4608, /*d=*/4096, /*n_hidden=*/2, /*n_cycles=*/5,
                /*lambda=*/1e-4f, /*use_tanh=*/false, /*use_rff=*/false,
                /*rff_gamma=*/0.01f, /*augment=*/false, /*use_tta=*/false,
                /*use_conv=*/true, /*conv_c_out=*/32);
        });
    return true;
}();
// deep_elm_conv64_L1: C_out=64, d0=9216, d=4096, 1 ELM hidden layer + readout.
[[maybe_unused]] static const bool _fayn_reg_deep_elm_conv64_L1 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_conv64_L1",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/9216, /*d=*/4096, /*n_hidden=*/1, /*n_cycles=*/5,
                /*lambda=*/1e-4f, /*use_tanh=*/false, /*use_rff=*/false,
                /*rff_gamma=*/0.01f, /*augment=*/false, /*use_tta=*/false,
                /*use_conv=*/true, /*conv_c_out=*/64);
        });
    return true;
}();
// deep_elm_conv64_L3: C_out=64, d0=9216, d=4096, 2 ELM hidden layers + readout.
[[maybe_unused]] static const bool _fayn_reg_deep_elm_conv64_L3 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_conv64_L3",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/9216, /*d=*/4096, /*n_hidden=*/2, /*n_cycles=*/5,
                /*lambda=*/1e-4f, /*use_tanh=*/false, /*use_rff=*/false,
                /*rff_gamma=*/0.01f, /*augment=*/false, /*use_tta=*/false,
                /*use_conv=*/true, /*conv_c_out=*/64);
        });
    return true;
}();
// deep_elm_conv128_L1: C_out=128, d0=18432, d=4096, 1 ELM hidden layer + readout.
[[maybe_unused]] static const bool _fayn_reg_deep_elm_conv128_L1 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_conv128_L1",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/18432, /*d=*/4096, /*n_hidden=*/1, /*n_cycles=*/5,
                /*lambda=*/1e-4f, /*use_tanh=*/false, /*use_rff=*/false,
                /*rff_gamma=*/0.01f, /*augment=*/false, /*use_tta=*/false,
                /*use_conv=*/true, /*conv_c_out=*/128);
        });
    return true;
}();
// Regularization sweep for conv64_L3 to close the 1.26% train-test gap.
[[maybe_unused]] static const bool _fayn_reg_deep_elm_conv64_L3_lam5e4 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_conv64_L3_lam5e4",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/9216, /*d=*/4096, /*n_hidden=*/2, /*n_cycles=*/5,
                /*lambda=*/5e-4f, /*use_tanh=*/false, /*use_rff=*/false,
                /*rff_gamma=*/0.01f, /*augment=*/false, /*use_tta=*/false,
                /*use_conv=*/true, /*conv_c_out=*/64);
        });
    return true;
}();
[[maybe_unused]] static const bool _fayn_reg_deep_elm_conv64_L3_lam1e3 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_conv64_L3_lam1e3",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/9216, /*d=*/4096, /*n_hidden=*/2, /*n_cycles=*/5,
                /*lambda=*/1e-3f, /*use_tanh=*/false, /*use_rff=*/false,
                /*rff_gamma=*/0.01f, /*augment=*/false, /*use_tta=*/false,
                /*use_conv=*/true, /*conv_c_out=*/64);
        });
    return true;
}();

// deep_elm_conv64_L1_learned: C_out=64 conv filters updated by ELM solve each cycle,
// 1 ELM hidden layer + readout. Tests whether learning conv improves over frozen.
[[maybe_unused]] static const bool _fayn_reg_deep_elm_conv64_L1_learned = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_conv64_L1_learned",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/9216, /*d=*/4096, /*n_hidden=*/1, /*n_cycles=*/5,
                /*lambda=*/1e-4f, /*use_tanh=*/false, /*use_rff=*/false,
                /*rff_gamma=*/0.01f, /*augment=*/false, /*use_tta=*/false,
                /*use_conv=*/true, /*conv_c_out=*/64, /*learn_conv=*/true);
        });
    return true;
}();
// deep_elm_conv64_L3_learned: C_out=64 learned conv, 2 ELM hidden layers + readout.
[[maybe_unused]] static const bool _fayn_reg_deep_elm_conv64_L3_learned = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_conv64_L3_learned",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/9216, /*d=*/4096, /*n_hidden=*/2, /*n_cycles=*/5,
                /*lambda=*/1e-4f, /*use_tanh=*/false, /*use_rff=*/false,
                /*rff_gamma=*/0.01f, /*augment=*/false, /*use_tta=*/false,
                /*use_conv=*/true, /*conv_c_out=*/64, /*learn_conv=*/true);
        });
    return true;
}();

// ---------------------------------------------------------------------------
// Deep ELM ensemble: M independent random projections W_0, each with their
// own ELM-solved hidden + readout. Inference averages logits over all members.
// ---------------------------------------------------------------------------
// 5 members, each d0=4096, d=4096, 1 hidden ELM layer + readout (L2 net).
[[maybe_unused]] static const bool _fayn_reg_deep_elm_ensemble_5x4096_L2 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_ensemble_5x4096_L2",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMEnsembleExperiment>(
                cfg, "data/mnist",
                /*n_members=*/5, /*d0=*/4096, /*d=*/4096,
                /*n_hidden=*/1, /*n_cycles=*/5, /*lambda=*/1e-4f);
        });
    return true;
}();
// 3 members, each d0=4096, d=4096, 2 hidden ELM layers + readout (L3 net).
[[maybe_unused]] static const bool _fayn_reg_deep_elm_ensemble_3x4096_L3 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_ensemble_3x4096_L3",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMEnsembleExperiment>(
                cfg, "data/mnist",
                /*n_members=*/3, /*d0=*/4096, /*d=*/4096,
                /*n_hidden=*/2, /*n_cycles=*/5, /*lambda=*/1e-4f);
        });
    return true;
}();

// Multi-scale conv ensemble: 3 members with K=3, K=5, K=7 (one per scale), n_hidden=1 (L2).
// d0 per member: 64×169=10816 (K=3), 64×144=9216 (K=5), 64×121=7744 (K=7).
[[maybe_unused]] static const bool _fayn_reg_deep_elm_multiscale3_ensemble_3xL2 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_multiscale3_ensemble_3xL2",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMEnsembleExperiment>(
                cfg, "data/mnist",
                /*n_members=*/3, /*d0=*/0, /*d=*/4096,
                /*n_hidden=*/1, /*n_cycles=*/5, /*lambda=*/1e-4f,
                /*use_conv=*/true, /*conv_c_out=*/64,
                /*conv_k_per_member=*/std::vector<int>{3, 5, 7});
        });
    return true;
}();
// Multi-scale conv ensemble: 6 members with K=[3,3,5,5,7,7], n_hidden=1 (L2).
[[maybe_unused]] static const bool _fayn_reg_deep_elm_multiscale3_ensemble_6xL2 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_multiscale3_ensemble_6xL2",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMEnsembleExperiment>(
                cfg, "data/mnist",
                /*n_members=*/6, /*d0=*/0, /*d=*/4096,
                /*n_hidden=*/1, /*n_cycles=*/5, /*lambda=*/1e-4f,
                /*use_conv=*/true, /*conv_c_out=*/64,
                /*conv_k_per_member=*/std::vector<int>{3, 3, 5, 5, 7, 7});
        });
    return true;
}();

// Conv64 ensemble: 3 members, C_out=64, n_hidden=2 (L3 net), n_cycles=5.
// d0=9216 (64 filters × 144 spatial positions after 2×2 max-pool).
[[maybe_unused]] static const bool _fayn_reg_deep_elm_conv64_ensemble_3xL3 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_conv64_ensemble_3xL3",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMEnsembleExperiment>(
                cfg, "data/mnist",
                /*n_members=*/3, /*d0=*/9216, /*d=*/4096,
                /*n_hidden=*/2, /*n_cycles=*/5, /*lambda=*/1e-4f,
                /*use_conv=*/true, /*conv_c_out=*/64);
        });
    return true;
}();
// Conv64 ensemble: 5 members, C_out=64, n_hidden=1 (L2 net), n_cycles=5.
[[maybe_unused]] static const bool _fayn_reg_deep_elm_conv64_ensemble_5xL2 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_conv64_ensemble_5xL2",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMEnsembleExperiment>(
                cfg, "data/mnist",
                /*n_members=*/5, /*d0=*/9216, /*d=*/4096,
                /*n_hidden=*/1, /*n_cycles=*/5, /*lambda=*/1e-4f,
                /*use_conv=*/true, /*conv_c_out=*/64);
        });
    return true;
}();
// Conv64 ensemble: 10 members, C_out=64, n_hidden=1 (L2 net), n_cycles=5.
[[maybe_unused]] static const bool _fayn_reg_deep_elm_conv64_ensemble_10xL2 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_conv64_ensemble_10xL2",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMEnsembleExperiment>(
                cfg, "data/mnist",
                /*n_members=*/10, /*d0=*/9216, /*d=*/4096,
                /*n_hidden=*/1, /*n_cycles=*/5, /*lambda=*/1e-4f,
                /*use_conv=*/true, /*conv_c_out=*/64);
        });
    return true;
}();

// Feature-level augmentation (5 views = original + 4 axis-aligned shifts).
// 5-member conv64 L2 with 5-view augmentation.
[[maybe_unused]] static const bool _fayn_reg_deep_elm_conv64_aug_ensemble_5xL2 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_conv64_aug_ensemble_5xL2",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMEnsembleExperiment>(
                cfg, "data/mnist",
                /*n_members=*/5, /*d0=*/9216, /*d=*/4096,
                /*n_hidden=*/1, /*n_cycles=*/5, /*lambda=*/1e-4f,
                /*use_conv=*/true, /*conv_c_out=*/64,
                /*conv_k_per_member=*/std::vector<int>{},
                /*n_aug_views=*/5);
        });
    return true;
}();
// 10-member conv64 L2 with 5-view feature-level augmentation.
[[maybe_unused]] static const bool _fayn_reg_deep_elm_conv64_aug_ensemble_10xL2 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_conv64_aug_ensemble_10xL2",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMEnsembleExperiment>(
                cfg, "data/mnist",
                /*n_members=*/10, /*d0=*/9216, /*d=*/4096,
                /*n_hidden=*/1, /*n_cycles=*/5, /*lambda=*/1e-4f,
                /*use_conv=*/true, /*conv_c_out=*/64,
                /*conv_k_per_member=*/std::vector<int>{},
                /*n_aug_views=*/5);
        });
    return true;
}();
// 9-view augmentation: original + 4 axis-aligned + 4 diagonal 1-pixel shifts.
// 5-member conv64 L2 with 9-view augmentation.
[[maybe_unused]] static const bool _fayn_reg_deep_elm_conv64_9view_ensemble_5xL2 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_conv64_9view_ensemble_5xL2",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMEnsembleExperiment>(
                cfg, "data/mnist",
                /*n_members=*/5, /*d0=*/9216, /*d=*/4096,
                /*n_hidden=*/1, /*n_cycles=*/5, /*lambda=*/1e-4f,
                /*use_conv=*/true, /*conv_c_out=*/64,
                /*conv_k_per_member=*/std::vector<int>{},
                /*n_aug_views=*/9);
        });
    return true;
}();
// 10-member conv64 L2 with 9-view augmentation.
[[maybe_unused]] static const bool _fayn_reg_deep_elm_conv64_9view_ensemble_10xL2 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_conv64_9view_ensemble_10xL2",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMEnsembleExperiment>(
                cfg, "data/mnist",
                /*n_members=*/10, /*d0=*/9216, /*d=*/4096,
                /*n_hidden=*/1, /*n_cycles=*/5, /*lambda=*/1e-4f,
                /*use_conv=*/true, /*conv_c_out=*/64,
                /*conv_k_per_member=*/std::vector<int>{},
                /*n_aug_views=*/9);
        });
    return true;
}();
// C_out=128 with 5-view augmentation: regularize the wider front-end via 5× samples.
// 5-member conv128 L2 with augmentation.
[[maybe_unused]] static const bool _fayn_reg_deep_elm_conv128_aug_ensemble_5xL2 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_conv128_aug_ensemble_5xL2",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMEnsembleExperiment>(
                cfg, "data/mnist",
                /*n_members=*/5, /*d0=*/18432, /*d=*/4096,
                /*n_hidden=*/1, /*n_cycles=*/5, /*lambda=*/1e-4f,
                /*use_conv=*/true, /*conv_c_out=*/128,
                /*conv_k_per_member=*/std::vector<int>{},
                /*n_aug_views=*/5);
        });
    return true;
}();
// 10-member conv128 L2 with 5-view augmentation.
[[maybe_unused]] static const bool _fayn_reg_deep_elm_conv128_aug_ensemble_10xL2 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_conv128_aug_ensemble_10xL2",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMEnsembleExperiment>(
                cfg, "data/mnist",
                /*n_members=*/10, /*d0=*/18432, /*d=*/4096,
                /*n_hidden=*/1, /*n_cycles=*/5, /*lambda=*/1e-4f,
                /*use_conv=*/true, /*conv_c_out=*/128,
                /*conv_k_per_member=*/std::vector<int>{},
                /*n_aug_views=*/5);
        });
    return true;
}();

// Test-time augmentation (TTA): average logits over original + 4 pixel-shifted views.
// Same trained model as deep_elm_4096_L3; only inference changes.
[[maybe_unused]] static const bool _fayn_reg_deep_elm_tta_4096_L3 = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "deep_elm_tta_4096_L3",
        [](const fayn::ExperimentConfig& cfg) {
            return std::make_unique<fayn::DeepELMExperiment>(
                cfg, "data/mnist",
                /*d0=*/4096, /*d=*/4096, /*n_hidden=*/2, /*n_cycles=*/5,
                /*lambda=*/1e-4f, /*use_tanh=*/false, /*use_rff=*/false,
                /*rff_gamma=*/0.01f, /*augment=*/false, /*use_tta=*/true);
        });
    return true;
}();
}

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <experiment_name> [options]\n"
              << "Options:\n"
              << "  --epochs N        Number of epochs (default: 100)\n"
              << "  --batch-size N    Batch size (default: 128)\n"
              << "  --log PATH        Output log file (default: logs/<name>.jsonl)\n"
              << "  --no-mutations    Disable topology mutations\n"
              << "  --list            List available experiments\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string first_arg = argv[1];

    if (first_arg == "--list") {
        std::cout << "Registered experiments:\n";
        fayn::ExperimentRegistry::instance().list(std::cout);
        return 0;
    }

    if (first_arg == "--help" || first_arg == "-h") {
        print_usage(argv[0]);
        return 0;
    }

    fayn::ExperimentConfig cfg;
    cfg.name     = first_arg;
    cfg.log_path = "logs/" + first_arg + ".jsonl";

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--epochs" && i + 1 < argc) {
            cfg.epochs = std::stoull(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            cfg.batch_size = std::stoull(argv[++i]);
        } else if (arg == "--log" && i + 1 < argc) {
            cfg.log_path = argv[++i];
        } else if (arg == "--no-mutations") {
            cfg.enable_mutations = false;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    try {
        auto exp = fayn::ExperimentRegistry::instance().create(cfg.name, cfg);
        exp->execute();
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
