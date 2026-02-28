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
