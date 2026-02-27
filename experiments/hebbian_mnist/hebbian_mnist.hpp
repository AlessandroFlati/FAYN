#pragma once

#include "../experiment.hpp"
#include "src/core/loss.hpp"
#include "src/ops/dense.hpp"
#include "src/ops/activations.hpp"
#include "src/ops/hebbian_updater.hpp"
#include "src/io/mnist_loader.hpp"

#include <memory>
#include <string>

namespace fayn {

// ---------------------------------------------------------------------------
// HebbianMnistExperiment
//
// Topology: Dense(784,256,bias) -> ReLU -> Dense(256,10,no bias)
//
// Learning rule (reward-modulated Hebbian, no backprop):
//   After each mini-batch, emit RewardEvent{reward = -cross_entropy}.
//   HebbianUpdater subscriber applies:
//     ΔW ∝ reward × post^T @ pre   (Global routing mode)
//   Rows of W are L2-normalised every `normalize_every` batches.
//
// Configuration:
//   lr              - Hebbian learning rate (default 0.01)
//   mnist_dir       - directory containing the four MNIST binary files
//   normalize_every - normalise weights every N batches (default 1)
// ---------------------------------------------------------------------------
class HebbianMnistExperiment : public Experiment {
public:
    explicit HebbianMnistExperiment(
        const ExperimentConfig& cfg,
        const std::string& mnist_dir      = "data/mnist",
        float              lr             = 0.01f,
        int                normalize_every = 1);

protected:
    void  setup()                  override;
    float run_epoch(size_t epoch)  override;

private:
    std::string                   mnist_dir_;
    float                         lr_              = 0.01f;
    int                           normalize_every_ = 1;
    size_t                        step_            = 0;

    std::unique_ptr<HebbianUpdater> updater_;
    LossFn                          loss_fn_;
};

} // namespace fayn
