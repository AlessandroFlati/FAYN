#pragma once

#include "../experiment.hpp"
#include "src/core/loss.hpp"
#include "src/ops/dense.hpp"
#include "src/ops/activations.hpp"
#include "src/ops/hebbian_updater.hpp"
#include "src/ops/one_hot.hpp"
#include "src/io/mnist_loader.hpp"

#include <memory>
#include <string>

namespace fayn {

// ---------------------------------------------------------------------------
// HebbianMnistExperiment
//
// Topology: Dense(784,256,bias) -> ReLU -> Dense(256,10,no bias)
//
// Learning rule (random projection + supervised Hebbian readout, no backprop):
//   Hidden layer (d0): frozen at Kaiming random init. Random ReLU projections
//     of the 784-dim input provide adequate discriminative features (~78% acc).
//     Local Hebbian on d0 is explicitly disabled — it learns PCA-like features
//     that are less class-discriminative and destabilise the readout.
//   Readout layer (d1): SupervisedHebbian mode — one-hot targets used as the
//     post-synaptic signal. Each class weight row is pulled toward the hidden
//     representations of that class (nearest-centroid learning on the sphere).
//     ΔW[label_row] ∝ lr × hidden   (other rows unchanged)
//   d1 rows are L2-normalised every `normalize_every` batches.
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

    std::shared_ptr<DenseLayer>   d0_;     // 784 -> 256, Local Hebbian
    std::shared_ptr<DenseLayer>   d1_;     // 256 -> 10,  SupervisedHebbian
    std::unique_ptr<HebbianUpdater> updater_;
};

} // namespace fayn
