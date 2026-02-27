#pragma once

#include "../experiment.hpp"
#include "src/ops/dense.hpp"
#include "src/ops/activations.hpp"
#include "src/ops/hebbian_updater.hpp"
#include "src/ops/one_hot.hpp"
#include "src/io/mnist_loader.hpp"

#include <memory>
#include <string>
#include <vector>

namespace fayn {

// ---------------------------------------------------------------------------
// EnsembleHebbianMnistExperiment
//
// Topology: K independent networks, each:
//   Dense(784, 256, bias) -> ReLU -> Dense(256, 10, no bias)
//
// Learning rule: identical to HebbianMnistExperiment for each member.
//   d0: frozen Kaiming random projection (independent random seed per member)
//   d1: SupervisedHebbian — one-hot targets as post-synaptic signal
//       (nearest-centroid learning on the L2-normalised sphere)
//
// Ensemble inference: sum logits from all K members, then argmax.
// Each member's random d0 covers a different subspace; summing votes reduces
// variance, pushing ensemble accuracy toward the ELM ceiling (~93-96%).
//
// Configuration:
//   num_networks    - number of ensemble members (default 10)
//   lr              - Hebbian learning rate (default 0.01)
//   mnist_dir       - directory containing the four MNIST binary files
//   normalize_every - normalise d1 weight rows every N batches (default 1)
// ---------------------------------------------------------------------------
class EnsembleHebbianMnistExperiment : public Experiment {
public:
    explicit EnsembleHebbianMnistExperiment(
        const ExperimentConfig& cfg,
        const std::string&      mnist_dir       = "data/mnist",
        float                   lr              = 0.01f,
        int                     num_networks    = 10,
        int                     normalize_every = 1);

protected:
    void  setup()                  override;
    float run_epoch(size_t epoch)  override;

private:
    struct Member {
        std::unique_ptr<Graph>          graph;
        std::shared_ptr<DenseLayer>     d0;
        std::shared_ptr<DenseLayer>     d1;
        std::unique_ptr<HebbianUpdater> updater;
    };

    std::string          mnist_dir_;
    float                lr_              = 0.01f;
    int                  num_networks_    = 10;
    int                  normalize_every_ = 1;
    size_t               step_            = 0;

    std::vector<Member>  members_;
};

} // namespace fayn
