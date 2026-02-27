#pragma once

#include "../experiment.hpp"
#include "src/ops/dense.hpp"
#include "src/ops/activations.hpp"
#include "src/ops/hebbian.hpp"
#include "src/io/mnist_loader.hpp"
#include "src/cuda/stream_pool.hpp"

#include <array>
#include <string>

namespace fayn {

// ---------------------------------------------------------------------------
// HebbianMnistExperiment
//
// Topology: Dense(784,256) -> ReLU -> Dense(256,10)
//
// Learning rule (pure Hebbian, no backprop):
//   For each dense layer after each mini-batch:
//     W += lr * post^T @ pre / batch
//     Rows of W are L2-normalised to prevent weight explosion.
//
// This is a baseline to demonstrate the framework end-to-end.
// Expect accuracy to plateau below gradient-based methods; the goal is
// verifying the infrastructure (stats, events, logging) works correctly.
//
// Configuration (via ExperimentConfig + constructor args):
//   lr              - Hebbian learning rate (default 0.01)
//   mnist_dir       - directory containing the four MNIST binary files
//   normalize_every - normalise weights every N batches (default 1)
// ---------------------------------------------------------------------------
class HebbianMnistExperiment : public Experiment {
public:
    explicit HebbianMnistExperiment(
        const ExperimentConfig& cfg,
        const std::string& mnist_dir = "data/mnist",
        float              lr             = 0.01f,
        int                normalize_every = 1);

protected:
    void  setup()            override;
    float run_epoch(size_t epoch) override;

private:
    // Compute top-1 accuracy for one batch: argmax(output) vs labels.
    // output:  [batch, 10] BF16, on device -> copied to host for argmax.
    // labels:  [batch]     Int32, on device -> copied to host for comparison.
    float batch_accuracy(const Tensor& output, const Tensor& labels);

    std::string mnist_dir_;
    float       lr_             = 0.01f;
    int         normalize_every_ = 1;

    // Pointers into the graph for direct access during the update step.
    DenseLayer* dense0_ = nullptr;   // node 0: Dense(784, 256)
    DenseLayer* dense1_ = nullptr;   // node 2: Dense(256, 10)
};

} // namespace fayn
