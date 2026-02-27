#include "hebbian_mnist.hpp"
#include "tools/registry.hpp"

#include "src/stats/event_bus.hpp"
#include "src/stats/events.hpp"

#include <fmt/core.h>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace fayn {

HebbianMnistExperiment::HebbianMnistExperiment(
    const ExperimentConfig& cfg,
    const std::string&      mnist_dir,
    float                   lr,
    int                     normalize_every)
    : Experiment(cfg)
    , mnist_dir_(mnist_dir)
    , lr_(lr)
    , normalize_every_(normalize_every)
{}

// ---------------------------------------------------------------------------
// setup: build the graph and wire up the reward-modulated Hebbian updater.
// ---------------------------------------------------------------------------
void HebbianMnistExperiment::setup() {
    graph_ = std::make_unique<Graph>();

    // Node 0: Dense 784 -> 256
    auto d0 = std::make_shared<DenseLayer>(784, 256, /*bias=*/true);
    d0->set_cache_activations(true);
    graph_->add_node(d0);

    // Node 1: ReLU
    graph_->add_node(make_activation_layer(ActivationType::ReLU));

    // Node 2: Dense 256 -> 10
    auto d1 = std::make_shared<DenseLayer>(256, 10, /*bias=*/false);
    d1->set_cache_activations(true);
    int n2 = graph_->add_node(d1);

    graph_->add_edge(0, 1);
    graph_->add_edge(1, n2);

    // Reward-modulated Hebbian updater: subscribes to RewardEvent (sync),
    // applies ΔW ∝ reward × post^T @ pre for both dense layers.
    updater_ = std::make_unique<HebbianUpdater>(std::vector<HebbianUpdater::LayerConfig>{
        { d0, lr_, HebbianUpdater::RoutingMode::Global, /*normalize=*/true, normalize_every_ },
        { d1, lr_, HebbianUpdater::RoutingMode::Global, /*normalize=*/true, normalize_every_ },
    });

    loss_fn_ = fayn::cross_entropy;

    // Data source.
    data_ = std::make_unique<MnistLoader>(
        mnist_dir_ + "/train-images-idx3-ubyte",
        mnist_dir_ + "/train-labels-idx1-ubyte");
    data_->set_output_device(Device::CUDA);
    data_->set_input_dtype(DType::BFloat16);
}

// ---------------------------------------------------------------------------
// run_epoch: forward pass + emit RewardEvent per batch.
// The HebbianUpdater subscriber handles all weight updates.
// ---------------------------------------------------------------------------
float HebbianMnistExperiment::run_epoch(size_t epoch) {
    data_->reset();

    const size_t n_batches = data_->batches_per_epoch(cfg_.batch_size);
    float  total_acc = 0.f;

    for (size_t b = 0; b < n_batches; ++b) {
        Batch batch = data_->next_batch(cfg_.batch_size);

        std::vector<std::pair<int, Tensor>> fwd_inputs;
        fwd_inputs.emplace_back(0, std::move(batch.inputs));
        auto outputs = graph_->forward(std::move(fwd_inputs));

        if (outputs.empty())
            throw std::runtime_error("HebbianMnistExperiment: graph produced no output");

        const float loss = loss_fn_(outputs[0], batch.targets);
        const float acc  = fayn::accuracy(outputs[0], batch.targets);
        total_acc += acc;

        // Emit reward = -loss. The HebbianUpdater subscriber fires synchronously,
        // applies the weight update, and returns before the next batch begins.
        RewardEvent ev;
        ev.step   = step_++;
        ev.reward = -loss;
        EventBus::instance().emit(ev);
    }

    const float epoch_acc = n_batches > 0
        ? total_acc / static_cast<float>(n_batches) : 0.f;

    std::cout << fmt::format("epoch {:3d}  acc={:.4f}  loss_proxy={:.4f}\n",
                             epoch, epoch_acc,
                             n_batches > 0 ? (1.f - epoch_acc) : 0.f);
    return epoch_acc;
}

} // namespace fayn
