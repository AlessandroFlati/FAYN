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
    int                     normalize_every,
    int                     hidden_dim)
    : Experiment(cfg)
    , mnist_dir_(mnist_dir)
    , lr_(lr)
    , normalize_every_(normalize_every)
    , hidden_dim_(hidden_dim)
{}

// ---------------------------------------------------------------------------
// setup: build the graph and wire up the two-stage local Hebbian updater.
// ---------------------------------------------------------------------------
void HebbianMnistExperiment::setup() {
    graph_ = std::make_unique<Graph>();

    // Node 0: Dense 784 -> hidden_dim_ (frozen random projection)
    d0_ = std::make_shared<DenseLayer>(784, static_cast<size_t>(hidden_dim_), /*bias=*/true);
    d0_->set_cache_activations(true);
    graph_->add_node(d0_);

    // Node 1: ReLU
    graph_->add_node(make_activation_layer(ActivationType::ReLU));

    // Node 2: Dense hidden_dim_ -> 10 (SupervisedHebbian — target one-hot as post)
    d1_ = std::make_shared<DenseLayer>(static_cast<size_t>(hidden_dim_), 10, /*bias=*/false);
    d1_->set_cache_activations(true);
    d1_->enable_fp32_weights();
    int n2 = graph_->add_node(d1_);

    graph_->add_edge(0, 1);
    graph_->add_edge(1, n2);

    // d0: frozen — Kaiming random features are stable and discriminative enough.
    //     Local Hebbian on d0 hurts because it learns input PCA features (not
    //     class-discriminative) and destabilises d1's class prototypes.
    // d1: SupervisedHebbian — one-hot targets as post; learns class prototypes
    //     over the fixed random projection from d0.
    updater_ = std::make_unique<HebbianUpdater>(std::vector<HebbianUpdater::LayerConfig>{{
        .layer           = d1_,
        .lr              = lr_,
        .mode            = HebbianUpdater::RoutingMode::SupervisedHebbian,
        .normalize       = true,
        .normalize_every = normalize_every_,
        .normalize_pre   = false,
        .lr_schedule     = {},
    }});

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

        const float acc = fayn::accuracy(outputs[0], batch.targets);
        total_acc += acc;

        // Supervised Hebbian: provide one-hot targets as post-synaptic signal
        // for the readout layer. The HebbianUpdater will pull each class row of
        // d1's weights toward the hidden representations of that class.
        d1_->set_target_activations(one_hot_encode(batch.targets, 10));

        // reward = 1.0: ignored by both Local (d0) and SupervisedHebbian (d1).
        RewardEvent ev;
        ev.step   = step_++;
        ev.reward = 1.0f;
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
