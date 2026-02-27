#include "ensemble_mnist.hpp"

#include "src/stats/event_bus.hpp"
#include "src/stats/events.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <fmt/core.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace fayn {

EnsembleHebbianMnistExperiment::EnsembleHebbianMnistExperiment(
    const ExperimentConfig& cfg,
    const std::string&      mnist_dir,
    float                   lr,
    int                     num_networks,
    int                     normalize_every)
    : Experiment(cfg)
    , mnist_dir_(mnist_dir)
    , lr_(lr)
    , num_networks_(num_networks)
    , normalize_every_(normalize_every)
{}

// ---------------------------------------------------------------------------
// setup: build K independent networks and register one HebbianUpdater per
// member. Data source is shared across all members (single MNIST loader).
// ---------------------------------------------------------------------------
void EnsembleHebbianMnistExperiment::setup() {
    members_.resize(static_cast<size_t>(num_networks_));

    for (auto& m : members_) {
        m.graph = std::make_unique<Graph>();

        // d0: frozen Kaiming random projection 784 -> 256.
        // Stats disabled: ensemble members don't need live monitoring.
        m.d0 = std::make_shared<DenseLayer>(784, 256, /*bias=*/true);
        m.d0->set_cache_activations(true);
        m.d0->set_compute_stats(false);
        m.graph->add_node(m.d0);

        // ReLU activation (stats disabled — no monitoring needed for ensemble).
        auto relu = make_activation_layer(ActivationType::ReLU);
        relu->set_compute_stats(false);
        m.graph->add_node(std::move(relu));

        // d1: readout 256 -> 10, trained with SupervisedHebbian.
        m.d1 = std::make_shared<DenseLayer>(256, 10, /*bias=*/false);
        m.d1->set_cache_activations(true);
        m.d1->set_compute_stats(false);
        const int n2 = m.graph->add_node(m.d1);

        m.graph->add_edge(0, 1);
        m.graph->add_edge(1, n2);

        m.updater = std::make_unique<HebbianUpdater>(
            std::vector<HebbianUpdater::LayerConfig>{{
                m.d1, lr_,
                HebbianUpdater::RoutingMode::SupervisedHebbian,
                /*normalize=*/true, normalize_every_,
            }});
    }

    // Single data source shared across members.
    data_ = std::make_unique<MnistLoader>(
        mnist_dir_ + "/train-images-idx3-ubyte",
        mnist_dir_ + "/train-labels-idx1-ubyte");
    data_->set_output_device(Device::CUDA);
    data_->set_input_dtype(DType::BFloat16);
}

// ---------------------------------------------------------------------------
// ensemble_accuracy: sum BF16 logits from K members host-side, argmax,
// compare to Int32 labels. Returns fraction of correct predictions.
// ---------------------------------------------------------------------------
static float ensemble_accuracy(
    const std::vector<Tensor*>& outputs,
    const Tensor&               target)
{
    if (outputs.empty()) return 0.f;

    const size_t batch = outputs[0]->shape[0];
    const size_t C     = outputs[0]->shape[1];
    const size_t K     = outputs.size();

    std::vector<float>          sum(batch * C, 0.f);
    std::vector<__nv_bfloat16>  buf(batch * C);

    for (size_t k = 0; k < K; ++k) {
        FAYN_CUDA_CHECK(cudaMemcpy(buf.data(), outputs[k]->data,
                                   batch * C * sizeof(__nv_bfloat16),
                                   cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < batch * C; ++i)
            sum[i] += static_cast<float>(buf[i]);
    }

    std::vector<int32_t> lbl_h(batch);
    FAYN_CUDA_CHECK(cudaMemcpy(lbl_h.data(), target.data,
                               batch * sizeof(int32_t),
                               cudaMemcpyDeviceToHost));

    int correct = 0;
    for (size_t i = 0; i < batch; ++i) {
        size_t pred = 0;
        float  best = sum[i * C];
        for (size_t j = 1; j < C; ++j) {
            if (sum[i * C + j] > best) { best = sum[i * C + j]; pred = j; }
        }
        if (static_cast<int32_t>(pred) == lbl_h[i]) ++correct;
    }
    return static_cast<float>(correct) / static_cast<float>(batch);
}

// ---------------------------------------------------------------------------
// run_epoch: for each batch, run K forward passes, compute ensemble accuracy
// from summed logits, then trigger all K HebbianUpdaters via a single event.
// ---------------------------------------------------------------------------
float EnsembleHebbianMnistExperiment::run_epoch(size_t epoch) {
    data_->reset();

    const size_t n_batches = data_->batches_per_epoch(cfg_.batch_size);
    float total_acc = 0.f;

    for (size_t b = 0; b < n_batches; ++b) {
        Batch batch = data_->next_batch(cfg_.batch_size);

        // Run K forward passes. For the first K-1 members, pass a D2D copy
        // of the input; move the original into the last member's forward call.
        std::vector<Tensor> member_outputs;
        member_outputs.reserve(members_.size());

        for (size_t k = 0; k < members_.size(); ++k) {
            const bool last = (k + 1 == members_.size());
            // borrow() creates a non-owning view: no D2D copy, no device sync.
            // DenseLayer::forward() only reads from the input data and the
            // async cache copy completes before forward() returns (stream sync
            // in compute_and_snapshot), so the original tensor outlives all uses.
            Tensor inp = last
                ? std::move(batch.inputs)
                : batch.inputs.borrow();

            std::vector<std::pair<int, Tensor>> fwd_inputs;
            fwd_inputs.emplace_back(0, std::move(inp));
            auto out = members_[k].graph->forward(std::move(fwd_inputs));
            if (out.empty())
                throw std::runtime_error(
                    "EnsembleHebbianMnistExperiment: member graph produced no output");
            member_outputs.push_back(std::move(out[0]));
        }

        // Ensemble accuracy: sum logits from all K members, then argmax.
        std::vector<Tensor*> out_ptrs;
        out_ptrs.reserve(member_outputs.size());
        for (auto& t : member_outputs) out_ptrs.push_back(&t);

        total_acc += ensemble_accuracy(out_ptrs, batch.targets);

        // Set one-hot targets for each member's readout layer.
        // one_hot_encode takes const Tensor& so batch.targets is not consumed.
        for (auto& m : members_)
            m.d1->set_target_activations(one_hot_encode(batch.targets, 10));

        // Emit one RewardEvent — all K HebbianUpdaters fire in sequence.
        RewardEvent ev;
        ev.step   = step_++;
        ev.reward = 1.0f;
        EventBus::instance().emit(ev);
    }

    const float epoch_acc = n_batches > 0
        ? total_acc / static_cast<float>(n_batches) : 0.f;

    std::cout << fmt::format("epoch {:3d}  acc={:.4f}\n", epoch, epoch_acc);
    return epoch_acc;
}

} // namespace fayn
