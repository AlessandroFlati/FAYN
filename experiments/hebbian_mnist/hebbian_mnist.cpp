#include "hebbian_mnist.hpp"
#include "tools/registry.hpp"

#include <algorithm>
#include <cstdint>
#include <fmt/core.h>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <cuda_bf16.h>

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
// setup: build the graph
// ---------------------------------------------------------------------------
void HebbianMnistExperiment::setup() {
    graph_ = std::make_unique<Graph>();

    // Node 0: Dense 784 -> 256
    auto d0 = std::make_shared<DenseLayer>(784, 256, /*bias=*/true);
    d0->set_cache_activations(true);
    int n0 = graph_->add_node(d0);

    // Node 1: ReLU
    int n1 = graph_->add_node(make_activation_layer(ActivationType::ReLU));

    // Node 2: Dense 256 -> 10
    auto d1 = std::make_shared<DenseLayer>(256, 10, /*bias=*/false);
    d1->set_cache_activations(true);
    int n2 = graph_->add_node(d1);

    graph_->add_edge(n0, n1);
    graph_->add_edge(n1, n2);

    dense0_ = d0.get();
    dense1_ = d1.get();

    // Data source.
    data_ = std::make_unique<MnistLoader>(
        mnist_dir_ + "/train-images-idx3-ubyte",
        mnist_dir_ + "/train-labels-idx1-ubyte");
    data_->set_output_device(Device::CUDA);
    data_->set_input_dtype(DType::BFloat16);
}

// ---------------------------------------------------------------------------
// run_epoch
// ---------------------------------------------------------------------------
float HebbianMnistExperiment::run_epoch(size_t epoch) {
    data_->reset();

    const size_t n_batches = data_->batches_per_epoch(cfg_.batch_size);
    float  total_acc = 0.f;
    size_t n_samples = 0;

    for (size_t b = 0; b < n_batches; ++b) {
        Batch batch = data_->next_batch(cfg_.batch_size);
        const size_t actual_batch = batch.inputs.shape[0];

        // Forward pass through the graph (node 0 is the input node).
        // Tensor is move-only, so we cannot use initializer_list syntax.
        std::vector<std::pair<int, Tensor>> fwd_inputs;
        fwd_inputs.emplace_back(0, std::move(batch.inputs));
        auto outputs = graph_->forward(std::move(fwd_inputs));
        if (outputs.empty())
            throw std::runtime_error("HebbianMnistExperiment: graph produced no output");

        Tensor& output = outputs[0];   // [actual_batch, 10] BF16

        total_acc += batch_accuracy(output, batch.targets) * static_cast<float>(actual_batch);
        n_samples += actual_batch;

        // Hebbian updates on both dense layers.
        {
            StreamPool::Guard g0;
            hebbian_update(dense0_->weights(),
                           dense0_->last_input(),
                           dense0_->last_output(),
                           lr_, g0.stream());
            if ((b + 1) % normalize_every_ == 0)
                normalize_weights_rows(dense0_->weights(), 1e-8f, g0.stream());
            FAYN_CUDA_CHECK(cudaStreamSynchronize(g0.stream()));
        }
        {
            StreamPool::Guard g1;
            hebbian_update(dense1_->weights(),
                           dense1_->last_input(),
                           dense1_->last_output(),
                           lr_, g1.stream());
            if ((b + 1) % normalize_every_ == 0)
                normalize_weights_rows(dense1_->weights(), 1e-8f, g1.stream());
            FAYN_CUDA_CHECK(cudaStreamSynchronize(g1.stream()));
        }
    }

    const float epoch_acc = n_samples > 0
        ? total_acc / static_cast<float>(n_samples) : 0.f;

    std::cout << fmt::format("epoch {:3d}  acc={:.4f}  samples={}\n",
                             epoch, epoch_acc, n_samples);
    return epoch_acc;
}

// ---------------------------------------------------------------------------
// batch_accuracy
// ---------------------------------------------------------------------------
float HebbianMnistExperiment::batch_accuracy(const Tensor& output, const Tensor& labels) {
    const size_t batch = output.shape[0];
    const size_t n_cls = output.shape[1];

    std::vector<__nv_bfloat16> out_bf16(batch * n_cls);
    FAYN_CUDA_CHECK(cudaMemcpy(out_bf16.data(), output.data,
                               batch * n_cls * sizeof(__nv_bfloat16),
                               cudaMemcpyDeviceToHost));

    std::vector<int32_t> lbl(batch);
    FAYN_CUDA_CHECK(cudaMemcpy(lbl.data(), labels.data,
                               batch * sizeof(int32_t),
                               cudaMemcpyDeviceToHost));

    int correct = 0;
    for (size_t i = 0; i < batch; ++i) {
        size_t pred = 0;
        float  best = static_cast<float>(out_bf16[i * n_cls]);
        for (size_t j = 1; j < n_cls; ++j) {
            float v = static_cast<float>(out_bf16[i * n_cls + j]);
            if (v > best) { best = v; pred = j; }
        }
        if (static_cast<int32_t>(pred) == lbl[i]) ++correct;
    }
    return static_cast<float>(correct) / static_cast<float>(batch);
}

} // namespace fayn

// ---------------------------------------------------------------------------
// Self-register.
// The macro constructor takes (const ExperimentConfig&); uses default
// mnist_dir ("data/mnist"), lr=0.01, normalize_every=1.
// ---------------------------------------------------------------------------
FAYN_REGISTER_EXPERIMENT("hebbian_mnist", fayn::HebbianMnistExperiment)
