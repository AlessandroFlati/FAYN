#include "src/core/device.hpp"
#include "src/core/loss.hpp"
#include "src/core/tensor.hpp"
#include "src/core/graph.hpp"
#include "src/ops/activations.hpp"
#include "src/ops/dense.hpp"
#include "src/ops/hebbian.hpp"
#include "src/ops/hebbian_updater.hpp"
#include "src/ops/one_hot.hpp"
#include "src/stats/activation_stats.hpp"
#include "src/stats/event_bus.hpp"
#include "src/stats/events.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static uint16_t f32_to_bf16_bits(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    return static_cast<uint16_t>(bits >> 16);
}

static float bf16_bits_to_f32(uint16_t bits) {
    uint32_t f32_bits = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &f32_bits, 4);
    return f;
}

static void PASS(const char* name) {
    std::cout << "  [PASS] " << name << "\n";
}

[[noreturn]] static void FAIL(const char* name, const char* why) {
    std::cerr << "  [FAIL] " << name << ": " << why << "\n";
    std::exit(1);
}

// ---------------------------------------------------------------------------
// Test 1: Tensor CPU/CUDA round-trip + metadata
// ---------------------------------------------------------------------------
static void test_tensor_roundtrip() {
    const char* TEST = "tensor_roundtrip";

    fayn::Tensor cpu = fayn::Tensor::make({4}, fayn::DType::Float32, fayn::Device::CPU);
    float* p = static_cast<float*>(cpu.data);
    p[0] = 1.0f;  p[1] = -2.0f;  p[2] = 3.14f;  p[3] = -0.5f;

    if (cpu.numel()  != 4)  FAIL(TEST, "numel wrong");
    if (cpu.nbytes() != 16) FAIL(TEST, "nbytes wrong");

    fayn::Tensor gpu  = cpu.to(fayn::Device::CUDA);
    fayn::Tensor back = gpu.to(fayn::Device::CPU);
    const float* q = static_cast<const float*>(back.data);

    for (int i = 0; i < 4; ++i)
        if (std::abs(q[i] - p[i]) > 1e-6f)
            FAIL(TEST, "data mismatch after CPU->CUDA->CPU round-trip");

    PASS(TEST);
}

// ---------------------------------------------------------------------------
// Test 2: ReLU on BF16 CUDA tensor
// ---------------------------------------------------------------------------
static void test_relu_bf16() {
    const char* TEST = "relu_bf16";

    const size_t N = 8;
    float vals[N] = { 1.f, -1.f, 0.f, -2.f, 0.5f, -0.5f, 3.f, -3.f };

    std::vector<uint16_t> host_in(N);
    for (size_t i = 0; i < N; ++i)
        host_in[i] = f32_to_bf16_bits(vals[i]);

    fayn::Tensor gpu = fayn::Tensor::make({N}, fayn::DType::BFloat16, fayn::Device::CUDA);
    FAYN_CUDA_CHECK(cudaMemcpy(gpu.data, host_in.data(), N * 2, cudaMemcpyHostToDevice));

    fayn::apply_relu(gpu);

    std::vector<uint16_t> host_out(N);
    FAYN_CUDA_CHECK(cudaMemcpy(host_out.data(), gpu.data, N * 2, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        float expected = vals[i] > 0.f ? vals[i] : 0.f;
        float got      = bf16_bits_to_f32(host_out[i]);
        if (std::abs(got - expected) > 0.02f)   // BF16 precision ~1e-2
            FAIL(TEST, "ReLU output mismatch");
    }

    PASS(TEST);
}

// ---------------------------------------------------------------------------
// Test 3: DenseLayer forward — correct output shape and device
// ---------------------------------------------------------------------------
static void test_dense_forward_shape() {
    const char* TEST = "dense_forward_shape";

    // Use dimensions that are multiples of 16 to satisfy Blackwell BF16
    // tensor-core alignment requirements (m, n, k all >= 16 and % 16 == 0).
    const size_t BATCH = 32, IN = 64, OUT = 32;

    fayn::DenseLayer layer(IN, OUT, /*use_bias=*/true);
    layer.set_id(0);

    // Fill with +1.0 in BF16 — non-zero input ensures a non-zero output.
    fayn::Tensor x = fayn::Tensor::make({BATCH, IN}, fayn::DType::BFloat16, fayn::Device::CUDA);
    {
        const uint16_t one = f32_to_bf16_bits(1.0f);
        std::vector<uint16_t> host(BATCH * IN, one);
        FAYN_CUDA_CHECK(cudaMemcpy(x.data, host.data(), BATCH * IN * 2, cudaMemcpyHostToDevice));
    }
    fayn::Tensor y = layer.forward(x);

    if (y.shape.size() != 2)                    FAIL(TEST, "output ndim != 2");
    if (y.shape[0] != BATCH)                    FAIL(TEST, "output batch dim wrong");
    if (y.shape[1] != OUT)                      FAIL(TEST, "output feature dim wrong");
    if (y.device != fayn::Device::CUDA)         FAIL(TEST, "output not on CUDA");
    if (y.dtype  != fayn::DType::BFloat16)      FAIL(TEST, "output dtype wrong");

    // Verify GEMM produced non-zero results (zero output = GEMM broken).
    std::vector<uint16_t> y_host(BATCH * OUT);
    FAYN_CUDA_CHECK(cudaMemcpy(y_host.data(), y.data, BATCH * OUT * 2, cudaMemcpyDeviceToHost));
    float max_abs = 0.f;
    for (uint16_t bits : y_host) max_abs = std::max(max_abs, std::abs(bf16_bits_to_f32(bits)));
    if (max_abs < 1e-4f) FAIL(TEST, "DenseLayer output is all-zero — GEMM is broken");

    PASS(TEST);
}

// ---------------------------------------------------------------------------
// Test 4: Activation stats — all-zero tensor should have dead_ratio = 1
// ---------------------------------------------------------------------------
static void test_activation_stats_all_dead() {
    const char* TEST = "activation_stats_all_dead";

    const size_t BATCH = 8, NEURONS = 32;

    // Tensor::make zero-initialises, so every activation is 0 (dead).
    fayn::Tensor acts = fayn::Tensor::make(
        {BATCH, NEURONS}, fayn::DType::BFloat16, fayn::Device::CUDA);

    fayn::BatchStatsBuffers buf;
    buf.init(NEURONS);
    fayn::compute_activation_stats(acts, buf, /*dead_threshold=*/0.01f, nullptr);

    float ratio = static_cast<float>(buf.h_dead_count) / static_cast<float>(NEURONS);
    if (ratio < 0.99f)
        FAIL(TEST, "expected dead_ratio ≈ 1.0 for all-zero input");

    PASS(TEST);
}

// ---------------------------------------------------------------------------
// Test 5: Activation stats — uniform +1 tensor should have dead_ratio = 0
// ---------------------------------------------------------------------------
static void test_activation_stats_no_dead() {
    const char* TEST = "activation_stats_no_dead";

    const size_t BATCH = 8, NEURONS = 32;

    fayn::Tensor acts = fayn::Tensor::make(
        {BATCH, NEURONS}, fayn::DType::BFloat16, fayn::Device::CUDA);

    // Fill with +1.0 in BF16.
    const uint16_t one = f32_to_bf16_bits(1.0f);
    std::vector<uint16_t> host(BATCH * NEURONS, one);
    FAYN_CUDA_CHECK(cudaMemcpy(acts.data, host.data(),
                               BATCH * NEURONS * 2, cudaMemcpyHostToDevice));

    fayn::BatchStatsBuffers buf;
    buf.init(NEURONS);
    fayn::compute_activation_stats(acts, buf, /*dead_threshold=*/0.01f, nullptr);

    if (buf.h_dead_count != 0)
        FAIL(TEST, "expected dead_count = 0 for all-ones input");

    // Mean should be close to 1.0 for every neuron.
    for (size_t j = 0; j < NEURONS; ++j)
        if (std::abs(buf.h_neuron_mean[j] - 1.0f) > 0.05f)
            FAIL(TEST, "neuron_mean far from 1.0");

    PASS(TEST);
}

// ---------------------------------------------------------------------------
// Test 6: Hebbian update — weights must change after non-zero forward
// ---------------------------------------------------------------------------
static void test_hebbian_weight_change() {
    const char* TEST = "hebbian_weight_change";

    // Dimensions must be multiples of 16 for Blackwell BF16 tensor cores.
    const size_t BATCH = 16, IN = 32, OUT = 16;

    fayn::DenseLayer layer(IN, OUT, /*use_bias=*/false);
    layer.set_cache_activations(true);
    layer.set_id(0);

    // Non-zero input: fill with 1.0 in BF16.
    fayn::Tensor x = fayn::Tensor::make({BATCH, IN}, fayn::DType::BFloat16, fayn::Device::CUDA);
    {
        const uint16_t one = f32_to_bf16_bits(1.0f);
        std::vector<uint16_t> host(BATCH * IN, one);
        FAYN_CUDA_CHECK(cudaMemcpy(x.data, host.data(), BATCH * IN * 2, cudaMemcpyHostToDevice));
    }

    layer.forward(x);

    // Snapshot weights before update.
    const size_t W_N = OUT * IN;
    std::vector<uint16_t> before(W_N), after_w(W_N);
    FAYN_CUDA_CHECK(cudaMemcpy(before.data(), layer.weights().data,
                               W_N * 2, cudaMemcpyDeviceToHost));

    fayn::hebbian_update(layer.weights(), layer.last_input(), layer.last_output(),
                         /*lr=*/0.1f, nullptr);
    FAYN_CUDA_CHECK(cudaDeviceSynchronize());

    FAYN_CUDA_CHECK(cudaMemcpy(after_w.data(), layer.weights().data,
                               W_N * 2, cudaMemcpyDeviceToHost));

    bool any_changed = false;
    for (size_t i = 0; i < W_N; ++i)
        if (before[i] != after_w[i]) { any_changed = true; break; }

    if (!any_changed)
        FAIL(TEST, "weights did not change after hebbian_update");

    PASS(TEST);
}

// ---------------------------------------------------------------------------
// Test 7: EventBus — ActivationEvent received synchronously
// ---------------------------------------------------------------------------
static void test_eventbus_sync_dispatch() {
    const char* TEST = "eventbus_sync_dispatch";

    std::atomic<int> fired{0};
    int captured_layer_id = -99;

    auto id = fayn::EventBus::instance().subscribe<fayn::ActivationEvent>(
        [&](const fayn::ActivationEvent& ev) {
            ++fired;
            captured_layer_id = ev.layer_id;
        },
        fayn::DispatchMode::Sync);

    fayn::DenseLayer layer(4, 4, false);
    layer.set_id(77);
    fayn::Tensor x = fayn::Tensor::make({2, 4}, fayn::DType::BFloat16, fayn::Device::CUDA);
    layer.forward(x);

    fayn::EventBus::instance().unsubscribe(id);

    if (fired.load() == 0)
        FAIL(TEST, "ActivationEvent was never received");
    if (captured_layer_id != 77)
        FAIL(TEST, "layer_id in event is wrong");

    PASS(TEST);
}

// ---------------------------------------------------------------------------
// Test 8: Full graph forward — MLP topology, synthetic input
// ---------------------------------------------------------------------------
static void test_graph_forward_mlp() {
    const char* TEST = "graph_forward_mlp";

    fayn::Graph graph;

    // Use tensor-core-aligned dimensions (multiples of 16).
    auto d0 = std::make_shared<fayn::DenseLayer>(32, 64, true);
    auto d1 = std::make_shared<fayn::DenseLayer>(64, 16, false);

    int n0 = graph.add_node(d0);
    int n1 = graph.add_node(fayn::make_activation_layer(fayn::ActivationType::ReLU));
    int n2 = graph.add_node(d1);

    graph.add_edge(n0, n1);
    graph.add_edge(n1, n2);

    fayn::Tensor x = fayn::Tensor::make({16, 32}, fayn::DType::BFloat16, fayn::Device::CUDA);
    // Fill with 1.0 so the forward pass produces visible non-trivial values.
    {
        const uint16_t one = f32_to_bf16_bits(1.0f);
        std::vector<uint16_t> host(16 * 32, one);
        FAYN_CUDA_CHECK(cudaMemcpy(x.data, host.data(), 16 * 32 * 2, cudaMemcpyHostToDevice));
    }

    std::vector<std::pair<int, fayn::Tensor>> inputs;
    inputs.emplace_back(n0, std::move(x));
    auto outputs = graph.forward(std::move(inputs));

    if (outputs.size() != 1)      FAIL(TEST, "expected exactly 1 output tensor");
    if (outputs[0].shape[0] != 16) FAIL(TEST, "output batch dim wrong");
    if (outputs[0].shape[1] != 16) FAIL(TEST, "output feature dim wrong");

    PASS(TEST);
}

// ---------------------------------------------------------------------------
// Test 9: cross_entropy — uniform logits should give CE = log(C)
// ---------------------------------------------------------------------------
static void test_loss_cross_entropy() {
    const char* TEST = "loss_cross_entropy";

    // output: [4, 3] BF16, all zeros (uniform logits → softmax = 1/3 each)
    // target: [4] Int32, labels [0, 1, 2, 0]
    const size_t BATCH = 4, C = 3;

    fayn::Tensor output = fayn::Tensor::make({BATCH, C}, fayn::DType::BFloat16, fayn::Device::CUDA);
    // Tensor::make zero-initialises, so all logits are 0.0 in BF16.

    fayn::Tensor target = fayn::Tensor::make({BATCH}, fayn::DType::Int32, fayn::Device::CUDA);
    int32_t lbl_h[4] = {0, 1, 2, 0};
    FAYN_CUDA_CHECK(cudaMemcpy(target.data, lbl_h, BATCH * sizeof(int32_t), cudaMemcpyHostToDevice));

    const float ce = fayn::cross_entropy(output, target);

    // Uniform softmax over 3 classes: CE = -log(1/3) = log(3) ≈ 1.0986
    const float expected = std::log(static_cast<float>(C));
    if (std::abs(ce - expected) > 0.05f)
        FAIL(TEST, "cross_entropy value far from log(C) for uniform logits");

    PASS(TEST);
}

// ---------------------------------------------------------------------------
// Test 10: HebbianUpdater — weights change when RewardEvent is emitted
// ---------------------------------------------------------------------------
static void test_hebbian_updater_fires() {
    const char* TEST = "hebbian_updater_fires";

    const size_t BATCH = 16, IN = 32, OUT = 16;

    auto layer = std::make_shared<fayn::DenseLayer>(IN, OUT, /*use_bias=*/false);
    layer->set_cache_activations(true);
    layer->set_id(0);

    // Forward pass to populate last_input_ / last_output_.
    fayn::Tensor x = fayn::Tensor::make({BATCH, IN}, fayn::DType::BFloat16, fayn::Device::CUDA);
    {
        const uint16_t one = f32_to_bf16_bits(1.0f);
        std::vector<uint16_t> host(BATCH * IN, one);
        FAYN_CUDA_CHECK(cudaMemcpy(x.data, host.data(), BATCH * IN * 2, cudaMemcpyHostToDevice));
    }
    layer->forward(x);

    // Snapshot weights before the update.
    const size_t W_N = OUT * IN;
    std::vector<uint16_t> before(W_N), after_w(W_N);
    FAYN_CUDA_CHECK(cudaMemcpy(before.data(), layer->weights().data,
                               W_N * 2, cudaMemcpyDeviceToHost));

    // Create updater (subscribes to RewardEvent on construction).
    fayn::HebbianUpdater updater({{layer, /*lr=*/0.1f,
                                   fayn::HebbianUpdater::RoutingMode::Global,
                                   /*normalize=*/false}});

    // Emit reward — updater fires synchronously and applies hebbian_update.
    fayn::RewardEvent ev;
    ev.step   = 0;
    ev.reward = 1.0f;
    fayn::EventBus::instance().emit(ev);

    FAYN_CUDA_CHECK(cudaMemcpy(after_w.data(), layer->weights().data,
                               W_N * 2, cudaMemcpyDeviceToHost));

    bool any_changed = false;
    for (size_t i = 0; i < W_N; ++i)
        if (before[i] != after_w[i]) { any_changed = true; break; }

    if (!any_changed)
        FAIL(TEST, "weights did not change after RewardEvent emission");

    PASS(TEST);
    // updater destructor unsubscribes.
}

// ---------------------------------------------------------------------------
// Test 11: one_hot_encode — correct one-hot matrix on GPU
// ---------------------------------------------------------------------------
static void test_one_hot_encode() {
    const char* TEST = "one_hot_encode";

    // Labels [0, 2, 1, 3] → one-hot [4, 4]
    const size_t BATCH = 4, C = 4;
    int32_t lbl_h[BATCH] = {0, 2, 1, 3};

    fayn::Tensor labels = fayn::Tensor::make({BATCH}, fayn::DType::Int32, fayn::Device::CUDA);
    FAYN_CUDA_CHECK(cudaMemcpy(labels.data, lbl_h, BATCH * sizeof(int32_t),
                               cudaMemcpyHostToDevice));

    fayn::Tensor oh = fayn::one_hot_encode(labels, C);
    FAYN_CUDA_CHECK(cudaDeviceSynchronize());

    if (oh.shape.size() != 2 || oh.shape[0] != BATCH || oh.shape[1] != C)
        FAIL(TEST, "output shape wrong");
    if (oh.dtype  != fayn::DType::BFloat16) FAIL(TEST, "output dtype wrong");
    if (oh.device != fayn::Device::CUDA)    FAIL(TEST, "output not on CUDA");

    std::vector<uint16_t> oh_h(BATCH * C);
    FAYN_CUDA_CHECK(cudaMemcpy(oh_h.data(), oh.data, BATCH * C * 2,
                               cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < BATCH; ++i) {
        for (size_t j = 0; j < C; ++j) {
            float got      = bf16_bits_to_f32(oh_h[i * C + j]);
            float expected = (j == static_cast<size_t>(lbl_h[i])) ? 1.0f : 0.0f;
            if (std::abs(got - expected) > 0.01f)
                FAIL(TEST, "one-hot value mismatch");
        }
    }

    PASS(TEST);
}

// ---------------------------------------------------------------------------
// Test 12: SupervisedHebbian — readout row for correct class must move toward
//          the hidden representation after one update.
// ---------------------------------------------------------------------------
static void test_supervised_hebbian() {
    const char* TEST = "supervised_hebbian";

    // Small network: IN=32, OUT=4 (4 classes).
    const size_t BATCH = 16, IN = 32, OUT = 4;

    auto layer = std::make_shared<fayn::DenseLayer>(IN, OUT, /*use_bias=*/false);
    layer->set_cache_activations(true);
    layer->set_id(0);

    // Forward pass: all-ones input.
    fayn::Tensor x = fayn::Tensor::make({BATCH, IN}, fayn::DType::BFloat16, fayn::Device::CUDA);
    {
        const uint16_t one = f32_to_bf16_bits(1.0f);
        std::vector<uint16_t> host(BATCH * IN, one);
        FAYN_CUDA_CHECK(cudaMemcpy(x.data, host.data(), BATCH * IN * 2, cudaMemcpyHostToDevice));
    }
    layer->forward(x);

    // Snapshot weights before update.
    const size_t W_N = OUT * IN;
    std::vector<uint16_t> before(W_N), after_w(W_N);
    FAYN_CUDA_CHECK(cudaMemcpy(before.data(), layer->weights().data,
                               W_N * 2, cudaMemcpyDeviceToHost));

    // One-hot targets: all samples assigned to class 0.
    int32_t lbl_h[BATCH];
    for (size_t i = 0; i < BATCH; ++i) lbl_h[i] = 0;
    fayn::Tensor labels = fayn::Tensor::make({BATCH}, fayn::DType::Int32, fayn::Device::CUDA);
    FAYN_CUDA_CHECK(cudaMemcpy(labels.data, lbl_h, BATCH * sizeof(int32_t),
                               cudaMemcpyHostToDevice));
    layer->set_target_activations(fayn::one_hot_encode(labels, OUT));

    // Updater with SupervisedHebbian, no normalization.
    fayn::HebbianUpdater updater({{layer, /*lr=*/0.1f,
                                   fayn::HebbianUpdater::RoutingMode::SupervisedHebbian,
                                   /*normalize=*/false}});
    fayn::RewardEvent ev;
    ev.step = 0; ev.reward = 1.0f;
    fayn::EventBus::instance().emit(ev);

    FAYN_CUDA_CHECK(cudaMemcpy(after_w.data(), layer->weights().data,
                               W_N * 2, cudaMemcpyDeviceToHost));

    // Row 0 (correct class) must change; rows 1-3 must be unchanged
    // (one-hot for class 0 → post for rows 1-3 is 0).
    bool row0_changed = false;
    for (size_t j = 0; j < IN; ++j)
        if (before[j] != after_w[j]) { row0_changed = true; break; }

    bool other_rows_changed = false;
    for (size_t i = 1; i < OUT; ++i)
        for (size_t j = 0; j < IN; ++j)
            if (before[i * IN + j] != after_w[i * IN + j]) { other_rows_changed = true; break; }

    if (!row0_changed)       FAIL(TEST, "correct-class row did not change");
    if (other_rows_changed)  FAIL(TEST, "wrong-class rows changed (should be zero update)");

    PASS(TEST);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "=== FAYN smoke test ===\n";

    FAYN_CUDA_CHECK(cudaSetDevice(0));

    test_tensor_roundtrip();
    test_relu_bf16();
    test_dense_forward_shape();
    test_activation_stats_all_dead();
    test_activation_stats_no_dead();
    test_hebbian_weight_change();
    test_eventbus_sync_dispatch();
    test_graph_forward_mlp();
    test_loss_cross_entropy();
    test_hebbian_updater_fires();
    test_one_hot_encode();
    test_supervised_hebbian();

    std::cout << "=== All tests passed ===\n";
    return 0;
}
