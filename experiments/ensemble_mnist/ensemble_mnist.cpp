#include "ensemble_mnist.hpp"

#include "src/stats/event_bus.hpp"
#include "src/stats/events.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <fmt/core.h>

#include <algorithm>
#include <cmath>
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
    int                     normalize_every,
    float                   d0_init_scale,
    int64_t                 seed,
    bool                    normalize_pre,
    float                   lr_final,
    bool                    row_normalize,
    float                   weight_decay,
    int                     hidden_dim,
    bool                    use_delta_rule)
    : Experiment(cfg)
    , mnist_dir_(mnist_dir)
    , lr_(lr)
    , num_networks_(num_networks)
    , normalize_every_(normalize_every)
    , d0_init_scale_(d0_init_scale)
    , seed_(seed)
    , normalize_pre_(normalize_pre)
    , lr_final_(lr_final)
    , row_normalize_(row_normalize)
    , weight_decay_(weight_decay)
    , hidden_dim_(hidden_dim)
    , use_delta_rule_(use_delta_rule)
{}

// ---------------------------------------------------------------------------
// setup: build K independent networks and register one HebbianUpdater per
// member. Data source is shared across all members (single MNIST loader).
// ---------------------------------------------------------------------------
void EnsembleHebbianMnistExperiment::setup() {
    if (seed_ >= 0) reset_kaiming_seed(static_cast<uint64_t>(seed_));

    // Build data source first so we can compute total_steps for LR schedule.
    data_ = std::make_unique<MnistLoader>(
        mnist_dir_ + "/train-images-idx3-ubyte",
        mnist_dir_ + "/train-labels-idx1-ubyte");
    data_->set_output_device(Device::CUDA);
    data_->set_input_dtype(DType::BFloat16);

    // Cosine LR schedule: lr(t) = lr0 + (lr1 - lr0) * 0.5 * (1 - cos(pi*t))
    // t goes from 0 to 1 over total_steps; returns lr0 at t=0, lr1 at t=1.
    std::function<float(size_t)> lr_sched;
    if (lr_final_ >= 0.f) {
        const size_t n_batches   = data_->size() / cfg_.batch_size;
        const size_t total_steps = cfg_.epochs * n_batches;
        const float  lr0 = lr_, lr1 = lr_final_;
        const size_t T   = total_steps > 0 ? total_steps : 1;
        lr_sched = [lr0, lr1, T](size_t step) -> float {
            const float t = std::min(static_cast<float>(step) / static_cast<float>(T), 1.f);
            return lr0 + (lr1 - lr0) * 0.5f * (1.f - std::cos(float(M_PI) * t));
        };
    }

    members_.resize(static_cast<size_t>(num_networks_));

    for (auto& m : members_) {
        m.graph = std::make_unique<Graph>();

        // d0: frozen random projection 784 -> hidden_dim_.
        // Stats disabled: ensemble members don't need live monitoring.
        m.d0 = std::make_shared<DenseLayer>(784, static_cast<size_t>(hidden_dim_), /*bias=*/true, d0_init_scale_);
        m.d0->set_cache_activations(true);
        m.d0->set_compute_stats(false);
        m.graph->add_node(m.d0);

        // ReLU activation (stats disabled — no monitoring needed for ensemble).
        auto relu = make_activation_layer(ActivationType::ReLU);
        relu->set_compute_stats(false);
        m.graph->add_node(std::move(relu));

        // d1: readout hidden_dim_ -> 10, trained with SupervisedHebbian.
        m.d1 = std::make_shared<DenseLayer>(static_cast<size_t>(hidden_dim_), 10, /*bias=*/false);
        m.d1->set_cache_activations(true);
        m.d1->set_compute_stats(false);
        const int n2 = m.graph->add_node(m.d1);

        m.graph->add_edge(0, 1);
        m.graph->add_edge(1, n2);

        HebbianUpdater::LayerConfig lcfg;
        lcfg.layer           = m.d1;
        lcfg.lr              = lr_;
        lcfg.mode            = use_delta_rule_
                                   ? HebbianUpdater::RoutingMode::DeltaRule
                                   : HebbianUpdater::RoutingMode::SupervisedHebbian;
        lcfg.normalize       = row_normalize_;
        lcfg.normalize_every = normalize_every_;
        lcfg.normalize_pre   = normalize_pre_;
        lcfg.lr_schedule     = lr_sched;   // empty if no scheduling
        lcfg.weight_decay    = weight_decay_;

        m.updater = std::make_unique<HebbianUpdater>(
            std::vector<HebbianUpdater::LayerConfig>{ std::move(lcfg) });
    }
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

// ---------------------------------------------------------------------------
// Gaussian elimination with partial pivoting.
// A: [n, n] row-major FP32 — modified in-place (LU factored).
// b: [n, C] row-major FP32 — overwritten with solution W_solve.
// Throws if A is (near-)singular.
// ---------------------------------------------------------------------------
static void gauss_solve(std::vector<float>& A, std::vector<float>& b, int n, int C) {
    for (int k = 0; k < n; ++k) {
        // Partial pivot: find row with largest |A[i, k]| for i >= k.
        int pivot = k;
        for (int i = k + 1; i < n; ++i)
            if (std::abs(A[i * n + k]) > std::abs(A[pivot * n + k]))
                pivot = i;
        if (std::abs(A[pivot * n + k]) < 1e-8f)
            throw std::runtime_error("elm_fit: H^T H is singular (degenerate features)");
        // Swap rows k and pivot in A and b.
        for (int j = 0; j < n; ++j) std::swap(A[k * n + j], A[pivot * n + j]);
        for (int j = 0; j < C; ++j) std::swap(b[k * C + j], b[pivot * C + j]);
        // Eliminate column k below the diagonal.
        float inv = 1.f / A[k * n + k];
        for (int i = k + 1; i < n; ++i) {
            float f = A[i * n + k] * inv;
            for (int j = k; j < n; ++j) A[i * n + j] -= f * A[k * n + j];
            for (int j = 0; j < C; ++j)  b[i * C + j] -= f * b[k * C + j];
        }
    }
    // Back-substitution.
    for (int k = n - 1; k >= 0; --k) {
        float inv = 1.f / A[k * n + k];
        for (int j = 0; j < C; ++j) b[k * C + j] *= inv;
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < C; ++j)
                b[i * C + j] -= A[i * n + k] * b[k * C + j];
    }
}

// ---------------------------------------------------------------------------
// ELMEnsembleExperiment
// ---------------------------------------------------------------------------
ELMEnsembleExperiment::ELMEnsembleExperiment(
    const ExperimentConfig& cfg,
    const std::string& mnist_dir,
    int num_networks,
    float d0_init_scale,
    int64_t seed,
    int hidden_dim)
    : Experiment(cfg)
    , mnist_dir_(mnist_dir)
    , num_networks_(num_networks)
    , d0_init_scale_(d0_init_scale)
    , seed_(seed)
    , hidden_dim_(hidden_dim)
{
    FAYN_CUBLAS_CHECK(cublasCreate(&cublas_));
}

ELMEnsembleExperiment::~ELMEnsembleExperiment() {
    if (cublas_) cublasDestroy(cublas_);
}

void ELMEnsembleExperiment::setup() {
    if (seed_ >= 0) reset_kaiming_seed(static_cast<uint64_t>(seed_));
    members_.resize(static_cast<size_t>(num_networks_));

    for (auto& m : members_) {
        m.graph = std::make_unique<Graph>();

        m.d0 = std::make_shared<DenseLayer>(784, static_cast<size_t>(hidden_dim_), /*bias=*/true, d0_init_scale_);
        m.d0->set_compute_stats(false);
        m.graph->add_node(m.d0);

        auto relu = make_activation_layer(ActivationType::ReLU);
        relu->set_compute_stats(false);
        m.graph->add_node(std::move(relu));

        m.d1 = std::make_shared<DenseLayer>(static_cast<size_t>(hidden_dim_), 10, /*bias=*/false);
        m.d1->set_compute_stats(false);
        const int n2 = m.graph->add_node(m.d1);

        m.graph->add_edge(0, 1);
        m.graph->add_edge(1, n2);
    }

    data_ = std::make_unique<MnistLoader>(
        mnist_dir_ + "/train-images-idx3-ubyte",
        mnist_dir_ + "/train-labels-idx1-ubyte");
    data_->set_output_device(Device::CUDA);
    data_->set_input_dtype(DType::BFloat16);
}

// ---------------------------------------------------------------------------
// elm_fit: one data pass per member.
//
// For each member k:
//   1. Run all N training samples through frozen d0_k + ReLU → H_k [N, d] FP32.
//   2. (Shared across members) Build T [N, C] one-hot FP32.
//   3. Normal equations on GPU via cuBLAS:
//        A_k = H_k^T H_k  [d, d]
//        b_k = H_k^T T    [d, C]
//   4. Solve A_k W_k = b_k on CPU (Gaussian elimination, [d,d] = [256,256]).
//   5. Transpose W_k [d, C] → [C, d], cast to BF16, write to d1_k->weights().
//
// cuBLAS convention note:
//   Row-major [N, d] data = col-major [d, N] with ld = d.
//   A_k = H_k^T H_k in row-major
//       = H_cm * H_cm^T in col-major   → cublasSgemm(N, T, d, d, N, H, d, H, d, A, d)
//   b_k = H_k^T T in row-major
//       = H_cm * T_cm^T in col-major   → cublasSgemm(N, T, d, C, N, H, d, T, C, b, d)
//   Both results are col-major. A_k is symmetric (A^T = A) so its col-major
//   and row-major flat arrays are identical. b_k must be transposed from
//   col-major [d, C] to row-major [d, C] before passing to gauss_solve.
// ---------------------------------------------------------------------------
void ELMEnsembleExperiment::elm_fit() {
    const size_t fit_bs   = cfg_.batch_size;
    // Floor-divide: drop the last partial batch for uniform-size allocation.
    const size_t n_batches = data_->size() / fit_bs;
    const size_t N        = n_batches * fit_bs;
    const size_t d        = static_cast<size_t>(hidden_dim_);
    const size_t C        = 10;

    // ---- Build T [N, C] one-hot FP32 on device ----
    std::vector<float>   T_host(N * C, 0.f);
    std::vector<int32_t> lbl_buf(fit_bs);
    data_->reset();
    for (size_t b = 0; b < n_batches; ++b) {
        Batch batch = data_->next_batch(fit_bs);
        FAYN_CUDA_CHECK(cudaMemcpy(lbl_buf.data(), batch.targets.data,
                                   fit_bs * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < fit_bs; ++i)
            T_host[(b * fit_bs + i) * C + static_cast<size_t>(lbl_buf[i])] = 1.f;
    }
    Tensor T_dev = Tensor::make({N, C}, DType::Float32, Device::CUDA);
    FAYN_CUDA_CHECK(cudaMemcpy(T_dev.data, T_host.data(),
                               N * C * sizeof(float), cudaMemcpyHostToDevice));
    T_host.clear();

    // Reusable host buffers (allocated once, reused per member).
    std::vector<__nv_bfloat16> bf16_buf(fit_bs * d);
    std::vector<float>          H_host(N * d);

    for (auto& m : members_) {
        // ---- Collect H_k [N, d] FP32 ----
        data_->reset();
        for (size_t b = 0; b < n_batches; ++b) {
            Batch batch = data_->next_batch(fit_bs);
            // Forward through d0 only (not the full graph).
            // borrow() avoids a D2D copy; forward() syncs its stream before returning.
            auto h = m.d0->forward(batch.inputs.borrow());  // BF16 [fit_bs, d]
            apply_relu(h, /*stream=*/nullptr);
            FAYN_CUDA_CHECK(cudaStreamSynchronize(nullptr));
            FAYN_CUDA_CHECK(cudaMemcpy(bf16_buf.data(), h.data,
                                       fit_bs * d * sizeof(__nv_bfloat16),
                                       cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < fit_bs * d; ++i)
                H_host[b * fit_bs * d + i] = __bfloat162float(bf16_buf[i]);
        }

        Tensor H_dev = Tensor::make({N, d}, DType::Float32, Device::CUDA);
        FAYN_CUDA_CHECK(cudaMemcpy(H_dev.data, H_host.data(),
                                   N * d * sizeof(float), cudaMemcpyHostToDevice));

        // ---- Normal equations on GPU ----
        Tensor A_dev = Tensor::make({d, d}, DType::Float32, Device::CUDA);
        Tensor b_dev = Tensor::make({d, C}, DType::Float32, Device::CUDA);
        const float alpha = 1.f, beta = 0.f;

        FAYN_CUBLAS_CHECK(cublasSgemm(
            cublas_, CUBLAS_OP_N, CUBLAS_OP_T,
            (int)d, (int)d, (int)N, &alpha,
            (float*)H_dev.data, (int)d,
            (float*)H_dev.data, (int)d,
            &beta, (float*)A_dev.data, (int)d));

        FAYN_CUBLAS_CHECK(cublasSgemm(
            cublas_, CUBLAS_OP_N, CUBLAS_OP_T,
            (int)d, (int)C, (int)N, &alpha,
            (float*)H_dev.data, (int)d,
            (float*)T_dev.data, (int)C,
            &beta, (float*)b_dev.data, (int)d));

        FAYN_CUDA_CHECK(cudaDeviceSynchronize());

        // Copy A and b to CPU.
        std::vector<float> A_cm(d * d), b_cm(d * C);
        FAYN_CUDA_CHECK(cudaMemcpy(A_cm.data(), A_dev.data,
                                   d * d * sizeof(float), cudaMemcpyDeviceToHost));
        FAYN_CUDA_CHECK(cudaMemcpy(b_cm.data(), b_dev.data,
                                   d * C * sizeof(float), cudaMemcpyDeviceToHost));

        // A is symmetric (H^T H) so its col-major and row-major flat arrays
        // are identical — pass A_cm directly as row-major to gauss_solve.
        // Convert b from col-major [d, C] to row-major [d, C]:
        //   b_rm[i * C + j] = b_cm[i + j * d]
        std::vector<float> b_rm(d * C);
        for (size_t i = 0; i < d; ++i)
            for (size_t j = 0; j < C; ++j)
                b_rm[i * C + j] = b_cm[i + j * d];

        gauss_solve(A_cm, b_rm, (int)d, (int)C);

        // b_rm now holds W_solve [d, C] row-major.
        // d1->weights() shape is [C, d] row-major: W[j, i] = W_solve[i, j].
        // Cast to BF16, transpose into weights layout.
        std::vector<__nv_bfloat16> W_bf16(C * d);
        for (size_t i = 0; i < d; ++i)
            for (size_t j = 0; j < C; ++j)
                W_bf16[j * d + i] = __float2bfloat16(b_rm[i * C + j]);

        FAYN_CUDA_CHECK(cudaMemcpy(m.d1->weights().data, W_bf16.data(),
                                   C * d * sizeof(__nv_bfloat16),
                                   cudaMemcpyHostToDevice));
    }

    fmt::print("elm_fit: solved {} members ({} samples each)\n", members_.size(), N);
}

float ELMEnsembleExperiment::run_epoch(size_t epoch) {
    if (!fitted_) {
        elm_fit();
        fitted_ = true;
    }

    data_->reset();
    const size_t n_batches = data_->batches_per_epoch(cfg_.batch_size);
    float total_acc = 0.f;

    for (size_t b = 0; b < n_batches; ++b) {
        Batch batch = data_->next_batch(cfg_.batch_size);
        std::vector<Tensor> member_outputs;
        member_outputs.reserve(members_.size());

        for (size_t k = 0; k < members_.size(); ++k) {
            const bool last = (k + 1 == members_.size());
            Tensor inp = last ? std::move(batch.inputs) : batch.inputs.borrow();
            std::vector<std::pair<int, Tensor>> fwd;
            fwd.emplace_back(0, std::move(inp));
            auto out = members_[k].graph->forward(std::move(fwd));
            if (out.empty())
                throw std::runtime_error(
                    "ELMEnsembleExperiment: member graph produced no output");
            member_outputs.push_back(std::move(out[0]));
        }

        std::vector<Tensor*> ptrs;
        for (auto& t : member_outputs) ptrs.push_back(&t);
        total_acc += ensemble_accuracy(ptrs, batch.targets);
    }

    const float acc = n_batches > 0 ? total_acc / (float)n_batches : 0.f;
    fmt::print("epoch {:3d}  acc={:.4f}\n", epoch, acc);
    return acc;
}

} // namespace fayn
