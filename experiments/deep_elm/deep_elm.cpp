#include "deep_elm.hpp"

#include "src/core/loss.hpp"
#include "src/ops/activations.hpp"

#include <cuda_bf16.h>
#include <fmt/format.h>

#include <stdexcept>
#include <vector>

namespace fayn {

DeepELMExperiment::DeepELMExperiment(
    const ExperimentConfig& cfg,
    std::string data_path,
    int d0,
    int d1,
    int n_cycles,
    float lambda1,
    float lambda2)
    : Experiment(cfg)
    , data_path_(std::move(data_path))
    , d0_(d0)
    , d1_(d1)
    , n_cycles_(n_cycles)
    , lambda1_(lambda1)
    , lambda2_(lambda2)
{}

void DeepELMExperiment::setup() {
    // Build data loader (training set only — evaluation runs on train set).
    data_ = std::make_unique<MnistLoader>(
        data_path_ + "/train-images-idx3-ubyte",
        data_path_ + "/train-labels-idx1-ubyte");
    data_->set_output_device(Device::CUDA);
    data_->set_input_dtype(DType::BFloat16);

    // W_0: frozen random projection [784 → d0].
    w0_ = std::make_shared<DenseLayer>(784, static_cast<size_t>(d0_));
    w0_->set_compute_stats(false);

    // W_1: inner learned layer [d0 → d1], FP32 weights for full-precision solve.
    w1_ = std::make_shared<DenseLayer>(static_cast<size_t>(d0_),
                                       static_cast<size_t>(d1_),
                                       /*use_bias=*/false);
    w1_->set_compute_stats(false);
    w1_->enable_fp32_weights();

    // W_2: readout layer [d1 → 10], FP32 weights.
    w2_ = std::make_shared<DenseLayer>(static_cast<size_t>(d1_), 10,
                                       /*use_bias=*/false);
    w2_->set_compute_stats(false);
    w2_->enable_fp32_weights();

    precompute_h0_t();
}

// ---------------------------------------------------------------------------
// precompute_h0_t: one pass through the training set to build:
//   H0_dev_ [N_fit, d0] FP32 — forward through frozen W_0 + ReLU
//   T_dev_  [N_fit, 10] FP32 — one-hot class labels
// Follows the same batch-then-host-assemble pattern as ELMEnsembleExperiment::elm_fit.
// ---------------------------------------------------------------------------
void DeepELMExperiment::precompute_h0_t() {
    const size_t fit_bs   = cfg_.batch_size;
    const size_t n_batches = data_->size() / fit_bs;  // floor-divide, drop partial batch
    N_fit_ = n_batches * fit_bs;

    const size_t d0 = static_cast<size_t>(d0_);
    const size_t C  = 10;

    std::vector<float>          H0_host(N_fit_ * d0, 0.f);
    std::vector<float>          T_host(N_fit_ * C, 0.f);
    std::vector<__nv_bfloat16>  bf16_buf(fit_bs * d0);
    std::vector<int32_t>        lbl_buf(fit_bs);

    data_->reset();
    for (size_t b = 0; b < n_batches; ++b) {
        Batch batch = data_->next_batch(fit_bs);

        // Labels: copy Int32 from device to host.
        FAYN_CUDA_CHECK(cudaMemcpy(lbl_buf.data(), batch.targets.data,
                                   fit_bs * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < fit_bs; ++i)
            T_host[(b * fit_bs + i) * C + static_cast<size_t>(lbl_buf[i])] = 1.f;

        // Forward through frozen W_0, apply ReLU, copy BF16 → host → FP32.
        Tensor h = w0_->forward(batch.inputs);
        apply_relu(h, /*stream=*/nullptr);
        FAYN_CUDA_CHECK(cudaStreamSynchronize(nullptr));
        FAYN_CUDA_CHECK(cudaMemcpy(bf16_buf.data(), h.data,
                                   fit_bs * d0 * sizeof(__nv_bfloat16),
                                   cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < fit_bs * d0; ++i)
            H0_host[b * fit_bs * d0 + i] = __bfloat162float(bf16_buf[i]);
    }

    // Upload assembled matrices to device as FP32.
    H0_dev_ = Tensor::make({(size_t)N_fit_, (size_t)d0},
                            DType::Float32, Device::CUDA);
    FAYN_CUDA_CHECK(cudaMemcpy(H0_dev_.data, H0_host.data(),
                               N_fit_ * d0 * sizeof(float), cudaMemcpyHostToDevice));

    T_dev_ = Tensor::make({(size_t)N_fit_, (size_t)C},
                           DType::Float32, Device::CUDA);
    FAYN_CUDA_CHECK(cudaMemcpy(T_dev_.data, T_host.data(),
                               N_fit_ * C * sizeof(float), cudaMemcpyHostToDevice));

    fmt::print("deep_elm: precomputed H0 [{}, {}] and T [{}, {}]\n",
               N_fit_, d0, N_fit_, C);
}

// ---------------------------------------------------------------------------
// compute_h1: H_1 = ReLU(H_0 @ W_1^T) using FP32 GEMM directly.
// ---------------------------------------------------------------------------
Tensor DeepELMExperiment::compute_h1() const {
    return solver_.relu_forward(H0_dev_, w1_->weights_fp32());
}

// ---------------------------------------------------------------------------
// write_fp32_weights: copy W_fp32 [d_out, d_in] device → layer.weights_fp32().
// ---------------------------------------------------------------------------
void DeepELMExperiment::write_fp32_weights(DenseLayer& layer, const Tensor& W_fp32) {
    const size_t n = W_fp32.numel();
    FAYN_CUDA_CHECK(cudaMemcpy(layer.weights_fp32().data, W_fp32.data,
                               n * sizeof(float), cudaMemcpyDeviceToDevice));
    FAYN_CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// evaluate: accuracy on the training set using the current W_1 and W_2.
// Pipeline: BF16 input → W_0 → ReLU → W_1 (FP32 path) → ReLU → W_2 (FP32 path) → logits
// ---------------------------------------------------------------------------
float DeepELMExperiment::evaluate() {
    data_->reset();
    const size_t n_batches = data_->batches_per_epoch(cfg_.batch_size);
    float total_acc = 0.f;

    for (size_t b = 0; b < n_batches; ++b) {
        Batch batch = data_->next_batch(cfg_.batch_size);

        Tensor h0 = w0_->forward(batch.inputs);
        apply_relu(h0, /*stream=*/nullptr);

        Tensor h1 = w1_->forward(h0);     // uses FP32 weights path; BF16 output
        apply_relu(h1, /*stream=*/nullptr);

        Tensor logits = w2_->forward(h1); // BF16 [batch, 10]

        total_acc += fayn::accuracy(logits, batch.targets);
    }

    return n_batches > 0 ? total_acc / static_cast<float>(n_batches) : 0.f;
}

// ---------------------------------------------------------------------------
// run_epoch: on epoch 0, run n_cycles alternating ELM solves.
//            All subsequent epochs just re-evaluate (network is fixed after fit).
// ---------------------------------------------------------------------------
float DeepELMExperiment::run_epoch(size_t epoch) {
    if (epoch == 0) {
        for (int cycle = 0; cycle < n_cycles_; ++cycle) {
            // Step 1: solve W_2 optimal for current W_1 features.
            Tensor H1 = compute_h1();
            Tensor W2 = solver_.solve(H1, T_dev_, lambda2_);
            write_fp32_weights(*w2_, W2);

            // Evaluate with consistent (W_1, W_2) pair.
            float acc = evaluate();
            fmt::print("  cycle {:2d}  acc={:.4f}\n", cycle, acc);

            // Step 2: back-project targets through W_2, solve W_1.
            Tensor H1_tgt = solver_.propagate_target(W2, T_dev_);
            Tensor W1     = solver_.solve(H0_dev_, H1_tgt, lambda1_);
            write_fp32_weights(*w1_, W1);
            // W_2 is now stale for the new W_1; next cycle will re-solve it.
        }
        // Final W_2 re-solve to match the last W_1 update.
        Tensor H1_final = compute_h1();
        Tensor W2_final = solver_.solve(H1_final, T_dev_, lambda2_);
        write_fp32_weights(*w2_, W2_final);
    }

    const float acc = evaluate();
    fmt::print("epoch {:3d}  acc={:.4f}\n", epoch, acc);
    return acc;
}

} // namespace fayn
