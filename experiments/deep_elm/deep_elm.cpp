#include "deep_elm.hpp"

#include "src/core/loss.hpp"
#include "src/ops/activations.hpp"

#include <cuda_bf16.h>
#include <fmt/format.h>

#include <memory>
#include <stdexcept>
#include <vector>

namespace fayn {

// ===========================================================================
// DeepELMExperiment implementation
// ===========================================================================

DeepELMExperiment::DeepELMExperiment(
    const ExperimentConfig& cfg,
    std::string data_path,
    int   d0,
    int   d,
    int   n_hidden,
    int   n_cycles,
    float lambda)
    : Experiment(cfg)
    , data_path_(std::move(data_path))
    , d0_(d0)
    , d_(d)
    , n_hidden_(n_hidden)
    , n_cycles_(n_cycles)
    , lambda_(lambda)
{}

void DeepELMExperiment::setup() {
    data_ = std::make_unique<MnistLoader>(
        data_path_ + "/train-images-idx3-ubyte",
        data_path_ + "/train-labels-idx1-ubyte");
    data_->set_output_device(Device::CUDA);
    data_->set_input_dtype(DType::BFloat16);

    test_data_ = std::make_unique<MnistLoader>(
        data_path_ + "/t10k-images-idx3-ubyte",
        data_path_ + "/t10k-labels-idx1-ubyte");
    test_data_->set_output_device(Device::CUDA);
    test_data_->set_input_dtype(DType::BFloat16);

    // W_0: frozen random projection [784 → d0].
    w0_ = std::make_shared<DenseLayer>(784, static_cast<size_t>(d0_));
    w0_->set_compute_stats(false);

    // hidden_[k]: n_hidden ELM layers.
    //   hidden_[0]: [d0 → d]
    //   hidden_[k] (k>0): [d → d]
    hidden_.resize(static_cast<size_t>(n_hidden_));
    for (int k = 0; k < n_hidden_; ++k) {
        const size_t in_dim = (k == 0) ? static_cast<size_t>(d0_)
                                       : static_cast<size_t>(d_);
        hidden_[static_cast<size_t>(k)] = std::make_shared<DenseLayer>(
            in_dim, static_cast<size_t>(d_), /*use_bias=*/false);
        hidden_[static_cast<size_t>(k)]->set_compute_stats(false);
        hidden_[static_cast<size_t>(k)]->enable_fp32_weights();
    }

    // readout: ELM [d → 10], FP32 weights.
    readout_ = std::make_shared<DenseLayer>(
        static_cast<size_t>(d_), 10, /*use_bias=*/false);
    readout_->set_compute_stats(false);
    readout_->enable_fp32_weights();

    precompute_h0_t();
}

// ---------------------------------------------------------------------------
// precompute_h0_t: one pass through the training set to build:
//   H0_dev_ [N_fit, d0] FP32 — forward through frozen W_0 + ReLU
//   T_dev_  [N_fit, 10] FP32 — one-hot class labels
// ---------------------------------------------------------------------------
void DeepELMExperiment::precompute_h0_t() {
    const size_t fit_bs   = cfg_.batch_size;
    const size_t n_batches = data_->size() / fit_bs;
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

        FAYN_CUDA_CHECK(cudaMemcpy(lbl_buf.data(), batch.targets.data,
                                   fit_bs * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < fit_bs; ++i)
            T_host[(b * fit_bs + i) * C + static_cast<size_t>(lbl_buf[i])] = 1.f;

        Tensor h = w0_->forward(batch.inputs);
        apply_relu(h, /*stream=*/nullptr);
        FAYN_CUDA_CHECK(cudaStreamSynchronize(nullptr));
        FAYN_CUDA_CHECK(cudaMemcpy(bf16_buf.data(), h.data,
                                   fit_bs * d0 * sizeof(__nv_bfloat16),
                                   cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < fit_bs * d0; ++i)
            H0_host[b * fit_bs * d0 + i] = __bfloat162float(bf16_buf[i]);
    }

    H0_dev_ = Tensor::make({N_fit_, d0}, DType::Float32, Device::CUDA);
    FAYN_CUDA_CHECK(cudaMemcpy(H0_dev_.data, H0_host.data(),
                               N_fit_ * d0 * sizeof(float), cudaMemcpyHostToDevice));

    T_dev_ = Tensor::make({N_fit_, C}, DType::Float32, Device::CUDA);
    FAYN_CUDA_CHECK(cudaMemcpy(T_dev_.data, T_host.data(),
                               N_fit_ * C * sizeof(float), cudaMemcpyHostToDevice));

    fmt::print("deep_elm: precomputed H0 [{}, {}] and T [{}, {}] ({} hidden layers)\n",
               N_fit_, d0, N_fit_, C, n_hidden_);
}

// ---------------------------------------------------------------------------
// compute_hidden_activations: returns H[0..n_hidden-1] where
//   H[k] = ReLU(H[k-1] @ hidden_[k]^T)   (H[-1] = H0_dev_)
// ---------------------------------------------------------------------------
std::vector<Tensor> DeepELMExperiment::compute_hidden_activations() const {
    std::vector<Tensor> H;
    H.reserve(static_cast<size_t>(n_hidden_));
    for (int k = 0; k < n_hidden_; ++k) {
        const Tensor& prev = (k == 0) ? H0_dev_ : H[static_cast<size_t>(k - 1)];
        H.push_back(solver_.relu_forward(prev, hidden_[static_cast<size_t>(k)]->weights_fp32()));
    }
    return H;
}

// ---------------------------------------------------------------------------
// write_fp32_weights: copy W_fp32 [d_out, d_in] device → layer.weights_fp32().
// ---------------------------------------------------------------------------
void DeepELMExperiment::write_fp32_weights(DenseLayer& layer, const Tensor& W_fp32) {
    FAYN_CUDA_CHECK(cudaMemcpy(layer.weights_fp32().data, W_fp32.data,
                               W_fp32.numel() * sizeof(float),
                               cudaMemcpyDeviceToDevice));
    FAYN_CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// evaluate: accuracy on the given data source using current hidden_ and readout_.
// ---------------------------------------------------------------------------
float DeepELMExperiment::evaluate(DataSource& ds) {
    ds.reset();
    const size_t n_batches = ds.batches_per_epoch(cfg_.batch_size);
    float total_acc = 0.f;

    for (size_t b = 0; b < n_batches; ++b) {
        Batch batch = ds.next_batch(cfg_.batch_size);

        Tensor h = w0_->forward(batch.inputs);
        apply_relu(h, /*stream=*/nullptr);

        for (auto& layer : hidden_) {
            h = layer->forward(h);
            apply_relu(h, /*stream=*/nullptr);
        }

        Tensor logits = readout_->forward(h);
        total_acc += fayn::accuracy(logits, batch.targets);
    }

    return n_batches > 0 ? total_acc / static_cast<float>(n_batches) : 0.f;
}

// ---------------------------------------------------------------------------
// run_epoch: on epoch 0, run n_cycles alternating ELM solves.
//            Subsequent epochs just re-evaluate (network fixed after fit).
//
// Cycle structure (backward target propagation):
//   Forward: H = compute_hidden_activations()
//   Solve readout: W_r = solve(H.back(), T, λ)
//   Evaluate.
//   Backward: starting from readout targets T, propagate down each hidden:
//     T_owned = propagate_target(W_curr, T_curr); apply_relu
//     W_k     = solve(H_pre, T_owned, λ)
//     W_curr  = W_k  (for next layer down)
// ---------------------------------------------------------------------------
float DeepELMExperiment::run_epoch(size_t epoch) {
    if (epoch == 0) {
        for (int cycle = 0; cycle < n_cycles_; ++cycle) {
            auto H = compute_hidden_activations();

            // Solve readout optimal for current hidden activations.
            Tensor W_r = solver_.solve(H.back(), T_dev_, lambda_);
            write_fp32_weights(*readout_, W_r);

            fmt::print("  cycle {:2d}  train={:.4f}\n", cycle, evaluate(*data_));

            // Back-propagate targets from readout down to hidden_[0].
            const Tensor*           T_curr = &T_dev_;
            const Tensor*           W_curr = &readout_->weights_fp32();
            std::unique_ptr<Tensor> T_owned;

            for (int k = n_hidden_ - 1; k >= 0; --k) {
                T_owned = std::make_unique<Tensor>(
                    solver_.propagate_target(*W_curr, *T_curr));
                apply_relu(*T_owned, /*stream=*/nullptr);
                T_curr = T_owned.get();

                const Tensor& H_pre =
                    (k == 0) ? H0_dev_ : H[static_cast<size_t>(k - 1)];
                Tensor W_k = solver_.solve(H_pre, *T_curr, lambda_);
                write_fp32_weights(*hidden_[static_cast<size_t>(k)], W_k);
                W_curr = &hidden_[static_cast<size_t>(k)]->weights_fp32();
            }
        }

        // Final readout re-solve consistent with last hidden update.
        auto H_final = compute_hidden_activations();
        Tensor W_r_final = solver_.solve(H_final.back(), T_dev_, lambda_);
        write_fp32_weights(*readout_, W_r_final);
    }

    const float train_acc = evaluate(*data_);
    const float test_acc  = evaluate(*test_data_);
    fmt::print("epoch {:3d}  train={:.4f}  test={:.4f}\n", epoch, train_acc, test_acc);
    return test_acc;
}

// ===========================================================================
// HybridElmHebbianExperiment implementation
// ===========================================================================

HybridElmHebbianExperiment::HybridElmHebbianExperiment(
    const ExperimentConfig& cfg,
    std::string data_path,
    int   d0,
    int   d,
    int   n_hidden,
    float lambda2,
    float lr_w,
    bool  elm_init)
    : Experiment(cfg)
    , data_path_(std::move(data_path))
    , d0_(d0)
    , d_(d)
    , n_hidden_(n_hidden)
    , lambda2_(lambda2)
    , lr_w_(lr_w)
    , elm_init_(elm_init)
{}

void HybridElmHebbianExperiment::setup() {
    data_ = std::make_unique<MnistLoader>(
        data_path_ + "/train-images-idx3-ubyte",
        data_path_ + "/train-labels-idx1-ubyte");
    data_->set_output_device(Device::CUDA);
    data_->set_input_dtype(DType::BFloat16);

    test_data_ = std::make_unique<MnistLoader>(
        data_path_ + "/t10k-images-idx3-ubyte",
        data_path_ + "/t10k-labels-idx1-ubyte");
    test_data_->set_output_device(Device::CUDA);
    test_data_->set_input_dtype(DType::BFloat16);

    w0_ = std::make_shared<DenseLayer>(784, static_cast<size_t>(d0_));
    w0_->set_compute_stats(false);

    hidden_.resize(static_cast<size_t>(n_hidden_));
    for (int k = 0; k < n_hidden_; ++k) {
        const size_t in_dim = (k == 0) ? static_cast<size_t>(d0_)
                                       : static_cast<size_t>(d_);
        hidden_[static_cast<size_t>(k)] = std::make_shared<DenseLayer>(
            in_dim, static_cast<size_t>(d_), /*use_bias=*/false);
        hidden_[static_cast<size_t>(k)]->set_compute_stats(false);
        hidden_[static_cast<size_t>(k)]->enable_fp32_weights();
    }

    readout_ = std::make_shared<DenseLayer>(
        static_cast<size_t>(d_), 10, /*use_bias=*/false);
    readout_->set_compute_stats(false);
    readout_->enable_fp32_weights();

    precompute_h0_t();
}

void HybridElmHebbianExperiment::precompute_h0_t() {
    const size_t fit_bs   = cfg_.batch_size;
    const size_t n_batches = data_->size() / fit_bs;
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

        FAYN_CUDA_CHECK(cudaMemcpy(lbl_buf.data(), batch.targets.data,
                                   fit_bs * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < fit_bs; ++i)
            T_host[(b * fit_bs + i) * C + static_cast<size_t>(lbl_buf[i])] = 1.f;

        Tensor h = w0_->forward(batch.inputs);
        apply_relu(h, /*stream=*/nullptr);
        FAYN_CUDA_CHECK(cudaStreamSynchronize(nullptr));
        FAYN_CUDA_CHECK(cudaMemcpy(bf16_buf.data(), h.data,
                                   fit_bs * d0 * sizeof(__nv_bfloat16),
                                   cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < fit_bs * d0; ++i)
            H0_host[b * fit_bs * d0 + i] = __bfloat162float(bf16_buf[i]);
    }

    H0_dev_ = Tensor::make({N_fit_, d0}, DType::Float32, Device::CUDA);
    FAYN_CUDA_CHECK(cudaMemcpy(H0_dev_.data, H0_host.data(),
                               N_fit_ * d0 * sizeof(float), cudaMemcpyHostToDevice));

    T_dev_ = Tensor::make({N_fit_, C}, DType::Float32, Device::CUDA);
    FAYN_CUDA_CHECK(cudaMemcpy(T_dev_.data, T_host.data(),
                               N_fit_ * C * sizeof(float), cudaMemcpyHostToDevice));

    fmt::print("hybrid_elm_hebb: precomputed H0 [{}, {}] and T [{}, {}] ({} hidden layers)\n",
               N_fit_, d0, N_fit_, C, n_hidden_);
}

std::vector<Tensor> HybridElmHebbianExperiment::compute_hidden_activations() const {
    std::vector<Tensor> H;
    H.reserve(static_cast<size_t>(n_hidden_));
    for (int k = 0; k < n_hidden_; ++k) {
        const Tensor& prev = (k == 0) ? H0_dev_ : H[static_cast<size_t>(k - 1)];
        H.push_back(solver_.relu_forward(prev, hidden_[static_cast<size_t>(k)]->weights_fp32()));
    }
    return H;
}

void HybridElmHebbianExperiment::write_fp32_weights(DenseLayer& layer,
                                                     const Tensor& W_fp32) {
    FAYN_CUDA_CHECK(cudaMemcpy(layer.weights_fp32().data, W_fp32.data,
                               W_fp32.numel() * sizeof(float),
                               cudaMemcpyDeviceToDevice));
    FAYN_CUDA_CHECK(cudaDeviceSynchronize());
}

float HybridElmHebbianExperiment::evaluate(DataSource& ds) {
    ds.reset();
    const size_t n_batches = ds.batches_per_epoch(cfg_.batch_size);
    float total_acc = 0.f;

    for (size_t b = 0; b < n_batches; ++b) {
        Batch batch = ds.next_batch(cfg_.batch_size);

        Tensor h = w0_->forward(batch.inputs);
        apply_relu(h, /*stream=*/nullptr);

        for (auto& layer : hidden_) {
            h = layer->forward(h);
            apply_relu(h, /*stream=*/nullptr);
        }

        Tensor logits = readout_->forward(h);
        total_acc += fayn::accuracy(logits, batch.targets);
    }

    return n_batches > 0 ? total_acc / static_cast<float>(n_batches) : 0.f;
}

// ---------------------------------------------------------------------------
// run_epoch: per-epoch ELM readout re-solve + gradient steps on hidden layers.
// If elm_init_=true, epoch 0 first warm-starts all hidden via one ELM cycle.
//
// Backward propagation loop (shared by both ELM warm-start and per-epoch gradient):
//   T_curr = T_dev_ (from readout)
//   W_curr = readout_->weights_fp32()
//   for k from n_hidden-1 down to 0:
//     T_owned = propagate_target(W_curr, T_curr); apply_relu
//     H_pre   = (k==0) ? H0_dev_ : H[k-1]
//     [ELM:     W_k = solve(H_pre, T_owned, λ); write_fp32_weights]
//     [Hebb:    gradient_step(W_k, H_pre, H[k], T_owned, lr)]
//     W_curr  = W_k->weights_fp32()
// ---------------------------------------------------------------------------
float HybridElmHebbianExperiment::run_epoch(size_t epoch) {
    // Optional ELM warm-start on epoch 0.
    if (epoch == 0 && elm_init_) {
        auto H = compute_hidden_activations();
        Tensor W_r = solver_.solve(H.back(), T_dev_, lambda2_);
        write_fp32_weights(*readout_, W_r);

        const Tensor*           T_curr = &T_dev_;
        const Tensor*           W_curr = &readout_->weights_fp32();
        std::unique_ptr<Tensor> T_owned;

        for (int k = n_hidden_ - 1; k >= 0; --k) {
            T_owned = std::make_unique<Tensor>(
                solver_.propagate_target(*W_curr, *T_curr));
            apply_relu(*T_owned, /*stream=*/nullptr);
            T_curr = T_owned.get();

            const Tensor& H_pre =
                (k == 0) ? H0_dev_ : H[static_cast<size_t>(k - 1)];
            Tensor W_k = solver_.solve(H_pre, *T_curr, 1e-4f);
            write_fp32_weights(*hidden_[static_cast<size_t>(k)], W_k);
            W_curr = &hidden_[static_cast<size_t>(k)]->weights_fp32();
        }

        // Re-solve readout consistent with warm-started hidden.
        auto H2 = compute_hidden_activations();
        Tensor W_r2 = solver_.solve(H2.back(), T_dev_, lambda2_);
        write_fp32_weights(*readout_, W_r2);
        fmt::print("  elm_init  train={:.4f}\n", evaluate(*data_));
    }

    // Per-epoch: re-solve readout, gradient steps on all hidden layers.
    auto H = compute_hidden_activations();
    Tensor W_r = solver_.solve(H.back(), T_dev_, lambda2_);
    write_fp32_weights(*readout_, W_r);

    const Tensor*           T_curr = &T_dev_;
    const Tensor*           W_curr = &readout_->weights_fp32();
    std::unique_ptr<Tensor> T_owned;

    for (int k = n_hidden_ - 1; k >= 0; --k) {
        T_owned = std::make_unique<Tensor>(
            solver_.propagate_target(*W_curr, *T_curr));
        apply_relu(*T_owned, /*stream=*/nullptr);
        T_curr = T_owned.get();

        const Tensor& H_pre =
            (k == 0) ? H0_dev_ : H[static_cast<size_t>(k - 1)];
        solver_.gradient_step(
            hidden_[static_cast<size_t>(k)]->weights_fp32(),
            H_pre, H[static_cast<size_t>(k)], *T_curr, lr_w_);
        W_curr = &hidden_[static_cast<size_t>(k)]->weights_fp32();
    }

    const float train_acc = evaluate(*data_);
    const float test_acc  = evaluate(*test_data_);
    fmt::print("epoch {:3d}  train={:.4f}  test={:.4f}\n", epoch, train_acc, test_acc);
    return test_acc;
}

// ===========================================================================
// AdmmElmExperiment implementation
// ===========================================================================

AdmmElmExperiment::AdmmElmExperiment(
    const ExperimentConfig& cfg,
    std::string data_path,
    int   d0,
    int   d,
    int   n_hidden,
    int   n_admm,
    float lambda,
    float rho,
    float mu,
    float leaky_alpha)
    : Experiment(cfg)
    , data_path_(std::move(data_path))
    , d0_(d0), d_(d), n_hidden_(n_hidden), n_admm_(n_admm)
    , lambda_(lambda), rho_(rho), mu_(mu), leaky_alpha_(leaky_alpha)
{}

void AdmmElmExperiment::setup() {
    data_ = std::make_unique<MnistLoader>(
        data_path_ + "/train-images-idx3-ubyte",
        data_path_ + "/train-labels-idx1-ubyte");
    data_->set_output_device(Device::CUDA);
    data_->set_input_dtype(DType::BFloat16);

    test_data_ = std::make_unique<MnistLoader>(
        data_path_ + "/t10k-images-idx3-ubyte",
        data_path_ + "/t10k-labels-idx1-ubyte");
    test_data_->set_output_device(Device::CUDA);
    test_data_->set_input_dtype(DType::BFloat16);

    w0_ = std::make_shared<DenseLayer>(784, static_cast<size_t>(d0_));
    w0_->set_compute_stats(false);

    hidden_.resize(static_cast<size_t>(n_hidden_));
    for (int k = 0; k < n_hidden_; ++k) {
        const size_t in_dim = (k == 0) ? static_cast<size_t>(d0_)
                                       : static_cast<size_t>(d_);
        hidden_[static_cast<size_t>(k)] = std::make_shared<DenseLayer>(
            in_dim, static_cast<size_t>(d_), /*bias=*/false);
        hidden_[static_cast<size_t>(k)]->set_compute_stats(false);
        hidden_[static_cast<size_t>(k)]->enable_fp32_weights();
    }

    readout_ = std::make_shared<DenseLayer>(
        static_cast<size_t>(d_), 10, /*bias=*/false);
    readout_->set_compute_stats(false);
    readout_->enable_fp32_weights();

    precompute_h0_t();

    // Allocate Z_k state tensors and initialize from a forward pass
    // with the random weights: Z_k^(0) = H_{k-1} @ W_k^T.
    Z_.clear();
    const size_t d_sz = static_cast<size_t>(d_);
    for (int k = 0; k < n_hidden_; ++k)
        Z_.push_back(Tensor::make({N_fit_, d_sz}, DType::Float32, Device::CUDA));
    u_zero_ = Tensor::make({N_fit_, d_sz}, DType::Float32, Device::CUDA);
    FAYN_CUDA_CHECK(cudaMemset(u_zero_.data, 0, N_fit_ * d_sz * sizeof(float)));

    {
        std::vector<Tensor> H_init;
        H_init.reserve(static_cast<size_t>(n_hidden_));
        const Tensor* H_prev = &H0_dev_;
        for (int k = 0; k < n_hidden_; ++k) {
            const size_t ki = static_cast<size_t>(k);
            Tensor A = solver_.linear_forward(*H_prev, hidden_[ki]->weights_fp32());
            FAYN_CUDA_CHECK(cudaMemcpy(Z_[ki].data, A.data,
                N_fit_ * d_sz * sizeof(float), cudaMemcpyDeviceToDevice));
            H_init.push_back(Tensor::make({N_fit_, d_sz}, DType::Float32, Device::CUDA));
            FAYN_CUDA_CHECK(cudaMemcpy(H_init.back().data, Z_[ki].data,
                N_fit_ * d_sz * sizeof(float), cudaMemcpyDeviceToDevice));
            apply_leaky_relu(H_init.back(), leaky_alpha_);
            H_prev = &H_init.back();
        }
    }
}

void AdmmElmExperiment::precompute_h0_t() {
    const size_t fit_bs   = cfg_.batch_size;
    const size_t n_batches = data_->size() / fit_bs;
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

        FAYN_CUDA_CHECK(cudaMemcpy(lbl_buf.data(), batch.targets.data,
                                   fit_bs * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < fit_bs; ++i)
            T_host[(b * fit_bs + i) * C + static_cast<size_t>(lbl_buf[i])] = 1.f;

        // H0 uses ReLU (consistent with how this frozen projection layer is defined).
        Tensor h = w0_->forward(batch.inputs);
        apply_relu(h, /*stream=*/nullptr);
        FAYN_CUDA_CHECK(cudaStreamSynchronize(nullptr));
        FAYN_CUDA_CHECK(cudaMemcpy(bf16_buf.data(), h.data,
                                   fit_bs * d0 * sizeof(__nv_bfloat16),
                                   cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < fit_bs * d0; ++i)
            H0_host[b * fit_bs * d0 + i] = __bfloat162float(bf16_buf[i]);
    }

    H0_dev_ = Tensor::make({N_fit_, d0}, DType::Float32, Device::CUDA);
    FAYN_CUDA_CHECK(cudaMemcpy(H0_dev_.data, H0_host.data(),
                               N_fit_ * d0 * sizeof(float), cudaMemcpyHostToDevice));

    T_dev_ = Tensor::make({N_fit_, C}, DType::Float32, Device::CUDA);
    FAYN_CUDA_CHECK(cudaMemcpy(T_dev_.data, T_host.data(),
                               N_fit_ * C * sizeof(float), cudaMemcpyHostToDevice));

    fmt::print("admm_elm: precomputed H0 [{}, {}] and T [{}, {}] ({} hidden layers)\n",
               N_fit_, d0, N_fit_, C, n_hidden_);
}

std::vector<Tensor> AdmmElmExperiment::get_hidden_activations() const {
    std::vector<Tensor> H;
    H.reserve(static_cast<size_t>(n_hidden_));
    const size_t d_sz = static_cast<size_t>(d_);
    for (int k = 0; k < n_hidden_; ++k) {
        Tensor h = Tensor::make({N_fit_, d_sz}, DType::Float32, Device::CUDA);
        FAYN_CUDA_CHECK(cudaMemcpy(h.data, Z_[static_cast<size_t>(k)].data,
            N_fit_ * d_sz * sizeof(float), cudaMemcpyDeviceToDevice));
        apply_leaky_relu(h, leaky_alpha_);
        H.push_back(std::move(h));
    }
    return H;
}

void AdmmElmExperiment::write_fp32_weights(DenseLayer& layer, const Tensor& W_fp32) {
    auto& W_layer = layer.weights_fp32();
    const size_t n = W_fp32.numel();
    FAYN_CUDA_CHECK(cudaMemcpy(W_layer.data, W_fp32.data,
        n * sizeof(float), cudaMemcpyDeviceToDevice));
}

float AdmmElmExperiment::evaluate(DataSource& ds) {
    ds.reset();
    const size_t n_batches = ds.batches_per_epoch(cfg_.batch_size);
    float total_acc = 0.f;
    for (size_t b = 0; b < n_batches; ++b) {
        Batch batch = ds.next_batch(cfg_.batch_size);
        // w0_ is the frozen ReLU projection — matches how H0_dev_ is built.
        Tensor h = w0_->forward(batch.inputs);
        apply_relu(h, /*stream=*/nullptr);
        // Trained hidden layers use LeakyReLU.
        for (auto& layer : hidden_) {
            h = layer->forward(h);
            apply_leaky_relu(h, leaky_alpha_);
        }
        Tensor logits = readout_->forward(h);
        total_acc += fayn::accuracy(logits, batch.targets);
    }
    return n_batches > 0 ? total_acc / static_cast<float>(n_batches) : 0.f;
}

// ---------------------------------------------------------------------------
// run_epoch: epoch 0 runs n_admm_ proximal ALS iterations; subsequent epochs
// just re-evaluate (network is fixed after the fit).
//
// Proximal ALS iteration (no dual variables — coordinate descent on penalty):
//   1. H_k = LeakyReLU(Z_k)                                  [activations from Z]
//   2. W_k  = solve(H_{k-1}, Z_k, lambda/rho)                [W fits current Z_k]
//   3. W_r  = solve(H_n, T, lambda)                           [ELM readout]
//   4. A_k  = H_{k-1} @ W_k^T                                [new linear pre-acts]
//   5. T_k  = Gram-propagate targets: T_{n-1}=prop(W_r,T); T_{k-1}=prop(W_k,T_k)
//   6. Z_k  = element-wise argmin (rho/2)(z-A_k)^2+(mu/2)(sigma(z)-T_k)^2
//             — closed-form case split (see AdmmElmExperiment header comment)
// ---------------------------------------------------------------------------
float AdmmElmExperiment::run_epoch(size_t epoch) {
    if (epoch == 0) {
        const size_t d_sz = static_cast<size_t>(d_);

        for (int iter = 0; iter < n_admm_; ++iter) {
            // ---- Step 1: H_k = LeakyReLU(Z_k) ----
            std::vector<Tensor> H(static_cast<size_t>(n_hidden_));
            for (int k = 0; k < n_hidden_; ++k) {
                const size_t ki = static_cast<size_t>(k);
                H[ki] = Tensor::make({N_fit_, d_sz}, DType::Float32, Device::CUDA);
                FAYN_CUDA_CHECK(cudaMemcpy(H[ki].data, Z_[ki].data,
                    N_fit_ * d_sz * sizeof(float), cudaMemcpyDeviceToDevice));
                apply_leaky_relu(H[ki], leaky_alpha_);
            }

            // ---- Step 2: W-update — ELM solve, target = current Z_k (no dual) ----
            {
                const Tensor* H_prev = &H0_dev_;
                for (int k = 0; k < n_hidden_; ++k) {
                    const size_t ki = static_cast<size_t>(k);
                    Tensor W_k = solver_.solve(*H_prev, Z_[ki], lambda_ / rho_);
                    write_fp32_weights(*hidden_[ki], W_k);
                    H_prev = &H[ki];
                }
            }

            // ---- Step 3: Readout update ----
            {
                Tensor W_r = solver_.solve(H.back(), T_dev_, lambda_);
                write_fp32_weights(*readout_, W_r);
            }

            // ---- Step 4: A_k = H_{k-1} @ W_k^{new,T} ----
            std::vector<Tensor> A(static_cast<size_t>(n_hidden_));
            {
                const Tensor* H_prev = &H0_dev_;
                for (int k = 0; k < n_hidden_; ++k) {
                    const size_t ki = static_cast<size_t>(k);
                    A[ki] = solver_.linear_forward(*H_prev, hidden_[ki]->weights_fp32());
                    H_prev = &H[ki];
                }
            }

            // ---- Step 5: Top-down target propagation ----
            // T_k is the post-activation target for hidden layer k.
            std::vector<Tensor> T_targets(static_cast<size_t>(n_hidden_));
            {
                Tensor T_curr = solver_.propagate_target(
                    readout_->weights_fp32(), T_dev_, lambda_);
                for (int k = n_hidden_ - 1; k >= 0; --k) {
                    const size_t ki = static_cast<size_t>(k);
                    T_targets[ki] = std::move(T_curr);
                    if (k > 0)
                        T_curr = solver_.propagate_target(
                            hidden_[ki]->weights_fp32(), T_targets[ki], lambda_);
                }
            }

            // ---- Step 6: Z-update — element-wise case-split minimizer ----
            // u_zero_ is a pre-allocated zero tensor (proximal ALS: no dual accumulation).
            for (int k = 0; k < n_hidden_; ++k) {
                const size_t ki = static_cast<size_t>(k);
                admm_z_update(Z_[ki], A[ki], u_zero_, T_targets[ki],
                              rho_, mu_, leaky_alpha_);
            }
            FAYN_CUDA_CHECK(cudaDeviceSynchronize());

            fmt::print("  iter {:3d}  train={:.4f}\n", iter, evaluate(*data_));
        }

        // Final readout re-solve with converged activations.
        {
            std::vector<Tensor> H_final = get_hidden_activations();
            Tensor W_r = solver_.solve(H_final.back(), T_dev_, lambda_);
            write_fp32_weights(*readout_, W_r);
        }
    }

    const float train_acc = evaluate(*data_);
    const float test_acc  = evaluate(*test_data_);
    fmt::print("epoch {:3d}  train={:.4f}  test={:.4f}\n", epoch, train_acc, test_acc);
    return test_acc;
}

} // namespace fayn
