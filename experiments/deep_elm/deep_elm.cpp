#include "deep_elm.hpp"

#include "src/core/loss.hpp"
#include "src/ops/activations.hpp"

#include <cuda_bf16.h>
#include <fmt/format.h>

#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace fayn {

// ---------------------------------------------------------------------------
// shift_mnist_bf16: shift each 28×28 image in a batch of N flat BF16 vectors.
//
// dir 0 (left):  output[row][col] = input[row][col+1]  (content shifts left)
// dir 1 (right): output[row][col] = input[row][col-1]  (content shifts right)
// dir 2 (up):    output[row][col] = input[row+1][col]  (content shifts up)
// dir 3 (down):  output[row][col] = input[row-1][col]  (content shifts down)
//
// Out-of-bounds pixels are filled with 0 (zero-padding).
// ---------------------------------------------------------------------------
static void shift_mnist_bf16(
    const __nv_bfloat16* src,   // [N, 784]
    __nv_bfloat16*       dst,   // [N, 784]
    size_t N,
    int direction)
{
    static const __nv_bfloat16 kZero = __float2bfloat16(0.f);
    for (size_t i = 0; i < N; ++i) {
        const __nv_bfloat16* s = src + i * 784;
        __nv_bfloat16*       d = dst + i * 784;
        for (int row = 0; row < 28; ++row) {
            for (int col = 0; col < 28; ++col) {
                int sr = row, sc = col;
                switch (direction) {
                    case 0: sc = col + 1; break;  // left
                    case 1: sc = col - 1; break;  // right
                    case 2: sr = row + 1; break;  // up
                    case 3: sr = row - 1; break;  // down
                }
                d[row * 28 + col] = (sr >= 0 && sr < 28 && sc >= 0 && sc < 28)
                    ? s[sr * 28 + sc] : kZero;
            }
        }
    }
}

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
    float lambda,
    bool  use_tanh,
    bool  use_rff,
    float rff_gamma,
    bool  augment,
    bool  use_tta,
    bool  use_conv,
    int   conv_c_out,
    bool  learn_conv)
    : Experiment(cfg)
    , data_path_(std::move(data_path))
    , d0_(d0)
    , d_(d)
    , n_hidden_(n_hidden)
    , n_cycles_(n_cycles)
    , lambda_(lambda)
    , use_tanh_(use_tanh)
    , use_rff_(use_rff)
    , rff_gamma_(rff_gamma)
    , augment_(augment)
    , use_tta_(use_tta)
    , use_conv_(use_conv)
    , conv_c_out_(conv_c_out)
    , learn_conv_(learn_conv)
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

    if (use_conv_) {
        // Convolutional front-end: 5×5 filters, Kaiming init, frozen.
        // Output features: conv_c_out_ * 144  (after ReLU + 2×2 max-pool).
        // d0_ is overridden to match; the d0 constructor parameter is ignored.
        conv_front_ = std::make_unique<ConvFrontend>(conv_c_out_, /*max_pool=*/true);
        d0_ = conv_front_->output_features();
    } else {
        // W_0: frozen random projection [784 → d0].
        w0_ = std::make_shared<DenseLayer>(784, static_cast<size_t>(d0_));
        w0_->set_compute_stats(false);

        // RFF: overwrite W_0 weights with N(0, rff_gamma*I) and bias with U[0, 2π].
        if (use_rff_) {
            w0_->enable_fp32_weights();
            const size_t d0 = static_cast<size_t>(d0_);
            std::mt19937 rng(42);
            std::normal_distribution<float> nd(0.f, std::sqrt(rff_gamma_));
            std::vector<float> W_host(d0 * 784);
            for (auto& v : W_host) v = nd(rng);
            FAYN_CUDA_CHECK(cudaMemcpy(w0_->weights_fp32().data, W_host.data(),
                                       d0 * 784 * sizeof(float), cudaMemcpyHostToDevice));
            constexpr float k2Pi = 6.28318530718f;
            std::uniform_real_distribution<float> ud(0.f, k2Pi);
            std::vector<float> b_host(d0);
            for (auto& v : b_host) v = ud(rng);
            FAYN_CUDA_CHECK(cudaMemcpy(w0_->bias().data, b_host.data(),
                                       d0 * sizeof(float), cudaMemcpyHostToDevice));
        }
    }

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
// precompute_h0_t: one (or five) passes through the training set to build:
//   H0_dev_ [N_fit, d0] FP32 — forward through frozen W_0 + activation
//   T_dev_  [N_fit, 10] FP32 — one-hot class labels
//
// If augment_=true, N_fit is 5× the dataset size: the original samples
// followed by 4 pixel-shifted copies (left/right/up/down by 1 pixel).
// All 5 copies share the same labels. This is standard training-time
// augmentation: the ELM solves for a readout that is invariant to ±1-pixel
// translations, which are common between training and test distributions.
// ---------------------------------------------------------------------------
void DeepELMExperiment::precompute_h0_t() {
    const size_t fit_bs   = cfg_.batch_size;
    const size_t n_batches = data_->size() / fit_bs;
    N_fit_ = n_batches * fit_bs;

    const size_t d0 = static_cast<size_t>(d0_);
    const size_t C  = 10;
    const int    n_aug = augment_ ? 5 : 1;  // 1 original + 4 shifts

    std::vector<float>          H0_host(N_fit_ * d0, 0.f);
    std::vector<float>          T_host(N_fit_ * C, 0.f);
    std::vector<__nv_bfloat16>  bf16_buf(fit_bs * d0);
    std::vector<int32_t>        lbl_buf(fit_bs);

    // ---- Pass 0: original images ----
    data_->reset();
    for (size_t b = 0; b < n_batches; ++b) {
        Batch batch = data_->next_batch(fit_bs);

        FAYN_CUDA_CHECK(cudaMemcpy(lbl_buf.data(), batch.targets.data,
                                   fit_bs * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < fit_bs; ++i)
            T_host[(b * fit_bs + i) * C + static_cast<size_t>(lbl_buf[i])] = 1.f;

        if (use_conv_) {
            // ConvFrontend outputs FP32 directly — no BF16→FP32 conversion needed.
            Tensor h = conv_front_->forward(batch.inputs);  // FP32 [bs, d0]
            FAYN_CUDA_CHECK(cudaMemcpy(H0_host.data() + b * fit_bs * d0,
                                       h.data,
                                       fit_bs * d0 * sizeof(float),
                                       cudaMemcpyDeviceToHost));
            FAYN_CUDA_CHECK(cudaDeviceSynchronize());
        } else {
            Tensor h = w0_->forward(batch.inputs);
            if (use_rff_) apply_cos(h, /*stream=*/nullptr);
            else          apply_relu(h, /*stream=*/nullptr);
            FAYN_CUDA_CHECK(cudaStreamSynchronize(nullptr));
            FAYN_CUDA_CHECK(cudaMemcpy(bf16_buf.data(), h.data,
                                       fit_bs * d0 * sizeof(__nv_bfloat16),
                                       cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < fit_bs * d0; ++i)
                H0_host[b * fit_bs * d0 + i] = __bfloat162float(bf16_buf[i]);
        }
    }

    // Allocate device tensors for n_aug × N_fit_ rows.
    H0_dev_ = Tensor::make({static_cast<size_t>(n_aug) * N_fit_, d0},
                           DType::Float32, Device::CUDA);
    T_dev_  = Tensor::make({static_cast<size_t>(n_aug) * N_fit_, C},
                           DType::Float32, Device::CUDA);

    FAYN_CUDA_CHECK(cudaMemcpy(H0_dev_.data, H0_host.data(),
                               N_fit_ * d0 * sizeof(float), cudaMemcpyHostToDevice));
    FAYN_CUDA_CHECK(cudaMemcpy(T_dev_.data, T_host.data(),
                               N_fit_ * C * sizeof(float), cudaMemcpyHostToDevice));

    // ---- Passes 1-4: shifted images ----
    if (augment_) {
        std::vector<__nv_bfloat16> input_host(fit_bs * 784);
        std::vector<__nv_bfloat16> shifted_host(fit_bs * 784);

        for (int dir = 0; dir < 4; ++dir) {
            data_->reset();
            std::fill(H0_host.begin(), H0_host.end(), 0.f);

            for (size_t b = 0; b < n_batches; ++b) {
                Batch batch = data_->next_batch(fit_bs);

                // Copy inputs to host, shift, re-upload.
                FAYN_CUDA_CHECK(cudaMemcpy(input_host.data(), batch.inputs.data,
                    fit_bs * 784 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
                shift_mnist_bf16(input_host.data(), shifted_host.data(), fit_bs, dir);
                Tensor shifted_dev = Tensor::make({fit_bs, 784},
                                                 DType::BFloat16, Device::CUDA);
                FAYN_CUDA_CHECK(cudaMemcpy(shifted_dev.data, shifted_host.data(),
                    fit_bs * 784 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

                Tensor h = w0_->forward(shifted_dev);
                if (use_rff_) apply_cos(h, /*stream=*/nullptr);
                else          apply_relu(h, /*stream=*/nullptr);
                FAYN_CUDA_CHECK(cudaStreamSynchronize(nullptr));
                FAYN_CUDA_CHECK(cudaMemcpy(bf16_buf.data(), h.data,
                    fit_bs * d0 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < fit_bs * d0; ++i)
                    H0_host[b * fit_bs * d0 + i] = __bfloat162float(bf16_buf[i]);
            }

            // Write shifted H0 to slice (dir+1) of H0_dev_ and T_dev_.
            FAYN_CUDA_CHECK(cudaMemcpy(
                static_cast<float*>(H0_dev_.data) + (size_t)(dir + 1) * N_fit_ * d0,
                H0_host.data(), N_fit_ * d0 * sizeof(float), cudaMemcpyHostToDevice));
            FAYN_CUDA_CHECK(cudaMemcpy(
                static_cast<float*>(T_dev_.data) + (size_t)(dir + 1) * N_fit_ * C,
                T_host.data(), N_fit_ * C * sizeof(float), cudaMemcpyHostToDevice));
        }
        N_fit_ *= 5;  // report total rows
    }

    fmt::print("deep_elm{}{}{}: precomputed H0 [{}, {}] and T [{}, {}] ({} hidden layers)\n",
               use_rff_ ? "/rff" : "", augment_ ? "/aug5" : "",
               use_conv_ ? "/conv" : "",
               N_fit_, d0, N_fit_, C, n_hidden_);
}

// ---------------------------------------------------------------------------
// recompute_h0_conv: refresh H0_dev_ after the conv filters have been updated.
// Iterates the training data once, runs conv_front_->forward(), and writes
// the FP32 output directly into H0_dev_ (device-to-device copy per batch).
// T_dev_ is unchanged (labels are fixed).
// ---------------------------------------------------------------------------
void DeepELMExperiment::recompute_h0_conv() {
    data_->reset();
    const size_t bs        = cfg_.batch_size;
    const size_t n_batches = data_->size() / bs;
    const size_t d0        = static_cast<size_t>(d0_);

    for (size_t b = 0; b < n_batches; ++b) {
        Batch  batch = data_->next_batch(bs);
        Tensor h     = conv_front_->forward(batch.inputs);   // FP32 [bs, d0]
        FAYN_CUDA_CHECK(cudaMemcpy(
            static_cast<float*>(H0_dev_.data) + b * bs * d0,
            h.data, bs * d0 * sizeof(float),
            cudaMemcpyDeviceToDevice));
    }
    FAYN_CUDA_CHECK(cudaDeviceSynchronize());
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
        if (use_tanh_)
            H.push_back(solver_.tanh_forward(prev, hidden_[static_cast<size_t>(k)]->weights_fp32()));
        else
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
//
// Conv path (use_conv_=true): runs fully in FP32 via solver_ methods.
//   Logits are downloaded to CPU; argmax computed there.
// Non-conv path: uses BF16 DenseLayer inference + accuracy() helper.
// ---------------------------------------------------------------------------
float DeepELMExperiment::evaluate(DataSource& ds) {
    ds.reset();
    const size_t n_batches = ds.batches_per_epoch(cfg_.batch_size);

    if (use_conv_) {
        // Full FP32 inference path for conv front-end.
        constexpr size_t C = 10;
        size_t n_correct = 0, n_total = 0;
        std::vector<float>   logits_host;
        std::vector<int32_t> tgt_host;

        for (size_t b = 0; b < n_batches; ++b) {
            Batch batch = ds.next_batch(cfg_.batch_size);
            const size_t bs = batch.inputs.shape[0];

            // Conv → FP32 [bs, d0].
            Tensor h = conv_front_->forward(batch.inputs);

            // Hidden layers (FP32 GEMM + relu_forward).
            for (auto& layer : hidden_)
                h = solver_.relu_forward(h, layer->weights_fp32());

            // Readout (FP32 GEMM, no activation).
            Tensor logits = solver_.linear_forward(h, readout_->weights_fp32());

            // Download logits and targets.
            logits_host.resize(bs * C);
            tgt_host.resize(bs);
            FAYN_CUDA_CHECK(cudaMemcpy(logits_host.data(), logits.data,
                                       bs * C * sizeof(float), cudaMemcpyDeviceToHost));
            FAYN_CUDA_CHECK(cudaMemcpy(tgt_host.data(), batch.targets.data,
                                       bs * sizeof(int32_t), cudaMemcpyDeviceToHost));
            FAYN_CUDA_CHECK(cudaDeviceSynchronize());

            for (size_t i = 0; i < bs; ++i) {
                const float* row = logits_host.data() + i * C;
                int pred = 0;
                for (int c = 1; c < (int)C; ++c)
                    if (row[c] > row[pred]) pred = c;
                if (pred == tgt_host[i]) ++n_correct;
            }
            n_total += bs;
        }
        return n_total > 0 ? static_cast<float>(n_correct) / static_cast<float>(n_total) : 0.f;
    }

    // Non-conv path: BF16 DenseLayer inference.
    float total_acc = 0.f;
    for (size_t b = 0; b < n_batches; ++b) {
        Batch batch = ds.next_batch(cfg_.batch_size);

        // W0 activation matches precomputed H0: ReLU normally, cos for RFF.
        Tensor h = w0_->forward(batch.inputs);
        if (use_rff_)
            apply_cos(h, /*stream=*/nullptr);
        else
            apply_relu(h, /*stream=*/nullptr);

        for (auto& layer : hidden_) {
            h = layer->forward(h);
            if (use_tanh_)
                apply_tanh(h, /*stream=*/nullptr);
            else
                apply_relu(h, /*stream=*/nullptr);
        }

        Tensor logits = readout_->forward(h);
        total_acc += fayn::accuracy(logits, batch.targets);
    }
    return n_batches > 0 ? total_acc / static_cast<float>(n_batches) : 0.f;
}

// ---------------------------------------------------------------------------
// evaluate_tta: test-time augmentation — average logits over 5 shifted views.
//
// For each test batch, runs the forward pass 5 times:
//   view 0: original pixels
//   views 1-4: 1-pixel shifts (left, right, up, down)
// Logits are accumulated as FP32 on CPU, averaged, then argmax is computed.
//
// This is separate from augment_ (training augmentation). TTA never changes
// the trained weights; it only averages predictions at inference time.
// ---------------------------------------------------------------------------
float DeepELMExperiment::evaluate_tta(DataSource& ds) {
    ds.reset();
    const size_t n_batches = ds.batches_per_epoch(cfg_.batch_size);
    size_t n_correct = 0, n_total = 0;
    constexpr size_t C = 10;

    for (size_t b = 0; b < n_batches; ++b) {
        Batch batch = ds.next_batch(cfg_.batch_size);
        const size_t bs = batch.inputs.shape[0];

        // Download original inputs to CPU (BF16 [bs, 784]).
        std::vector<__nv_bfloat16> inp_host(bs * 784);
        FAYN_CUDA_CHECK(cudaMemcpy(inp_host.data(), batch.inputs.data,
                                   bs * 784 * sizeof(__nv_bfloat16),
                                   cudaMemcpyDeviceToHost));
        FAYN_CUDA_CHECK(cudaDeviceSynchronize());

        // Accumulate FP32 logits over 5 views.
        std::vector<float> logit_sum(bs * C, 0.f);

        auto add_view = [&](const __nv_bfloat16* pixels) {
            Tensor inp_dev = Tensor::make({bs, 784}, DType::BFloat16, Device::CUDA);
            FAYN_CUDA_CHECK(cudaMemcpy(inp_dev.data, pixels,
                                       bs * 784 * sizeof(__nv_bfloat16),
                                       cudaMemcpyHostToDevice));
            Tensor h = w0_->forward(inp_dev);
            if (use_rff_) apply_cos(h, nullptr); else apply_relu(h, nullptr);
            for (auto& layer : hidden_) {
                h = layer->forward(h);
                if (use_tanh_) apply_tanh(h, nullptr); else apply_relu(h, nullptr);
            }
            Tensor logits = readout_->forward(h);
            std::vector<__nv_bfloat16> logits_host(bs * C);
            FAYN_CUDA_CHECK(cudaMemcpy(logits_host.data(), logits.data,
                                       bs * C * sizeof(__nv_bfloat16),
                                       cudaMemcpyDeviceToHost));
            FAYN_CUDA_CHECK(cudaDeviceSynchronize());
            for (size_t i = 0; i < bs * C; ++i)
                logit_sum[i] += __bfloat162float(logits_host[i]);
        };

        add_view(inp_host.data());

        std::vector<__nv_bfloat16> shifted(bs * 784);
        for (int dir = 0; dir < 4; ++dir) {
            shift_mnist_bf16(inp_host.data(), shifted.data(), bs, dir);
            add_view(shifted.data());
        }

        // Download targets.
        std::vector<int32_t> tgt_host(bs);
        FAYN_CUDA_CHECK(cudaMemcpy(tgt_host.data(), batch.targets.data,
                                   bs * sizeof(int32_t), cudaMemcpyDeviceToHost));
        FAYN_CUDA_CHECK(cudaDeviceSynchronize());

        // Argmax on averaged logits.
        for (size_t i = 0; i < bs; ++i) {
            const float* row = logit_sum.data() + i * C;
            int pred = 0;
            for (int c = 1; c < (int)C; ++c)
                if (row[c] > row[pred]) pred = c;
            if (pred == tgt_host[i]) ++n_correct;
        }
        n_total += bs;
    }

    return n_total > 0 ? static_cast<float>(n_correct) / static_cast<float>(n_total) : 0.f;
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

            // Solve readout: use H0_dev_ directly when n_hidden=0 (no ELM hidden layers).
            const Tensor& top_H = H.empty() ? H0_dev_ : H.back();
            Tensor W_r = solver_.solve(top_H, T_dev_, lambda_);
            write_fp32_weights(*readout_, W_r);

            fmt::print("  cycle {:2d}  train={:.4f}\n", cycle, evaluate(*data_));

            // Back-propagate targets from readout down to hidden_[0].
            const Tensor*           T_curr = &T_dev_;
            const Tensor*           W_curr = &readout_->weights_fp32();
            std::unique_ptr<Tensor> T_owned;

            for (int k = n_hidden_ - 1; k >= 0; --k) {
                T_owned = std::make_unique<Tensor>(
                    solver_.propagate_target(*W_curr, *T_curr));
                // ReLU: clip propagated targets to valid range [0, inf).
                // tanh: no clipping — tanh can target any value in (-1,1) and
                //       the solve uses raw propagated targets as regression targets.
                if (!use_tanh_)
                    apply_relu(*T_owned, /*stream=*/nullptr);
                T_curr = T_owned.get();

                const Tensor& H_pre =
                    (k == 0) ? H0_dev_ : H[static_cast<size_t>(k - 1)];
                Tensor W_k = solver_.solve(H_pre, *T_curr, lambda_);
                write_fp32_weights(*hidden_[static_cast<size_t>(k)], W_k);
                W_curr = &hidden_[static_cast<size_t>(k)]->weights_fp32();
            }
        }

        // Learned conv: update conv filters ONCE after all ELM cycles have
        // converged. Changing W_conv updates H0_dev_, which invalidates the
        // existing hidden weights. A re-adaptation pass (one full ELM cycle)
        // then restores accuracy with the improved conv features.
        if (use_conv_ && learn_conv_) {
            // Propagate the final target back through hidden_[0] to get
            // T_0 [N_fit_, d0_] = target for H0_dev_ (conv output).
            // Uses GPU Gram path for non-square W_0 [d_, d0_].
            const Tensor& W_0_fp32 = hidden_[0]->weights_fp32();

            // Re-compute the final propagated target T_1 [N_fit_, d_].
            auto H_pre_conv = compute_hidden_activations();
            const Tensor& top_H_pre = H_pre_conv.empty() ? H0_dev_ : H_pre_conv.back();
            Tensor W_r_tmp = solver_.solve(top_H_pre, T_dev_, lambda_);
            write_fp32_weights(*readout_, W_r_tmp);

            const Tensor* T_c  = &T_dev_;
            const Tensor* W_c  = &readout_->weights_fp32();
            std::unique_ptr<Tensor> T_prop_owned;
            for (int k = n_hidden_ - 1; k >= 0; --k) {
                T_prop_owned = std::make_unique<Tensor>(
                    solver_.propagate_target(*W_c, *T_c));
                if (!use_tanh_) apply_relu(*T_prop_owned, nullptr);
                T_c = T_prop_owned.get();
                W_c = &hidden_[static_cast<size_t>(k)]->weights_fp32();
            }

            // T_c now = T_1 [N_fit_, d_] — propagate through W_0 to H0 space.
            Tensor T_0_full = solver_.propagate_target(W_0_fp32, *T_c);
            if (!use_tanh_) apply_relu(T_0_full, nullptr);

            // Accumulate the im2col Gram and cross-correlation over training data.
            conv_front_->reset_gram();
            data_->reset();
            const size_t bs            = cfg_.batch_size;
            const size_t n_fit_batches = N_fit_ / bs;
            const size_t d0            = static_cast<size_t>(d0_);
            const float* T0_ptr        = static_cast<const float*>(T_0_full.data);
            for (size_t bi = 0; bi < n_fit_batches; ++bi) {
                Batch  batch = data_->next_batch(bs);
                Tensor col   = conv_front_->compute_im2col(batch.inputs);
                Tensor T0_b  = Tensor::make({bs, d0}, DType::Float32, Device::CUDA);
                FAYN_CUDA_CHECK(cudaMemcpy(
                    T0_b.data,
                    T0_ptr + bi * bs * d0,
                    bs * d0 * sizeof(float),
                    cudaMemcpyDeviceToDevice));
                Tensor T0_up = conv_front_->upsample_pool_target(T0_b);
                conv_front_->accumulate_gram(col, T0_up);
            }
            conv_front_->solve_gram(lambda_);
            recompute_h0_conv();
            fmt::print("  [conv update]  train={:.4f}\n", evaluate(*data_));

            // Re-adaptation: run n_cycles of ELM to re-fit hidden + readout to
            // the new H0_dev_ produced by the updated conv filters.
            for (int cycle = 0; cycle < n_cycles_; ++cycle) {
                auto H_re = compute_hidden_activations();
                const Tensor& top_H_re = H_re.empty() ? H0_dev_ : H_re.back();
                Tensor W_r_re = solver_.solve(top_H_re, T_dev_, lambda_);
                write_fp32_weights(*readout_, W_r_re);

                fmt::print("  readapt {:2d}  train={:.4f}\n", cycle, evaluate(*data_));

                const Tensor* T_re  = &T_dev_;
                const Tensor* W_re  = &readout_->weights_fp32();
                std::unique_ptr<Tensor> T_re_owned;
                for (int k = n_hidden_ - 1; k >= 0; --k) {
                    T_re_owned = std::make_unique<Tensor>(
                        solver_.propagate_target(*W_re, *T_re));
                    if (!use_tanh_) apply_relu(*T_re_owned, nullptr);
                    T_re = T_re_owned.get();
                    const Tensor& H_prev =
                        (k == 0) ? H0_dev_ : H_re[static_cast<size_t>(k - 1)];
                    Tensor W_k = solver_.solve(H_prev, *T_re, lambda_);
                    write_fp32_weights(*hidden_[static_cast<size_t>(k)], W_k);
                    W_re = &hidden_[static_cast<size_t>(k)]->weights_fp32();
                }
            }
        }

        // Final readout re-solve consistent with last hidden update.
        auto H_final = compute_hidden_activations();
        const Tensor& top_H_final = H_final.empty() ? H0_dev_ : H_final.back();
        Tensor W_r_final = solver_.solve(top_H_final, T_dev_, lambda_);
        write_fp32_weights(*readout_, W_r_final);
    }

    const float train_acc = evaluate(*data_);
    const float test_acc  = use_tta_ ? evaluate_tta(*test_data_) : evaluate(*test_data_);
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
    float leaky_alpha,
    bool  use_tanh)
    : Experiment(cfg)
    , data_path_(std::move(data_path))
    , d0_(d0), d_(d), n_hidden_(n_hidden), n_admm_(n_admm)
    , lambda_(lambda), rho_(rho), mu_(mu), leaky_alpha_(leaky_alpha)
    , use_tanh_(use_tanh)
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
            if (use_tanh_)
                apply_tanh(H_init.back(), /*stream=*/nullptr);
            else
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
        if (use_tanh_)
            apply_tanh(h, /*stream=*/nullptr);
        else
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
        // Trained hidden layers: LeakyReLU or tanh depending on use_tanh_.
        for (auto& layer : hidden_) {
            h = layer->forward(h);
            if (use_tanh_)
                apply_tanh(h, /*stream=*/nullptr);
            else
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
            // ---- Step 1: H_k = sigma(Z_k)  [LeakyReLU or tanh] ----
            std::vector<Tensor> H(static_cast<size_t>(n_hidden_));
            for (int k = 0; k < n_hidden_; ++k) {
                const size_t ki = static_cast<size_t>(k);
                H[ki] = Tensor::make({N_fit_, d_sz}, DType::Float32, Device::CUDA);
                FAYN_CUDA_CHECK(cudaMemcpy(H[ki].data, Z_[ki].data,
                    N_fit_ * d_sz * sizeof(float), cudaMemcpyDeviceToDevice));
                if (use_tanh_)
                    apply_tanh(H[ki], /*stream=*/nullptr);
                else
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
            // Readout W [10, d] is non-square → CPU Gram path.
            // Hidden W [d, d]: Gram + Cholesky with lambda_prop scaled to W norm.
            // Both direct-LU and Gram+Cholesky diverge at d=1024 depth-4 due to
            // target amplitude cascade (ADMM analog of exploding gradients):
            // each propagation step amplifies targets → tanh saturation → near-rank-1 H
            // → large-norm W → further amplification. This is a known limitation of
            // deep ADMM/target-propagation for tanh activations beyond d=512.
            const float lambda_prop = 1e-4f;
            std::vector<Tensor> T_targets(static_cast<size_t>(n_hidden_));
            {
                Tensor T_curr = solver_.propagate_target(
                    readout_->weights_fp32(), T_dev_, lambda_prop);
                for (int k = n_hidden_ - 1; k >= 0; --k) {
                    const size_t ki = static_cast<size_t>(k);
                    T_targets[ki] = std::move(T_curr);
                    if (k > 0)
                        T_curr = solver_.propagate_target(
                            hidden_[ki]->weights_fp32(), T_targets[ki], lambda_prop);
                }
            }

            // ---- Step 6: Z-update ----
            // LeakyReLU: element-wise case-split minimizer (exact for piecewise linear sigma).
            // tanh: linear blend in pre-activation space via atanh (exact for linear sigma approx).
            // u_zero_ passed for LeakyReLU path (proximal ALS: no dual accumulation).
            for (int k = 0; k < n_hidden_; ++k) {
                const size_t ki = static_cast<size_t>(k);
                if (use_tanh_)
                    admm_z_update_tanh(Z_[ki], A[ki], T_targets[ki], rho_, mu_);
                else
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

// ===========================================================================
// DeepELMEnsembleExperiment implementation
// ===========================================================================

DeepELMEnsembleExperiment::DeepELMEnsembleExperiment(
    const ExperimentConfig& cfg,
    std::string data_path,
    int              n_members,
    int              d0,
    int              d,
    int              n_hidden,
    int              n_cycles,
    float            lambda,
    bool             use_conv,
    int              conv_c_out,
    std::vector<int> conv_k_per_member,
    bool             use_aug)
    : Experiment(cfg)
    , data_path_(std::move(data_path))
    , n_members_(n_members)
    , d0_(d0)
    , d_(d)
    , n_hidden_(n_hidden)
    , n_cycles_(n_cycles)
    , lambda_(lambda)
    , use_conv_(use_conv)
    , conv_c_out_(conv_c_out)
    , conv_k_per_member_(std::move(conv_k_per_member))
    , use_aug_(use_aug)
{
    if (!conv_k_per_member_.empty() &&
        static_cast<int>(conv_k_per_member_.size()) != n_members_)
        throw std::invalid_argument(
            "DeepELMEnsembleExperiment: conv_k_per_member size must equal n_members");
    if (use_aug_ && !use_conv_)
        throw std::invalid_argument(
            "DeepELMEnsembleExperiment: use_aug requires use_conv=true");
}

void DeepELMEnsembleExperiment::setup() {
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

    // Build T_dev_ (one-hot labels, FP32) with one pass over training data.
    const size_t fit_bs    = cfg_.batch_size;
    const size_t n_batches = data_->size() / fit_bs;
    N_fit_ = n_batches * fit_bs;

    std::vector<float>   T_host(N_fit_ * 10, 0.f);
    std::vector<int32_t> lbl_buf(fit_bs);

    data_->reset();
    for (size_t b = 0; b < n_batches; ++b) {
        Batch batch = data_->next_batch(fit_bs);
        FAYN_CUDA_CHECK(cudaMemcpy(lbl_buf.data(), batch.targets.data,
                                   fit_bs * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < fit_bs; ++i)
            T_host[(b * fit_bs + i) * 10 + static_cast<size_t>(lbl_buf[i])] = 1.f;
    }
    T_dev_ = Tensor::make({N_fit_, 10}, DType::Float32, Device::CUDA);
    FAYN_CUDA_CHECK(cudaMemcpy(T_dev_.data, T_host.data(),
                               N_fit_ * 10 * sizeof(float),
                               cudaMemcpyHostToDevice));

    // Build T_aug_dev_ = T_dev_ repeated 5× (for feature-level augmentation).
    if (use_aug_) {
        T_aug_dev_ = Tensor::make({5 * N_fit_, 10}, DType::Float32, Device::CUDA);
        for (int v = 0; v < 5; ++v) {
            FAYN_CUDA_CHECK(cudaMemcpy(
                static_cast<float*>(T_aug_dev_.data) + static_cast<size_t>(v) * N_fit_ * 10,
                T_dev_.data, N_fit_ * 10 * sizeof(float),
                cudaMemcpyDeviceToDevice));
        }
        FAYN_CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Create M ensemble members. Each w0 / conv_front gets a unique Kaiming seed
    // via the global seed_counter auto-increment in DenseLayer / ConvFrontend.
    members_.resize(static_cast<size_t>(n_members_));
    for (int m = 0; m < n_members_; ++m) {
        Member& mem = members_[static_cast<size_t>(m)];

        if (use_conv_) {
            const int k_m = conv_k_per_member_.empty()
                            ? 5 : conv_k_per_member_[static_cast<size_t>(m)];
            mem.conv_front = std::make_unique<ConvFrontend>(
                conv_c_out_, /*max_pool=*/true, k_m);
            mem.d0 = mem.conv_front->output_features();
        } else {
            mem.d0 = d0_;
            mem.w0 = std::make_shared<DenseLayer>(784, static_cast<size_t>(mem.d0));
            mem.w0->set_compute_stats(false);
        }

        mem.hidden.resize(static_cast<size_t>(n_hidden_));
        for (int k = 0; k < n_hidden_; ++k) {
            const size_t in_dim = (k == 0) ? static_cast<size_t>(mem.d0)
                                           : static_cast<size_t>(d_);
            mem.hidden[static_cast<size_t>(k)] = std::make_shared<DenseLayer>(
                in_dim, static_cast<size_t>(d_), /*use_bias=*/false);
            mem.hidden[static_cast<size_t>(k)]->set_compute_stats(false);
            mem.hidden[static_cast<size_t>(k)]->enable_fp32_weights();
        }

        mem.readout = std::make_shared<DenseLayer>(
            static_cast<size_t>(d_), 10, /*use_bias=*/false);
        mem.readout->set_compute_stats(false);
        mem.readout->enable_fp32_weights();
    }

    const bool multi_scale = use_conv_ && !conv_k_per_member_.empty();
    fmt::print("deep_elm_ensemble{}: {} members, d={}, n_hidden={}, N_fit={}\n",
               multi_scale ? "/conv-multiscale" : (use_conv_ ? "/conv" : ""),
               n_members_, d_, n_hidden_, N_fit_);
}

// ---------------------------------------------------------------------------
// compute_member_h0: one pass through training data to build H0 [N_fit, d0]
// (FP32 on device) for a single member's frozen W_0 (or ConvFrontend).
// ---------------------------------------------------------------------------
Tensor DeepELMEnsembleExperiment::compute_member_h0(const Member& m) {
    const size_t fit_bs    = cfg_.batch_size;
    const size_t n_batches = data_->size() / fit_bs;
    const size_t d0        = static_cast<size_t>(m.d0);

    if (use_aug_) {
        // Augmented conv path: 5 views — original + 4 pixel shifts.
        // H0_dev [5*N_fit, d0]: slots 0..N_fit-1 = original, N_fit..5*N_fit-1 = shifts.
        Tensor H0_dev = Tensor::make({5 * N_fit_, d0}, DType::Float32, Device::CUDA);
        std::vector<__nv_bfloat16> bf16_buf(fit_bs * 784);
        std::vector<__nv_bfloat16> shifted_buf(fit_bs * 784);
        Tensor x_shifted = Tensor::make({fit_bs, 784}, DType::BFloat16, Device::CUDA);

        data_->reset();
        for (size_t b = 0; b < n_batches; ++b) {
            Batch batch = data_->next_batch(fit_bs);

            // Original view → slot 0.
            {
                Tensor h = m.conv_front->forward(batch.inputs);
                FAYN_CUDA_CHECK(cudaMemcpy(
                    static_cast<float*>(H0_dev.data) + b * fit_bs * d0,
                    h.data, fit_bs * d0 * sizeof(float),
                    cudaMemcpyDeviceToDevice));
            }

            // 4 shifted views.
            FAYN_CUDA_CHECK(cudaStreamSynchronize(nullptr));
            FAYN_CUDA_CHECK(cudaMemcpy(bf16_buf.data(), batch.inputs.data,
                                       fit_bs * 784 * sizeof(__nv_bfloat16),
                                       cudaMemcpyDeviceToHost));
            for (int dir = 0; dir < 4; ++dir) {
                shift_mnist_bf16(bf16_buf.data(), shifted_buf.data(), fit_bs, dir);
                FAYN_CUDA_CHECK(cudaMemcpy(x_shifted.data, shifted_buf.data(),
                                           fit_bs * 784 * sizeof(__nv_bfloat16),
                                           cudaMemcpyHostToDevice));
                Tensor h = m.conv_front->forward(x_shifted);
                const size_t off = (static_cast<size_t>(dir + 1) * N_fit_ + b * fit_bs) * d0;
                FAYN_CUDA_CHECK(cudaMemcpy(
                    static_cast<float*>(H0_dev.data) + off,
                    h.data, fit_bs * d0 * sizeof(float),
                    cudaMemcpyDeviceToDevice));
            }
        }
        FAYN_CUDA_CHECK(cudaDeviceSynchronize());
        return H0_dev;
    }

    Tensor H0_dev = Tensor::make({N_fit_, d0}, DType::Float32, Device::CUDA);

    if (use_conv_) {
        // Conv path: forward() returns FP32 directly → device-to-device copy per batch.
        data_->reset();
        for (size_t b = 0; b < n_batches; ++b) {
            Batch  batch = data_->next_batch(fit_bs);
            Tensor h     = m.conv_front->forward(batch.inputs);  // FP32 [fit_bs, d0]
            FAYN_CUDA_CHECK(cudaMemcpy(
                static_cast<float*>(H0_dev.data) + b * fit_bs * d0,
                h.data, fit_bs * d0 * sizeof(float),
                cudaMemcpyDeviceToDevice));
        }
        FAYN_CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        // Non-conv path: BF16 from w0->forward, convert to FP32 via CPU.
        std::vector<float>         H0_host(N_fit_ * d0);
        std::vector<__nv_bfloat16> bf16_buf(fit_bs * d0);

        data_->reset();
        for (size_t b = 0; b < n_batches; ++b) {
            Batch  batch = data_->next_batch(fit_bs);
            Tensor h     = m.w0->forward(batch.inputs);
            apply_relu(h, /*stream=*/nullptr);
            FAYN_CUDA_CHECK(cudaStreamSynchronize(nullptr));
            FAYN_CUDA_CHECK(cudaMemcpy(bf16_buf.data(), h.data,
                                       fit_bs * d0 * sizeof(__nv_bfloat16),
                                       cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < fit_bs * d0; ++i)
                H0_host[b * fit_bs * d0 + i] = __bfloat162float(bf16_buf[i]);
        }
        FAYN_CUDA_CHECK(cudaMemcpy(H0_dev.data, H0_host.data(),
                                   N_fit_ * d0 * sizeof(float),
                                   cudaMemcpyHostToDevice));
    }
    return H0_dev;
}

// ---------------------------------------------------------------------------
// compute_member_hidden: forward pass through hidden layers starting from H0.
// ---------------------------------------------------------------------------
std::vector<Tensor> DeepELMEnsembleExperiment::compute_member_hidden(
    const Tensor& H0, const Member& m) const
{
    std::vector<Tensor> H;
    H.reserve(m.hidden.size());
    for (const auto& layer : m.hidden) {
        const Tensor& prev = H.empty() ? H0 : H.back();
        H.push_back(solver_.relu_forward(prev, layer->weights_fp32()));
    }
    return H;
}

// ---------------------------------------------------------------------------
// run_member_cycles: n_cycles of alternating ELM solve for one member,
// using the pre-computed H0 for that member's W_0.
// ---------------------------------------------------------------------------
void DeepELMEnsembleExperiment::run_member_cycles(Member& m, const Tensor& H0,
                                                    const Tensor& T) {
    for (int cycle = 0; cycle < n_cycles_; ++cycle) {
        auto H = compute_member_hidden(H0, m);
        const Tensor& top_H = H.empty() ? H0 : H.back();
        Tensor W_r = solver_.solve(top_H, T, lambda_);
        write_fp32_weights(*m.readout, W_r);

        const Tensor*           T_curr = &T;
        const Tensor*           W_curr = &m.readout->weights_fp32();
        std::unique_ptr<Tensor> T_owned;

        for (int k = static_cast<int>(m.hidden.size()) - 1; k >= 0; --k) {
            T_owned = std::make_unique<Tensor>(
                solver_.propagate_target(*W_curr, *T_curr));
            apply_relu(*T_owned, /*stream=*/nullptr);
            T_curr = T_owned.get();
            const Tensor& H_pre = (k == 0) ? H0 : H[static_cast<size_t>(k - 1)];
            Tensor W_k = solver_.solve(H_pre, *T_curr, lambda_);
            write_fp32_weights(*m.hidden[static_cast<size_t>(k)], W_k);
            W_curr = &m.hidden[static_cast<size_t>(k)]->weights_fp32();
        }
    }

    // Final readout re-solve consistent with the last hidden update.
    auto H_final = compute_member_hidden(H0, m);
    const Tensor& top_H_final = H_final.empty() ? H0 : H_final.back();
    Tensor W_r_final = solver_.solve(top_H_final, T, lambda_);
    write_fp32_weights(*m.readout, W_r_final);
}

// ---------------------------------------------------------------------------
// write_fp32_weights: copy W_fp32 device → layer.weights_fp32().
// ---------------------------------------------------------------------------
void DeepELMEnsembleExperiment::write_fp32_weights(
    DenseLayer& layer, const Tensor& W_fp32)
{
    FAYN_CUDA_CHECK(cudaMemcpy(layer.weights_fp32().data, W_fp32.data,
                               W_fp32.numel() * sizeof(float),
                               cudaMemcpyDeviceToDevice));
    FAYN_CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// evaluate_member: accuracy for a single member.
// Conv path: FP32 inference (conv_front + solver_.relu_forward).
// Non-conv path: BF16 inference (DenseLayer::forward).
// ---------------------------------------------------------------------------
float DeepELMEnsembleExperiment::evaluate_member(const Member& m, DataSource& ds) {
    ds.reset();
    const size_t n_batches = ds.batches_per_epoch(cfg_.batch_size);

    if (use_conv_) {
        constexpr size_t C = 10;
        size_t n_correct = 0, n_total = 0;
        std::vector<float>   logits_host;
        std::vector<int32_t> tgt_host;

        for (size_t b = 0; b < n_batches; ++b) {
            Batch batch = ds.next_batch(cfg_.batch_size);
            const size_t bs = batch.inputs.shape[0];
            Tensor h = m.conv_front->forward(batch.inputs);  // FP32 [bs, d0]
            for (const auto& layer : m.hidden)
                h = solver_.relu_forward(h, layer->weights_fp32());
            Tensor logits = solver_.linear_forward(h, m.readout->weights_fp32());

            logits_host.resize(bs * C);
            tgt_host.resize(bs);
            FAYN_CUDA_CHECK(cudaMemcpy(logits_host.data(), logits.data,
                                       bs * C * sizeof(float), cudaMemcpyDeviceToHost));
            FAYN_CUDA_CHECK(cudaMemcpy(tgt_host.data(), batch.targets.data,
                                       bs * sizeof(int32_t), cudaMemcpyDeviceToHost));
            FAYN_CUDA_CHECK(cudaDeviceSynchronize());
            for (size_t i = 0; i < bs; ++i) {
                const float* row = logits_host.data() + i * C;
                int pred = 0;
                for (int c = 1; c < (int)C; ++c)
                    if (row[c] > row[pred]) pred = c;
                if (pred == static_cast<int>(tgt_host[i])) ++n_correct;
            }
            n_total += bs;
        }
        return n_total > 0 ? static_cast<float>(n_correct) / static_cast<float>(n_total) : 0.f;
    }

    // Non-conv path: BF16 DenseLayer inference.
    float total_acc = 0.f;
    for (size_t b = 0; b < n_batches; ++b) {
        Batch  batch = ds.next_batch(cfg_.batch_size);
        Tensor h     = m.w0->forward(batch.inputs);
        apply_relu(h, /*stream=*/nullptr);
        for (const auto& layer : m.hidden) {
            h = layer->forward(h);
            apply_relu(h, /*stream=*/nullptr);
        }
        Tensor logits = m.readout->forward(h);
        total_acc += fayn::accuracy(logits, batch.targets);
    }
    return n_batches > 0 ? total_acc / static_cast<float>(n_batches) : 0.f;
}

// ---------------------------------------------------------------------------
// evaluate: ensemble accuracy — average logits from all M members.
// Conv path: FP32 logit accumulation directly.
// Non-conv path: BF16→FP32 conversion per member then accumulate.
// ---------------------------------------------------------------------------
float DeepELMEnsembleExperiment::evaluate(DataSource& ds) {
    ds.reset();
    const size_t n_batches = ds.batches_per_epoch(cfg_.batch_size);
    constexpr size_t C = 10;
    size_t n_correct = 0, n_total = 0;

    for (size_t b = 0; b < n_batches; ++b) {
        Batch        batch = ds.next_batch(cfg_.batch_size);
        const size_t bs   = batch.inputs.shape[0];

        std::vector<float> logit_sum(bs * C, 0.f);
        std::vector<float> logits_fp32(bs * C);

        if (use_conv_) {
            for (const auto& m : members_) {
                Tensor h = m.conv_front->forward(batch.inputs);  // FP32 [bs, d0]
                for (const auto& layer : m.hidden)
                    h = solver_.relu_forward(h, layer->weights_fp32());
                Tensor logits = solver_.linear_forward(h, m.readout->weights_fp32());
                FAYN_CUDA_CHECK(cudaMemcpy(logits_fp32.data(), logits.data,
                                           bs * C * sizeof(float), cudaMemcpyDeviceToHost));
                FAYN_CUDA_CHECK(cudaDeviceSynchronize());
                for (size_t i = 0; i < bs * C; ++i)
                    logit_sum[i] += logits_fp32[i];
            }
        } else {
            std::vector<__nv_bfloat16> logits_bf16(bs * C);
            for (const auto& m : members_) {
                Tensor h = m.w0->forward(batch.inputs);
                apply_relu(h, /*stream=*/nullptr);
                for (const auto& layer : m.hidden) {
                    h = layer->forward(h);
                    apply_relu(h, /*stream=*/nullptr);
                }
                Tensor logits = m.readout->forward(h);
                FAYN_CUDA_CHECK(cudaMemcpy(logits_bf16.data(), logits.data,
                                           bs * C * sizeof(__nv_bfloat16),
                                           cudaMemcpyDeviceToHost));
                FAYN_CUDA_CHECK(cudaDeviceSynchronize());
                for (size_t i = 0; i < bs * C; ++i)
                    logit_sum[i] += __bfloat162float(logits_bf16[i]);
            }
        }

        std::vector<int32_t> tgt_host(bs);
        FAYN_CUDA_CHECK(cudaMemcpy(tgt_host.data(), batch.targets.data,
                                   bs * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost));
        FAYN_CUDA_CHECK(cudaDeviceSynchronize());

        for (size_t i = 0; i < bs; ++i) {
            const float* row = logit_sum.data() + i * C;
            int pred = 0;
            for (int c = 1; c < (int)C; ++c)
                if (row[c] > row[pred]) pred = c;
            if (pred == static_cast<int>(tgt_host[i])) ++n_correct;
        }
        n_total += bs;
    }
    return n_total > 0 ? static_cast<float>(n_correct) / static_cast<float>(n_total) : 0.f;
}

// ---------------------------------------------------------------------------
// run_epoch: on epoch 0, train each member sequentially (each gets its own
// H0 computed fresh, then n_cycles of ELM solve). H0 is freed after each
// member to avoid holding M×[N_fit,d0] simultaneously.
// Subsequent epochs just re-evaluate (all weights fixed after epoch 0).
// ---------------------------------------------------------------------------
float DeepELMEnsembleExperiment::run_epoch(size_t epoch) {
    if (epoch == 0) {
        const Tensor& T_train = use_aug_ ? T_aug_dev_ : T_dev_;
        for (int m_idx = 0; m_idx < n_members_; ++m_idx) {
            const int member_d0 = members_[static_cast<size_t>(m_idx)].d0;
            const size_t h0_rows = use_aug_ ? 5 * N_fit_ : N_fit_;
            fmt::print("  [member {:2d}] computing H0 [{}, {}]...\n",
                       m_idx, h0_rows, member_d0);
            Tensor H0 = compute_member_h0(members_[static_cast<size_t>(m_idx)]);
            run_member_cycles(members_[static_cast<size_t>(m_idx)], H0, T_train);
            const float m_test = evaluate_member(
                members_[static_cast<size_t>(m_idx)], *test_data_);
            fmt::print("  [member {:2d}] test={:.4f}\n", m_idx, m_test);
            // H0 goes out of scope here → freed (cudaFree via Tensor RAII).
        }
    }

    const float train_acc = evaluate(*data_);
    const float test_acc  = evaluate(*test_data_);
    fmt::print("epoch {:3d}  train={:.4f}  test={:.4f}\n", epoch, train_acc, test_acc);
    return test_acc;
}

} // namespace fayn
