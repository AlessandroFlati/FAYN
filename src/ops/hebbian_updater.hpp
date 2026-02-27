#pragma once

#include "dense.hpp"
#include "hebbian.hpp"
#include "../stats/event_bus.hpp"
#include "../stats/events.hpp"
#include "../cuda/stream_pool.hpp"

#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

namespace fayn {

// ---------------------------------------------------------------------------
// HebbianUpdater
//
// EventBus subscriber that applies reward-modulated Hebbian weight updates
// whenever a RewardEvent is emitted. The training loop never calls
// hebbian_update() directly — it only calls graph.forward() and emits a
// RewardEvent. This class handles the rest.
//
// Routing modes:
//   Local            — pure Hebbian: ΔW ∝ pre × post (reward ignored, lr always applied)
//   Global           — reward-modulated: ΔW ∝ reward × pre × post
//   SupervisedHebbian — uses target_activations (one-hot) as post instead of
//                       last_output; reward scalar ignored, lr always applied.
//                       Pulls each class weight row toward hidden reps of that class.
//   DeltaRule        — uses (target − output) as post (gradient of MSE loss).
//                       Self-stabilizing: updates → 0 as ŷ → T. Converges
//                       iteratively to the ELM solution (H^T H)^{-1} H^T T.
//
// Usage:
//   auto updater = std::make_unique<HebbianUpdater>({{
//       .layer = dense_layer_ptr,
//       .lr    = 0.01f,
//       .mode  = HebbianUpdater::RoutingMode::Global,
//   }});
//   // updater auto-unsubscribes in destructor
// ---------------------------------------------------------------------------
class HebbianUpdater {
public:
    enum class RoutingMode { Local, Global, SupervisedHebbian, DeltaRule };

    struct LayerConfig {
        std::shared_ptr<DenseLayer>   layer;
        float                         lr              = 0.01f;
        RoutingMode                   mode            = RoutingMode::Global;
        bool                          normalize       = true;
        int                           normalize_every = 1;
        // L2-normalise each pre-synaptic feature vector before the outer
        // product. Makes H^T H ≈ (N/d)·I, so the Hebbian direction better
        // approximates the ELM solution (H^T H)^{-1} H^T T.
        bool                          normalize_pre   = false;
        // Optional per-step LR schedule. If set, called with ev.step and its
        // return value replaces cfg.lr. Useful for cosine/linear annealing.
        std::function<float(size_t)>  lr_schedule;
        // Pre-step weight decay: W ← W*(1-decay) before each Hebbian update.
        // Soft alternative to normalize (hard sphere projection). At steady
        // state W* ∝ H^T T (same direction as row-norm), but allows non-unit
        // magnitudes and smoother dynamics. Use normalize=false with this.
        float                         weight_decay    = 0.f;
    };

    explicit HebbianUpdater(std::vector<LayerConfig> layers)
        : layers_(std::move(layers))
        , step_counters_(layers_.size(), 0)
    {
        for (const auto& cfg : layers_)
            if (!cfg.layer) throw std::invalid_argument("HebbianUpdater: null layer in config");

        sub_id_ = EventBus::instance().subscribe<RewardEvent>(
            [this](const RewardEvent& ev){ on_reward(ev); },
            DispatchMode::Sync);
    }

    ~HebbianUpdater() {
        EventBus::instance().unsubscribe(sub_id_);
    }

    HebbianUpdater(const HebbianUpdater&)            = delete;
    HebbianUpdater& operator=(const HebbianUpdater&) = delete;

private:
    void on_reward(const RewardEvent& ev) {
        // Ensure all GPU work from the forward pass (GEMM + async activation
        // cache copies) has completed before reading last_input_/last_output_.
        // DenseLayer::forward() uses non-blocking streams which do not
        // synchronise with any other stream automatically.
        FAYN_CUDA_CHECK(cudaDeviceSynchronize());

        StreamPool::Guard guard;
        cudaStream_t stream = guard.stream();

        for (size_t i = 0; i < layers_.size(); ++i) {
            auto& cfg = layers_[i];
            if (!cfg.layer->cache_activations()) continue;

            // Skip if no forward pass has been done yet.
            if (cfg.layer->last_input().data == nullptr) continue;

            // LR: use schedule if provided, else constant cfg.lr.
            float effective_lr = cfg.lr_schedule ? cfg.lr_schedule(ev.step) : cfg.lr;
            if (cfg.mode == RoutingMode::Global)
                effective_lr *= ev.reward;
            // Local and SupervisedHebbian always use base lr unchanged.

            if (effective_lr == 0.f) continue;

            // Select post-synaptic signal based on routing mode.
            // DeltaRule: post = target − output (gradient of MSE).
            //   Self-stabilizing (updates → 0 as ŷ → T) and converges to ELM solution.
            // SupervisedHebbian: post = one-hot target (normalized centroid learning).
            // Otherwise: post = last_output (unsupervised Hebbian).
            Tensor delta_post;
            const Tensor* post_ptr;
            if (cfg.mode == RoutingMode::SupervisedHebbian &&
                cfg.layer->has_target_activations()) {
                post_ptr = &cfg.layer->target_activations();
            } else if (cfg.mode == RoutingMode::DeltaRule &&
                       cfg.layer->has_target_activations()) {
                delta_post = tensor_subtract_bf16(
                    cfg.layer->target_activations(), cfg.layer->last_output(), stream);
                post_ptr = &delta_post;
            } else {
                post_ptr = &cfg.layer->last_output();
            }

            // Pre-synaptic activations: optionally L2-normalise each feature
            // vector (row) so that H^T H ≈ (N/d)·I, making the Hebbian
            // update direction a better proxy for the ELM solution.
            Tensor pre_normed;
            const Tensor* pre_ptr = &cfg.layer->last_input();
            if (cfg.normalize_pre) {
                const Tensor& src = cfg.layer->last_input();
                pre_normed = Tensor::make(src.shape, src.dtype, Device::CUDA);
                FAYN_CUDA_CHECK(cudaMemcpyAsync(
                    pre_normed.data, src.data, src.nbytes(),
                    cudaMemcpyDeviceToDevice, stream));
                normalize_weights_rows(pre_normed, 1e-8f, stream);
                pre_ptr = &pre_normed;
            }

            // Pre-step weight decay: W ← (1-decay)*W before Hebbian update.
            if (cfg.weight_decay > 0.f)
                weight_decay_weights(cfg.layer->weights(), cfg.weight_decay, stream);

            hebbian_update(cfg.layer->weights(),
                           *pre_ptr,
                           *post_ptr,
                           effective_lr, stream);

            ++step_counters_[i];
            if (cfg.normalize && (step_counters_[i] % cfg.normalize_every == 0))
                normalize_weights_rows(cfg.layer->weights(), 1e-8f, stream);
        }

        FAYN_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::vector<LayerConfig> layers_;
    std::vector<int>         step_counters_;
    SubID                    sub_id_;
};

} // namespace fayn
