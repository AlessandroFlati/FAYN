#pragma once

#include "dense.hpp"
#include "hebbian.hpp"
#include "../stats/event_bus.hpp"
#include "../stats/events.hpp"
#include "../cuda/stream_pool.hpp"

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
//   Local  — pure Hebbian: ΔW ∝ pre × post (reward scalar is ignored)
//   Global — reward-modulated: ΔW ∝ reward × pre × post
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
    enum class RoutingMode { Local, Global };

    struct LayerConfig {
        std::shared_ptr<DenseLayer> layer;
        float       lr              = 0.01f;
        RoutingMode mode            = RoutingMode::Global;
        bool        normalize       = true;
        int         normalize_every = 1;
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

            float effective_lr = cfg.lr;
            if (cfg.mode == RoutingMode::Global)
                effective_lr *= ev.reward;

            if (effective_lr == 0.f) continue;

            hebbian_update(cfg.layer->weights(),
                           cfg.layer->last_input(),
                           cfg.layer->last_output(),
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
