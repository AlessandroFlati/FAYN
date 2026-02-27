#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace fayn {

// ---------------------------------------------------------------------------
// StatsSnapshot: a small, copyable summary of per-layer activation statistics.
// All statistics are stored in float32 regardless of the tensor's dtype.
// ---------------------------------------------------------------------------
struct StatsSnapshot {
    int    layer_id   = -1;
    size_t step       = 0;       // global forward-pass counter

    // Per-neuron EMA mean of activations (size = num_outputs).
    std::vector<float> ema_mean;

    // Per-neuron EMA variance of activations.
    std::vector<float> ema_var;

    // Fraction of neurons considered "dead" (|activation| < dead_threshold).
    float dead_ratio  = 0.0f;

    // EMA of the mean absolute activation magnitude.
    float ema_magnitude = 0.0f;
};

// ---------------------------------------------------------------------------
// Event types emitted on the EventBus during a forward pass.
// All events are small, copyable value types.
// ---------------------------------------------------------------------------

// Emitted by a Layer after each forward() call.
struct ActivationEvent {
    int           layer_id = -1;
    size_t        step     = 0;
    StatsSnapshot stats;
};

// Emitted by the MutationEngine when it proposes a structural change.
struct MutationProposalEvent {
    int         layer_id  = -1;
    size_t      step      = 0;
    std::string op_type;          // "add_node", "remove_node", "add_edge", etc.
    std::string rationale;        // human-readable reason
};

// Emitted after a mutation has been applied to the graph.
struct MutationAppliedEvent {
    size_t      step      = 0;
    std::string op_type;
    bool        accepted  = false;
};

// Emitted at the start and end of each epoch by the Experiment runner.
struct EpochEvent {
    size_t epoch        = 0;
    bool   begin        = true;   // true = epoch start, false = epoch end
    float  metric_value = 0.0f;   // task metric at epoch end (if begin == false)
};

// Emitted when the task emits a scalar reward / fitness signal.
// Multiple named signals can be emitted in the same step; subscribers
// may filter by name if they only react to a specific signal.
struct RewardEvent {
    std::string name   = "reward";
    size_t      step   = 0;
    float       reward = 0.0f;
};

} // namespace fayn
