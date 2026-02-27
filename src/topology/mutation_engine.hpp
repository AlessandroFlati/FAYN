#pragma once

#include "mutation_ops.hpp"
#include "../core/graph.hpp"
#include "../stats/event_bus.hpp"
#include "../stats/events.hpp"
#include "../stats/ema.hpp"

#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace fayn {

// ---------------------------------------------------------------------------
// MutationThreshold: a rule that maps a scalar metric to a MutationOp.
//
// When metric_fn(stats) satisfies comparator(value, threshold),
// the factory is called to produce the mutation to apply.
// ---------------------------------------------------------------------------
struct MutationThreshold {
    // Extract a scalar from the stats snapshot.
    std::function<float(const StatsSnapshot&)>  metric_fn;

    // Comparison: return true if the mutation should fire.
    // e.g. [](float v, float t){ return v > t; }  for "if dead_ratio > 0.3"
    std::function<bool(float value, float threshold)> comparator;

    float threshold = 0.0f;

    // Produce the mutation to apply when the condition fires.
    std::function<MutationOp(const StatsSnapshot&, const Graph&)> factory;

    std::string name;   // human-readable label for logging
};

// ---------------------------------------------------------------------------
// MutationEngine: subscribes to the EventBus and applies structural
// mutations to registered graphs when threshold conditions are met.
//
// Each graph must be registered explicitly. The engine holds a non-owning
// pointer to each graph; the caller is responsible for graph lifetime.
//
// Rules are added via add_rule(). Each rule is checked on every
// ActivationEvent for the corresponding graph.
// ---------------------------------------------------------------------------
class MutationEngine {
public:
    MutationEngine();
    ~MutationEngine();

    // Register a graph to be managed. Returns a graph handle ID.
    int register_graph(Graph* graph);

    // Deregister a graph (e.g. before destroying it).
    void deregister_graph(int handle_id);

    // Add a mutation rule. Rules are checked in insertion order.
    void add_rule(MutationThreshold rule);

    // Remove all rules.
    void clear_rules();

    // Disable/enable automatic mutation (e.g. during population evaluation).
    void set_enabled(bool enabled) { enabled_ = enabled; }
    bool enabled() const { return enabled_; }

    MutationEngine(const MutationEngine&) = delete;
    MutationEngine& operator=(const MutationEngine&) = delete;

private:
    void on_activation_event(const ActivationEvent& ev);
    bool apply_mutation(Graph& graph, const MutationOp& op,
                        const std::string& rule_name, size_t step);

    struct GraphEntry {
        Graph* graph   = nullptr;
        bool   active  = true;
    };

    std::vector<GraphEntry>       graphs_;
    std::vector<MutationThreshold> rules_;
    std::mutex                    mutex_;
    SubID                         sub_id_ = INVALID_SUB_ID;
    bool                          enabled_ = true;

    // Per-layer EMA stats (keyed by layer_id).
    std::unordered_map<int, StatsSnapshot> layer_stats_;
};

// ---------------------------------------------------------------------------
// Built-in rule factories for common patterns.
// ---------------------------------------------------------------------------
namespace rules {

// Remove a dead node when its dead_ratio exceeds the threshold.
inline MutationThreshold dead_neuron_prune(int node_id, float threshold = 0.5f) {
    return {
        [](const StatsSnapshot& s) { return s.dead_ratio; },
        [](float v, float t) { return v > t; },
        threshold,
        [node_id](const StatsSnapshot&, const Graph&) -> MutationOp {
            return RemoveNode{ node_id };
        },
        "dead_neuron_prune"
    };
}

} // namespace rules

} // namespace fayn
