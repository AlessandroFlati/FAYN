#include "mutation_engine.hpp"

#include <stdexcept>

namespace fayn {

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------
MutationEngine::MutationEngine() {
    // Subscribe synchronously so mutations happen inline during forward pass.
    sub_id_ = EventBus::instance().subscribe<ActivationEvent>(
        [this](const ActivationEvent& ev) { on_activation_event(ev); },
        DispatchMode::Sync);
}

MutationEngine::~MutationEngine() {
    if (sub_id_ != INVALID_SUB_ID) {
        EventBus::instance().unsubscribe(sub_id_);
    }
}

// ---------------------------------------------------------------------------
// Graph registration
// ---------------------------------------------------------------------------
int MutationEngine::register_graph(Graph* graph) {
    if (!graph) throw std::invalid_argument("register_graph: null graph pointer");
    std::lock_guard<std::mutex> lock(mutex_);
    int id = static_cast<int>(graphs_.size());
    graphs_.push_back({ graph, true });
    return id;
}

void MutationEngine::deregister_graph(int handle_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (handle_id < 0 || static_cast<size_t>(handle_id) >= graphs_.size())
        throw std::out_of_range("deregister_graph: invalid handle");
    graphs_[handle_id].active = false;
}

// ---------------------------------------------------------------------------
// Rule management
// ---------------------------------------------------------------------------
void MutationEngine::add_rule(MutationThreshold rule) {
    std::lock_guard<std::mutex> lock(mutex_);
    rules_.push_back(std::move(rule));
}

void MutationEngine::clear_rules() {
    std::lock_guard<std::mutex> lock(mutex_);
    rules_.clear();
}

// ---------------------------------------------------------------------------
// Event handler
// ---------------------------------------------------------------------------
void MutationEngine::on_activation_event(const ActivationEvent& ev) {
    if (!enabled_) return;

    std::lock_guard<std::mutex> lock(mutex_);

    // Update cached stats for this layer.
    layer_stats_[ev.layer_id] = ev.stats;

    // Check rules against all active graphs.
    for (auto& entry : graphs_) {
        if (!entry.active || !entry.graph) continue;

        for (const auto& rule : rules_) {
            float value = rule.metric_fn(ev.stats);
            if (rule.comparator(value, rule.threshold)) {
                MutationOp op = rule.factory(ev.stats, *entry.graph);
                apply_mutation(*entry.graph, op, rule.name, ev.step);
                break;  // apply at most one rule per event per graph
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Apply a mutation op to a graph and emit events.
// ---------------------------------------------------------------------------
bool MutationEngine::apply_mutation(Graph& graph, const MutationOp& op,
                                    const std::string& rule_name, size_t step) {
    bool accepted = false;
    try {
        std::visit([&](auto&& m) {
            using T = std::decay_t<decltype(m)>;
            if constexpr (std::is_same_v<T, NoOp>) {
                // Nothing to do.
            } else if constexpr (std::is_same_v<T, RemoveNode>) {
                graph.remove_node(m.node_id);
                accepted = true;
            } else if constexpr (std::is_same_v<T, AddEdge>) {
                graph.add_edge(m.src, m.dst);
                accepted = true;
            } else if constexpr (std::is_same_v<T, RemoveEdge>) {
                graph.remove_edge(m.edge_id);
                accepted = true;
            } else if constexpr (std::is_same_v<T, RewireEdge>) {
                graph.rewire_edge(m.edge_id, m.new_dst);
                accepted = true;
            } else if constexpr (std::is_same_v<T, AddNode>) {
                // AddNode requires instantiating a layer; delegate to caller
                // via a MutationProposalEvent so the Experiment can handle it.
                // The MutationEngine does not own layer factories.
                MutationProposalEvent proposal;
                proposal.step      = step;
                proposal.op_type   = "add_node";
                proposal.rationale = "rule: " + rule_name;
                EventBus::instance().emit(proposal);
                // Not accepted here; Experiment must handle the proposal.
            } else if constexpr (std::is_same_v<T, SplitNode>) {
                // Same delegation for SplitNode.
                MutationProposalEvent proposal;
                proposal.step      = step;
                proposal.op_type   = "split_node";
                proposal.rationale = "rule: " + rule_name;
                EventBus::instance().emit(proposal);
            }
        }, op);
    } catch (const std::exception&) {
        // Mutation failed; emit a rejected event and continue.
        accepted = false;
    }

    MutationAppliedEvent applied;
    applied.step     = step;
    applied.op_type  = mutation_op_name(op);
    applied.accepted = accepted;
    EventBus::instance().emit(applied);

    return accepted;
}

} // namespace fayn
