#include "graph.hpp"

#include <algorithm>
#include <queue>
#include <stdexcept>
#include <unordered_map>

namespace fayn {

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------
void Graph::validate_node_id(int id) const {
    if (id < 0 || static_cast<size_t>(id) >= nodes_.size())
        throw std::out_of_range("Invalid node ID: " + std::to_string(id));
}

void Graph::validate_edge_id(int id) const {
    if (id < 0 || static_cast<size_t>(id) >= edges_.size())
        throw std::out_of_range("Invalid edge ID: " + std::to_string(id));
}

const Node& Graph::node(int id) const { validate_node_id(id); return nodes_[id]; }
Node&       Graph::node(int id)       { validate_node_id(id); return nodes_[id]; }
const Edge& Graph::edge(int id) const { validate_edge_id(id); return edges_[id]; }
Edge&       Graph::edge(int id)       { validate_edge_id(id); return edges_[id]; }

// ---------------------------------------------------------------------------
// Topology mutation
// ---------------------------------------------------------------------------
int Graph::add_node(LayerPtr layer) {
    if (!layer) throw std::invalid_argument("add_node: null LayerPtr");
    int id = static_cast<int>(nodes_.size());
    layer->set_id(id);
    nodes_.push_back({ std::move(layer), {}, /*active=*/true });
    in_degree_.push_back(0);
    return id;
}

int Graph::add_edge(int src_node, int dst_node) {
    validate_node_id(src_node);
    validate_node_id(dst_node);
    if (!nodes_[src_node].active || !nodes_[dst_node].active)
        throw std::runtime_error("add_edge: cannot connect inactive nodes");

    int id = static_cast<int>(edges_.size());
    edges_.push_back({ src_node, dst_node, /*active=*/true });
    nodes_[src_node].out_edges.push_back(id);
    ++in_degree_[dst_node];
    return id;
}

void Graph::remove_node(int node_id) {
    validate_node_id(node_id);
    // Deactivate all edges touching this node.
    for (int eid = 0; eid < static_cast<int>(edges_.size()); ++eid) {
        auto& e = edges_[eid];
        if (!e.active) continue;
        if (e.src == node_id || e.dst == node_id) {
            e.active = false;
            if (e.dst == node_id && nodes_[e.src].active) {
                auto& outs = nodes_[e.src].out_edges;
                outs.erase(std::remove(outs.begin(), outs.end(), eid), outs.end());
            }
            if (e.src == node_id) {
                --in_degree_[e.dst];
            }
        }
    }
    nodes_[node_id].active    = false;
    nodes_[node_id].out_edges = {};
    in_degree_[node_id]       = 0;
}

void Graph::remove_edge(int edge_id) {
    validate_edge_id(edge_id);
    Edge& e = edges_[edge_id];
    if (!e.active) return;
    e.active = false;
    auto& outs = nodes_[e.src].out_edges;
    outs.erase(std::remove(outs.begin(), outs.end(), edge_id), outs.end());
    --in_degree_[e.dst];
}

void Graph::rewire_edge(int edge_id, int new_dst) {
    validate_edge_id(edge_id);
    validate_node_id(new_dst);
    Edge& e = edges_[edge_id];
    if (!e.active) throw std::runtime_error("rewire_edge: edge is not active");
    --in_degree_[e.dst];
    e.dst = new_dst;
    ++in_degree_[new_dst];
}

// ---------------------------------------------------------------------------
// Topological sort (Kahn's algorithm)
// ---------------------------------------------------------------------------
std::vector<int> Graph::topological_order() const {
    const int n = static_cast<int>(nodes_.size());
    std::vector<int> deg(n, 0);

    // Recompute in-degrees from active edges only.
    for (const auto& e : edges_) {
        if (e.active && nodes_[e.src].active && nodes_[e.dst].active) {
            ++deg[e.dst];
        }
    }

    std::queue<int> q;
    for (int i = 0; i < n; ++i) {
        if (nodes_[i].active && deg[i] == 0) q.push(i);
    }

    std::vector<int> order;
    order.reserve(n);
    while (!q.empty()) {
        int u = q.front(); q.pop();
        order.push_back(u);
        for (int eid : nodes_[u].out_edges) {
            const Edge& e = edges_[eid];
            if (!e.active) continue;
            if (--deg[e.dst] == 0) q.push(e.dst);
        }
    }

    int active_count = 0;
    for (int i = 0; i < n; ++i) if (nodes_[i].active) ++active_count;
    if (static_cast<int>(order.size()) != active_count)
        throw std::runtime_error("Graph has a cycle; forward pass is undefined");

    return order;
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------
std::vector<Tensor> Graph::forward(std::vector<std::pair<int, Tensor>> inputs) {
    const std::vector<int> order = topological_order();

    // Store intermediate activations by node ID.
    std::unordered_map<int, Tensor> activations;

    // Seed with provided inputs.
    for (auto& [nid, t] : inputs) {
        activations.emplace(nid, std::move(t));
    }

    // Execute in topological order.
    for (int nid : order) {
        const Node& n = nodes_[nid];

        // Collect incoming tensors (from active in-edges).
        std::vector<const Tensor*> in_tensors;
        for (const auto& e : edges_) {
            if (e.active && e.dst == nid && nodes_[e.src].active) {
                auto it = activations.find(e.src);
                if (it != activations.end()) {
                    in_tensors.push_back(&it->second);
                }
            }
        }

        Tensor output;
        if (in_tensors.empty()) {
            // Input node: must have been seeded above.
            auto it = activations.find(nid);
            if (it == activations.end())
                throw std::runtime_error("Node " + std::to_string(nid) +
                                         " has no input and was not seeded");
            output = n.layer->forward(it->second);
        } else if (in_tensors.size() == 1) {
            output = n.layer->forward(*in_tensors[0]);
        } else {
            // Multiple inputs: sum element-wise (residual connection style).
            // For now only support two inputs with the same shape.
            // TODO: implement a generic multi-input merge strategy per layer.
            throw std::runtime_error("Multi-input merge not yet implemented for node " +
                                     std::to_string(nid));
        }
        activations.insert_or_assign(nid, std::move(output));
    }

    // Collect outputs: nodes with no active out-edges.
    std::vector<Tensor> outputs;
    for (int nid : order) {
        bool is_output = true;
        for (int eid : nodes_[nid].out_edges) {
            if (edges_[eid].active) { is_output = false; break; }
        }
        if (is_output) {
            outputs.push_back(std::move(activations.at(nid)));
        }
    }
    return outputs;
}

// ---------------------------------------------------------------------------
// Clone (shallow weight sharing)
// ---------------------------------------------------------------------------
std::unique_ptr<Graph> Graph::clone() const {
    auto g = std::make_unique<Graph>();
    g->nodes_    = nodes_;
    g->edges_    = edges_;
    g->in_degree_ = in_degree_;
    // Layer pointers are shared; weights are not deep-copied.
    return g;
}

} // namespace fayn
