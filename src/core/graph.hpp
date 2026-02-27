#pragma once

#include "layer.hpp"
#include "tensor.hpp"

#include <memory>
#include <stdexcept>
#include <vector>

namespace fayn {

// ---------------------------------------------------------------------------
// Graph node: wraps a Layer and records outgoing edge indices.
// ---------------------------------------------------------------------------
struct Node {
    LayerPtr         layer;
    std::vector<int> out_edges;   // indices into Graph::edges_
    bool             active = true;
};

// ---------------------------------------------------------------------------
// Graph edge: directed connection from one node to another.
// The tensor produced at 'src' is fed as input to 'dst'.
// Multiple input edges to a single dst are summed element-wise (residual).
// ---------------------------------------------------------------------------
struct Edge {
    int  src     = -1;
    int  dst     = -1;
    bool active  = true;
};

// ---------------------------------------------------------------------------
// Graph: directed acyclic graph of Layer nodes connected by tensor edges.
//
// Topology mutation (add/remove nodes and edges) is safe between forward
// passes but MUST NOT be called concurrently with forward().
// ---------------------------------------------------------------------------
class Graph {
public:
    Graph() = default;

    // Add a layer as a new node. Returns the node ID.
    int add_node(LayerPtr layer);

    // Add a directed edge from src to dst. Returns the edge ID.
    int add_edge(int src_node, int dst_node);

    // Deactivate a node (and all its edges). Does not remove from vectors
    // to preserve stable IDs. Use compact() to reclaim memory.
    void remove_node(int node_id);

    // Deactivate an edge.
    void remove_edge(int edge_id);

    // Rewire an existing edge to a new destination.
    void rewire_edge(int edge_id, int new_dst);

    // Execute a full forward pass.
    // inputs maps node IDs that have no incoming edges to their input tensors.
    // Returns the output tensors of nodes with no outgoing edges.
    std::vector<Tensor> forward(std::vector<std::pair<int, Tensor>> inputs);

    // Return a topological execution order (only active nodes).
    std::vector<int> topological_order() const;

    // Return a deep clone of the graph (same topology, shared layer weights).
    std::unique_ptr<Graph> clone() const;

    size_t num_nodes() const noexcept { return nodes_.size(); }
    size_t num_edges() const noexcept { return edges_.size(); }

    const Node& node(int id) const;
    Node&       node(int id);
    const Edge& edge(int id) const;
    Edge&       edge(int id);

private:
    std::vector<Node> nodes_;
    std::vector<Edge> edges_;

    // In-degree of each node (for topological sort).
    std::vector<int> in_degree_;

    void validate_node_id(int id) const;
    void validate_edge_id(int id) const;
};

} // namespace fayn
