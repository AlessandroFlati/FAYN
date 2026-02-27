#pragma once

#include <cstddef>
#include <string>
#include <variant>

namespace fayn {

// ---------------------------------------------------------------------------
// Mutation operations: the closed set of structural changes that can be
// applied to a Graph.
//
// Each op is a small value type describing the intent; the actual mutation
// is performed by Graph methods.
// ---------------------------------------------------------------------------

struct AddNode {
    std::string layer_type;   // e.g. "dense", "relu", "gelu"
    size_t      insert_after; // node ID after which to splice the new node
    size_t      out_features; // only meaningful for parameterized layers
};

struct RemoveNode {
    int node_id;
    // Edges are automatically deactivated; caller must reconnect if needed.
};

struct AddEdge {
    int src;
    int dst;
};

struct RemoveEdge {
    int edge_id;
};

struct RewireEdge {
    int edge_id;
    int new_dst;
};

struct SplitNode {
    int    node_id;
    size_t split_features;   // output width of the first half
};

struct NoOp {};

// ---------------------------------------------------------------------------
// MutationOp: type-erased variant over all mutation types.
// ---------------------------------------------------------------------------
using MutationOp = std::variant<
    NoOp,
    AddNode,
    RemoveNode,
    AddEdge,
    RemoveEdge,
    RewireEdge,
    SplitNode
>;

// Human-readable name for logging.
inline const char* mutation_op_name(const MutationOp& op) {
    return std::visit([](auto&& v) -> const char* {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, NoOp>)       return "no_op";
        if constexpr (std::is_same_v<T, AddNode>)    return "add_node";
        if constexpr (std::is_same_v<T, RemoveNode>) return "remove_node";
        if constexpr (std::is_same_v<T, AddEdge>)    return "add_edge";
        if constexpr (std::is_same_v<T, RemoveEdge>) return "remove_edge";
        if constexpr (std::is_same_v<T, RewireEdge>) return "rewire_edge";
        if constexpr (std::is_same_v<T, SplitNode>)  return "split_node";
        return "unknown";
    }, op);
}

} // namespace fayn
