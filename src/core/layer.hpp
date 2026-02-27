#pragma once

#include "tensor.hpp"
#include "../stats/events.hpp"

#include <memory>
#include <string>

namespace fayn {

// ---------------------------------------------------------------------------
// Layer: abstract base class for all network operations.
//
// Lifecycle:
//   1. Constructed with hyperparameters.
//   2. Graph::add_node() assigns an ID via set_id().
//   3. forward() is called during inference.
//   4. After forward(), an ActivationEvent is emitted on the EventBus
//      so that subscribers (Logger, MutationEngine) can react.
//
// Dtype convention:
//   - Activations: BFloat16 by default (optimal for Blackwell).
//   - Internal weight/bias buffers: up to the layer implementation.
//   - Stats: always accumulated in Float32.
// ---------------------------------------------------------------------------
class Layer {
public:
    virtual ~Layer() = default;

    // Execute the forward pass.
    // Implementations MUST emit an ActivationEvent on EventBus::instance()
    // after computing the output.
    virtual Tensor forward(const Tensor& x) = 0;

    // Human-readable name (e.g. "dense_0", "relu_1").
    virtual std::string name() const = 0;

    // Number of output neurons / channels.
    virtual size_t output_size() const = 0;

    // Number of trainable parameters (weight elements + bias elements).
    virtual size_t num_params() const = 0;

    // Assign a node ID. Called by Graph::add_node().
    void set_id(int id) { id_ = id; }
    int  id()     const { return id_; }

    // Disable per-forward-pass stats collection (no-op by default).
    // Implementations that compute stats should override this.
    virtual void set_compute_stats(bool) noexcept {}

    // Increment and return the per-layer step counter.
    size_t next_step() { return ++step_; }
    size_t step()      const { return step_; }

protected:
    int    id_   = -1;
    size_t step_ = 0;
};

using LayerPtr = std::shared_ptr<Layer>;

} // namespace fayn
