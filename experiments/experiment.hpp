#pragma once

#include "src/core/graph.hpp"
#include "src/io/data_source.hpp"
#include "src/io/logger.hpp"
#include "src/topology/mutation_engine.hpp"
#include "src/stats/event_bus.hpp"
#include "src/stats/events.hpp"

#include <memory>
#include <string>

namespace fayn {

// ---------------------------------------------------------------------------
// ExperimentConfig: shared hyperparameters passed to every experiment.
// ---------------------------------------------------------------------------
struct ExperimentConfig {
    std::string name          = "experiment";
    size_t      epochs        = 100;
    size_t      batch_size    = 128;
    std::string log_path      = "logs/experiment.jsonl";
    bool        enable_mutations = true;
};

// ---------------------------------------------------------------------------
// Experiment: base class for all FAYN experiments.
//
// Subclass this and override setup() and run_epoch() to define:
//   - The network graph
//   - The data source(s)
//   - The learning paradigm (Hebbian, evolutionary, perturbation, etc.)
//   - Mutation rules
//
// Call execute() from the runner to run the full experiment.
//
// Example:
//   class HebbianMnistExperiment : public Experiment {
//     void setup()            override { ... }
//     float run_epoch(size_t) override { ... }
//   };
// ---------------------------------------------------------------------------
class Experiment {
public:
    explicit Experiment(ExperimentConfig cfg);
    virtual ~Experiment() = default;

    // Run the full experiment: setup() then run_epoch() for each epoch.
    void execute();

    const ExperimentConfig& config() const { return cfg_; }

    Experiment(const Experiment&) = delete;
    Experiment& operator=(const Experiment&) = delete;

protected:
    // Override to build the graph, register mutation rules, set up data, etc.
    virtual void setup() = 0;

    // Override to run one epoch. Return a scalar metric (e.g. accuracy, reward).
    // 'epoch' is 0-indexed.
    virtual float run_epoch(size_t epoch) = 0;

    // Convenience: emit an EpochEvent.
    void emit_epoch_begin(size_t epoch);
    void emit_epoch_end(size_t epoch, float metric);

    // Convenience: emit a RewardEvent.
    void emit_reward(size_t step, float reward);

    ExperimentConfig           cfg_;
    std::unique_ptr<Graph>     graph_;
    std::unique_ptr<DataSource> data_;
    std::unique_ptr<Logger>    logger_;
    MutationEngine             mutation_engine_;
};

} // namespace fayn
