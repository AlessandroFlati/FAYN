#pragma once

#include "../experiments/experiment.hpp"

#include <functional>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace fayn {

// ---------------------------------------------------------------------------
// ExperimentRegistry: maps string names to factory functions.
// Experiments self-register at static initialisation via FAYN_REGISTER_EXPERIMENT.
// ---------------------------------------------------------------------------
using ExperimentFactory = std::function<std::unique_ptr<Experiment>(const ExperimentConfig&)>;

class ExperimentRegistry {
public:
    static ExperimentRegistry& instance() {
        static ExperimentRegistry reg;
        return reg;
    }

    void register_experiment(const std::string& name, ExperimentFactory factory) {
        if (registry_.count(name))
            throw std::runtime_error("Duplicate experiment name: " + name);
        registry_[name] = std::move(factory);
    }

    std::unique_ptr<Experiment> create(const std::string& name,
                                       const ExperimentConfig& cfg) const {
        auto it = registry_.find(name);
        if (it == registry_.end())
            throw std::runtime_error("Unknown experiment: " + name +
                                     ". Run with --list to see registered experiments.");
        return it->second(cfg);
    }

    void list(std::ostream& os) const {
        for (const auto& [k, _] : registry_) os << "  " << k << "\n";
    }

private:
    std::unordered_map<std::string, ExperimentFactory> registry_;
};

} // namespace fayn

// ---------------------------------------------------------------------------
// FAYN_REGISTER_EXPERIMENT(name, ExperimentClass)
//
// Place exactly once in the experiment's .cpp file.
// The ExperimentClass must be constructible from (const ExperimentConfig&).
//
// Example:
//   FAYN_REGISTER_EXPERIMENT("hebbian_mnist", fayn::HebbianMnistExperiment)
// ---------------------------------------------------------------------------
// Two-level macro so __COUNTER__ expands before token-pasting.
// This avoids issues with namespaced class names (e.g. fayn::Foo) in ##.
#define FAYN_REGISTER_EXPERIMENT_IMPL(name, ExperimentClass, counter)           \
    namespace {                                                                  \
    struct _ExperimentRegistrar_##counter {                                     \
        _ExperimentRegistrar_##counter() {                                      \
            fayn::ExperimentRegistry::instance().register_experiment(           \
                name,                                                            \
                [](const fayn::ExperimentConfig& cfg) {                         \
                    return std::make_unique<ExperimentClass>(cfg);              \
                });                                                              \
        }                                                                        \
    } _registrar_instance_##counter;                                            \
    }

#define FAYN_REGISTER_EXPERIMENT(name, ExperimentClass) \
    FAYN_REGISTER_EXPERIMENT_IMPL(name, ExperimentClass, __COUNTER__)
