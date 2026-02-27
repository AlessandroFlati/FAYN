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
// FAYN_REGISTER_EXPERIMENT(name, ExperimentClass, unique_id)
//
// Call in runner.cpp for each experiment that should be available at runtime.
// The ExperimentClass must be constructible from (const ExperimentConfig&).
// unique_id must be a valid C++ identifier, unique within runner.cpp.
//
// GCC does not pre-scan predefined macros (__COUNTER__, __LINE__) when they
// appear as macro arguments, so a user-supplied unique_id is required.
//
// Example:
//   FAYN_REGISTER_EXPERIMENT("hebbian_mnist", fayn::HebbianMnistExperiment,
//                             hebbian_mnist)
// ---------------------------------------------------------------------------
#define FAYN_REGISTER_EXPERIMENT(name, ExperimentClass, unique_id)              \
    namespace {                                                                  \
    [[maybe_unused]] static const bool _fayn_reg_##unique_id = []() {           \
        fayn::ExperimentRegistry::instance().register_experiment(               \
            name,                                                                \
            [](const fayn::ExperimentConfig& cfg) {                             \
                return std::make_unique<ExperimentClass>(cfg);                  \
            });                                                                  \
        return true;                                                             \
    }();                                                                         \
    }
