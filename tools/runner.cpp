#include "registry.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Experiment registration.
// Include each experiment header and register it here. Static initializers
// in runner.cpp are guaranteed to run; registering here avoids the
// --whole-archive linker requirement for experiment static libraries.
// Add one block per experiment as it is created.
// ---------------------------------------------------------------------------
#include "experiments/hebbian_mnist/hebbian_mnist.hpp"
#include "experiments/ensemble_mnist/ensemble_mnist.hpp"

FAYN_REGISTER_EXPERIMENT("hebbian_mnist",  fayn::HebbianMnistExperiment,         hebbian_mnist)
FAYN_REGISTER_EXPERIMENT("ensemble_mnist", fayn::EnsembleHebbianMnistExperiment, ensemble_mnist)
FAYN_REGISTER_EXPERIMENT("elm_ensemble",   fayn::ELMEnsembleExperiment,          elm_ensemble)

// Scaled-init variants: d0 weights drawn from Uniform(-scale, +scale) where
// scale = sqrt(2/fan_in) * init_scale.  With init_scale ≈ sqrt(fan_in/2) = 19.8
// the range becomes Uniform(-1, 1), matching standard ELM initialisation.
// Pre-activation std ≈ 4.9 vs ≈ 0.27 for Kaiming — much more informative features.
namespace {
[[maybe_unused]] static const bool _fayn_reg_ensemble_mnist_scaled = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "ensemble_mnist_scaled",
        [](const fayn::ExperimentConfig& cfg) {
            constexpr float   kScale = 19.8f;  // sqrt(784/2) → Uniform(-1,1) range
            constexpr int64_t kSeed  = 42;     // reproducible d0 projections
            return std::make_unique<fayn::EnsembleHebbianMnistExperiment>(
                cfg, "data/mnist", /*lr=*/0.01f, /*K=*/10, /*norm_every=*/1,
                kScale, kSeed);
        });
    return true;
}();
[[maybe_unused]] static const bool _fayn_reg_elm_ensemble_scaled = []() {
    fayn::ExperimentRegistry::instance().register_experiment(
        "elm_ensemble_scaled",
        [](const fayn::ExperimentConfig& cfg) {
            constexpr float   kScale = 19.8f;
            constexpr int64_t kSeed  = 42;     // same projections as ensemble_mnist_scaled
            return std::make_unique<fayn::ELMEnsembleExperiment>(
                cfg, "data/mnist", /*K=*/10, kScale, kSeed);
        });
    return true;
}();
}

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <experiment_name> [options]\n"
              << "Options:\n"
              << "  --epochs N        Number of epochs (default: 100)\n"
              << "  --batch-size N    Batch size (default: 128)\n"
              << "  --log PATH        Output log file (default: logs/<name>.jsonl)\n"
              << "  --no-mutations    Disable topology mutations\n"
              << "  --list            List available experiments\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string first_arg = argv[1];

    if (first_arg == "--list") {
        std::cout << "Registered experiments:\n";
        fayn::ExperimentRegistry::instance().list(std::cout);
        return 0;
    }

    if (first_arg == "--help" || first_arg == "-h") {
        print_usage(argv[0]);
        return 0;
    }

    fayn::ExperimentConfig cfg;
    cfg.name     = first_arg;
    cfg.log_path = "logs/" + first_arg + ".jsonl";

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--epochs" && i + 1 < argc) {
            cfg.epochs = std::stoull(argv[++i]);
        } else if (arg == "--batch-size" && i + 1 < argc) {
            cfg.batch_size = std::stoull(argv[++i]);
        } else if (arg == "--log" && i + 1 < argc) {
            cfg.log_path = argv[++i];
        } else if (arg == "--no-mutations") {
            cfg.enable_mutations = false;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    try {
        auto exp = fayn::ExperimentRegistry::instance().create(cfg.name, cfg);
        exp->execute();
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
