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

FAYN_REGISTER_EXPERIMENT("hebbian_mnist", fayn::HebbianMnistExperiment)

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
