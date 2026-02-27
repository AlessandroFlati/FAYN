#include "experiment.hpp"

#include <filesystem>
#include <stdexcept>

namespace fayn {

Experiment::Experiment(ExperimentConfig cfg)
    : cfg_(std::move(cfg)) {}

void Experiment::execute() {
    // Ensure log directory exists.
    if (!cfg_.log_path.empty()) {
        std::filesystem::path log_dir =
            std::filesystem::path(cfg_.log_path).parent_path();
        if (!log_dir.empty())
            std::filesystem::create_directories(log_dir);

        logger_ = std::make_unique<Logger>(cfg_.log_path);
        logger_->attach();
    }

    setup();

    if (cfg_.enable_mutations && graph_) {
        mutation_engine_.register_graph(graph_.get());
    }

    for (size_t epoch = 0; epoch < cfg_.epochs; ++epoch) {
        emit_epoch_begin(epoch);
        float metric = run_epoch(epoch);
        emit_epoch_end(epoch, metric);
    }

    if (logger_) {
        EventBus::instance().flush();
        logger_->detach();
    }
}

void Experiment::emit_epoch_begin(size_t epoch) {
    EpochEvent ev;
    ev.epoch = epoch;
    ev.begin = true;
    EventBus::instance().emit(ev);
}

void Experiment::emit_epoch_end(size_t epoch, float metric) {
    EpochEvent ev;
    ev.epoch        = epoch;
    ev.begin        = false;
    ev.metric_value = metric;
    EventBus::instance().emit(ev);
}

void Experiment::emit_reward(size_t step, float reward) {
    RewardEvent ev;
    ev.step   = step;
    ev.reward = reward;
    EventBus::instance().emit(ev);
}

} // namespace fayn
