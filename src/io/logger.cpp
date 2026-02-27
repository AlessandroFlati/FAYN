#include "logger.hpp"
#include <stdexcept>

namespace fayn {

Logger::Logger(const std::string& path, size_t flush_every)
    : path_(path), flush_every_(flush_every) {
    file_.open(path, std::ios::out | std::ios::trunc);
    if (!file_)
        throw std::runtime_error("Logger: cannot open " + path);
}

Logger::~Logger() {
    detach();
    flush();
}

void Logger::attach() {
    auto& bus = EventBus::instance();
    sub_activation_   = bus.subscribe<ActivationEvent>(
        [this](const ActivationEvent& ev)    { on_activation_event(ev); },
        DispatchMode::Async);
    sub_mut_applied_  = bus.subscribe<MutationAppliedEvent>(
        [this](const MutationAppliedEvent& ev) { on_mutation_applied(ev); },
        DispatchMode::Async);
    sub_mut_proposal_ = bus.subscribe<MutationProposalEvent>(
        [this](const MutationProposalEvent& ev){ on_mutation_proposal(ev); },
        DispatchMode::Async);
    sub_epoch_        = bus.subscribe<EpochEvent>(
        [this](const EpochEvent& ev)          { on_epoch_event(ev); },
        DispatchMode::Async);
    sub_reward_       = bus.subscribe<RewardEvent>(
        [this](const RewardEvent& ev)         { on_reward_event(ev); },
        DispatchMode::Async);
}

void Logger::detach() {
    auto& bus = EventBus::instance();
    for (SubID id : {sub_activation_, sub_mut_applied_,
                     sub_mut_proposal_, sub_epoch_, sub_reward_}) {
        if (id != INVALID_SUB_ID) bus.unsubscribe(id);
    }
    sub_activation_   = INVALID_SUB_ID;
    sub_mut_applied_  = INVALID_SUB_ID;
    sub_mut_proposal_ = INVALID_SUB_ID;
    sub_epoch_        = INVALID_SUB_ID;
    sub_reward_       = INVALID_SUB_ID;
}

void Logger::flush() {
    std::lock_guard<std::mutex> lock(mutex_);
    file_.flush();
}

void Logger::write(const nlohmann::json& entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    file_ << entry.dump() << '\n';
    ++write_count_;
    if (write_count_ % flush_every_ == 0) file_.flush();
}

// ---------------------------------------------------------------------------
// Event handlers
// ---------------------------------------------------------------------------
void Logger::on_activation_event(const ActivationEvent& ev) {
    nlohmann::json j;
    j["type"]       = "activation";
    j["step"]       = ev.step;
    j["layer_id"]   = ev.layer_id;
    j["dead_ratio"] = ev.stats.dead_ratio;
    j["magnitude"]  = ev.stats.ema_magnitude;
    write(j);
}

void Logger::on_mutation_applied(const MutationAppliedEvent& ev) {
    nlohmann::json j;
    j["type"]     = "mutation_applied";
    j["step"]     = ev.step;
    j["op"]       = ev.op_type;
    j["accepted"] = ev.accepted;
    write(j);
}

void Logger::on_mutation_proposal(const MutationProposalEvent& ev) {
    nlohmann::json j;
    j["type"]      = "mutation_proposal";
    j["step"]      = ev.step;
    j["layer_id"]  = ev.layer_id;
    j["op"]        = ev.op_type;
    j["rationale"] = ev.rationale;
    write(j);
}

void Logger::on_epoch_event(const EpochEvent& ev) {
    nlohmann::json j;
    j["type"]  = "epoch";
    j["epoch"] = ev.epoch;
    j["begin"] = ev.begin;
    if (!ev.begin) j["metric"] = ev.metric_value;
    write(j);
}

void Logger::on_reward_event(const RewardEvent& ev) {
    nlohmann::json j;
    j["type"]   = "reward";
    j["step"]   = ev.step;
    j["reward"] = ev.reward;
    write(j);
}

} // namespace fayn
