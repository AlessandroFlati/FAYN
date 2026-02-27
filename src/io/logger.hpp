#pragma once

#include "../stats/event_bus.hpp"
#include "../stats/events.hpp"

#include <nlohmann/json.hpp>
#include <fstream>
#include <mutex>
#include <string>

namespace fayn {

// ---------------------------------------------------------------------------
// Logger: subscribes to EventBus (async) and writes structured JSON lines
// to a log file, one JSON object per line (JSONL / NDJSON format).
//
// Each JSON entry has a "type" field indicating the event class.
//
// File is flushed periodically (every flush_every events) and on destruction.
// ---------------------------------------------------------------------------
class Logger {
public:
    // Opens the log file (truncating if it exists).
    // flush_every: number of entries between automatic flush().
    explicit Logger(const std::string& path, size_t flush_every = 100);
    ~Logger();

    // Start subscribing to EventBus. Must be called after construction.
    void attach();

    // Stop subscribing and flush.
    void detach();

    // Manually flush the file buffer.
    void flush();

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

private:
    void write(const nlohmann::json& entry);

    void on_activation_event   (const ActivationEvent& ev);
    void on_mutation_applied   (const MutationAppliedEvent& ev);
    void on_mutation_proposal  (const MutationProposalEvent& ev);
    void on_epoch_event        (const EpochEvent& ev);
    void on_reward_event       (const RewardEvent& ev);

    std::string   path_;
    std::ofstream file_;
    std::mutex    mutex_;
    size_t        flush_every_;
    size_t        write_count_ = 0;

    SubID sub_activation_  = INVALID_SUB_ID;
    SubID sub_mut_applied_ = INVALID_SUB_ID;
    SubID sub_mut_proposal_= INVALID_SUB_ID;
    SubID sub_epoch_       = INVALID_SUB_ID;
    SubID sub_reward_      = INVALID_SUB_ID;
};

} // namespace fayn
