#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <typeindex>
#include <unordered_map>
#include <vector>
#include <condition_variable>
#include <future>

namespace fayn {

using SubID = uint64_t;
static constexpr SubID INVALID_SUB_ID = ~SubID{0};

enum class DispatchMode : uint8_t {
    // Handler is called inline in the thread that called emit().
    Sync  = 0,
    // Handler is posted to a background worker queue.
    Async = 1,
};

// ---------------------------------------------------------------------------
// EventBus: type-erased publish-subscribe bus.
//
// Supports per-subscriber dispatch mode (sync or async).
// Thread-safe for concurrent emit() calls.
//
// Usage:
//   auto& bus = EventBus::instance();
//   SubID id = bus.subscribe<ActivationEvent>([](const ActivationEvent& e) {
//       // handle
//   }, DispatchMode::Async);
//   bus.emit(ActivationEvent{...});
//   bus.unsubscribe(id);
// ---------------------------------------------------------------------------
class EventBus {
public:
    static EventBus& instance();

    // Subscribe a handler for EventT.
    // Returns a SubID that can be passed to unsubscribe().
    template<typename EventT>
    SubID subscribe(std::function<void(const EventT&)> handler,
                    DispatchMode mode = DispatchMode::Sync);

    // Remove a previously registered subscription.
    // Safe to call from within a handler (takes effect after current emit).
    void unsubscribe(SubID id);

    // Emit an event to all subscribers of type EventT.
    template<typename EventT>
    void emit(const EventT& event);

    // Block until all pending async tasks have been dispatched.
    void flush();

    // Disable copying.
    EventBus(const EventBus&) = delete;
    EventBus& operator=(const EventBus&) = delete;

    ~EventBus();

private:
    EventBus();

    struct HandlerEntry {
        SubID                            id;
        DispatchMode                     mode;
        bool                             valid = true;
        std::function<void(const void*)> handler;
    };

    std::unordered_map<std::type_index, std::vector<HandlerEntry>> handlers_;
    mutable std::mutex handlers_mutex_;
    SubID next_id_ = 0;

    // Async dispatch worker.
    struct AsyncTask { std::function<void()> fn; };
    std::queue<AsyncTask>    async_queue_;
    std::mutex               async_mutex_;
    std::condition_variable  async_cv_;
    std::thread              async_worker_;
    bool                     stop_ = false;

    void async_worker_loop();
};

// ---------------------------------------------------------------------------
// Template implementations
// ---------------------------------------------------------------------------
template<typename EventT>
SubID EventBus::subscribe(std::function<void(const EventT&)> handler,
                          DispatchMode mode) {
    std::lock_guard<std::mutex> lock(handlers_mutex_);
    SubID id = next_id_++;
    handlers_[std::type_index(typeid(EventT))].push_back({
        id,
        mode,
        /*valid=*/true,
        [h = std::move(handler)](const void* ptr) {
            h(*static_cast<const EventT*>(ptr));
        }
    });
    return id;
}

template<typename EventT>
void EventBus::emit(const EventT& event) {
    std::lock_guard<std::mutex> lock(handlers_mutex_);
    auto it = handlers_.find(std::type_index(typeid(EventT)));
    if (it == handlers_.end()) return;

    for (auto& entry : it->second) {
        if (!entry.valid) continue;

        if (entry.mode == DispatchMode::Sync) {
            entry.handler(&event);
        } else {
            // Copy the event for safe async dispatch.
            auto copy      = std::make_shared<EventT>(event);
            auto& h        = entry.handler;
            std::lock_guard<std::mutex> alock(async_mutex_);
            async_queue_.push({ [copy, h]() { h(copy.get()); } });
            async_cv_.notify_one();
        }
    }
}

} // namespace fayn
