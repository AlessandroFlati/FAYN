#include "event_bus.hpp"

#include <algorithm>

namespace fayn {

// ---------------------------------------------------------------------------
// Singleton
// ---------------------------------------------------------------------------
EventBus& EventBus::instance() {
    static EventBus bus;
    return bus;
}

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------
EventBus::EventBus() {
    async_worker_ = std::thread([this]() { async_worker_loop(); });
}

EventBus::~EventBus() {
    {
        std::lock_guard<std::mutex> lock(async_mutex_);
        stop_ = true;
        async_cv_.notify_all();
    }
    if (async_worker_.joinable()) {
        async_worker_.join();
    }
}

// ---------------------------------------------------------------------------
// Async worker
// ---------------------------------------------------------------------------
void EventBus::async_worker_loop() {
    while (true) {
        std::unique_lock<std::mutex> lock(async_mutex_);
        async_cv_.wait(lock, [this]() { return stop_ || !async_queue_.empty(); });

        if (stop_ && async_queue_.empty()) break;

        AsyncTask task = std::move(async_queue_.front());
        async_queue_.pop();
        lock.unlock();

        task.fn();
    }
}

// ---------------------------------------------------------------------------
// unsubscribe
// ---------------------------------------------------------------------------
void EventBus::unsubscribe(SubID id) {
    std::lock_guard<std::mutex> lock(handlers_mutex_);
    for (auto& [type, entries] : handlers_) {
        for (auto& e : entries) {
            if (e.id == id) {
                e.valid = false;
                return;
            }
        }
    }
    // Silently ignore unknown IDs to allow defensive cleanup.
}

// ---------------------------------------------------------------------------
// flush: wait for all pending async tasks to complete
// ---------------------------------------------------------------------------
void EventBus::flush() {
    // Use shared_ptr so the lambda is copy-constructible (required by std::function).
    auto promise = std::make_shared<std::promise<void>>();
    std::future<void> future = promise->get_future();
    {
        std::lock_guard<std::mutex> lock(async_mutex_);
        async_queue_.push({ [promise]() { promise->set_value(); } });
        async_cv_.notify_one();
    }
    future.wait();
}

} // namespace fayn
