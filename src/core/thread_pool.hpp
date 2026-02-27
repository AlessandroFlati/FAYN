#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

namespace fayn {

// ---------------------------------------------------------------------------
// ThreadPool: fixed-size pool of std::threads with a task queue.
//
// Usage:
//   ThreadPool pool(4);
//   auto fut = pool.submit([]() { return expensive_computation(); });
//   auto result = fut.get();
// ---------------------------------------------------------------------------
class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads) {
        if (num_threads == 0)
            throw std::invalid_argument("ThreadPool: num_threads must be > 0");
        workers_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this]() { worker_loop(); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }
    }

    // Submit a callable and return a future for its result.
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>>
    {
        using RetT = std::invoke_result_t<F, Args...>;
        auto task = std::make_shared<std::packaged_task<RetT()>>(
            [fn = std::forward<F>(f),
             ...a = std::forward<Args>(args)]() mutable { return fn(std::move(a)...); }
        );
        std::future<RetT> fut = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_) throw std::runtime_error("ThreadPool is stopped");
            tasks_.push([task]() { (*task)(); });
        }
        cv_.notify_one();
        return fut;
    }

    size_t num_threads() const noexcept { return workers_.size(); }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

private:
    void worker_loop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

    std::vector<std::thread>          workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex                        mutex_;
    std::condition_variable           cv_;
    bool                              stop_ = false;
};

} // namespace fayn
