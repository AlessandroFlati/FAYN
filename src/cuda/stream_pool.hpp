#pragma once

#include "../core/device.hpp"

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

namespace fayn {

// ---------------------------------------------------------------------------
// StreamPool: manages a fixed set of CUDA streams.
//
// Streams are acquired before launching CUDA work and released when done.
// If the pool is exhausted, acquire() blocks until a stream is returned.
//
// Usage (manual):
//   auto& pool = StreamPool::instance();
//   cudaStream_t s = pool.acquire();
//   // ... launch kernels on s ...
//   pool.release(s);
//
// Usage (RAII):
//   StreamPool::Guard guard;
//   kernelA<<<..., guard.stream()>>>(...);
// ---------------------------------------------------------------------------
class StreamPool {
public:
    // Default pool size. Tune based on experiment population size.
    static constexpr size_t DEFAULT_POOL_SIZE = 8;

    static StreamPool& instance(size_t pool_size = DEFAULT_POOL_SIZE);

    // Acquire a stream (blocks if pool is empty).
    cudaStream_t acquire();

    // Return a stream to the pool.
    void release(cudaStream_t stream);

    // RAII guard: acquires on construction, releases on destruction.
    class Guard {
    public:
        explicit Guard(StreamPool& pool = StreamPool::instance())
            : pool_(pool), stream_(pool.acquire()) {}
        ~Guard() { pool_.release(stream_); }

        cudaStream_t stream() const noexcept { return stream_; }

        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;
        Guard(Guard&&) = delete;

    private:
        StreamPool&  pool_;
        cudaStream_t stream_;
    };

    size_t pool_size() const noexcept { return streams_.size(); }

    StreamPool(const StreamPool&) = delete;
    StreamPool& operator=(const StreamPool&) = delete;

    ~StreamPool();

private:
    explicit StreamPool(size_t pool_size);

    std::vector<cudaStream_t> streams_;
    std::queue<cudaStream_t>  available_;
    std::mutex                mutex_;
    std::condition_variable   cv_;
};

} // namespace fayn
