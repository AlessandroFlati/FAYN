#include "stream_pool.hpp"

namespace fayn {

StreamPool& StreamPool::instance(size_t pool_size) {
    static StreamPool pool(pool_size);
    return pool;
}

StreamPool::StreamPool(size_t pool_size) {
    if (pool_size == 0)
        throw std::invalid_argument("StreamPool: pool_size must be > 0");
    streams_.reserve(pool_size);
    for (size_t i = 0; i < pool_size; ++i) {
        cudaStream_t s;
        FAYN_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        streams_.push_back(s);
        available_.push(s);
    }
}

StreamPool::~StreamPool() {
    for (auto s : streams_) {
        cudaStreamDestroy(s);  // ignore error in destructor
    }
}

cudaStream_t StreamPool::acquire() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]() { return !available_.empty(); });
    cudaStream_t s = available_.front();
    available_.pop();
    return s;
}

void StreamPool::release(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    available_.push(stream);
    cv_.notify_one();
}

} // namespace fayn
