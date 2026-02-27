#pragma once

#include "../core/tensor.hpp"

#include <cstddef>
#include <memory>
#include <utility>

namespace fayn {

// ---------------------------------------------------------------------------
// Batch: a single mini-batch of inputs and targets.
// ---------------------------------------------------------------------------
struct Batch {
    Tensor inputs;   // [batch_size, ...]
    Tensor targets;  // [batch_size, ...] or [batch_size] for class labels
};

// ---------------------------------------------------------------------------
// DataSource: abstract interface for training data.
//
// Implementations: MnistLoader, CsvLoader, any user-defined source.
// ---------------------------------------------------------------------------
class DataSource {
public:
    virtual ~DataSource() = default;

    // Return the next batch. If the dataset is exhausted, wraps around.
    virtual Batch next_batch(size_t batch_size) = 0;

    // Reset to the beginning of the dataset.
    virtual void reset() = 0;

    // Total number of samples.
    virtual size_t size() const = 0;

    // Number of batches per epoch for the given batch_size.
    size_t batches_per_epoch(size_t batch_size) const {
        return (size() + batch_size - 1) / batch_size;
    }

    // Target device for output tensors. Default: CUDA.
    void set_output_device(Device d) { output_device_ = d; }
    Device output_device() const     { return output_device_; }

    // Target dtype for input tensors. Default: BFloat16.
    void set_input_dtype(DType d)  { input_dtype_ = d; }
    DType input_dtype() const      { return input_dtype_; }

protected:
    Device output_device_ = Device::CUDA;
    DType  input_dtype_   = DType::BFloat16;
};

} // namespace fayn
