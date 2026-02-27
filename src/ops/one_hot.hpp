#pragma once

#include "../core/tensor.hpp"

#include <cuda_runtime.h>

namespace fayn {

// ---------------------------------------------------------------------------
// one_hot_encode: encode Int32 class labels into a BF16 one-hot matrix.
//
//   labels    — [batch]          Int32,    CUDA device
//   num_classes — number of classes C
//   returns   — [batch, C]       BFloat16, CUDA device
//               out[i, labels[i]] = 1.0, all other entries = 0.0
// ---------------------------------------------------------------------------
Tensor one_hot_encode(const Tensor& labels, size_t num_classes,
                      cudaStream_t stream = nullptr);

} // namespace fayn
