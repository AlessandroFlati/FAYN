#include "one_hot.hpp"

#include "../core/device.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace fayn {

// One thread per output element: out[sample * C + cls] = (labels[sample] == cls).
__global__ static void one_hot_kernel(
    __nv_bfloat16* __restrict__ out,
    const int32_t* __restrict__ labels,
    size_t batch_size,
    size_t num_classes)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * num_classes) return;

    const size_t sample = idx / num_classes;
    const size_t cls    = idx % num_classes;
    out[idx] = __float2bfloat16(
        labels[sample] == static_cast<int32_t>(cls) ? 1.0f : 0.0f);
}

Tensor one_hot_encode(const Tensor& labels, size_t num_classes, cudaStream_t stream) {
    if (labels.device != Device::CUDA)
        throw std::invalid_argument("one_hot_encode: labels must be on CUDA");
    if (labels.dtype != DType::Int32)
        throw std::invalid_argument("one_hot_encode: labels must be Int32");
    if (labels.shape.size() != 1)
        throw std::invalid_argument("one_hot_encode: labels must be 1D [batch]");

    const size_t batch = labels.shape[0];
    Tensor out = Tensor::make({batch, num_classes}, DType::BFloat16, Device::CUDA);

    const size_t total = batch * num_classes;
    const size_t block = 256;
    const size_t grid  = (total + block - 1) / block;

    one_hot_kernel<<<grid, block, 0, stream>>>(
        static_cast<__nv_bfloat16*>(out.data),
        static_cast<const int32_t*>(labels.data),
        batch, num_classes);

    return out;
}

} // namespace fayn
