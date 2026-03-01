#include "cifar_loader.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace fayn {

static constexpr size_t CIFAR_N_PER_FILE  = 10000;
static constexpr size_t CIFAR_PIXELS      = 3 * 32 * 32;  // 3072
static constexpr size_t CIFAR_RECORD_SIZE = 1 + CIFAR_PIXELS;  // 3073

void CifarLoader::load_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("CifarLoader: cannot open " + path);

    std::vector<uint8_t> raw(CIFAR_RECORD_SIZE * CIFAR_N_PER_FILE);
    f.read(reinterpret_cast<char*>(raw.data()),
           static_cast<std::streamsize>(raw.size()));
    if (!f) throw std::runtime_error("CifarLoader: truncated data in " + path);

    const size_t base = images_.size();
    images_.resize(base + CIFAR_N_PER_FILE);
    labels_.resize(base + CIFAR_N_PER_FILE);

    for (size_t i = 0; i < CIFAR_N_PER_FILE; ++i) {
        const uint8_t* rec = raw.data() + i * CIFAR_RECORD_SIZE;
        labels_[base + i] = rec[0];
        images_[base + i].resize(CIFAR_PIXELS);
        for (size_t j = 0; j < CIFAR_PIXELS; ++j)
            images_[base + i][j] = static_cast<float>(rec[1 + j]) / 255.0f;
    }
}

CifarLoader::CifarLoader(const std::string& cifar_dir, bool train) {
    if (train) {
        for (int b = 1; b <= 5; ++b)
            load_file(cifar_dir + "/data_batch_" + std::to_string(b) + ".bin");
    } else {
        load_file(cifar_dir + "/test_batch.bin");
    }
}

size_t CifarLoader::size()  const { return images_.size(); }
void   CifarLoader::reset()       { cursor_ = 0; }

Batch CifarLoader::next_batch(size_t batch_size) {
    const size_t n      = images_.size();
    const size_t actual = std::min(batch_size, n - (cursor_ % n));
    const size_t start  = cursor_ % n;
    cursor_ += actual;

    std::vector<float>   inp_fp32(actual * CIFAR_PIXELS);
    std::vector<int32_t> tgt_i32(actual);

    for (size_t i = 0; i < actual; ++i) {
        std::memcpy(inp_fp32.data() + i * CIFAR_PIXELS,
                    images_[(start + i) % n].data(),
                    CIFAR_PIXELS * sizeof(float));
        tgt_i32[i] = static_cast<int32_t>(labels_[(start + i) % n]);
    }

    Tensor inputs  = Tensor::make({actual, CIFAR_PIXELS}, input_dtype_, Device::CPU);
    Tensor targets = Tensor::make({actual},               DType::Int32,  Device::CPU);

    switch (input_dtype_) {
        case DType::Float32:
            std::memcpy(inputs.data, inp_fp32.data(),
                        actual * CIFAR_PIXELS * sizeof(float));
            break;
        case DType::BFloat16: {
            auto* dst = static_cast<__nv_bfloat16*>(inputs.data);
            for (size_t i = 0; i < actual * CIFAR_PIXELS; ++i)
                dst[i] = __float2bfloat16(inp_fp32[i]);
            break;
        }
        case DType::Float16: {
            auto* dst = static_cast<__half*>(inputs.data);
            for (size_t i = 0; i < actual * CIFAR_PIXELS; ++i)
                dst[i] = __float2half(inp_fp32[i]);
            break;
        }
        default:
            throw std::runtime_error("CifarLoader: unsupported input dtype");
    }

    std::memcpy(targets.data, tgt_i32.data(), actual * sizeof(int32_t));
    return { inputs.to(output_device_), targets.to(output_device_) };
}

} // namespace fayn
