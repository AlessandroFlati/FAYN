#include "mnist_loader.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace fayn {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
uint32_t MnistLoader::read_big_endian_u32(const uint8_t* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
           (uint32_t(p[2]) <<  8) |  uint32_t(p[3]);
}

// ---------------------------------------------------------------------------
// File loading
// ---------------------------------------------------------------------------
void MnistLoader::load_images(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("MnistLoader: cannot open " + path);

    uint8_t header[16];
    f.read(reinterpret_cast<char*>(header), 16);
    if (!f) throw std::runtime_error("MnistLoader: truncated header in " + path);

    uint32_t magic  = read_big_endian_u32(header + 0);
    uint32_t n      = read_big_endian_u32(header + 4);
    uint32_t rows   = read_big_endian_u32(header + 8);
    uint32_t cols   = read_big_endian_u32(header + 12);

    if (magic != 0x00000803)
        throw std::runtime_error("MnistLoader: invalid image magic number in " + path);

    rows_ = rows;
    cols_ = cols;
    const size_t pixels = rows * cols;

    images_.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        std::vector<uint8_t> raw(pixels);
        f.read(reinterpret_cast<char*>(raw.data()), pixels);
        if (!f) throw std::runtime_error("MnistLoader: truncated image data");
        images_[i].resize(pixels);
        for (size_t j = 0; j < pixels; ++j) {
            images_[i][j] = static_cast<float>(raw[j]) / 255.0f;
        }
    }
}

void MnistLoader::load_labels(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("MnistLoader: cannot open " + path);

    uint8_t header[8];
    f.read(reinterpret_cast<char*>(header), 8);
    if (!f) throw std::runtime_error("MnistLoader: truncated label header in " + path);

    uint32_t magic = read_big_endian_u32(header + 0);
    uint32_t n     = read_big_endian_u32(header + 4);

    if (magic != 0x00000801)
        throw std::runtime_error("MnistLoader: invalid label magic number in " + path);

    labels_.resize(n);
    f.read(reinterpret_cast<char*>(labels_.data()), n);
    if (!f) throw std::runtime_error("MnistLoader: truncated label data");
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
MnistLoader::MnistLoader(const std::string& images_path,
                         const std::string& labels_path) {
    load_images(images_path);
    load_labels(labels_path);
    if (images_.size() != labels_.size())
        throw std::runtime_error("MnistLoader: image/label count mismatch");
}

// ---------------------------------------------------------------------------
// DataSource interface
// ---------------------------------------------------------------------------
size_t MnistLoader::size() const { return images_.size(); }

void MnistLoader::reset() { cursor_ = 0; }

Batch MnistLoader::next_batch(size_t batch_size) {
    const size_t n       = images_.size();
    const size_t pixels  = rows_ * cols_;
    const size_t actual  = std::min(batch_size, n - (cursor_ % n));

    const size_t start = cursor_ % n;
    cursor_ += actual;

    // Build FP32 host buffers.
    std::vector<float>   inp_fp32(actual * pixels);
    std::vector<int32_t> tgt_i32(actual);

    for (size_t i = 0; i < actual; ++i) {
        std::memcpy(inp_fp32.data() + i * pixels,
                    images_[(start + i) % n].data(),
                    pixels * sizeof(float));
        tgt_i32[i] = static_cast<int32_t>(labels_[(start + i) % n]);
    }

    // Convert input to target dtype.
    Tensor inputs  = Tensor::make({actual, pixels}, input_dtype_, Device::CPU);
    Tensor targets = Tensor::make({actual},         DType::Int32,  Device::CPU);

    switch (input_dtype_) {
        case DType::Float32:
            std::memcpy(inputs.data, inp_fp32.data(), actual * pixels * sizeof(float));
            break;
        case DType::BFloat16: {
            auto* dst = static_cast<__nv_bfloat16*>(inputs.data);
            for (size_t i = 0; i < actual * pixels; ++i)
                dst[i] = __float2bfloat16(inp_fp32[i]);
            break;
        }
        case DType::Float16: {
            auto* dst = static_cast<__half*>(inputs.data);
            for (size_t i = 0; i < actual * pixels; ++i)
                dst[i] = __float2half(inp_fp32[i]);
            break;
        }
        default:
            throw std::runtime_error("MnistLoader: unsupported input dtype");
    }

    std::memcpy(targets.data, tgt_i32.data(), actual * sizeof(int32_t));

    // Transfer to output device.
    return { inputs.to(output_device_), targets.to(output_device_) };
}

} // namespace fayn
