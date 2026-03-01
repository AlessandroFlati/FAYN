#pragma once

#include "data_source.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fayn {

// ---------------------------------------------------------------------------
// CifarLoader: reads the CIFAR-10 binary batch format.
//
// File format (each batch file, no header):
//   10 000 records of 3073 bytes each:
//     Byte 0:       label (0-9)
//     Bytes 1-3072: pixels in CHW layout — R[32×32] + G[32×32] + B[32×32]
//
// Training: 5 files (data_batch_1.bin to data_batch_5.bin) = 50 000 samples.
// Test:     1 file  (test_batch.bin)                        = 10 000 samples.
//
// Pixel values are normalised to [0, 1] float32 before dtype conversion.
// Inputs are returned as [batch_size, 3072] BF16 (default) in CHW layout.
// ---------------------------------------------------------------------------
class CifarLoader : public DataSource {
public:
    // cifar_dir: directory containing the CIFAR-10 binary batch files.
    // train: if true, load data_batch_{1..5}.bin; if false, load test_batch.bin.
    CifarLoader(const std::string& cifar_dir, bool train = true);

    Batch  next_batch(size_t batch_size) override;
    void   reset()                       override;
    size_t size()                        const override;

    size_t num_classes() const { return 10; }
    size_t channels()    const { return 3;  }
    size_t height()      const { return 32; }
    size_t width()       const { return 32; }
    size_t input_dim()   const { return 3 * 32 * 32; }  // 3072

private:
    std::vector<std::vector<float>> images_;  // [n, 3072] float32 normalised, CHW
    std::vector<uint8_t>            labels_;  // [n]
    size_t cursor_ = 0;

    void load_file(const std::string& path);
};

} // namespace fayn
