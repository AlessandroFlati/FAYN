#pragma once

#include "data_source.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fayn {

// ---------------------------------------------------------------------------
// MnistLoader: reads the original MNIST binary format.
//
// File format (IDX):
//   images: magic(4) | n_images(4) | rows(4) | cols(4) | pixels[n*784] uint8
//   labels: magic(4) | n_labels(4) | labels[n]                           uint8
//
// Pixel values are normalised to [0, 1] float32 before dtype conversion.
// ---------------------------------------------------------------------------
class MnistLoader : public DataSource {
public:
    MnistLoader(const std::string& images_path,
                const std::string& labels_path);

    Batch  next_batch(size_t batch_size) override;
    void   reset()                       override;
    size_t size()                        const override;

    size_t num_classes()  const { return 10; }
    size_t image_height() const { return rows_; }
    size_t image_width()  const { return cols_; }
    size_t input_dim()    const { return rows_ * cols_; }

private:
    std::vector<std::vector<float>> images_;  // [n, H*W] float32 normalised
    std::vector<uint8_t>            labels_;  // [n]

    size_t rows_   = 0;
    size_t cols_   = 0;
    size_t cursor_ = 0;

    static uint32_t read_big_endian_u32(const uint8_t* p);
    void load_images(const std::string& path);
    void load_labels(const std::string& path);
};

} // namespace fayn
