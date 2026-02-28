#pragma once

// ---------------------------------------------------------------------------
// CIW (Cluster-based Input Weight) initialisation for DenseLayer d0.
//
// Replaces the Kaiming uniform random weights with cluster centroids learned
// from training data via mini-batch k-means.  Each row of the weight matrix
// is set to a normalised centroid, giving data-informed feature detectors
// instead of purely random projections.
//
// API:
//   // Load all training images as float32 [N × 784] in [0, 1].
//   auto images = fayn::load_mnist_images_float32("train-images-idx3-ubyte");
//
//   // Initialise d0 for each ensemble member with a different seed.
//   fayn::ciw_init(m.d0_layer, images, /*seed=*/42 + k);
//
// Reference:
//   Huang et al. "Extreme Learning Machines: a survey", Int. J. Mach.
//   Learn. & Cyber., 2011. CIW-ELM achieves ~99% on MNIST with 15 K neurons.
// ---------------------------------------------------------------------------

#include "dense.hpp"
#include "../core/device.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fayn {

// ---------------------------------------------------------------------------
// load_mnist_images_float32
//
// Reads the MNIST IDX3-ubyte file at `path` and returns all N images as a
// flat float32 vector of shape [N, 784], normalised to [0, 1].
// Throws std::runtime_error on any I/O or format error.
// ---------------------------------------------------------------------------
inline std::vector<float> load_mnist_images_float32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("load_mnist_images_float32: cannot open " + path);

    auto read_be32 = [&]() -> int32_t {
        uint8_t b[4];
        f.read(reinterpret_cast<char*>(b), 4);
        return static_cast<int32_t>(
            (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) |
            (uint32_t(b[2]) <<  8) |  uint32_t(b[3]));
    };

    const int32_t magic = read_be32();
    if (magic != 0x00000803)
        throw std::runtime_error("load_mnist_images_float32: bad magic in " + path);

    const int32_t n    = read_be32();
    const int32_t rows = read_be32();
    const int32_t cols = read_be32();
    if (n <= 0 || rows <= 0 || cols <= 0)
        throw std::runtime_error("load_mnist_images_float32: invalid dimensions");

    const size_t dim   = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    const size_t total = static_cast<size_t>(n) * dim;

    std::vector<uint8_t> raw(total);
    f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(total));
    if (!f)
        throw std::runtime_error("load_mnist_images_float32: truncated file " + path);

    std::vector<float> data(total);
    for (size_t i = 0; i < total; ++i)
        data[i] = static_cast<float>(raw[i]) * (1.0f / 255.0f);

    return data;
}

// ---------------------------------------------------------------------------
// kmeans_minibatch (internal)
//
// Runs mini-batch k-means on data [N × D] for M clusters.
// Returns centroids as a flat float32 vector [M × D], row-major.
//
// Algorithm (sklearn mini-batch k-means):
//   1. Initialise M centroids by sampling M distinct rows from data.
//   2. Repeat n_iter times:
//        a. Sample batch_sz rows from data (uniform, with replacement).
//        b. Assign each sample to the nearest centroid (argmin L2²).
//        c. Update each claimed centroid:
//             count_j += 1;  c_j += (1/count_j) * (x - c_j)
//           (equivalent to running mean; early updates dominate).
//   3. Return final centroids.
// ---------------------------------------------------------------------------
inline std::vector<float> kmeans_minibatch(
    const float* data,
    size_t       N,
    size_t       D,
    size_t       M,
    int64_t      seed,
    int          n_iter   = 100,
    int          batch_sz = 256)
{
    if (M > N)
        throw std::invalid_argument("kmeans_minibatch: more clusters than samples");

    std::mt19937_64 rng(static_cast<uint64_t>(seed));

    // Step 1: initialise centroids from M randomly sampled, distinct data rows.
    std::vector<size_t> perm(N);
    std::iota(perm.begin(), perm.end(), 0u);
    std::shuffle(perm.begin(), perm.end(), rng);

    std::vector<float> centroids(M * D);
    for (size_t j = 0; j < M; ++j)
        std::copy(data + perm[j] * D, data + perm[j] * D + D,
                  centroids.data() + j * D);

    std::vector<float>  counts(M, 0.f);
    std::uniform_int_distribution<size_t> dist(0, N - 1);

    // Step 2: mini-batch update loop.
    for (int it = 0; it < n_iter; ++it) {
        // Sample a mini-batch of row indices (with replacement).
        std::vector<size_t> batch(static_cast<size_t>(batch_sz));
        for (auto& idx : batch) idx = dist(rng);

        // Assignment: find nearest centroid for each mini-batch sample.
        std::vector<size_t> assign(static_cast<size_t>(batch_sz));
        for (int bi = 0; bi < batch_sz; ++bi) {
            const float* x       = data + batch[bi] * D;
            float        best_d2 = std::numeric_limits<float>::max();
            size_t       best_j  = 0;
            for (size_t j = 0; j < M; ++j) {
                const float* c  = centroids.data() + j * D;
                float        d2 = 0.f;
                for (size_t d = 0; d < D; ++d) {
                    const float diff = x[d] - c[d];
                    d2 += diff * diff;
                }
                if (d2 < best_d2) { best_d2 = d2; best_j = j; }
            }
            assign[bi] = best_j;
        }

        // Update: running-mean centroid update.
        for (int bi = 0; bi < batch_sz; ++bi) {
            const size_t j   = assign[bi];
            const float* x   = data + batch[bi] * D;
            float*       c   = centroids.data() + j * D;
            counts[j]       += 1.f;
            const float  lr  = 1.f / counts[j];
            for (size_t d = 0; d < D; ++d)
                c[d] += lr * (x[d] - c[d]);
        }
    }

    return centroids;
}

// ---------------------------------------------------------------------------
// ciw_init
//
// Initialise a DenseLayer's input weights with CIW centroids:
//   1. Run mini-batch k-means on `training_data` [N × in_features] with
//      M = layer.out_features() clusters.
//   2. L2-normalise each centroid row.
//   3. Upload to the layer's weight tensor (FP32 if enabled, else BF16).
//
// Parameters:
//   layer         — the DenseLayer to initialise (typically d0, frozen).
//   training_data — flat float32 [N × in_features] in [0, 1].
//   seed          — RNG seed; use a different seed per ensemble member for
//                   diversity (e.g., base_seed + member_index).
//   n_iter        — total mini-batch k-means iterations (default 100).
//   batch_sz      — samples per mini-batch iteration (default 256).
// ---------------------------------------------------------------------------
inline void ciw_init(
    DenseLayer&               layer,
    const std::vector<float>& training_data,
    int64_t                   seed     = 42,
    int                       n_iter   = 100,
    int                       batch_sz = 256)
{
    const size_t D = layer.in_features();
    const size_t M = layer.out_features();

    if (D == 0 || M == 0)
        throw std::invalid_argument("ciw_init: layer has zero dimensions");
    if (training_data.size() % D != 0)
        throw std::invalid_argument("ciw_init: training_data size not divisible by in_features");

    const size_t N = training_data.size() / D;
    if (N == 0)
        throw std::invalid_argument("ciw_init: empty training data");

    // Run mini-batch k-means.
    auto centroids = kmeans_minibatch(training_data.data(), N, D, M,
                                      seed, n_iter, batch_sz);

    // L2-normalise each centroid row.
    for (size_t j = 0; j < M; ++j) {
        float* c      = centroids.data() + j * D;
        float  sum_sq = 0.f;
        for (size_t d = 0; d < D; ++d) sum_sq += c[d] * c[d];
        if (sum_sq > 1e-12f) {
            const float inv = 1.f / std::sqrt(sum_sq);
            for (size_t d = 0; d < D; ++d) c[d] *= inv;
        }
    }

    // Upload to device weight tensor.
    if (layer.has_fp32_weights()) {
        FAYN_CUDA_CHECK(cudaMemcpy(
            layer.weights_fp32().data, centroids.data(),
            M * D * sizeof(float), cudaMemcpyHostToDevice));
    } else {
        // Convert float32 → BF16 via bit truncation (rounds toward zero).
        // Sufficient precision for weight initialisation.
        std::vector<uint16_t> bf16_buf(M * D);
        for (size_t i = 0; i < M * D; ++i) {
            uint32_t bits;
            std::memcpy(&bits, &centroids[i], 4);
            bf16_buf[i] = static_cast<uint16_t>(bits >> 16);
        }
        FAYN_CUDA_CHECK(cudaMemcpy(
            layer.weights().data, bf16_buf.data(),
            M * D * sizeof(uint16_t), cudaMemcpyHostToDevice));
    }
}

} // namespace fayn
