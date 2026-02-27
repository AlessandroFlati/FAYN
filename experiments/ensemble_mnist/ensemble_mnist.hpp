#pragma once

#include "../experiment.hpp"
#include "src/ops/dense.hpp"
#include "src/ops/activations.hpp"
#include "src/ops/hebbian_updater.hpp"
#include "src/ops/one_hot.hpp"
#include "src/io/mnist_loader.hpp"

#include <memory>
#include <string>
#include <vector>

namespace fayn {

// ---------------------------------------------------------------------------
// EnsembleHebbianMnistExperiment
//
// Topology: K independent networks, each:
//   Dense(784, 256, bias) -> ReLU -> Dense(256, 10, no bias)
//
// Learning rule: identical to HebbianMnistExperiment for each member.
//   d0: frozen Kaiming random projection (independent random seed per member)
//   d1: SupervisedHebbian — one-hot targets as post-synaptic signal
//       (nearest-centroid learning on the L2-normalised sphere)
//
// Ensemble inference: sum logits from all K members, then argmax.
// Each member's random d0 covers a different subspace; summing votes reduces
// variance, pushing ensemble accuracy toward the ELM ceiling (~93-96%).
//
// Configuration:
//   num_networks    - number of ensemble members (default 10)
//   lr              - Hebbian learning rate (default 0.01)
//   mnist_dir       - directory containing the four MNIST binary files
//   normalize_every - normalise d1 weight rows every N batches (default 1)
// ---------------------------------------------------------------------------
class EnsembleHebbianMnistExperiment : public Experiment {
public:
    // seed: if >= 0, resets the global Kaiming seed counter to this value
    //       before building members — ensures reproducible d0 projections.
    // normalize_pre: if true, L2-normalise each pre-synaptic feature vector
    //       before the Hebbian outer product (makes H^T H ≈ (N/d)·I).
    // lr_final: if >= 0, apply cosine decay from lr to lr_final over all
    //       training steps; if < 0, use constant lr.
    explicit EnsembleHebbianMnistExperiment(
        const ExperimentConfig& cfg,
        const std::string&      mnist_dir       = "data/mnist",
        float                   lr              = 0.01f,
        int                     num_networks    = 10,
        int                     normalize_every = 1,
        float                   d0_init_scale   = 1.0f,
        int64_t                 seed            = -1,
        bool                    normalize_pre   = false,
        float                   lr_final        = -1.f,
        bool                    row_normalize   = true,
        float                   weight_decay    = 0.f,
        int                     hidden_dim      = 256);

protected:
    void  setup()                  override;
    float run_epoch(size_t epoch)  override;

private:
    struct Member {
        std::unique_ptr<Graph>          graph;
        std::shared_ptr<DenseLayer>     d0;
        std::shared_ptr<DenseLayer>     d1;
        std::unique_ptr<HebbianUpdater> updater;
    };

    std::string          mnist_dir_;
    float                lr_              = 0.01f;
    int                  num_networks_    = 10;
    int                  normalize_every_ = 1;
    float                d0_init_scale_   = 1.0f;
    int64_t              seed_            = -1;
    bool                 normalize_pre_   = false;
    float                lr_final_        = -1.f;
    bool                 row_normalize_   = true;
    float                weight_decay_    = 0.f;
    int                  hidden_dim_      = 256;
    size_t               step_            = 0;

    std::vector<Member>  members_;
};

// ---------------------------------------------------------------------------
// ELMEnsembleExperiment
//
// Same K-member topology as EnsembleHebbianMnistExperiment, but the d1
// readout weights are computed analytically via the normal equations:
//
//   W_k = (H_k^T H_k)^{-1} H_k^T T
//
// where H_k is the [N, 256] matrix of post-ReLU activations from frozen d0_k
// and T is the [N, 10] one-hot label matrix.  One pass over the training set
// per member; no iterative updates.
//
// Expected training accuracy: ~93-96% (each member near the single-network
// ELM ceiling; ensemble further reduces variance).
// ---------------------------------------------------------------------------
class ELMEnsembleExperiment : public Experiment {
public:
    explicit ELMEnsembleExperiment(
        const ExperimentConfig& cfg,
        const std::string& mnist_dir  = "data/mnist",
        int num_networks              = 10,
        float d0_init_scale           = 1.0f,
        int64_t seed                  = -1,
        int hidden_dim                = 256);
    ~ELMEnsembleExperiment() override;

protected:
    void  setup()                  override;
    float run_epoch(size_t epoch)  override;

private:
    // Run one full pass per member: collect H_k, solve normal equations,
    // write optimal W_k to d1->weights().
    void elm_fit();

    struct Member {
        std::unique_ptr<Graph>      graph;
        std::shared_ptr<DenseLayer> d0;
        std::shared_ptr<DenseLayer> d1;
    };

    std::string         mnist_dir_;
    int                 num_networks_;
    float               d0_init_scale_;
    int64_t             seed_;
    int                 hidden_dim_  = 256;
    std::vector<Member> members_;
    bool                fitted_  = false;
    cublasHandle_t      cublas_  = nullptr;
};

} // namespace fayn
