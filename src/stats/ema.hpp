#pragma once

#include <cstddef>
#include <vector>
#include <cmath>
#include <stdexcept>

namespace fayn {

// ---------------------------------------------------------------------------
// EmaScalar: exponential moving average of a single scalar value.
//
//   mu_t  = alpha * x_t + (1 - alpha) * mu_{t-1}
//   var_t = alpha * (x_t - mu_{t-1})^2 + (1 - alpha) * var_{t-1}
//
// Bias correction is applied during the warm-up phase so that early samples
// do not produce artificially small estimates.
// ---------------------------------------------------------------------------
class EmaScalar {
public:
    explicit EmaScalar(float alpha = 0.05f) : alpha_(alpha) {}

    void update(float x) {
        ++n_;
        // Welford-style EMA with bias correction.
        float old_mu = mu_;
        mu_          = alpha_ * x + (1.0f - alpha_) * mu_;
        var_         = (1.0f - alpha_) * (var_ + alpha_ * (x - old_mu) * (x - old_mu));
    }

    float mean()     const noexcept { return mu_; }
    float variance() const noexcept { return var_; }
    float stddev()   const noexcept { return std::sqrt(var_); }
    size_t count()   const noexcept { return n_; }

    void reset() { mu_ = 0.0f; var_ = 0.0f; n_ = 0; }

private:
    float  alpha_ = 0.05f;
    float  mu_    = 0.0f;
    float  var_   = 0.0f;
    size_t n_     = 0;
};

// ---------------------------------------------------------------------------
// EmaVector: per-element EMA over a fixed-length vector.
// Intended for per-neuron activation statistics.
// ---------------------------------------------------------------------------
class EmaVector {
public:
    EmaVector() = default;

    explicit EmaVector(size_t size, float alpha = 0.05f)
        : alpha_(alpha), mu_(size, 0.0f), var_(size, 0.0f) {}

    // Update with a batch of per-neuron values.
    void update(const float* values, size_t n) {
        if (n != mu_.size())
            throw std::invalid_argument("EmaVector::update: size mismatch");
        for (size_t i = 0; i < n; ++i) {
            float old_mu = mu_[i];
            mu_[i]  = alpha_ * values[i] + (1.0f - alpha_) * mu_[i];
            var_[i] = (1.0f - alpha_) * (var_[i] + alpha_ * (values[i] - old_mu) * (values[i] - old_mu));
        }
        ++n_;
    }

    const std::vector<float>& mean()     const noexcept { return mu_; }
    const std::vector<float>& variance() const noexcept { return var_; }
    size_t                    size()     const noexcept { return mu_.size(); }
    size_t                    count()    const noexcept { return n_; }

    void resize(size_t n, float alpha = 0.05f) {
        alpha_ = alpha;
        mu_.assign(n, 0.0f);
        var_.assign(n, 0.0f);
        n_ = 0;
    }

    void reset() {
        std::fill(mu_.begin(),  mu_.end(),  0.0f);
        std::fill(var_.begin(), var_.end(), 0.0f);
        n_ = 0;
    }

private:
    float              alpha_ = 0.05f;
    std::vector<float> mu_;
    std::vector<float> var_;
    size_t             n_     = 0;
};

} // namespace fayn
