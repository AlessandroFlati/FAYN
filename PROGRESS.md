# FAYN — Development Progress

Progress is recorded as semantic checkpoints: what became possible or true,
not which files changed. Newest entries at the top.

---

## [CP-15] Larger init scale reveals ELM gap — 86% vs 81% Hebbian
**Date:** 2026-02-27

Parameterised d0 random projection init: `init_scale` multiplies the Kaiming
uniform range (`±sqrt(2/fan_in)`). Two new experiments registered:
- `ensemble_mnist_scaled` — Hebbian ensemble with `init_scale = 19.8` (≈ Uniform(−1,1))
- `elm_ensemble_scaled`   — ELM ensemble with `init_scale = 19.8`, same projections

Both variants pin the Kaiming seed counter to 42 via `reset_kaiming_seed(42)` in
`setup()`, ensuring identical d0 projections for a fair comparison.

**Results (K=10, batch_size=512, seed=42, init_scale=19.8):**

| Method | Epochs | Train acc |
|---|---|---|
| ELM ensemble, Kaiming init (CP-14) | 1 | ~78–80% |
| Hebbian ensemble, Kaiming init (CP-13) | 40 | 79.1% |
| Hebbian ensemble, scaled init | 40 | **81.4%** |
| ELM ensemble, scaled init | 1 | **86.3%** |

**Why scaled init helps:**
Pre-activation std rises from 0.27 (Kaiming) to ~4.9 (scale 19.8).
Active ReLU features carry much more discriminative signal.

**Why ELM beats Hebbian by +5% with scaled init (but not with Kaiming):**
With Kaiming features, Hebbian converges to approximately the ELM optimum after
40 epochs.  With larger features, the Hebbian delta-rule update (`ΔW ∝ (target −
output) · input`) is 73× larger in magnitude; the iterative optimisation
oscillates and converges to a suboptimal plateau (~81%) far below the exact
normal-equations solution (~86%).

**Infrastructure additions:**
- `DenseLayer(in, out, bias, init_scale=1.0f)` — 4th optional constructor arg
- `reset_kaiming_seed(uint64_t)` — exposed from `dense.hpp` for reproducibility
- `seed` parameter on both ensemble experiment constructors

---

## [CP-14] ELM readout implemented — confirms Hebbian converges to optimal solution
**Date:** 2026-02-27

`ELMEnsembleExperiment` replaces iterative HebbianUpdater with the one-shot
Extreme Learning Machine readout: normal equations solved analytically in one
pass over the training data.

**Algorithm per member k:**
1. Collect `H_k [N, 256]` FP32 by running all N training samples through frozen
   `d0_k + ReLU` (BF16→FP32 conversion on CPU per batch).
2. Build `T [N, 10]` one-hot FP32 (shared across all members).
3. GPU normal equations via cuBLAS SGEMM:
   `A_k = H_k^T H_k  [256, 256]`  and  `b_k = H_k^T T  [256, 10]`
4. CPU Gaussian elimination (`A_k W_k = b_k`, 256×256 — trivially fast).
5. Transpose + BF16 cast → write `W_k` to `d1_k->weights()`.

**Results (K=10, batch_size=512):**
- Training accuracy: **~76–80%** depending on random seed (d0 projection quality)
- Per-epoch variance ±2–3% from non-deterministic BF16 cuBLAS accumulation
- Compare: Hebbian ensemble 40 epochs → 79.1%

**Key finding:** ELM ≈ Hebbian. After 40 epochs of SupervisedHebbian, the
weights have converged to approximately the optimal linear readout. The
bottleneck is **feature quality**, not readout rule:
- Kaiming uniform scale ≈ `sqrt(2/784) ≈ 0.05` → pre-activation std ≈ 0.27
- ~50% of ReLU features are dead per sample → limited discriminative capacity
- With standard ELM init (e.g. `Uniform(-1, 1)`), accuracy would be ~90-95%

**Next**: improve feature representation — larger init scale, deeper projections,
or a principled feature-selection method (e.g. greedy ELM).

---

## [CP-13] Ensemble of K random projections reaches ~79% — variance reduction confirmed
**Date:** 2026-02-27

`EnsembleHebbianMnistExperiment` runs K=10 independent networks (each: frozen
random d0 + SupervisedHebbian d1) and combines logits by summation before argmax.

**40-epoch learning curve (K=10, batch_size=512, lr=0.01):**
```
epoch   0  acc=0.3075        ← slow start (larger batch)
epoch   5  acc=0.5493
epoch  10  acc=0.6464
epoch  20  acc=0.7659
epoch  25  acc=0.7823        ← surpasses single-network 78% plateau
epoch  29  acc=0.7900        ← peak vicinity
epoch  36  acc=0.7934        ← peak observed
epoch  39  acc=0.7912        ← plateau: oscillates ~79.1% ± 0.003
```

**Comparison:**

| Configuration | Plateau accuracy | Epochs to plateau |
|---|---|---|
| Single network (batch=128) | ~78.0% | 5–6 |
| Ensemble K=10 (batch=512) | ~79.1% | 25–30 |
| Theoretical ELM ceiling (same arch) | ~93–96% | 1 pass |

**Key finding:** Variance reduction from K=10 independent random projections
yields ~+1% accuracy over the single network. The improvement is real but modest
because the dominant bottleneck is **readout sub-optimality** (nearest-centroid
learning vs. the globally-optimal linear separator), not feature diversity.

With a better readout (delta-rule, pseudoinverse, perturbation learning) the
ensemble gap would widen — each member would be closer to ELM ceiling (~95%) and
the ensemble average would reduce residual variance further.

**Performance on WSL2 (sm_120, CUDA 12.9):**
K=10 ensemble runs at ~20 seconds/epoch with batch_size=512. The bottleneck is
CUDA stream synchronization overhead per layer per batch (WSL2 incurs ~5ms per
`cudaStreamSynchronize` call, × 3 layers × K members = ~150ms/batch at batch=128;
reduced to ~40ms/batch at batch=512 via fewer batches). Stats collection is
disabled for ensemble members via `set_compute_stats(false)` to avoid K×3×5
blocking D2H copies per batch.

**Infrastructure added:**
- `experiments/ensemble_mnist/ensemble_mnist.{hpp,cpp}` — `EnsembleHebbianMnistExperiment`
- `src/core/tensor.hpp/cpp` — `Tensor::borrow()` (non-owning view, zero-copy input sharing)
- `src/core/layer.hpp` — `Layer::set_compute_stats(bool)` virtual (default no-op)
- `src/ops/dense.hpp/cu` — `DenseLayer::set_compute_stats(bool)` override; skip D2H stats when false (still stream-syncs for correctness)
- `src/ops/activations.cu` — `ActivationLayer::set_compute_stats(bool)` override; same pattern
- `src/ops/dense.cu` — atomic seed counter in `kaiming_uniform_bf16` (ensures each ensemble member gets a distinct random d0 projection)
- `tools/registry.hpp` — `FAYN_REGISTER_EXPERIMENT` macro changed to 3-arg form (name, class, unique_id) to avoid GCC `__COUNTER__` prescan limitation

**What is now possible:** Population experiments (evolutionary, perturbation) can
run K independent networks efficiently. The `borrow()` + `set_compute_stats(false)`
pattern eliminates per-member D2D copy overhead and monitoring overhead.
The next frontier: better readout rules per member (delta-rule, ELM pseudoinverse)
to close the ~15% gap to backprop.

---

## [CP-12] Supervised Hebbian readout achieves 78% on MNIST without backprop
**Date:** 2026-02-27

`HebbianMnistExperiment` reaches ~78% train accuracy using:
- **d0** (784→256): frozen Kaiming random projection — no updates. Random ReLU features
  provide adequate discriminability.
- **d1** (256→10): `SupervisedHebbian` mode — one-hot class labels used as post-synaptic
  signal. Each class weight row is pulled toward hidden representations of that class
  (nearest-centroid learning on the L2-normalised sphere). No backpropagation at any
  point in the pipeline.

**Full 50-epoch learning curve (batch_size=128, lr=0.01):**
```
epoch   0  acc=0.3756        ← epoch 0: random init, some structure already visible
epoch   1  acc=0.6226        ← most learning happens here
epoch   2  acc=0.7156
epoch   3  acc=0.7515
epoch   4  acc=0.7657
epoch   5  acc=0.7705        ← convergence zone begins
epoch   9  acc=0.7777
epoch  10  acc=0.7793
epoch  26  acc=0.7805        ← peak observed
epoch  49  acc=0.7789        ← plateau: oscillates within ±0.003 indefinitely
```

**Benchmark comparison — MNIST 784→256(ReLU)→10 architecture:**

| Method | Update rule | Readout | Acc (train) | Acc (test) | Epochs to plateau |
|---|---|---|---|---|---|
| **FAYN SupervisedHebbian** | none (d0 frozen) + Hebbian nearest-centroid | nearest centroid on L2 sphere | **~78%** | est. ~75–76% | 5–6 |
| Random chance | — | — | 10% | 10% | — |
| Nearest centroid (raw pixels, no hidden layer) | — | centroid in pixel space | — | ~82% | 1 pass |
| Linear classifier (raw pixels) | SGD | logistic regression | — | ~92% | ~10–30 |
| ELM (same arch: random d0, optimal linear readout) | none (d0 frozen) + pseudoinverse | linear (exact optimum) | — | ~93–96% | 1 pass |
| Feedback Alignment (same arch) | FA approximation | linear | — | ~94% | ~50 |
| MLP + backprop (same arch, SGD) | backprop | linear (softmax) | ~99% | ~97–98% | ~20–50 |

Notes:
- FAYN and ELM use identical d0 (random ReLU projection). The ~15–18% gap to ELM is the cost
  of using online nearest-centroid vs. the globally-optimal linear readout (pseudoinverse).
  Online Hebbian nearest-centroid converges to the class centroid directions, not the
  optimal linear separator.
- "Test" column for FAYN is an estimate; no test-set evaluation is implemented yet.
- Literature test numbers assume standard MNIST 10k test set; train numbers vary by run.

**Key negative finding:** Local Hebbian on d0 (unsupervised feature learning) hurts:
it learns PCA-like features that capture input variance, not class discriminability,
AND continuously shifts representations so d1 chases a moving target. Even at 10×
lower lr for d0, training never escapes random chance.

**Gap analysis vs. ELM ceiling (~95%):**
The ~17% gap breaks down as:
1. **Readout sub-optimality** (~10–12%): Hebbian nearest-centroid vs. optimal linear.
   Fix: replace Hebbian with a delta-rule or use a small number of perturbation steps.
2. **Feature sub-optimality** (~5–6%): Kaiming random features vs. trained features.
   Fix: supervised feature learning (perturbation, contrastive Hebbian, BCM).

**What is now possible:** Backprop-free learning is demonstrated at a meaningful
accuracy level. `RoutingMode::SupervisedHebbian` and `one_hot_encode()` are production-ready.
The next frontier is closing the gap to ELM with a better readout update rule.

**New infrastructure:**
- `src/ops/one_hot.hpp/cu` — `one_hot_encode(labels, C)` CUDA kernel
- `src/ops/dense.hpp` — `set_target_activations()` / `has_target_activations()` / `target_activations()`
- `src/ops/hebbian_updater.hpp` — `RoutingMode::SupervisedHebbian`
- 2 new smoke tests (12 total, all pass): `one_hot_encode`, `supervised_hebbian`

---

## [CP-11] HebbianMnistExperiment runs end-to-end on MNIST
**Date:** 2026-02-27

The full training loop runs on real MNIST data (60,000 train images, BF16 on CUDA).
5 epochs complete in seconds on sm_120 (Blackwell). Output:
```
epoch   0  acc=0.1049  loss_proxy=0.8951
epoch   1  acc=0.1055  loss_proxy=0.8945
epoch   2  acc=0.1035  loss_proxy=0.8965
epoch   3  acc=0.1049  loss_proxy=0.8951
epoch   4  acc=0.1052  loss_proxy=0.8948
```
~10.5% accuracy (random-chance baseline for 10 classes) confirms the pipeline is wired correctly but the reward-modulated Hebbian rule has not yet learned. Expected — the Hebbian rule modulated by `-cross_entropy` needs tuning.

Also fixed: CMake CUDA device-link propagation bug. Pure-CXX static libraries that transitively depend on CUDA separable-compiled libs (fayn_stats, fayn_ops) were receiving spurious device-link steps in CMake 3.28, embedding `cmake_device_link.o` stubs in their archives. This caused nvlink to miss device registrations at the runner link step. Fixed by setting `CUDA_RESOLVE_DEVICE_SYMBOLS OFF` on all pure-CXX intermediary libraries.

**What is now possible:** End-to-end experiments can be launched with `./fayn <name> --epochs N`. The infrastructure is validated on real data. Next steps: tune the Hebbian learning rule, add eligibility traces, or run for more epochs to observe learning.

---

## [CP-10] Reward pipeline implemented: event-driven training loop
**Date:** 2026-02-27

The reward pipeline is fully wired. The training loop now only calls `graph.forward()` and emits `RewardEvent` — all weight updates are side effects of EventBus subscribers.

Deliverables:
- **`src/core/loss.hpp`** — `LossFn` type alias + `cross_entropy`, `accuracy`, `mse` (inline, CPU-side, handle BF16 output + Int32 label tensors)
- **`src/ops/hebbian_updater.hpp`** — `HebbianUpdater` EventBus subscriber; subscribes to `RewardEvent` (sync), applies `ΔW ∝ reward × pre × post` per registered layer; `RoutingMode::Local` (no reward scaling) and `RoutingMode::Global` (reward-scaled lr); unsubscribes in destructor
- **`src/stats/events.hpp`** — `RewardEvent` gains a `name` field for multi-signal support
- **`HebbianMnistExperiment`** refactored: `run_epoch()` emits `RewardEvent{reward=-cross_entropy}` per batch; `HebbianUpdater` fires synchronously; no direct `hebbian_update()` calls in the loop
- **2 new smoke tests** (10 total, all pass): `loss_cross_entropy`, `hebbian_updater_fires`

**What is now possible:** Any experiment can plug in a different loss function, routing mode, or update subscriber without changing the training loop. The infrastructure is ready for perturbation, evolutionary, and contrastive Hebbian experiments.

---

## [CP-09] Reward/loss pipeline design complete
**Date:** 2026-02-27

The role of the loss/reward signal in FAYN is now fully specified: the framework
uses a loss or reward scalar to *modulate* weight updates — what it eliminates is
backpropagation, not supervision.

Decisions made (full rationale in `docs/DESIGN.md` §2.31–2.38):
- **Signal forms**: per-sample, per-batch, delayed RL, and multiple parallel named signals are all supported. `RewardEvent` carries the scalar through the EventBus.
- **Update rules**: all four paradigms are in scope — reward-modulated Hebbian, perturbation, evolutionary selection, and contrastive Hebbian.
- **Loss functions**: cross-entropy, MSE, accuracy/ranking, environment reward, and custom callables via `LossFn = std::function<float(output, target)>`.
- **Signal routing**: layers can operate in Local (pure Hebbian, no reward), Global (reward-scaled Hebbian), or Hierarchical (eligibility trace) mode.
- **Update timing**: asynchronous / event-driven — all weight updates are side effects of EventBus subscribers, never called directly in the training loop.
- **Temporal credit (RL)**: eligibility traces (`e[t] = λ·e[t-1] + pre·post`; `ΔW = lr·r·e`).
- **Readout**: last graph node (sink), trained with the same rules as all other layers.
- **Transport**: `RewardEvent` for internal subscribers + `run_epoch()` return value for the CLI runner.

**What is now possible:** The learning paradigm can be designed at a level of
abstraction above individual kernels. An experiment only needs to emit a
`RewardEvent` after each batch; the choice of Hebbian vs. perturbation vs.
evolutionary update is a subscriber configuration, not a code change.

---

## [CP-08] Smoke test suite: all 8 tests pass on Linux/WSL
**Date:** 2026-02-27

The full smoke-test suite (`tools/smoke_test.cu`) now passes cleanly on the
Linux/WSL build (sm_120, CUDA 12.9, `/usr/local/cuda-12.9`).

Tests covered:
1. `tensor_roundtrip` — host↔device round-trip for a BF16 tensor
2. `relu_bf16` — in-place ReLU on a BF16 tensor; negative values clamped
3. `dense_forward_shape` — DenseLayer produces correct output shape
4. `activation_stats_all_dead` — LayerStats detects 100% dead neurons
5. `activation_stats_no_dead` — LayerStats detects 0% dead neurons
6. `hebbian_weight_change` — a single hebbian_update() call visibly changes weights
7. `eventbus_sync_dispatch` — sync EventBus subscriber fires before emit() returns
8. `graph_forward_mlp` — end-to-end forward pass through a 3-node MLP graph

**What is now possible:** Any new layer, op, or graph change can be sanity-checked
against all eight tests before merging. The tests also serve as a regression guard
for the CUDA stream, graph execution, and stats subsystems.

---

## [CP-07] Hebbian learning is functional on Blackwell GPU
**Date:** 2026-02-27

`hebbian_update()` (outer-product weight update) and `normalize_weights_rows()`
(per-row L2 normalisation) produce correct, non-zero results on sm_120.

The implementation uses a custom fused CUDA kernel (`hebbian_fused_kernel`) that
accumulates the batch outer product in FP32 and updates weights in BF16 in one
pass. This bypasses cuBLAS entirely for the update step.

**What is now possible:** Hebbian learning experiments can run. A `DenseLayer`
with `set_cache_activations(true)` captures pre/post activations; a training loop
can call `hebbian_update(weights, last_input, last_output, lr)` after each forward
pass and `normalize_weights_rows(weights)` periodically.

---

## [CP-06] Activation statistics pipeline is end-to-end
**Date:** 2026-02-27

Every `DenseLayer` and `ActivationLayer` forward call now computes per-neuron
statistics (mean, variance, absolute mean, dead ratio) on the GPU, updates EMA
state, and emits a fully populated `ActivationEvent` to the `EventBus`.

Statistics are accumulated via a custom CUDA reduction kernel
(`compute_activation_stats`) using one block per neuron with FP32 shared-memory
tree reduction. EMA state lives in `LayerStats` (per-layer) using
`EmaVector`/`EmaScalar` with configurable alpha.

**What is now possible:** The `MutationEngine` can fire topology rules based on
live activation statistics without any additional instrumentation. Downstream
analysis (e.g., dead neuron fraction over training) is available in `StatsSnapshot`
on every forward step.

---

## [CP-05] Global EventBus connects layers to subscribers
**Date:** 2026-02-27

The singleton `EventBus` is operational. Layers emit typed events
(`ActivationEvent`, `EpochEvent`, `RewardEvent`); subscribers register handlers
with either `DispatchMode::Sync` (inline) or `DispatchMode::Async` (background
thread). `flush()` drains the async queue synchronously.

**What is now possible:** Multiple independent consumers (Logger, MutationEngine,
future wandb exporter) can react to forward-pass events without any coupling to
layer code. New event types and subscribers can be added at any time.

---

## [CP-04] DenseLayer (GEMM + bias) is operational
**Date:** 2026-02-27

`DenseLayer` performs `y = x W^T + b` using `cublasGemmEx` with BF16 inputs and
FP32 accumulation. Weights are Kaiming-uniform initialised. Bias is FP32 and
added via a custom `bias_add_kernel`.

The row-major ↔ column-major duality is used: a row-major `M[m,n]` is passed to
cuBLAS as a column-major `M^T[n,m]` with leading dimension `n`, yielding
`(OP_T, OP_N)` as the correct transpose pair for `y = x W^T`.

**What is now possible:** Multi-layer MLP graphs can execute a full forward pass.
DenseLayer is the workhorse op for all MLP-style experiments including
`HebbianMnistExperiment`.

---

## [CP-03] Graph executes forward passes in topological order
**Date:** 2026-02-27

`Graph::forward()` accepts `{node_id, Tensor}` pairs as inputs, runs all active
nodes in Kahn-sorted order, and returns the tensors at sink nodes (nodes with no
active outgoing edges).

Topology mutation (`add_node`, `add_edge`, `remove_node`, `remove_edge`,
`rewire_edge`) is live. `clone()` produces a shallow copy (shared weights) for
population experiments.

**What is now possible:** Arbitrary DAG networks — not just sequential chains —
can be built in code and executed. Topology can be mutated between forward passes
without rebuilding the network from scratch.

---

## [CP-02] Core tensor abstraction is stable
**Date:** 2026-02-27

`Tensor` is a non-copyable, movable N-dimensional strided buffer owning a
host or device allocation. `Tensor::make(shape, dtype, device)` is the canonical
factory. `Tensor::to(Device)` deep-copies across host/device.

`DType` supports `Float32`, `BFloat16`, `Float16`, `Int32`, `Int8`.
`FAYN_CUDA_CHECK` and `FAYN_CUBLAS_CHECK` macros enforce fail-fast error handling
on all CUDA calls.

**What is now possible:** Any CUDA kernel or cuBLAS call can operate on
`Tensor` objects. Shape-checked layer constructors and runtime dimension
validation are possible against `tensor.shape`.

---

## [CP-01] Project scaffolded from design interview
**Date:** 2026-02-27

Repository structure, CMake build system, and module boundaries established
from a 30-question design interview (recorded in `docs/DESIGN.md`). All
architectural decisions — precision strategy, graph representation, EventBus
dispatch model, CUDA stream policy, experiment registration pattern, dependency
policy — are documented with explicit rationale.

Initial CMake targets created: `fayn_core`, `fayn_stats`, `fayn_cuda`,
`fayn_ops`, `fayn_topology`, `fayn_io`, `fayn_experiments`, `fayn` (runner),
`fayn_smoke` (tests).

**What is now possible:** New modules and experiments can be added within a
well-defined dependency graph. The build system handles CUDA separable compilation
and FetchContent dependencies (`nlohmann_json`, `fmt`) automatically.
