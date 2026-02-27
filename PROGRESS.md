# FAYN — Development Progress

Progress is recorded as semantic checkpoints: what became possible or true,
not which files changed. Newest entries at the top.

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
