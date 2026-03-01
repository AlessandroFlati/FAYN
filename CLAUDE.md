# FAYN — Claude Code Instructions

## Project Overview

FAYN is a C++20 + CUDA research framework for **forward-only neural networks**.
No backpropagation. Explores Hebbian, topology-mutation, perturbation-based, and
evolutionary learning paradigms on Blackwell (RTX 5090 / sm_120) GPU hardware.

Full design rationale lives in `docs/DESIGN.md`.

---

## Build

### Primary build: Linux / WSL2

```bash
# One-shot configure + build
./build-linux.sh

# Or manually:
export PATH="/usr/local/cuda-12.9/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH"
mkdir -p build-linux && cd build-linux
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo \
         -DCMAKE_CUDA_ARCHITECTURES=120 \
         -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc
cmake --build . -j$(nproc)
```

### Run smoke tests

```bash
LD_LIBRARY_PATH="/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH" \
  ./build-linux/tools/fayn_smoke
```

All 12 tests must pass before committing. If output is missing due to buffering,
prefix with `stdbuf -oL -eL`.

### Windows MSVC build (secondary)

Use the VS-bundled cmake from WSL:
```bash
"/mnt/c/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE/\
CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe" \
  --build "$(wslpath -w /mnt/c/Users/aless/PycharmProjects/FAYN/build)" \
  --config RelWithDebInfo
```

The Windows build last had an unresolved exit-code-9 crash (stale intermediates);
the Linux build is canonical.

---

## Hardware & CUDA Requirements

- GPU: RTX 5090 (Blackwell GB202, sm_120)
- CUDA Toolkit: ≥ 12.8 (WSL: `/usr/local/cuda-12.9`)
- `CMAKE_CUDA_ARCHITECTURES=120` — no older GPU compatibility maintained
- PATH / LD_LIBRARY_PATH for CUDA are already in `~/.bashrc` (lines 157–158)

---

## Repo Structure

```
FAYN/
  src/
    core/       # tensor.hpp/cpp, graph.hpp/cpp, layer.hpp, device.hpp, thread_pool.hpp
    stats/      # event_bus.hpp/cpp, events.hpp, ema.hpp, activation_stats.hpp/cu
    cuda/       # stream_pool.hpp/cpp
    ops/        # activations.hpp/cu, dense.hpp/cu, hebbian.hpp/cu
    topology/   # mutation_ops.hpp, mutation_engine.hpp/cpp
    io/         # data_source.hpp, mnist_loader.hpp/cpp, logger.hpp/cpp
  experiments/  # experiment.hpp/cpp (base class), hebbian_mnist/
  tools/        # runner.cpp, registry.hpp, smoke_test.cu
  docs/         # DESIGN.md (full architecture Q&A)
  build-linux/  # Linux build artifacts (gitignored)
  build/        # Windows MSVC build artifacts (gitignored)
  build-linux.sh
  PROGRESS.md
```

## CMake Library Targets

| Target | Contents |
|---|---|
| `fayn_core` | tensor, graph, layer, thread_pool |
| `fayn_stats` | EventBus, EMA, events, activation_stats (CUDA) |
| `fayn_cuda` | StreamPool |
| `fayn_ops` | activations, dense, hebbian (all CUDA) |
| `fayn_topology` | MutationEngine, MutationOps |
| `fayn_io` | DataSource, MnistLoader, Logger |
| `fayn_experiments` | Experiment base + concrete subclasses |
| `fayn` (exe) | CLI runner |
| `fayn_smoke` (exe) | Smoke test suite |

---

## Coding Conventions

- **Language**: C++20, CUDA 12.x. All code and comments in English.
- **Precision**: BF16 for layer inputs/outputs/weights; FP32 for stats, bias, EMA.
- **Error handling**: Fail fast. Use `std::runtime_error` / `std::invalid_argument`.
  Never add silent fallbacks or default values.
- **CUDA calls**: Always wrap with `FAYN_CUDA_CHECK(expr)` and `FAYN_CUBLAS_CHECK(expr)`.
- **Streams**: Always use `StreamPool::Guard` to acquire a stream. Never use the
  default (null) stream in production paths.
- **Map insertions**: Use `insert_or_assign`, not `emplace`, when the key may already
  exist and the value must be overwritten (e.g., `Graph::forward()` activations map).
- **Layer IDs**: Assigned by `Graph::add_node()`; never set manually elsewhere.
- **EventBus**: `emit()` is thread-safe. Mutation subscribers use `DispatchMode::Sync`;
  logger uses `DispatchMode::Async`. Call `EventBus::flush()` at experiment end.
- No emoji anywhere in C++ code, comments, or output.

---

## Key Architecture Decisions

### Tensor
- Strided N-dim, non-copyable, movable. Owns its buffer via RAII (`cudaFree` / `free`).
- `Tensor::to(Device)` always deep-copies (no unified memory).
- `TensorView` is a POD struct (CUDA-safe), max 8 dims.

### Graph
- Adjacency list: `Node { LayerPtr; out_edges; active }`, `Edge { src; dst; active }`.
- Node/edge IDs are stable indices; removals set `active=false` (no compaction).
- Topological execution via Kahn's algorithm (active elements only).
- `clone()` shares `LayerPtr` instances (weights not deep-copied) for population use.

### Layer
- Abstract base with `virtual Tensor forward(const Tensor&)`.
- Each concrete layer emits `ActivationEvent` to `EventBus` after computing output.
- `LayerPtr = std::shared_ptr<Layer>`.

### DenseLayer
- Weight GEMM: `cublasGemmEx` with `(OP_T, OP_N)` pattern (row-major duality trick).
- Weights: BF16 `[out, in]`; bias: FP32 `[out]`.
- Kaiming uniform init (FP32 on host → cast to BF16 → upload).
- Optional activation caching via `set_cache_activations(true)` / `last_input()` / `last_output()`.
  Caching uses `cudaMemcpyAsync` on the **same non-blocking stream** as the GEMM.

### Hebbian Update
- `hebbian_update()`: custom fused kernel `hebbian_fused_kernel` — computes outer product
  and weight delta in a single pass, no cuBLAS.
- **Why custom kernel**: `cublasGemmEx (OP_N, OP_T)` with BF16 silently produces zeros
  on Blackwell sm_120 (Windows CUDA 12.9). `(OP_T, OP_N)` works; `(OP_N, OP_T)` does not.
- `normalize_weights_rows()`: per-row L2 norm via block-reduce kernel.

### StreamPool
- Singleton holding 8 `cudaStreamNonBlocking` streams.
- **Critical**: non-blocking streams do NOT synchronize with the null/default stream.
  Any `cudaMemcpy` (null-stream) issued after a kernel on a non-blocking stream may
  execute before the kernel finishes. Always use `cudaMemcpyAsync` with the same stream.

### EventBus
- Singleton. One background thread drains the async queue.
- `flush()` posts a sentinel future and waits; guarantees all async events are processed.
- `std::promise` in async lambdas must be captured via `shared_ptr` (lambdas must be
  `CopyConstructible` for `std::function`).

### MutationEngine
- Subscribes to `ActivationEvent` (sync). Evaluates `MutationThreshold` rules in order.
- First-match-wins. At most one rule fires per event.
- `AddNode`/`SplitNode` proposals are re-emitted as `MutationProposalEvent` because
  the engine does not hold layer factory knowledge.

---

## Known Platform Gotchas

| Issue | Context | Fix |
|---|---|---|
| `cublasGemmEx (OP_N, OP_T)` BF16 → zeros | Windows CUDA 12.9, sm_120 | Use custom kernel instead |
| `cudaStreamNonBlocking` + `cudaMemcpy` race | Any | Use `cudaMemcpyAsync` with same stream |
| `emplace` silently fails on existing key | `std::unordered_map` | Use `insert_or_assign` |
| Stale `.device-link.obj` hash mismatch | CUDA separable compilation | Delete intermediate dir, rebuild |
| `std::promise` not `CopyConstructible` | `std::function` lambda capture | Capture via `shared_ptr` |
| C++20 template-lambda `[&]<typename T>` not valid in CUDA | nvcc | Use explicit switch-case per DType |
| `FAYN_REGISTER_EXPERIMENT` with namespaced class `::` | Macro token concat | Use `__COUNTER__` via two-level helper |
| Pure-CXX static lib gets spurious `cmake_device_link.o` | CMake 3.28 propagates CUDA device-link req through `PUBLIC` link deps | Set `CUDA_RESOLVE_DEVICE_SYMBOLS OFF` on all pure-CXX libs that transitively depend on CUDA separable libs |
| `fayn` runner CUDA device-link missing | Pure-CXX executable doesn't auto-get device-link step | Add a trivial `.cu` stub (`runner_cuda.cu`) to the executable and set `CUDA_SEPARABLE_COMPILATION ON` + `LINKER_LANGUAGE CUDA` on the target |

---

## Reward / Loss Pipeline

FAYN uses a loss or reward signal — what it eliminates is *backpropagation*, not supervision.
The scalar is used to modulate weight updates directly.

### Signal forms (all supported)
- Per-sample scalar, per-batch scalar, delayed RL scalar, multiple named signals in parallel
- `RewardEvent { std::string name; float value; size_t step; }` on EventBus
- Also returned from `run_epoch()` as the epoch metric for the CLI runner

### Update rules (all in scope)
| Rule | Mechanism | Status |
|---|---|---|
| Reward-modulated Hebbian | `ΔW ∝ r × pre × post` — reward scales the correlation | `hebbian_update()` exists; reward arg pending |
| Perturbation | Add noise; keep if loss improves | `PerturbationUpdater` planned |
| Evolutionary | Population fitness-based selection; `Population` class | planned |
| Contrastive Hebbian | Free phase vs. clamped phase; update from activity difference | planned |

### Loss functions (`src/core/loss.hpp`, planned)
`LossFn = std::function<float(const Tensor& output, const Tensor& target)>`
Standard implementations: `cross_entropy`, `mse`, `accuracy`. Custom callables registered per experiment.

### Signal routing per layer
- `RoutingMode::Local` — pure Hebbian, no reward consumed (unsupervised, earlier layers)
- `RoutingMode::Global` — layer subscribes to `RewardEvent`, scales Hebbian update by reward
- `RoutingMode::Hierarchical` — eligibility trace; reward applied when trace is non-zero

### Eligibility traces (RL credit assignment)
```
e[t] = λ × e[t-1] + pre[t] × post[t]
ΔW   = lr × r × e          (applied on RewardEvent)
```
`EligibilityTrace` planned in `src/ops/eligibility_trace.hpp`. λ is a per-layer hyperparameter.

### Readout
The last graph node (sink node) is the output by convention. No special readout class.
`Graph::forward()` returns tensors from all nodes with no active outgoing edges.

### Standard training loop pattern
```cpp
float run_epoch(size_t epoch) override {
    float total_loss = 0.f;
    for (auto& [x, y] : loader_) {
        auto outputs = graph_.forward({{0, x}});
        float loss   = loss_fn_(outputs[0], y);
        total_loss  += loss;
        RewardEvent ev; ev.step = step_++; ev.reward = -loss;
        EventBus::instance().emit(ev);
    }
    return total_loss / loader_.size();
}
```
All weight updates are side effects of EventBus subscribers — never called directly in the loop.

---

## Empirical Rules from Experiments (CP-32, MNIST)

Derived from CP-14 – CP-31. Mechanisms are general; magnitudes are MNIST-specific.
Full explanations and decision table: `PROGRESS.md [CP-32]`.

| # | Rule | Key numbers |
|---|---|---|
| R1 | **Width > depth for closed-form methods.** ELM scales strongly with d; depth adds +0.2–2.8% on top. | 256h→86%, 2048h→96%, 4096 L3→98% |
| R2 | **Hebbian is width-insensitive; ELM gap grows.** Converges to normalised class centroid — off the ELM solution sphere. Not fixable by width/LR/activation. | 256h→81%, 2048h→82%; gap 5%→14% |
| R3 | **ELM init is necessary for gradient fine-tuning.** Random init degrades; ELM warm-start improves. ELM sits near gradient-descent attractor. | −1% (random) vs +0.8% (ELM init) |
| R4 | **Frozen projection has a hard accuracy ceiling.** Invest in W₀ width before learned-layer depth; ceiling is information content of H₀, not solver capacity. | d0=8192 saturates at ~97.7% |
| R5 | **Depth only helps when single-layer is a bottleneck.** As d grows relative to input dim, a linear readout already near-memorises; depth adds nothing. | +2.8% at d=256, +0.2% at d=1024 |
| R6 | **ReLU target propagation is robust.** Cycle-2 transient collapse self-corrects. Approximate inversion is as good as exact at the fixed point. | L5 cycle-2 dip to 65–77%, then full recovery |
| R7 | **tanh gives smoother but not better convergence.** No oscillation; same or slightly lower final accuracy than ReLU at every width. | — |
| R8 | **Target propagation depth limited by amplitude cascade.** Each layer multiplies target magnitudes by κ(W). ELM+ReLU stable ≥4 layers; ADMM+tanh fails at depth-4 d=1024; LeakyReLU ADMM diverges at any depth >1. | κ(W)^4 explosion |
| R9 | **BF16 needs explicit precision management.** Updates ≈1e-5 are silently dropped in BF16 at W~0.1. Use FP32 weights; row-norm as side effect keeps W in representable range. | BF16 step ≈ 5×10⁻⁴ >> update |
| R10 | **Proximal ALS viable for 1–2 layers only.** Competitive with ELM at same depth. Fails at depth-4: tanh (amplitude cascade), LeakyReLU (unbounded Z), full ADMM with duals (oscillates). | — |

---

## Pending Work (see also PROGRESS.md)

Reward/loss pipeline:
- `src/core/loss.hpp` — `LossFn` type alias + `cross_entropy`, `mse`, `accuracy`
- Reward argument for `hebbian_update()` — scale delta by reward scalar
- `HеbbianUpdater` — EventBus subscriber owning the reward-modulated update loop
- `EligibilityTrace` — `src/ops/eligibility_trace.hpp`; per-synapse decaying trace tensor
- `DenseLayer::RoutingMode` — Local / Global / Hierarchical flag
- `PerturbationUpdater` — apply noise, evaluate loss, accept/reject
- Contrastive Hebbian two-phase forward pass API

Topology / evolution:
- `Population` class (`src/topology/population.hpp`) — N cloned graphs, fitness vector, selection

Data / IO:
- CSV data loader (`src/io/csv_loader.hpp`)
- Custom binary serialization / `GraphSerializer` (`src/io/`)

Graph:
- Multi-input merge strategy in `Graph::forward()` (currently throws for >1 in-edge)

Ops:
- `ConvLayer`, `SparseLayer`, `RecurrentLayer` scaffolding
- Mutual information / pairwise correlation kernels (opt-in, O(N²))

Experiments / tooling:
- Run `HebbianMnistExperiment` end-to-end on real MNIST data
- wandb exporter as an `EventBus` subscriber
- CI pipeline (deferred until framework stabilises)
