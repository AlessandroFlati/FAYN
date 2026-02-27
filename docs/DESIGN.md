# FAYN — Framework Design Document

**Created:** 2026-02-27 05:59 UTC
**Updated:** 2026-02-27 09:46 UTC
**Status:** Reward/loss pipeline design complete. First experiment pending.

---

## 1. Motivation and Philosophy

FAYN is an experimental C++ framework for **forward-only neural networks** that
deliberately excludes backpropagation and gradient-based weight updates. The goal
is to systematically explore alternative learning and adaptation mechanisms —
primarily statistical, information-theoretic, and topology-driven — and to build
the computational infrastructure needed to run those experiments at speed on
modern GPU hardware.

The framework is a **research toolbox**, not a production inference engine. Designs
favour clarity of the research signal over production robustness.

### Core thesis

Traditional deep learning conflates two separate concerns:

1. **Topology**: which neurons exist and how they are connected.
2. **Weights**: the numerical values on those connections.

Backprop optimises weights while holding topology fixed. FAYN decouples these and
asks: what if topology itself is the primary adaptive variable, driven by
statistical properties of forward activations — rather than a globally propagated
error signal?

---

## 2. Design Interview — Questions and Answers

All decisions below were established through an explicit design interview on
2026-02-27. Each section records the question, the answer given, and the
downstream architectural implication.

---

### 2.1 Learning paradigm

**Q:** What is the primary learning paradigm to explore?

**A:** All four — treated as non-mutually-exclusive axes of experimentation:
- Hebbian / correlation-based
- Topology mutation (add/prune neurons and edges)
- Perturbation-based (weight noise, keep beneficial changes)
- Evolutionary / population-based

**Implication:** The framework must support multiple paradigms running within the
same experiment. No single "update rule" is privileged in the API. The `Layer`
base class deliberately has no `update()` method; adaptation is external and
pluggable.

---

### 2.2 Architecture scope

**Q:** What network architecture types should be supported initially?

**A:** All four:
- Dense / MLP
- Convolutional (CNN)
- Sparse / graph-based
- Recurrent / temporal

**Implication:** The graph representation must handle arbitrary connectivity
(sparse and recurrent), not just sequential chains. The `Graph` class uses an
adjacency-list representation with explicit node/edge objects. Recurrent edges are
not yet handled in `forward()` (which requires a DAG); a recurrent unrolling or
explicit state-passing mechanism is a near-term TODO.

---

### 2.3 Target domain

**Q:** What is the primary domain / task?

**A:** Domain-agnostic toolbox. Initial experiments use:
- Reinforcement learning (reward signal replaces supervised loss)
- Classification benchmarks (MNIST, CIFAR-10)

**Implication:** The `Experiment` base class accepts any `DataSource` and reports
a generic scalar metric per epoch. A loss or reward function **is** required —
what FAYN eliminates is *backpropagation*, not the supervising signal. The loss
value modulates weight updates directly (reward-modulated Hebbian, perturbation
acceptance, evolutionary fitness) rather than being differentiated. The
`RewardEvent` on the `EventBus` carries the scalar signal; `EpochEvent` carries
the epoch-level metric for display.

---

### 2.4 Build system

**Q:** C++ standard and build system?

**A:** **C++20 + CMake 3.28+**

**Implication:** C++20 features in active use: `std::variant`, `std::visit`, C++20
lambda improvements, `std::filesystem`. CUDA standard set to 20 as well.
`--extended-lambda` and `--expt-relaxed-constexpr` are passed to `nvcc`.

---

### 2.5 CUDA / GPU strategy

**Q:** What is the CUDA strategy?

**A:** Use **cuBLAS / cuDNN for dense ops**; write **custom CUDA kernels only for
novel ops** (topology changes, correlation statistics, activation metrics).

**Implication:** `DenseLayer` uses cuBLASLt with BF16 inputs and FP32 compute.
Activation kernels are custom `.cu` templated on `__nv_bfloat16`, `__half`,
`float`. Stats reduction kernels (dead-neuron ratio, EMA mean/var) will be custom
CUDA kernels launched alongside each forward pass.

---

### 2.6 CUDA compute target

**Q:** Minimum CUDA compute capability?

**A:** **sm_120 (Blackwell, RTX 5090 / GB202)**. Requires CUDA Toolkit ≥ 12.8.

**Implication:** `CMAKE_CUDA_ARCHITECTURES 120` in the root CMakeLists. BF16
tensor cores native on Blackwell. FP8 (transformer engine) available for future
experiments. No backward compatibility with older GPUs is maintained.

---

### 2.7 Statistical / information-theoretic metrics

**Q:** What metrics drive learning and topology decisions?

**A:** All four:
- **Mutual information / entropy** — information content of neuron activations; basis for pruning redundant neurons.
- **Activation correlation / covariance** — detect co-firing patterns for Hebbian weight updates.
- **Dead / saturated neuron detection** — fraction of neurons with |activation| below a threshold; triggers rewiring.
- **Custom task-specific reward / fitness** — user-supplied scalar (RL reward, accuracy) carried via `RewardEvent`.

**Implication:** `StatsSnapshot` carries per-neuron `ema_mean`, `ema_var`,
`dead_ratio`, and `ema_magnitude`. Mutual information and full pairwise
correlation are not yet in `StatsSnapshot` (O(N²) memory cost); they will be
added as opt-in computed fields triggered per-layer when a subscriber requests
them.

---

### 2.8 Experiment tracking

**Q:** How to track and visualize experiments?

**A:** Weights & Biases if self-hosted; otherwise **custom in-process dashboard +
JSON/CSV logs**. Implemented the latter for now.

**Implication:** `Logger` subscribes to `EventBus` (async) and writes
JSONL (one JSON object per line) to a file. Each event type has a `"type"` field.
A separate Python script (not part of the C++ framework) can consume these logs
for plotting. A wandb exporter can be added as another `EventBus` subscriber
without touching any other code.

---

### 2.9 Topology mutation trigger

**Q:** How should mutation be triggered?

**A:** Each strategy (epoch-level, online, separate phase, user-controlled) is
itself part of the experimentation space — no single trigger is baked in.

**Implication:** `MutationEngine` exposes `MutationThreshold` rules that fire on
any `ActivationEvent`. The trigger granularity (per step, per N steps, per epoch)
is controlled by the threshold condition, which the user encodes in the rule's
`metric_fn` + `comparator` + `threshold`. No automatic scheduling exists in the
engine itself. `mutation_engine.set_enabled(false)` disables it entirely during
population evaluation phases.

---

### 2.10 Repo layout

**Q:** What repo/module structure?

**A:** **Monorepo** with `src/`, `experiments/`, `tests/`, `tools/`.

**Implication:** Directory tree confirmed:
```
FAYN/
  src/
    core/       # tensor, graph, layer primitives
    stats/      # EventBus, EMA, events
    cuda/       # StreamPool
    ops/        # activations, dense (CUDA)
    topology/   # MutationEngine, MutationOps
    io/         # data loaders, JSONL logger
  experiments/  # Experiment base class + concrete subclasses
  tests/        # validation scripts (no formal test framework)
  tools/        # CLI runner
  docs/         # this file
  CMakeLists.txt
```

---

### 2.11 CPU concurrency

**Q:** CPU-side parallelism model?

**A:** **std::thread + custom thread pool**

**Implication:** `ThreadPool` in `src/core/thread_pool.hpp`. Used for parallel
population evaluation (evolutionary experiments), data preprocessing, and async
EventBus dispatch. No OpenMP or TBB dependency.

---

### 2.12 Memory model

**Q:** Core tensor / buffer memory management?

**A:** **Explicit host/device buffers with manual transfers** (cudaMemcpy).

**Implication:** `Tensor::to(Device)` performs explicit `cudaMemcpy`. No unified
memory (cudaMallocManaged). The caller controls when data moves between host and
device. Hot-path data (weights, activations) lives permanently on device.
Stats snapshots are pulled to host only when `EventBus` subscribers need them.

---

### 2.13 Testing strategy

**Q:** Testing framework?

**A:** **No formal test framework** — experiment scripts validate behavior.

**Implication:** Correctness is validated by running end-to-end experiments and
checking output metrics. No gtest/Catch2 targets are added. A `tests/` directory
exists for future validation scripts.

---

### 2.14 Tensor design

**Q:** How should the core Tensor type be designed?

**A:** **Strided N-dimensional tensor** (like PyTorch / NumPy)

**Implication:** `Tensor` holds `data` (void*), `shape` (std::vector<size_t>),
`strides` (in elements, std::vector<size_t>), `DType`, `Device`. Non-copyable,
movable. `TensorView` is a POD struct (CUDA-safe) with fixed-size `shape` and
`strides` arrays (max dims = 8). Contiguous row-major tensors are the default;
strided views are supported but `contiguous()` is not yet implemented for
non-trivial strides.

---

### 2.15 Graph representation

**Q:** How should the network graph be represented?

**A:** **Adjacency list with typed edge/node structs**

```cpp
struct Node { LayerPtr layer; std::vector<int> out_edges; bool active; };
struct Edge { int src; int dst; bool active; };
```

**Implication:** Node and edge IDs are stable integers (indices into
`Graph::nodes_` and `Graph::edges_`). Removed nodes/edges are marked
`active=false` and not compacted. Topological sort (Kahn's algorithm) operates
only on active elements. `Graph::clone()` shares `LayerPtr` instances (weights
are not deep-copied) for use in population-based experiments where multiple
population members share base weights.

---

### 2.16 External dependencies

**Q:** Dependency policy?

**A:** **Minimal — only CUDA toolkit + CMake FetchContent for small headers**

**Implication:** Only two FetchContent deps:
- `nlohmann/json` v3.11.3 — for JSONL logger
- `fmtlib/fmt` v10.2.1 — for string formatting (not yet wired in, available)

No vcpkg, Conan, or vendored third_party/. CUDA toolkit (≥ 12.8) and cuBLASLt
are system requirements.

---

### 2.17 Floating-point precision

**Q:** Precision strategy?

**A:** **BF16 for activations, FP32 for stats accumulation**

**Implication:**
- Default `DType` for layer inputs/outputs: `DType::BFloat16`
- `DenseLayer` weights: BF16; bias: FP32
- cuBLASLt compute type: `CUBLAS_COMPUTE_32F` (FP32 accumulation inside GEMM)
- `StatsSnapshot` fields: all `float` (FP32)
- `EmaScalar` / `EmaVector`: FP32 internally

---

### 2.18 Experiment configuration

**Q:** How should experiments be configured?

**A:** **C++ experiment classes (code-as-config)**

```cpp
class MyExperiment : public Experiment {
    void  setup()            override { /* build graph, add rules */ }
    float run_epoch(size_t)  override { /* forward loop, return metric */ }
};
FAYN_REGISTER_EXPERIMENT("my_experiment", MyExperiment)
```

**Implication:** No JSON/TOML config files for the network graph or hyperparameters.
`ExperimentConfig` carries only primitive fields (name, epochs, batch_size,
log_path, enable_mutations). Graph construction is always code. This keeps the
compiler as the type-checker and eliminates a config parsing layer.

---

### 2.19 Layer abstraction

**Q:** How should individual layers / ops be abstracted?

**A:** **Abstract `Layer` base class with `virtual forward()`**

```cpp
class Layer {
public:
    virtual Tensor forward(const Tensor& x) = 0;
    virtual ~Layer() = default;
};
```

**Implication:** Runtime polymorphism via vtable. Acceptable given that the
forward-pass cost dominates vtable dispatch by many orders of magnitude.
`LayerPtr = std::shared_ptr<Layer>` used throughout the graph. Each concrete
layer class is responsible for emitting `ActivationEvent` to `EventBus` after
computing its output.

---

### 2.20 Stats accumulation / logging

**Q:** Stats and logging architecture?

**A:** **Global event bus** — layers emit events; subscribers log or mutate.

**Implication:**
- `EventBus` is a singleton.
- `Logger` subscribes with `DispatchMode::Async` (non-blocking for the forward pass).
- `MutationEngine` subscribes with `DispatchMode::Sync` (must act before the next forward call).
- Additional subscribers can be added at any time without modifying layer code.
- `EventBus::flush()` drains the async queue (called at experiment end).

---

### 2.21 Population management

**Q:** How should evolutionary / population-based experiments manage a population?

**A:** **Parallel population on multiple CUDA streams**

**Implication:** `StreamPool` (default 8 streams) provides streams for parallel
population member evaluation. Each population member acquires a stream via
`StreamPool::Guard`, runs its forward pass, and releases the stream.
`Graph::clone()` provides cheap copies for population members.
A dedicated `Population` class (holding N cloned graphs, a fitness vector, and
selection/crossover methods) is a planned addition to `src/topology/`.

---

### 2.22 Activation functions

**Q:** Initial activation function suite?

**A:** ReLU / LeakyReLU, GELU / SiLU, Tanh / Sigmoid, plus a user plugin API.

**Implication:** `apply_relu`, `apply_leaky_relu`, `apply_gelu`, `apply_silu`,
`apply_tanh`, `apply_sigmoid` are CUDA kernels in `src/ops/activations.cu`.
All are in-place and templated on dtype (`__nv_bfloat16`, `__half`, `float`).
`ActivationRegistry` accepts user-registered `std::function<void(Tensor&, cudaStream_t)>`
kernels by name. Registered under a string key; callable from any experiment.

---

### 2.23 Serialization

**Q:** How to checkpoint network state?

**A:** **Custom binary format** (hand-rolled initially)

**Implication:** Not yet implemented. Planned format:
- Header: magic bytes, version, node count, edge count
- Per-node: layer type tag, hyperparameters (int/float fields)
- Per-weight tensor: dtype, shape, raw bytes
- Topology section: edge list (src, dst, active)

Will be implemented in `src/io/` as `GraphSerializer` / `GraphDeserializer`.

---

### 2.24 Mutation engine design

**Q:** How should the mutation engine expose its decision logic?

**A:** **Event-driven** — `MutationEngine` subscribes to `EventBus`, acts on
thresholds.

**Implication:** `MutationThreshold` is a rule struct with:
- `metric_fn: StatsSnapshot -> float` — extract a scalar
- `comparator: (value, threshold) -> bool` — fire condition
- `factory: (StatsSnapshot, Graph) -> MutationOp` — produce the mutation

Rules are evaluated on every `ActivationEvent`. The engine applies at most one
rule per event per graph (first match wins). `AddNode` and `SplitNode` are
delegated back as `MutationProposalEvent` since they require layer factory
knowledge the engine does not have.

---

### 2.25 Stats window

**Q:** Statistics accumulation window model?

**A:** **Exponential moving average (EMA) per metric**

**Implication:** `EmaScalar` and `EmaVector` in `src/stats/ema.hpp`. Default
`alpha = 0.05`. Memory cost is O(neurons) regardless of run length. No ring
buffer or epoch-reset semantics. Each layer maintains its own EMA state
updated on every forward pass.

---

### 2.26 Python bindings

**Q:** pybind11 / Python bindings?

**A:** **None — pure C++ only**

**Implication:** No pybind11 dependency. Python is used only for post-hoc
analysis of JSONL log files. Public API uses no types that are intrinsically
non-bindable (no raw C arrays in public interfaces), so bindings could be added
later without redesign.

---

### 2.27 Data pipeline

**Q:** Data loading strategy?

**A:** **Minimal C++ loaders** for MNIST binary, CSV, and raw image directories.

**Implication:** `DataSource` abstract base with `next_batch()`, `reset()`,
`size()`. `MnistLoader` reads the IDX binary format (magic, count, rows, cols,
pixel bytes). Normalises to `[0, 1]` float, then converts to target dtype
(`BFloat16` by default) on CPU before uploading to device. A `CsvLoader` is
planned but not yet implemented.

---

### 2.28 EventBus dispatch mode per subscriber

**Q:** Should EventBus be sync or async?

**A:** **Configurable per subscriber at registration time**

**Implication:**
```cpp
bus.subscribe<ActivationEvent>(handler, DispatchMode::Sync);   // inline in emit()
bus.subscribe<EpochEvent>(handler,      DispatchMode::Async);  // posted to worker queue
```
`EventBus` has a single background worker thread draining an async task queue.
`flush()` posts a sentinel future to the queue and waits on it.

---

### 2.29 CUDA stream management

**Q:** How to manage CUDA streams across layers and population members?

**A:** **Stream pool managed by a central CUDA scheduler object**

**Implication:** `StreamPool` singleton holds `N` pre-created `cudaStreamNonBlocking`
streams (default N=8). `StreamPool::Guard` is a RAII wrapper that acquires a
stream on construction and releases it on destruction. Layers acquire a stream for
the duration of their forward pass. If the pool is exhausted, `acquire()` blocks.

---

### 2.30 CI / CD

**Q:** CI pipeline?

**A:** **None — local builds only for now**

**Implication:** No GitHub Actions workflows. Build and test locally against the
RTX 5090. CI can be added when the framework stabilises.

---

---

## 2.3x Reward / Loss Pipeline (design interview 2026-02-27 09:46 UTC)

The following questions and answers define how FAYN uses a reward or loss signal
to drive weight updates without backpropagation.

---

### 2.31 Signal form

**Q:** What does the reward/loss signal look like at runtime?

**A:** All four forms are supported — they are not mutually exclusive:
- **Scalar per sample** — per-example loss or reward before batching
- **Scalar per batch** — loss averaged over a mini-batch (e.g. cross-entropy)
- **Delayed scalar (RL)** — environment reward arriving after an action, not tied to a single forward pass
- **Multiple signals in parallel** — e.g. task loss + intrinsic novelty reward simultaneously

**Implication:** `RewardEvent` carries a named scalar (`std::string name; float value`)
so multiple signals can be emitted in the same step. Subscribers filter by name.
The experiment is responsible for computing and emitting each signal.

---

### 2.32 Update rules

**Q:** How does the reward/loss signal modulate weight updates?

**A:** All four update paradigms are in scope:
- **Reward-modulated Hebbian** — `ΔW ∝ r × pre × post`. The scalar reward gates the Hebbian correlation: positive reward reinforces co-firing, negative reward suppresses it.
- **Perturbation / node perturbation** — add noise to weights or activations; keep the perturbation if it improves the loss. No gradient required.
- **Evolutionary selection** — evaluate a population of weight variants; propagate the best-performing ones. Loss is a fitness score, not a gradient source.
- **Contrastive Hebbian** — run two forward passes (free phase vs. clamped/target phase); update weights based on the difference in activity between phases (cf. Contrastive Hebbian Learning, Equilibrium Propagation).

**Implication:** Each paradigm is an independent module:
- Reward-modulated Hebbian: `hebbian_update()` already exists; gains a `reward` scalar argument.
- Perturbation: `PerturbationUpdater` class (planned) applies noise and evaluates.
- Evolutionary: `Population` class (planned) holds N cloned graphs, runs them in parallel on separate CUDA streams, and selects survivors.
- Contrastive: a second forward pass with clamped output; the difference in `last_output_` between phases drives the update.

No single update rule is privileged in the `Layer` or `Graph` API. All rules are
external to the graph and operate on cached activations or cloned weight tensors.

---

### 2.33 Loss function

**Q:** How is the loss/reward scalar computed from the network output?

**A:** All four options are supported:
- **Standard supervised loss** — cross-entropy, MSE, hinge, etc., computed from network output vs. ground-truth labels. The scalar value modulates updates; it is never differentiated.
- **Accuracy / ranking signal** — binary or ordinal feedback (+1 / −1 / 0); used when a smooth loss is undesirable.
- **Environment reward (RL)** — external scalar from a Gym-style environment.
- **Custom / task-specific** — user-defined `std::function<float(const Tensor&, const Tensor&)>` registered per experiment.

**Implication:** A `LossFn = std::function<float(const Tensor& output, const Tensor& target)>`
type alias lives in `src/core/loss.hpp`. Standard implementations (cross_entropy,
mse, accuracy) are provided. The experiment passes its chosen `LossFn` to the
training loop; the framework never calls a loss function internally.

---

### 2.34 Signal routing

**Q:** Does every layer receive the same reward signal, or is it layer-specific?

**A:** Both **layer-local** and **hierarchical / propagated** routing are in scope.
They are not mutually exclusive — different layers in the same experiment can use
different routing strategies:

- **Layer-local only** — each layer sees only its own pre/post activations. No global signal. Classic Hebbian / Oja's rule. Used for unsupervised feature extraction in earlier layers.
- **Hierarchical / propagated** — the global reward is transformed layer-by-layer into a local teaching signal without backprop. One mechanism: a layer's *contribution to the output* is estimated from its cached activations and used to scale its local Hebbian update. Another: eligibility traces (see §2.36) naturally localise credit to recently co-firing synapses.

**Implication:** `DenseLayer` will expose a `set_reward_routing(RoutingMode)` flag:
- `RoutingMode::Local` — pure Hebbian, no reward signal consumed.
- `RoutingMode::Global` — layer subscribes to `RewardEvent` and scales its Hebbian update by the received reward.
- `RoutingMode::Hierarchical` — layer uses an eligibility trace; reward is applied when the trace is non-zero.

---

### 2.35 Update timing

**Q:** When does the weight update happen relative to the forward pass?

**A:** **Asynchronous / event-driven.** Updates are triggered by the `MutationEngine`
or a custom subscriber when a threshold condition is satisfied — decoupled from
the forward pass cadence. The subscriber accumulates activations and reward signals
internally and applies a weight update when its own criterion is met (e.g. every
N steps, or when the reward crosses a threshold).

**Implication:** The training loop in `Experiment::run_epoch()` only calls
`graph.forward()` and emits `RewardEvent`. It does not call `hebbian_update()` or
any other update function directly. All weight updates are side effects of
`EventBus` subscribers. This makes it trivial to swap update frequency without
touching the experiment code.

---

### 2.36 Temporal credit assignment (RL)

**Q:** How is temporal credit assignment handled for delayed rewards?

**A:** **Eligibility traces.** Each synapse maintains a decaying trace of recent
pre × post activity:
```
e[t] = λ × e[t-1] + pre[t] × post[t]
```
When the reward `r` arrives (possibly several steps later):
```
ΔW = lr × r × e
```
The trace naturally assigns more credit to recently co-firing synapses.

**Implication:** `EligibilityTrace` (planned, `src/ops/eligibility_trace.hpp`)
holds a device tensor `e` of the same shape as the weight matrix, updated on each
forward pass. `DenseLayer` allocates a trace when `RoutingMode::Hierarchical` is
set. On `RewardEvent`, the subscriber applies `ΔW = lr × r × e` and decays `e`.
The decay factor `λ` is a per-layer hyperparameter.

---

### 2.37 Readout layer

**Q:** Where does the readout layer live?

**A:** The **last graph node**, trained with the same reward-modulated rules as all
other layers. No separate fixed or differently-trained readout.

**Implication:** No special `ReadoutLayer` class. The final `DenseLayer` (or
`ActivationLayer`) in the graph is the output node by convention — `Graph::forward()`
returns tensors from all sink nodes (nodes with no active outgoing edges). The
experiment computes the loss from those output tensors.

---

### 2.38 Signal transport

**Q:** How should the reward/loss value be carried through the system?

**A:** **Both channels simultaneously:**
- `RewardEvent` on the `EventBus` — for internal subscribers (Hebbian updater, MutationEngine, Logger, eligibility trace manager).
- Return value from `run_epoch()` — the epoch-level metric returned to the CLI runner for display and external logging.

**Implication:** The standard pattern in every experiment:
```cpp
float run_epoch(size_t epoch) override {
    float total_loss = 0.f;
    for (auto& [x, y] : loader_) {
        auto outputs = graph_.forward({{0, x}});
        float loss   = loss_fn_(outputs[0], y);
        total_loss  += loss;

        RewardEvent ev;
        ev.step   = step_++;
        ev.reward = -loss;          // reward = negative loss
        EventBus::instance().emit(ev);
    }
    return total_loss / loader_.size();
}
```

---

## 3. System Architecture Diagram

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                        Experiment                               │
 │  setup() → run_epoch() → emit EpochEvent / RewardEvent          │
 └────────────────────────────┬────────────────────────────────────┘
                              │ owns
                              ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │                          Graph                                  │
 │  [Node] → [Node] → [Node]   (adjacency list, topological exec)  │
 │  add/remove_node, add/remove_edge, rewire_edge, clone()         │
 └─────────────┬─────────────────────────────────────┬────────────┘
               │ each Node owns                      │ forward() calls
               ▼                                     ▼
 ┌─────────────────────┐              ┌──────────────────────────┐
 │       Layer         │              │      StreamPool           │
 │  virtual forward()  │◄─acquires────│  8 x cudaStream_t         │
 │  emit ActivationEvt │              └──────────────────────────┘
 └─────────┬───────────┘
           │ emit()
           ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │                        EventBus (singleton)                     │
 │                                                                 │
 │   ActivationEvent ──sync──► MutationEngine                      │
 │   ActivationEvent ─async──► Logger (JSONL)                      │
 │   MutationProposalEvent ──► Experiment (user handles AddNode)   │
 │   EpochEvent      ─async──► Logger                              │
 │   RewardEvent     ─async──► Logger                              │
 └─────────────────────────────────────────────────────────────────┘

 ┌─────────────────────────────────────────────────────────────────┐
 │                     MutationEngine                              │
 │  [MutationThreshold rules]                                      │
 │  on ActivationEvent: check rules → apply MutationOp to Graph    │
 │  MutationOp variants: AddNode, RemoveNode, AddEdge, RemoveEdge, │
 │                        RewireEdge, SplitNode, NoOp              │
 └─────────────────────────────────────────────────────────────────┘

 Stats flow:
   Layer::forward() → CUDA kernel → activations on device
   → stats reduction kernel → StatsSnapshot (FP32, host)
   → ActivationEvent → EmaVector updates in MutationEngine

 Reward flow:
   Experiment::run_epoch()
     → graph.forward() → output tensor
     → loss_fn(output, target) → scalar loss
     → RewardEvent{reward = -loss}  ──────────────────────────────┐
                                                                   │
   EventBus subscribers reacting to RewardEvent:                  │
     sync  ──► HеbbianUpdater: ΔW = lr × reward × pre × post  ◄──┘
     sync  ──► EligibilityTraceManager: ΔW = lr × reward × e
     sync  ──► PerturbationUpdater: accept/reject noise step
     async ──► Logger (JSONL)
   return total_loss → CLI runner display
```

---

## 4. Module Dependencies

```
fayn_runner (executable)
  └── fayn_experiments
        └── fayn_io ─── nlohmann_json
        └── fayn_topology
              └── fayn_stats
                    └── fayn_core ─── CUDA::cudart
        └── fayn_ops ── CUDA::cublasLt
              └── fayn_cuda
```

No circular dependencies. `fayn_core` has no FAYN dependencies (only CUDA::cudart
for device.hpp).

---

## 5. Known Limitations and Planned Work

| Item | Status | Notes |
|---|---|---|
| `src/core/loss.hpp` | TODO | `LossFn` type alias + cross_entropy, mse, accuracy implementations |
| Reward-modulated `hebbian_update()` | TODO | Add `float reward` argument; scale delta by reward |
| `EligibilityTrace` | TODO | `src/ops/eligibility_trace.hpp`; per-synapse decaying trace tensor |
| `DenseLayer::RoutingMode` | TODO | Local / Global / Hierarchical routing flag |
| `HеbbianUpdater` subscriber | TODO | EventBus subscriber that owns the update loop |
| `PerturbationUpdater` | TODO | Apply noise, evaluate loss, accept/reject |
| Contrastive Hebbian support | TODO | Two-phase forward pass API in `Graph` or `Experiment` |
| `Population` class | TODO | `src/topology/population.hpp`; N cloned graphs, fitness vector, selection |
| CSV data loader | TODO | `src/io/csv_loader.hpp` |
| Recurrent graph support | TODO | Graph::forward() requires a DAG; need unrolling or state injection API |
| Multi-input merge | TODO | Graph::forward() throws for nodes with >1 active in-edge |
| GraphSerializer / Deserializer | TODO | Custom binary checkpoint format |
| Mutual information kernel | TODO | Stats: full MI is O(N²); needs opt-in activation |
| Pairwise correlation kernel | TODO | Same O(N²) concern; opt-in |
| First concrete experiment | TODO | HebbianMnistExperiment as integration test |
| ConvLayer | TODO | Not yet scaffolded |
| SparseLayer | TODO | Not yet scaffolded |
| RecurrentLayer | TODO | Not yet scaffolded |
| wandb exporter | TODO | Add as an EventBus subscriber; no changes to core |

---

## 6. Coding Conventions

- No emoji in C++ code, comments, or output.
- All code and comments in English.
- BF16 activations by default (optimal for Blackwell tensor cores).
- Stats always in FP32.
- Fail fast: no silent fallbacks. Raise `std::runtime_error` or `std::invalid_argument`.
- `FAYN_CUDA_CHECK(expr)` wraps all CUDA calls. `FAYN_CUBLAS_CHECK(expr)` wraps cuBLAS.
- `StreamPool::Guard` for all CUDA kernel launches (never use default stream in production paths).
- Layer IDs are assigned by `Graph::add_node()`; never set manually.
- `EventBus::emit()` is safe to call from any thread; internal mutex protects handler dispatch.
