// Minimal CUDA compilation unit for fayn_runner.
// Causes CMake to generate a CUDA device-link step, which is required
// when the executable links against CUDA separable-compiled libraries
// (fayn_ops, fayn_stats) but has no other CUDA source of its own.
namespace fayn {}
