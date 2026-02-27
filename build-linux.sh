#!/bin/bash
# Build FAYN natively on WSL2 (Linux/GCC/nvcc).
# Run after installing cuda-toolkit-12-9.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build-linux"

# Ensure Linux CUDA toolkit is on PATH.
if [ -d "/usr/local/cuda-12.9/bin" ]; then
    export PATH="/usr/local/cuda-12.9/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH"
elif [ -d "/usr/local/cuda/bin" ]; then
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
fi

NVCC_PATH=$(which nvcc 2>/dev/null || true)
if [ -z "$NVCC_PATH" ] || [[ "$NVCC_PATH" == *"cuda-bin"* ]]; then
    echo "ERROR: Linux nvcc not found. Install cuda-toolkit-12-9 first:"
    echo "  sudo apt-get install -y cuda-toolkit-12-9"
    exit 1
fi
echo "Using nvcc: $NVCC_PATH ($(nvcc --version | head -1))"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$SCRIPT_DIR" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_C_COMPILER=gcc

cmake --build . --parallel "$(nproc)"
echo ""
echo "Build complete. Run smoke test:"
echo "  $BUILD_DIR/tools/fayn_smoke"
