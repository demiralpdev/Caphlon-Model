#!/usr/bin/env bash
# Caphlon-2 — One-command builder for PrismML llama.cpp + PTQ1_SB (type 144) CUDA/Metal/AVX2/NEON patch
#
# Usage:
#   bash scripts/build-prism-llamacpp.sh [build_dir]
# Environment overrides:
#   CUDA_ARCH=75   (set e.g. 75 for Turing T4, 80 for A100, 86/89 for RTX 30/40; auto-detected on macOS Metal)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_FILE="$REPO_ROOT/patches/prismll_ptq1_sb_cuda_metal.patch"
PRISM_COMMIT="6447906bf96c1a023980d3453fb59daa6af3bb23"
WORK_DIR="${1:-$REPO_ROOT/build-prism}"

if [[ ! -f "$PATCH_FILE" ]]; then
  echo "error: patch not found at $PATCH_FILE" >&2
  exit 1
fi

mkdir -p "$WORK_DIR"
SRC_DIR="$WORK_DIR/llsrc"
BLD_DIR="$WORK_DIR/llbuild"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "== Downloading PrismML-Eng/llama.cpp ($PRISM_COMMIT) =="
  mkdir -p "$SRC_DIR"
  curl -fsS --proto '=https' --tlsv1.2 --retry 5 --http1.1 \
    "https://codeload.github.com/PrismML-Eng/llama.cpp/tar.gz/$PRISM_COMMIT" \
    -o "$WORK_DIR/prism_src.tar.gz"
  tar -xzf "$WORK_DIR/prism_src.tar.gz" -C "$SRC_DIR" --strip-components=1
  git -C "$SRC_DIR" init -q
  git -C "$SRC_DIR" apply "$PATCH_FILE"
  echo "== Applied PTQ1_SB patch: $(shasum -a 256 "$PATCH_FILE" | awk '{print $1}') =="
fi

CMAKE_ARGS=(
  -S "$SRC_DIR"
  -B "$BLD_DIR"
  -DCMAKE_BUILD_TYPE=Release
  -DLLAMA_CURL=OFF
  -DLLAMA_BUILD_UI=OFF
  -DLLAMA_USE_PREBUILT_UI=OFF
)

OS_NAME="$(uname -s)"
if [[ "$OS_NAME" == "Darwin" ]]; then
  echo "== Configuring for macOS (Apple Metal + ARM64 NEON / Accelerate) =="
  CMAKE_ARGS+=(-DGGML_METAL=ON)
else
  if command -v nvcc >/dev/null 2>&1 || [[ -x /usr/local/cuda/bin/nvcc ]]; then
    CUDA_ARCH="${CUDA_ARCH:-75}"
    echo "== Configuring for Linux CUDA (CMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH + AVX2) =="
    CMAKE_ARGS+=(
      -DGGML_CUDA=ON
      -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
      -DGGML_AVX2=ON
    )
    if [[ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]] && [[ ! -e /usr/lib/x86_64-linux-gnu/libcuda.so ]]; then
      sed -i 's|CUDA::cuda_driver|/usr/lib/x86_64-linux-gnu/libcuda.so.1|g' \
        "$SRC_DIR/ggml/src/ggml-cuda/CMakeLists.txt"
    fi
  else
    echo "== Configuring for CPU (AVX2) =="
    CMAKE_ARGS+=(-DGGML_AVX2=ON)
  fi
fi

cmake "${CMAKE_ARGS[@]}"
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
cmake --build "$BLD_DIR" --target llama-server llama-cli llama-perplexity llama-bench -j "$JOBS"

echo "== Build complete: binaries in $BLD_DIR/bin =="
ls -lh "$BLD_DIR/bin/llama-server" "$BLD_DIR/bin/llama-cli" "$BLD_DIR/bin/llama-bench" "$BLD_DIR/bin/llama-perplexity"
