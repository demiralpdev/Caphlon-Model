#!/usr/bin/env bash
# Caphlon-Model — tuned llama.cpp server launch (measured recipe, 2026-10-25)
#
# Usage:  ./run.sh /path/to/Caphlon-Model-IQ2_S.gguf [context]
# Requires: llama.cpp (llama-server) on PATH, built with CUDA/Metal/Vulkan.
set -euo pipefail

MODEL="${1:?usage: run.sh <model.gguf> [ctx]}"
CTX="${2:-32768}"
PORT="${PORT:-8080}"

if [ ! -f "$MODEL" ]; then
  echo "model not found: $MODEL" >&2
  exit 1
fi

# Optional KV squeeze — only when context is the binding constraint.
# Measured gain on this model: ~152 MB. Trade-off: greedy output is not
# byte-identical to f16 KV.
KV_FLAGS=()
if [ "${KV_QUANT:-0}" = "1" ]; then
  KV_FLAGS=(--cache-type-k q8_0 --cache-type-v q4_0)
fi

exec llama-server \
  -m "$MODEL" \
  -ngl "${NGL:-99}" \
  -c "$CTX" \
  -np 1 \
  -b 4096 -ub 1024 \
  --flash-attn on \
  --spec-type ngram-mod \
  --spec-ngram-mod-n-match 24 \
  --spec-ngram-mod-n-min 48 \
  --spec-ngram-mod-n-max 64 \
  "${KV_FLAGS[@]}" \
  --jinja \
  --host "${HOST:-127.0.0.1}" \
  --port "$PORT"
