#!/usr/bin/env bash
# Caphlon-2 — Perplexity & Speed verification protocol
#
# Usage:  ./scripts/eval-en40.sh <Caphlon-2-27B-PTQ1_SB.gguf> [corpus.txt]
# Verified Caphlon-2 (4.52 GiB PTQ1_SB) reference numbers (4x512 held-out):
#   English PPL (f16 KV / q8_0 KV) : 7.8556 / 7.8585
#   Turkish PPL (f16 KV / q8_0 KV) : 6.8824 / 6.8789
#   Tesla T4 speed (q8_0 KV, FA)   : pp512 = 277.95 tok/s, pp64 = 86.68 tok/s, tg32 = 16.14 tok/s
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:?usage: eval-en40.sh <model.gguf> [corpus.txt]}"
CORPUS="${2:-wiki.test.raw}"
PPL_BIN="${LLAMA_PPL:-$REPO_ROOT/build-prism/llbuild/bin/llama-perplexity}"

if ! command -v "$PPL_BIN" >/dev/null 2>&1 && [ ! -x "$PPL_BIN" ]; then
  PPL_BIN="llama-perplexity"
fi

if [ ! -f "$MODEL" ]; then echo "model not found: $MODEL" >&2; exit 1; fi
if [ ! -f "$CORPUS" ]; then echo "corpus not found: $CORPUS" >&2; exit 1; fi

exec "$PPL_BIN" \
  -m "$MODEL" \
  -f "$CORPUS" \
  -c 512 \
  --chunks "${CHUNKS:-4}" \
  -ngl "${NGL:-99}" \
  -t "${THREADS:-4}" \
  --no-warmup \
  -b 2048 -ub 512
