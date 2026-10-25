#!/usr/bin/env bash
# Caphlon-Model — English perplexity protocol (EN40)
#
# Usage:  ./eval-en40.sh <model.gguf> [wiki.test.raw]
# Produces a single PPL number comparable to the published reference:
#   reference (unmodified IQ2_S)  : 6.3638
#   Caphlon-Model                 : 6.6088  (+3.85%)
set -euo pipefail

MODEL="${1:?usage: eval-en40.sh <model.gguf> [wiki.test.raw]}"
CORPUS="${2:-wiki.test.raw}"

if [ ! -f "$MODEL" ];  then echo "model not found: $MODEL"  >&2; exit 1; fi
if [ ! -f "$CORPUS" ]; then echo "corpus not found: $CORPUS" >&2
  echo "download: https://huggingface.co/datasets/theleaf/planb (wiki.test.raw)" >&2
  exit 1
fi

echo "== protocol: -c 512 --chunks 40, no warmup, b=4096 ub=1024 ==" >&2

exec llama-perplexity \
  -m "$MODEL" \
  -f "$CORPUS" \
  -c 512 \
  --chunks 40 \
  -ngl "${NGL:-99}" \
  -t "${THREADS:-4}" \
  --no-warmup \
  -b 4096 -ub 1024
