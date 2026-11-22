#!/usr/bin/env bash
# Caphlon-2 (4.52 GiB PTQ1_SB) — Zero-miss Context OS & Flash-Attention llama-server launcher
#
# Usage:  ./scripts/run.sh /path/to/Caphlon-2-27B-PTQ1_SB.gguf [context] [slots]
# Requires: PrismML llama.cpp built with patches/prismll_ptq1_sb_cuda_metal.patch
#           (run `bash scripts/build-prism-llamacpp.sh` first).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:?usage: run.sh <Caphlon-2-27B-PTQ1_SB.gguf> [ctx] [slots]}"
CTX="${2:-32768}"
SLOTS="${3:-1}"
PORT="${PORT:-8080}"
SERVER_BIN="${LLAMA_SERVER:-$REPO_ROOT/build-prism/llbuild/bin/llama-server}"

if ! command -v "$SERVER_BIN" >/dev/null 2>&1 && [[ ! -x "$SERVER_BIN" ]]; then
  SERVER_BIN="llama-server"
fi

if [[ ! -f "$MODEL" ]]; then
  echo "model not found: $MODEL" >&2
  exit 1
fi

# q8_0 KV cache is verified lossless on Caphlon-2 (EN PPL 7.8585 vs f16 7.8556; TR PPL 6.8789 vs 6.8824)
KV_TYPE="${KV_TYPE:-q8_0}"

# Optional anti-hesitation logit-bias from metadata/Caphlon-2-27B-PTQ1_SB.meta.json
LB_FLAGS=()
META_JSON="$REPO_ROOT/metadata/Caphlon-2-27B-PTQ1_SB.meta.json"
if [[ "${ANTI_HESITATION:-1}" == "1" ]] && [[ -f "$META_JSON" ]] && command -v python3 >/dev/null 2>&1; then
  LB_STR="$(python3 -c '
import json, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    d = json.load(fh)
pairs = []
for e in d.get("tereddut_bias", []):
    b = float(e["bias"])
    pairs.append(str(e["yeni_id"]) + (f"+{b:g}" if b >= 0 else f"{b:g}"))
print(",".join(pairs))
' "$META_JSON")"
  if [[ -n "$LB_STR" ]]; then
    for pair in ${LB_STR//,/ }; do
      LB_FLAGS+=(--logit-bias "$pair")
    done
  fi
fi

exec "$SERVER_BIN" \
  -m "$MODEL" \
  -ngl "${NGL:-99}" \
  -c "$CTX" \
  -np "$SLOTS" \
  -b 2048 -ub 512 \
  -fa on \
  -ctk "$KV_TYPE" -ctv "$KV_TYPE" \
  --cache-ram 0 \
  --ctx-checkpoints 32 \
  -cms 128 \
  --no-context-shift \
  --jinja \
  --reasoning-format deepseek \
  "${LB_FLAGS[@]}" \
  --host "${HOST:-127.0.0.1}" \
  --port "$PORT"
