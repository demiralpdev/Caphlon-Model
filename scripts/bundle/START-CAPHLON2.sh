#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

CTX="${CTX:-16384}"
NGL="${NGL:-99}"
PORT="${PORT:-8080}"
MODEL="$DIR/Caphlon-2-27B-PTQ1_SB.gguf"
MODEL_URL="https://huggingface.co/ssuasd/Caphlon-2/resolve/main/Caphlon-2-27B-PTQ1_SB.gguf"
MODEL_BYTES=4856213824

file_size() {
  if [[ -f "$1" ]]; then
    stat -c%s "$1" 2>/dev/null || stat -f%z "$1" 2>/dev/null || echo 0
  else
    echo 0
  fi
}

if [[ "$(file_size "$MODEL")" -ne "$MODEL_BYTES" ]]; then
  echo "Downloading Caphlon-2 (4.52 GiB) from Hugging Face..."
  echo "This only happens once. Download is resumable if interrupted."
  curl -L -C - --fail --retry 5 -o "$MODEL" "$MODEL_URL"
  if [[ "$(file_size "$MODEL")" -ne "$MODEL_BYTES" ]]; then
    echo "ERROR: Download incomplete or failed. Re-run this script to resume." >&2
    exit 1
  fi
fi

echo "Starting Caphlon-2 on http://127.0.0.1:$PORT (OpenAI API: http://127.0.0.1:$PORT/v1)"

(
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      if command -v open >/dev/null 2>&1; then
        open "http://127.0.0.1:$PORT" >/dev/null 2>&1 || true
      elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "http://127.0.0.1:$PORT" >/dev/null 2>&1 || true
      fi
      break
    fi
    sleep 1
  done
) &

if [[ "$(uname -s)" == "Linux" ]]; then
  export LD_LIBRARY_PATH="$DIR/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec "$DIR/bin/llama-server" \
  -m "$MODEL" \
  -ngl "$NGL" \
  -c "$CTX" \
  -np 1 \
  -b 2048 -ub 512 \
  -fa on \
  -ctk q8_0 -ctv q8_0 \
  --cache-ram 0 \
  --ctx-checkpoints 32 \
  -cms 128 \
  --no-context-shift \
  --jinja \
  --reasoning-format deepseek \
  --logit-bias 13275-1.2 --logit-bias 13624-1.2 --logit-bias 3633-1.2 --logit-bias 11041-1.2 --logit-bias 62490-1.2 --logit-bias 58941-1.2 --logit-bias 62200-1.2 \
  --logit-bias 36496-1 --logit-bias 64160-1 --logit-bias 53703-1 --logit-bias 56978-1 --logit-bias 67148-1 --logit-bias 65560-1 \
  --logit-bias 32069-0.8 --logit-bias 46247-0.8 --logit-bias 3384-0.8 --logit-bias 55810-0.8 --logit-bias 17952-0.8 --logit-bias 8027-0.8 --logit-bias 29922-0.8 --logit-bias 52275-0.8 --logit-bias 10278-0.8 --logit-bias 6909-0.8 --logit-bias 20432-0.8 \
  --host 127.0.0.1 \
  --port "$PORT"
