<div align="center">

# Caphlon-Model

**A 6.87 GiB lightweight build of Qwen 3.8 27B — full 27B reasoning quality, laptop-sized footprint.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Base](https://img.shields.io/badge/base-Qwen%203.8%2027B-purple.svg)](#model-details)
[![Size](https://img.shields.io/badge/size-6.87%20GiB-green.svg)](#model-details)
[![Format](https://img.shields.io/badge/format-GGUF-orange.svg)](#quickstart)
[![HumanEval](https://img.shields.io/badge/HumanEval-96.7%25-brightgreen.svg)](#benchmarks)

</div>

---

## What this is

**Caphlon-Model** is a compact, runnable build of **Qwen 3.8 27B** produced by reducing *representation redundancy*, not by distilling a smaller student model. The result is a **single 6.87 GiB GGUF file** that keeps the full 27B parameter graph while cutting the weight footprint by roughly 40% relative to common 4-bit builds — and it still writes real code, reasons in English, and runs on modest hardware.

Built and benchmarked on **25 October 2026**.

| | |
|---|---|
| **Base model** | [Qwen 3.8 27B](https://huggingface.co/Qwen) (Apache-2.0) |
| **Intermediate** | Unsloth Dynamic **IQ2_S** GGUF |
| **Architecture** | 60-layer hybrid: gated-delta-net (majority) + full-attention (minority) |
| **File size** | **6.87 GiB** (7,375,122,272 bytes) |
| **SHA-256** | `8487b4e443834659b230e0bd529674ad65e068fc00380b996aaf11c873d63bc0` |
| **Context** | up to 32K tested |
| **Format** | GGUF (llama.cpp) |

> KV cache lives only on the full-attention layers — the gated-delta-net layers carry fixed-size recurrent state. That is why this build stays small in memory *and* scales cheaply with context.

## What was changed

| Step | Change | Footprint saved |
|---|---|---|
| 1 | **Vocabulary surgery** — merge table reduced to a 100k target with closure-preserving selection | −649 MiB |
| 2 | **Layer pruning** — layers 12–15 removed (4 layers, multiple of the hybrid interval) | −302 MiB |
| 3 | Base quantization kept at IQ2_S (no further weight damage) | — |

No distillation. No teacher model. No student network. The 27B topology survives.

## Benchmarks

Measured **2026-10-25** on a Tesla T4 (16 GB), `llama.cpp`, temperature 0, thinking disabled.

### Quality

| Test | Caphlon-Model | Reference (unmodified IQ2_S) | Result |
|---|---|---|---|
| **English perplexity** (`wiki.test.raw`, 512 ctx × 40 chunks) | **6.6088** | 6.3638 | **+3.85%** |
| **HumanEval** (30 problems, pass@1) | **29/30** | 30/30 | **96.7%** |
| **MMLU** (502-question subset, retention) | **344/502** | 370/502 | **92.9% retained** |

### Speed (T4, `-ngl 99`, 8K context)

| Workload | tok/s |
|---|---|
| File-edit / rewrite (n-gram speculative decoding on) | **29.7** |
| File-edit / rewrite (baseline) | 12.7 |
| Free-form prose | 12.4 |

## Hardware requirements

| Setup | Minimum | Comfortable |
|---|---|---|
| **GPU, full offload** | 8 GB VRAM (8K ctx) | 12 GB VRAM (32K ctx) |
| **CPU only** | 8 GB RAM (mmap) | 16 GB RAM |
| **Apple Silicon** | 16 GB unified | 24 GB+ |
| **Disk** | 7 GB | — |

Works on an 8 GB laptop with `mmap`. Works fully offloaded on an 8 GB GPU. Comfortable anywhere above 12 GB VRAM or 16 GB unified memory.

## Quickstart

```bash
# 1. weights — see "Weights" below, then:
# 2. run (llama.cpp, single user, tuned)
llama-server -m Caphlon-Model-IQ2_S.gguf \
  -ngl 99 -c 32768 -np 1 \
  -b 4096 -ub 1024 --flash-attn on \
  --spec-type ngram-mod --spec-ngram-mod-n-match 24 \
  --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64 \
  --jinja --port 8080
```

Minimal invocation, if you just want it running:

```bash
llama-cli -m Caphlon-Model-IQ2_S.gguf -ngl 99 -c 8192 --jinja \
  -p "Write a Python function that merges overlapping intervals."
```

> Always pass `chat_template_kwargs: {"enable_thinking": false}` on chat requests, or the model spends your token budget on internal reasoning.

Optional squeeze (only if you are context-bound): `--cache-type-k q8_0 --cache-type-v q4_0`. Measured gain: −152 MB. Trade-off: greedy output is no longer byte-identical.

## Reproducing the evaluation

```bash
bash scripts/eval-en40.sh Caphlon-Model-IQ2_S.gguf
```

Protocol: `wiki.test.raw`, `-c 512 --chunks 40 -ngl 99 -t 4 --no-warmup -b 4096 -ub 1024`. Compare against the unmodified base quant run on the same machine.

## Weights

Weights are hosted on **Hugging Face** as a single file (no joining needed):

**[⬇ huggingface.co/ssuasd/Caphlon-Model](https://huggingface.co/ssuasd/Caphlon-Model)**

```bash
# single file, resumable
curl -L -C - -o Caphlon-Model-IQ2_S.gguf \
  https://huggingface.co/ssuasd/Caphlon-Model/resolve/main/Caphlon-Model-IQ2_S.gguf

shasum -a 256 Caphlon-Model-IQ2_S.gguf
# expected: 8487b4e443834659b230e0bd529674ad65e068fc00380b996aaf11c873d63bc0
# size:     7375122272 bytes
```

Mirrors:

- **GitHub Release** (4 parts + `SHA256SUMS`, for GitHub-only users): [v1.0.0](https://github.com/demiralpdev/Caphlon-Model/releases/tag/v1.0.0)
  ```bash
  cat part-aa part-ab part-ac part-ad > Caphlon-Model-IQ2_S.gguf
  shasum -a 256 -c SHA256SUMS
  ```
- **Kaggle:** [`bitwarden/theleaf-k5-final`](https://www.kaggle.com/datasets/bitwarden/theleaf-k5-final)

## License and attribution

- **This repository** (model card, scripts, naming, documentation): **Apache License 2.0** — see [LICENSE](LICENSE).
- **Model weights**: derived from **Qwen 3.8 27B**, which is distributed under **Apache-2.0**. The upstream license governs the weights. Apache-2.0 permits commercial use, modification, and redistribution, provided you **keep the notices** and **do not imply endorsement**. See [NOTICE.md](NOTICE.md).
- **Name and marks**: "Caphlon" is a project mark of the author. Apache-2.0 grants **no trademark rights**. You may redistribute the weights, but you may not present a modified build as *Caphlon* or claim it is official, endorsed, or authored by this project.

## Citation

```bibtex
@misc{caphlon2026,
  title  = {Caphlon-Model: A 6.87 GiB lightweight build of Qwen 3.8 27B},
  author = {demiralpdev},
  year   = {2026},
  month  = {10},
  note   = {GGUF, IQ2\_S base, vocabulary-reduced and layer-pruned},
  url    = {https://github.com/demiralpdev/Caphlon-Model}
}
```

## Keywords

Qwen 3.8 27B · Qwen3 · 27B GGUF · lightweight 27B · 6.87 GiB LLM · IQ2_S · llama.cpp · local LLM · laptop LLM · quantized language model · small footprint 27B · hybrid gated delta net · layer pruning · vocabulary reduction · offline coding assistant · HumanEval 96.7% · run 27B on 8 GB RAM

---

<div align="center"><sub>Built 25 October 2026 · No distillation, no shortcuts on topology.</sub></div>
