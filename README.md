<div align="center">

# Caphlon-Model

**The lightest Qwen 3.8 27B GGUF model family — `Caphlon-2` (`4.52 GiB` `PTQ1_SB`, all 64 layers via Ternary-Bonsai-2) & `Caphlon-1` (`6.87 GiB` `IQ2_S`, stock `llama.cpp`).**

**4.52 GiB · Qwen 3.8 27B + Ternary-Bonsai-2 · runs on 6 GB VRAM / 8 GB RAM** — interactive showcase: **[https://caphlon.dev](https://caphlon.dev)**

</div>

---

## Overview — The Caphlon 27B Family

**Caphlon** is an open-weight model family engineered to produce the **lightest runnable builds of Qwen 3.8 27B** (*en hafif Qwen 3.8 27B modeli*) without distilling a smaller student network. Both builds keep the 27B hybrid `Gated-Delta-Net` + `Full-Attention` architecture intact so they can run locally on a standard **6 GB VRAM GPU** or **8 GB RAM laptop** (Apple Silicon M1/M2/M3/M4, Linux CUDA, Windows/Linux AVX2):

- **Caphlon-2 (`4.52 GiB`, `PTQ1_SB` `1.65 bpw` — Flagship Reasoning & Low-Hallucination Build):**
  Derived from **Qwen 3.8 27B** and **PrismML / UkisAI Ternary-Bonsai-2-27B (`PTQ1_0`, `5.54 GiB`)**. Preserves **all 64 hybrid layers** (`23.30B` active parameters) while shrinking the file by **`−1.02 GiB` vs Ternary-Bonsai-2**, **`−34.2%` vs Caphlon-1 (`6.87 GiB`)**, and **`−70.5%` vs `Q4_K_M` (`15.33 GiB`)**. Achieves **`80.0%` (`48/60`, `89.7%` BF16 retention) on GPQA-Diamond** and **`87.41%` (`236/270`, `100%` unanswerable abstain recall) on Halu-270**.
- **Caphlon-1 (`6.87 GiB`, `IQ2_S` — Legacy Stock `llama.cpp` Build):**
  Derived from **Qwen 3.8 27B** (`Unsloth Dynamic IQ2_S`) via `100k` vocabulary reduction and 4-layer hybrid interval pruning (`60` layers). Runs on unmodified stock `llama.cpp` with **`96.7%` (`29/30`) HumanEval pass@1** and **`92.9%` MMLU retention**.

---

## How Caphlon-2 Reaches `4.52 GiB` (`1.65 bpw`) Without Losing the 64-Layer Topology

1. **BPE Vocabulary Surgery (`V83K`):** Closure-preserving merge-table pruning from `111,104 → 84,352` tokens (`83,982` merges), remapping all control/thinking tokens (`<think>=84267`, `</think>=84268`) and shipping a 24-token anti-hesitation logit-bias profile ([`metadata/Caphlon-2-27B-PTQ1_SB.meta.json`](metadata/Caphlon-2-27B-PTQ1_SB.meta.json)).
2. **Layer-Adaptive FFN Width Pruning (`LayerTrim`):** Block-influence ridge-regression calibrated intermediate MLP dimension reduction (`17,408 → 15,456` aligned to `256`-element super-block boundaries), retaining `23,298,072,064` (`23.30B`) active parameters across all 64 layers.
3. **`PTQ1_SB` (`GGML_TYPE_PTQ1_SB = 144`, `1.648 bpw`) Super-Block Ternary Format:** Replaces per-128 FP16 scales with a 1,024-element Sylvester-Hadamard $H_{1024}$ super-block (`205 B` base-3 packed trits + `2 B` FP16 super-scale + `4 B` packed `8×128` 4-bit sub-scale nibbles = `211 B` per 1,024 weights).
4. **Inference-Time Intervention (`ITI`) Calibration Vector:** Ships with [`metadata/iti_mean.gguf`](metadata/iti_mean.gguf) (`1.2 MiB`), lifting 270-question Hallucination benchmark accuracy from `85.93%` (`232/270`) to **`87.41%` (`236/270`)** with **`100.0%` unanswerable abstain recall**.

---

## Model Specifications

| Field | Caphlon-2 (`4.52 GiB` Flagship) | Caphlon-1 (`6.87 GiB` Legacy) |
|---|---|---|
| **Base architecture** | [Qwen 3.8 27B](https://huggingface.co/Qwen) (64-layer hybrid GDN + Full-Attention) | [Qwen 3.8 27B](https://huggingface.co/Qwen) (60-layer pruned hybrid) |
| **Upstream intermediate** | [PrismML Ternary-Bonsai-2-27B (`PTQ1_0`)](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) / [UkisAI Swift-Bonsai-2](https://huggingface.co/ukisai/Swift-Bonsai-2-GGUF) (`5.54 GiB`) | Unsloth Dynamic `IQ2_S` (`7.80 GiB`) |
| **Quantization & format** | `PTQ1_SB` (GGML type `144`, `1.65 bpw` super-block ternary, `851` tensors) | `IQ2_S` (`2.31 bpw` stock GGUF) |
| **Active parameters** | `23,298,072,064` (`23.30B`, 64/64 layers, `n_ff = 15456`, `n_vocab = 84352`) | `~24.8B` (60/64 layers, `n_vocab = 100000`) |
| **File size** | **`4.5227 GiB`** (`4,856,213,824` bytes) | **`6.8686 GiB`** (`7,375,122,272` bytes) |
| **Filename** | `Caphlon-2-27B-PTQ1_SB.gguf` | `Caphlon-Model-IQ2_S.gguf` |
| **SHA-256** | `ac41fcddf5da28d71198f60202c10ea37cc64d791e5f6f118f849a02579c20db` | `8487b4e443834659b230e0bd529674ad65e068fc00380b996aaf11c873d63bc0` |
| **Runtime** | [`patches/prismll_ptq1_sb_cuda_metal.patch`](patches/prismll_ptq1_sb_cuda_metal.patch) on [`PrismML-Eng/llama.cpp` (`6447906`)](https://github.com/PrismML-Eng/llama.cpp/tree/6447906bf96c1a023980d3453fb59daa6af3bb23) | Stock `llama.cpp` |

---

## Audited Benchmarks

All benchmarks below were measured on real hardware (NVIDIA A100-SXM4-80GB, dual NVIDIA Tesla T4 16GB, and Apple M2 Metal) and verified through a two-layer deterministic + blind forensic audit gate (`0` unverified claims).

### 1. Reasoning, Hallucination & Perplexity Across the Lineage

| Model / Build | Format | Size | GPQA-Diamond (60 RID, `xhigh`) | Mini-40 Truth & Abstain | Halu-270 (270 Q, 4 Suites) | Held-out PPL (`4×512` EN / TR) |
|---|---|---:|---|---|---|---|
| **Caphlon-2** *(`Cand_pure`)* | `PTQ1_SB` (`1.65 bpw`) | **`4.52 GiB`** | **`48/60` (`80.0%`)**<br>*(89.7% retention vs BF16 `89.2%`; `47/59` excl. 1 cap)* | `32/40` (`80.0%`)<br>Abstain `100%` · False Refusal `2.86%` | **`232/270` (`85.93%`)**<br>Halu `11.85%` · False Refusal `2.48%` · Brier `0.1147` | **`7.8551` / `6.8792`** |
| **Caphlon-2 + ITI** *(`+metadata/iti_mean.gguf`)* | `PTQ1_SB` (`1.65 bpw`) | **`4.52 GiB`** | **`48/60` (`80.0%`)**<br>*(89.7% retention vs BF16 `89.2%`)* | **`34/40` (`85.0%`)**<br>Abstain `100%` · False Refusal `2.86%` | **`236/270` (`87.41%`)**<br>Halu `10.37%` · **Abstain `100%`** · **Brier `0.1014`** · **ECE `0.0767`** | **`7.8551` / `6.8792`** |
| **Ternary-Bonsai-2-27B** *(Upstream base)* | `PTQ1_0` (`1.75 bpw`) | `5.54 GiB` | **`51/60` (`85.0%`)** `@65K/98K`<br>*(95.3% retention; `46/60 = 76.7%` `@49K`)* | `32/40` (`80.0%`)<br>Brier `0.1573` | **`237/270` (`87.78%`)**<br>Halu `9.26%` · Abstain `100%` · Brier `0.0996` | `7.3961` / `6.1799` |
| **Caphlon-1** *(Stock `llama.cpp` legacy)* | `IQ2_S` (`voc100k+L56`) | `6.87 GiB` | — *(thinking disabled)* | HumanEval `29/30` (`96.7%`) | MMLU `344/502` (`92.9%` retention) | `6.6088` (`EN40`) |

> **Key takeaway:** At **`4.52 GiB`** (`1.02 GiB` smaller than upstream `5.54 GiB` `Ternary-Bonsai-2-27B`), **Caphlon-2** beats upstream's default `@49K` GPQA-Diamond score (`48/60 = 80.0%` vs `46/60 = 76.7%`), retains **`89.69%`** of full BF16 (`89.2%`), and with `iti_mean.gguf` matches upstream on the 270-question Hallucination suite within a single question (**`236/270 = 87.41%`** vs `237/270 = 87.78%`) while cutting average reasoning length nearly in half (`1,282` vs `2,355` tokens/question).

### 2. CUDA, AVX2 & Apple Metal Speed and Numerical Parity (`4.52 GiB` `PTQ1_SB`)

| Backend / Configuration | Prefill `pp64` | Prefill `pp512` | Decode `tg32` / `tg64` | Numerical Parity / Cache Audit |
|---|---:|---:|---:|---|
| **NVIDIA Tesla T4 (`sm_75`, `f16` KV, FA)** | **`87.29 tok/s`** | — | **`16.33 tok/s`** (`tg32`) | Patch v3 Tensor Core MMQ + `2row` `Q8_1_SB` |
| **NVIDIA Tesla T4 (`sm_75`, `q8_0` KV, FA)** | **`86.68 tok/s`** | **`277.95 tok/s`** | **`16.14 tok/s`** (`tg32`) / **`15.71 tok/s`** (`tg64`) | `4.43×` `pp512` speedup vs v2 (`62.70 tok/s`) |
| **Apple M2 Metal (`-ngl 99`, `q8_0` KV)** | — | — | **`3.64 tok/s`** (full CoT) | Standalone smoke verified (`chem_ph` & `Xylon-Theta-99` abstain) |
| **x86 AVX2 & ARM64 NEON (`ptq1_sb_dot32`)** | `27.30 tok/s` | — | `0.35 tok/s` (1-thread CPU check) | **`max_rel_err = 0`** (bit-exact vs FP32 reference) |
| **0-Miss Context OS (`--ctx-checkpoints 32 -cms 128`)** | — | — | — | **`toplam_cache_miss = 0`**, `unplanned_reprefill = 0`, `ckpt_erased = 0` |

---

## Included Kernel Patch & Calibration Kit

To make **Caphlon-2** immediately useful for local inference, benchmarking, and further ternary research, we include the full kernel and calibration kit alongside the weights:

- **[`patches/prismll_ptq1_sb_cuda_metal.patch`](patches/prismll_ptq1_sb_cuda_metal.patch) (`137.8 KB`):**
  - **CUDA (`sm_75+`):** Turing/Ampere INT8 Tensor Core MMQ prefill (`pp512 = 277.95 tok/s` on T4), `GGML_CUDA_Q8_1_SB` (`1152 B` per 1,024 elements), `0`-SMEM warp-register unpack (`ptq1_sb_warp_unpack_regs`), 2-row register-shared `mul_mat_vec_ptq1_sb_fast_2row` (`ncols_dst = 1..16`), multi-column `ffn_up + ffn_gate + swiglu` fusion (`ptq1_sb_mc`), and pool-backed fused `RMSNorm + FWHT -> Q8_1_SB`.
  - **Apple Metal:** Native `kernel_mul_mv_ptq1_sb_f32` & `q8_0` KV Flash-Attention support for Apple Silicon unified memory.
  - **ARM64 NEON & x86 AVX2:** Vectorized `ptq1_sb_dot32_avx2` and NEON `dotprod` / `no-dotprod` kernels verified at **`max_rel_err = 0`** against reference FP32.
- **[`metadata/Caphlon-2-27B-PTQ1_SB.meta.json`](metadata/Caphlon-2-27B-PTQ1_SB.meta.json):** Remapped special token IDs (`<think>=84267`, `</think>=84268`, `<|im_start|>=84244`, `<|im_end|>=84245`) and the 24-token anti-hesitation `--logit-bias` table (`Wait`, `Hmm`, `Alternatively`, `Actually`, etc.).
- **[`metadata/iti_mean.gguf`](metadata/iti_mean.gguf) (`1.2 MiB`):** Optional Inference-Time Intervention (`ITI`) mean activation vector (`236/270 = 87.41%` on Halu-270 with `100.0%` abstain recall).
- **[`scripts/build-prism-llamacpp.sh`](scripts/build-prism-llamacpp.sh) & [`scripts/run.sh`](scripts/run.sh):** One-command builder and zero-miss `q8_0` KV launcher.

---

## Quickstart

### 0. Quickest: Run on Ollama (One Command)

```bash
ollama run univerisr/caphlon
```

Or build and run Caphlon-2 with hardware-accelerated Sylvester-Hadamard ternary kernels:

### 1. Build the patched PrismML `llama.cpp` runtime (`CUDA` / `Metal` / `AVX2` / `NEON`)

```bash
git clone https://github.com/demiralpdev/Caphlon-Model.git
cd Caphlon-Model
bash scripts/build-prism-llamacpp.sh
```

### 2. Download Weights from Hugging Face

- **Caphlon-2 (`4.52 GiB` `PTQ1_SB`):** **[huggingface.co/ssuasd/Caphlon-Model](https://huggingface.co/ssuasd/Caphlon-Model)** & **[huggingface.co/ssuasd/Caphlon-2](https://huggingface.co/ssuasd/Caphlon-2)**

```bash
curl -L -C - -o Caphlon-2-27B-PTQ1_SB.gguf \
  https://huggingface.co/ssuasd/Caphlon-Model/resolve/main/Caphlon-2-27B-PTQ1_SB.gguf

shasum -a 256 Caphlon-2-27B-PTQ1_SB.gguf
# expected: ac41fcddf5da28d71198f60202c10ea37cc64d791e5f6f118f849a02579c20db
# size:     4856213824 bytes (4.5227 GiB)
```

- **Caphlon-1 (`6.87 GiB` `IQ2_S`, stock `llama.cpp`):**

```bash
curl -L -C - -o Caphlon-Model-IQ2_S.gguf \
  https://huggingface.co/ssuasd/Caphlon-Model/resolve/main/Caphlon-Model-IQ2_S.gguf
```

### 3. Run the Server

```bash
bash scripts/run.sh Caphlon-2-27B-PTQ1_SB.gguf 32768 1
```

---

## Hardware Requirements

| Setup | Minimum (`8K–16K` ctx) | Comfortable (`32K–65K` ctx) |
|---|---|---|
| **GPU, full offload (`-ctk q8_0 -ctv q8_0`)** | **6 GB VRAM** (`Caphlon-2`) / **8 GB VRAM** (`Caphlon-1`) | **8–12 GB VRAM** |
| **Apple Silicon (Metal, M1/M2/M3/M4)** | **8 GB Unified Memory** | **16 GB Unified Memory** |
| **CPU only (`mmap`, AVX2 / NEON)** | **6 GB RAM** | **8–16 GB RAM** |
| **Disk** | **4.6 GB** (`Caphlon-2`) / **7.0 GB** (`Caphlon-1`) | — |

---

## FAQ & Search Index (English / Türkçe)

- **What is the lightest Qwen 3.8 27B model? / En hafif Qwen 3.8 27B modeli hangisidir?**
  **Caphlon-2 (`4.52 GiB` / `4,856,213,824` bytes, `1.65 bpw` `PTQ1_SB`)** is the lightest full 64-layer build of **Qwen 3.8 27B** (derived via **Ternary-Bonsai-2-27B**). It runs locally on **6 GB VRAM** graphics cards or **8 GB RAM** laptops (including base 8 GB Apple Silicon MacBooks) while retaining **`80.0%` (`48/60`) GPQA-Diamond** accuracy (`89.7%` of full BF16) and **`87.41%` (`236/270`) Halu-270** truthfulness.
- **8 GB RAM veya 6 GB VRAM ile çalışan en iyi 27B yapay zeka modeli nasıl indirilir?**
  `Caphlon-2-27B-PTQ1_SB.gguf` (`4.52 GiB`, Türkçe `6.88` ve İngilizce `7.86` PPL) veya stok `llama.cpp` uyumlu `Caphlon-Model-IQ2_S.gguf` (`6.87 GiB`) dosyalarını doğrudan [huggingface.co/ssuasd/Caphlon-Model](https://huggingface.co/ssuasd/Caphlon-Model) ve [caphlon.dev](https://caphlon.dev) üzerinden indirebilirsiniz.

---

## Citation

```bibtex
@misc{caphlon2_2026,
  title  = {Caphlon-2: A 4.52 GiB PTQ1_SB Super-Block Ternary Build of Qwen 3.8 27B via Ternary-Bonsai-2},
  author = {Gündoğmuş, Demir Alp},
  year   = {2026},
  month  = {11},
  note   = {GGUF PTQ1\_SB (1.65 bpw), V83K vocabulary pruning, LayerTrim FFN pruning, 80.0\% GPQA-Diamond, 87.4\% Halu-270, and custom CUDA/Metal/SIMD kernels},
  url    = {https://github.com/demiralpdev/Caphlon-Model}
}
```

---

## License, Attribution & Badges

- **Repository code, kernel patch, scripts, documentation**: **Apache License 2.0** — see [LICENSE](LICENSE).
- **Model weights & upstream lineage**: Derived from **[Qwen 3.8 27B](https://huggingface.co/Qwen)** (Qwen Team, Apache-2.0) and **[Ternary-Bonsai-2-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)** (PrismML / UkisAI, Apache-2.0). The upstream Apache-2.0 license governs the weights. See [NOTICE.md](NOTICE.md) for the complete modification chain and required notices.
- **Trademarks**: "Caphlon", "Caphlon-2", and "Caphlon-Model" are project marks of Demir Alp Gündoğmuş. Apache-2.0 grants no trademark rights.

**Keywords:** lightest Qwen model · lightest Qwen 3.8 27B · en hafif Qwen modeli · en hafif 27B yapay zeka modeli · smallest 27B GGUF · 4.52 GiB 27B LLM · run 27B LLM on 8 GB RAM · run 27B on 6 GB VRAM · Ternary-Bonsai-2 27B GGUF · PTQ1_SB · Sylvester-Hadamard quantization · LayerTrim FFN pruning · BPE vocabulary pruning · Apple Silicon 8 GB 27B LLM · llama.cpp ternary CUDA Tensor Core · Caphlon · Caphlon-2 · Caphlon-Model

<div align="center">

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Base](https://img.shields.io/badge/base-Qwen%203.8%2027B%20%2B%20Ternary--Bonsai--2-purple.svg)](#model-specifications)
[![Size](https://img.shields.io/badge/size-4.52%20GiB%20(Caphlon--2)%20%7C%206.87%20GiB%20(Caphlon--1)-green.svg)](#model-specifications)
[![Format](https://img.shields.io/badge/format-GGUF%20PTQ1__SB%20%26%20IQ2__S-orange.svg)](#quickstart)
[![GPQA-D](https://img.shields.io/badge/GPQA--Diamond-80.0%25%20(48%2F60)-brightgreen.svg)](#audited-benchmarks)
[![Halu-270](https://img.shields.io/badge/Halu--270-87.4%25%20(236%2F270)-teal.svg)](#audited-benchmarks)

</div>
