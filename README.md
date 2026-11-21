---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - prism-ml/Ternary-Bonsai-2-27B-gguf
  - ukisai/Swift-Bonsai-2-GGUF
library_name: gguf
pipeline_tag: text-generation
language:
  - en
  - tr
tags:
  - gguf
  - caphlon
  - caphlon-2
  - qwen3.8
  - bonsai
  - ternary
  - ptq1_sb
  - hadamard
  - layertrim
  - vocabulary-pruning
  - llama.cpp
---

<div align="center">

# Caphlon-2 (`4.52 GiB` `PTQ1_SB`)

**A 4.52 GiB super-block ternary (`1.65 bpw`) build of Qwen 3.8 27B derived from Ternary-Bonsai-2 — preserving all 64 hybrid layers (`23.30B` active params) with custom CUDA Tensor Core, Apple Metal, ARM64 NEON & x86 AVX2 kernels.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Base](https://img.shields.io/badge/base-Qwen%203.8%2027B%20%2B%20Ternary--Bonsai--2-purple.svg)](#lineage--what-was-changed)
[![Size](https://img.shields.io/badge/size-4.52%20GiB%20(1.65%20bpw)-green.svg)](#model-details)
[![Format](https://img.shields.io/badge/format-GGUF%20PTQ1__SB-orange.svg)](#quickstart)
[![Mini-40](https://img.shields.io/badge/Mini--40%20Truth-87.5%25-brightgreen.svg)](#audited-benchmarks)
[![GPQA-D](https://img.shields.io/badge/GPQA--D-80.0%25%20Base%20%7C%2070.0%25%20%4022K-teal.svg)](#audited-benchmarks)

</div>

**4.52 GiB · Qwen 3.8 27B + Ternary-Bonsai-2 · fits in 6 GB VRAM / 8 GB RAM** — interactive showcase: **https://caphlon.dev**

---

## What this is

**Caphlon-2** is our second-generation ultra-compact build of **Qwen 3.8 27B**, built directly on top of the Sylvester-Hadamard $H_{1024}$ rotated **Ternary-Bonsai-2-27B (`PTQ1_0`, 5.54 GiB)** release from **PrismML / UkisAI**.

Instead of dropping whole transformer blocks or distilling a tiny student network, **Caphlon-2** keeps **all 64 hybrid layers** (`Gated-Delta-Net` recurrent layers + `Full-Attention` layers) and compresses the footprint from `5.54 GiB` down to **`4.5227 GiB` (`4,856,213,824` bytes, `−34.2%` smaller than Caphlon v1 `6.87 GiB` and `−70.5%` smaller than `Q4_K_M` `15.33 GiB`)** through four synergistic techniques:

1. **BPE Vocabulary Surgery (`V83K`):** Closure-preserving merge-table pruning from `111,104 → 84,352` tokens (`83,982` merges), remapping all control/thinking tokens (`<think>=84267`, `</think>=84268`) and shipping a 24-token anti-hesitation logit-bias profile ([`metadata/Caphlon-2-27B-PTQ1_SB.meta.json`](metadata/Caphlon-2-27B-PTQ1_SB.meta.json)).
2. **Layer-Adaptive FFN Width Pruning (`LayerTrim`):** Block-influence calibrated intermediate MLP dimension reduction (`17,408 → 15,456` aligned to `256`-element super-block boundaries), retaining `23,298,072,064` (`23.30B`) active parameters across all 64 layers.
3. **`PTQ1_SB` (`GGML_TYPE_PTQ1_SB = 144`, `1.648 bpw`) Super-Block Ternary Format:** Replaces per-128 FP16 scales with a 1,024-element Sylvester-Hadamard super-block (`205 B` base-3 packed trits + `2 B` FP16 super-scale + `4 B` packed `8×128` 4-bit sub-scale nibbles = `211 B` per 1,024 weights).
4. **Zero-Overhead In-Place Folded Stage-2 OPD LoRA:** On-Policy Distillation (`OPD`) LoRA adapters (`400` `lora_a`/`lora_b` pairs trained on A100 against the unpruned teacher mapping `84,276 / 84,352` student tokens) are mathematically rotated by $D \cdot H_{1024}$ and folded directly into the `PTQ1_SB` ternary super-blocks (`0 --lora` runtime penalty).

---

## Model Details

| Field | Value |
|---|---|
| **Base architecture** | [Qwen 3.8 27B](https://huggingface.co/Qwen) (64-layer hybrid Gated-Delta-Net + Full-Attention, Apache-2.0) |
| **Upstream ternary base** | [PrismML Ternary-Bonsai-2-27B (`PTQ1_0`)](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) / [UkisAI Swift-Bonsai-2](https://huggingface.co/ukisai/Swift-Bonsai-2-GGUF) (`5.54 GiB`, `1.75 bpw`) |
| **Quantization & format** | `PTQ1_SB` (GGML type `144`, `1.65 bpw` super-block ternary, `402` `PTQ1_SB` + `353` `F32` + `96` `BF16` = `851` tensors) |
| **Active parameters** | `23,298,072,064` (`23.30B`, 64/64 layers intact, `n_embd = 5120`, `n_ff = 15456`, `n_vocab = 84352`) |
| **Standalone file size** | **`4.5227 GiB`** (`4,856,213,824` bytes; tensor payload `4,853,038,336` bytes) |
| **SHA-256 (`Caphlon-2-27B-PTQ1_SB.gguf`)** | `fb898b3122aa12b1a0a73c3bfe3a8d7cfc213ce6646edec7692283a00cb41105` |
| **Runtime & kernel patch** | [`patches/prismll_ptq1_sb_cuda_metal.patch`](patches/prismll_ptq1_sb_cuda_metal.patch) on [`PrismML-Eng/llama.cpp` (`6447906`)](https://github.com/PrismML-Eng/llama.cpp/tree/6447906bf96c1a023980d3453fb59daa6af3bb23) |
| **Supported backends** | NVIDIA CUDA (`sm_75` Turing T4, `sm_80` A100, RTX 30/40), Apple Metal (M1/M2/M3/M4), ARM64 NEON, x86 AVX2 |

---

## What is Included in This Repository

To make **Caphlon-2** immediately useful for local inference, benchmarking, and further ternary research, we include the full kernel and metadata kit alongside the weights:

- **[`patches/prismll_ptq1_sb_cuda_metal.patch`](patches/prismll_ptq1_sb_cuda_metal.patch) (`137.8 KB`):**
  - **CUDA (`sm_75+`):** Turing/Ampere INT8 Tensor Core MMQ prefill (`pp512 = 277.95 tok/s` on T4, `4.43×` faster than tiled fallback), `GGML_CUDA_Q8_1_SB` (`1152 B` per 1,024 elements), `0`-SMEM warp-register unpack (`ptq1_sb_warp_unpack_regs`), 2-row register-shared `mul_mat_vec_ptq1_sb_fast_2row` (`ncols_dst = 1..16`), multi-column `ffn_up + ffn_gate + swiglu` fusion (`ptq1_sb_mc`), and pool-backed fused `RMSNorm + FWHT -> Q8_1_SB`.
  - **Apple Metal:** Native `kernel_mul_mv_ptq1_sb_f32` & `q8_0` KV Flash-Attention support for Apple Silicon unified memory.
  - **ARM64 NEON & x86 AVX2:** Vectorized `ptq1_sb_dot32_avx2` and NEON `dotprod` / `no-dotprod` kernels verified at **`max_rel_err = 0`** against reference FP32.
- **[`metadata/Caphlon-2-27B-PTQ1_SB.meta.json`](metadata/Caphlon-2-27B-PTQ1_SB.meta.json):** Remapped special token IDs (`<think>=84267`, `</think>=84268`, `<|im_start|>=84244`, `<|im_end|>=84245`) and the 24-token anti-hesitation `--logit-bias` table (`Wait`, `Hmm`, `Alternatively`, `Actually`, etc.) that curbs circular overthinking loops.
- **[`scripts/build-prism-llamacpp.sh`](scripts/build-prism-llamacpp.sh):** Automated builder that fetches `PrismML-Eng/llama.cpp` (`6447906`), applies `prismll_ptq1_sb_cuda_metal.patch`, and compiles `llama-server`, `llama-cli`, `llama-bench`, and `llama-perplexity` for macOS Metal or Linux CUDA/AVX2.
- **[`scripts/run.sh`](scripts/run.sh):** Launch script pre-configured with lossless `q8_0` KV cache (`-fa on -ctk q8_0 -ctv q8_0`), `0`-miss Context OS checkpointing (`--cache-ram 0 --ctx-checkpoints 32 -cms 128 --no-context-shift`), and automatic anti-hesitation logit-bias injection.

---

## Audited Benchmarks

All benchmarks below were measured on real hardware (NVIDIA A100-SXM4-80GB, dual NVIDIA Tesla T4 16GB on Kaggle, and Apple M2 Metal) and verified through a two-layer deterministic + blind forensic audit gate (`0` unverified claims).

### 1. Reasoning, Hallucination & Perplexity Across the Lineage

| Model / Build | Format | Size | GPQA-Diamond (60 RID, `xhigh`) | Mini-40 Truth & Abstain | Halu-270 (270 Q, 4 Suites) | Held-out PPL (`4×512` EN / TR) |
|---|---|---:|---|---|---|---|
| **Ternary-Bonsai-2-27B** *(Upstream)* | `PTQ1_0` (`1.75 bpw`) | `5.54 GiB` | **`51/60` (`85.0%`)** `@65K/98K`<br>*(95.3% vs BF16 `89.2%`; `46/60` `@49K`)* | `32/40` (`80.0%`)<br>Brier `0.1573` | **`237/270` (`87.8%`)**<br>Halu `9.26%` · Abstain `100%` · Brier `0.0996` | `7.3961` / `6.1799` |
| **Caphlon-2 Base** *(`cand_tam_bsqrt`)* | `PTQ1_SB` (`1.65 bpw`) | **`4.52 GiB`** | **`48/60` (`80.0%`)** `@65K` pure<br>*(89.7% retention vs BF16)* | `32/40` (`80.0%`) pure<br>**`34/40` (`85.0%`)** `+ITI` | `232/270` (`85.9%`) pure<br>**`236/270` (`87.4%`)** `+ITI` (`100%` abstain, Brier `0.1014`) | `7.8551` / `6.8792` |
| **Caphlon-2 Standalone** *(`Folded Stage-2 OPD`)* | `PTQ1_SB` (`1.65 bpw`) | **`4.52 GiB`** | **`42/60` (`70.0%`)** `@22K` PCCL+TIR<br>*(Physics `22/26 = 84.6%`, Bio `4/6`, Chem `16/28`)* | **`35/40` (`87.5%`)** *(OPD & TIR)*<br>Abstain `100%` · False Refusal `0.0%` · Brier `0.115` | **`224/270` (`83.0%`)** *(TIR)*<br>False Refusal `2.07%` · Brier `0.1277` | `7.8556` / `6.8824` (`f16` KV)<br>`7.8585` / `6.8789` (`q8_0` KV) |
| **Caphlon v1** *(Stock `llama.cpp` legacy)* | `IQ2_S` (`voc100k+L56`) | `6.87 GiB` | — *(thinking disabled)* | HumanEval `29/30` (`96.7%`) | MMLU `344/502` (`92.9%` retention) | `6.6088` (`EN40`) |

> **Why report both `@65K` Base (`48/60 = 80.0%`) and `@22K` PCCL+TIR (`42/60 = 70.0%`) on GPQA-Diamond?**
> GPQA-Diamond organic chemistry synthesis chains require deep reasoning traces (`20K–47K` tokens). Under a tight `22,528`-token ceiling on free dual-T4 GPUs (`k3_v3`), 9 long questions hit the `22K` cap and 7 questions triggered staged early termination (`3/7 = 42.9%` accuracy vs `39/53 = 73.6%` on normal completion). When given a `65,536`-token ceiling on A100 (`Cand_pure`), the exact same `4.52 GiB` `PTQ1_SB` architecture solves **`48/60` (`80.0%`)**.

### 2. CUDA, AVX2 & Apple Metal Speed and Numerical Parity (`4.52 GiB` Standalone GGUF)

| Backend / Configuration | Prefill `pp64` | Prefill `pp512` | Decode `tg32` / `tg64` | Numerical Parity / Cache Audit |
|---|---:|---:|---:|---|
| **NVIDIA Tesla T4 (`sm_75`, `f16` KV, FA)** | **`87.29 tok/s`** | — | **`16.33 tok/s`** (`tg32`) | Patch v3 Tensor Core MMQ + `2row` `Q8_1_SB` |
| **NVIDIA Tesla T4 (`sm_75`, `q8_0` KV, FA)** | **`86.68 tok/s`** | **`277.95 tok/s`** | **`16.14 tok/s`** (`tg32`) / **`15.71 tok/s`** (`tg64`) | `4.43×` `pp512` speedup vs v2 (`62.70 tok/s`) |
| **Apple M2 Metal (`-ngl 99`, `q8_0` KV, `0 --lora`)** | — | — | **`3.64 tok/s`** (full CoT) | Standalone smoke verified (`chem_ph` & `Xylon-Theta-99` abstain) |
| **x86 AVX2 & ARM64 NEON (`ptq1_sb_dot32`)** | `27.30 tok/s` | — | `0.35 tok/s` (1-thread CPU check) | **`max_rel_err = 0`** (bit-exact vs FP32 reference) |
| **0-Miss Context OS (`--ctx-checkpoints 32 -cms 128`)** | — | — | — | **`toplam_cache_miss = 0`**, `unplanned_reprefill = 0`, `ckpt_erased = 0` |

---

## Quickstart

### 1. Build the patched PrismML `llama.cpp` runtime (`CUDA` / `Metal` / `AVX2` / `NEON`)

```bash
git clone https://github.com/demiralpdev/Caphlon-Model.git
cd Caphlon-Model
bash scripts/build-prism-llamacpp.sh
```

### 2. Download `Caphlon-2-27B-PTQ1_SB.gguf` (`4.52 GiB`)

**[⬇ huggingface.co/ssuasd/Caphlon-2](https://huggingface.co/ssuasd/Caphlon-2)**

```bash
curl -L -C - -o Caphlon-2-27B-PTQ1_SB.gguf \
  https://huggingface.co/ssuasd/Caphlon-2/resolve/main/Caphlon-2-27B-PTQ1_SB.gguf

shasum -a 256 Caphlon-2-27B-PTQ1_SB.gguf
# expected: fb898b3122aa12b1a0a73c3bfe3a8d7cfc213ce6646edec7692283a00cb41105
# size:     4856213824 bytes (4.5227 GiB)
```

*(Looking for the legacy `6.87 GiB` `IQ2_S` build for stock `llama.cpp`? It remains available at [`ssuasd/Caphlon-Model`](https://huggingface.co/ssuasd/Caphlon-Model) and [GitHub Release `v1.0.0`](https://github.com/demiralpdev/Caphlon-Model/releases/tag/v1.0.0).)*

### 3. Run the server (with lossless `q8_0` KV cache, 0-miss Context OS & anti-hesitation logit-bias)

```bash
bash scripts/run.sh Caphlon-2-27B-PTQ1_SB.gguf 32768 1
```

Or invoke `llama-cli` directly:

```bash
./build-prism/llbuild/bin/llama-cli \
  -m Caphlon-2-27B-PTQ1_SB.gguf \
  -ngl 99 -c 32768 -fa on -ctk q8_0 -ctv q8_0 --jinja \
  -p "Calculate the pH of a 0.001 M HCl solution at 25 C. State your final answer clearly."
```

---

## Hardware Requirements

| Setup | Minimum (`8K–16K` ctx) | Comfortable (`32K–65K` ctx) |
|---|---|---|
| **GPU, full offload (`-ctk q8_0 -ctv q8_0`)** | **6 GB VRAM** | **8–12 GB VRAM** |
| **Apple Silicon (Metal)** | **8 GB Unified Memory** | **16 GB Unified Memory** |
| **CPU only (`mmap`, AVX2 / NEON)** | **6 GB RAM** | **8–16 GB RAM** |
| **Disk** | **4.6 GB** | — |

Because 48 of the 64 layers are linear-complexity `Gated-Delta-Net` recurrent layers and only 16 layers use `Full-Attention` KV cache, **Caphlon-2** scales to `32K–65K` context windows within a fraction of the KV memory required by dense 27B transformers.

---

## License and Attribution

- **Repository code, kernel patch, scripts, documentation**: **Apache License 2.0** — see [LICENSE](LICENSE).
- **Model weights & upstream lineage**: Derived from **[Qwen 3.8 27B](https://huggingface.co/Qwen)** (Qwen Team, Apache-2.0) and **[Ternary-Bonsai-2-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)** (PrismML / UkisAI, Apache-2.0). The upstream Apache-2.0 license governs the weights. See [NOTICE.md](NOTICE.md) for the complete modification chain and required notices.
- **Trademarks**: "Caphlon" and "Caphlon-2" are project marks of Demir Alp Gündoğmuş. Apache-2.0 grants no trademark rights.

## Citation

```bibtex
@misc{caphlon2_2026,
  title  = {Caphlon-2: A 4.52 GiB PTQ1_SB Super-Block Ternary Build of Qwen 3.8 27B via Ternary-Bonsai-2},
  author = {Gündoğmuş, Demir Alp},
  year   = {2026},
  month  = {11},
  note   = {GGUF PTQ1\_SB (1.65 bpw), V83K vocabulary pruning, LayerTrim FFN pruning, folded OPD LoRA, and custom CUDA/Metal/SIMD kernels},
  url    = {https://github.com/demiralpdev/Caphlon-Model}
}
```
