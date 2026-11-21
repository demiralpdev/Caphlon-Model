# NOTICE

Caphlon-2 (Caphlon-Model v2.0.0)
Copyright 2026 Demir Alp Gündoğmuş (demiralpdev)

This repository's original contents (model cards, documentation, CUDA/Metal/SIMD kernel patches,
scripts, naming, and project files) are licensed under the Apache License, Version 2.0.
See LICENSE.

## Upstream and derivative-work notice

The model weights distributed under the name "Caphlon-2" (and "Caphlon-Model") are a derivative
of **Qwen 3.8 27B** (published by the Qwen team) and **Ternary-Bonsai-2-27B** (published by
PrismML / UkisAI), both distributed under the **Apache License, Version 2.0**. The upstream
license governs the weights and any redistribution of them; nothing in this repository narrows,
extends, or replaces those upstream terms.

Modification chain (Caphlon-2 — 4.52 GiB `PTQ1_SB`, `cand_tam_bsqrt` / `Cand_pure`):

1. **Qwen 3.8 27B** (upstream base architecture, Apache-2.0) — https://huggingface.co/Qwen
2. **Ternary-Bonsai-2-27B (`PTQ1_0`, 5.54 GiB)** (Sylvester-Hadamard $H_{1024}$ 1.75 bpw ternary intermediate, Apache-2.0) — https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf & https://huggingface.co/ukisai/Swift-Bonsai-2-GGUF
3. **Caphlon-2 (`PTQ1_SB`, 4.52 GiB / 4,856,213,824 bytes, SHA-256 `ac41fcddf5da28d71198f60202c10ea37cc64d791e5f6f118f849a02579c20db`)** changes by this project:
   - BPE vocabulary closure-preserving pruning (`111,104 → 84,352` tokens, `83,982` merges) + anti-hesitation logit-bias map (`metadata/Caphlon-2-27B-PTQ1_SB.meta.json`)
   - Layer-adaptive FFN intermediate width pruning (`17,408 → 15,456` via block-influence ridge calibration, keeping all 64 hybrid layers and `23.30B` active parameters)
   - `PTQ1_SB` (GGML type `144`, `1.648 bpw`) super-block ternary quantization (`8×128` 4-bit sub-scale nibbles per 1,024-weight Sylvester-Hadamard block)
   - Optional Inference-Time Intervention truthfulness direction vector (`metadata/iti_mean.gguf`, `1.2 MiB`) lifting 270-question Hallucination benchmark accuracy to `87.41%` (`236/270`) with `100%` unanswerable abstain recall
   - Custom CUDA (`sm_75`/`sm_80` Tensor Core prefill + `GGML_CUDA_Q8_1_SB` 2-row decode), Apple Metal, ARM64 NEON, and x86 AVX2 kernels (`patches/prismll_ptq1_sb_cuda_metal.patch`)

Modification chain (Caphlon v1 — 6.87 GiB `IQ2_S`, legacy stock `llama.cpp` build):

1. Qwen 3.8 27B (upstream, Apache-2.0) — https://huggingface.co/Qwen
2. Unsloth Dynamic IQ2_S quantization of the above (intermediate)
3. Vocabulary merge-table reduction (`100k` target) and hybrid-interval layer pruning (layers 12–15)

## Required attributions

If you redistribute these weights or a modified build:

- retain this NOTICE and the upstream license text;
- credit **Qwen 3.8 27B** (Qwen team) and **Ternary-Bonsai-2-27B** (PrismML / UkisAI) in your attribution chain;
- state clearly which changes you made;
- do not remove or alter copyright, patent, trademark, or attribution notices.

Apache-2.0 also requires that you do not use the names, logos, or marks of the
licensors to endorse your derivative. See the trademark section below.

## Trademark and endorsement

"**Caphlon**", "**Caphlon-2**", and "Caphlon-Model" are project marks of Demir Alp Gündoğmuş.
The Apache License 2.0 grants **no trademark rights** (see Section 6 of the
license).

Therefore:

- You MAY redistribute the weights, modified or unmodified, under Apache-2.0.
- You MAY use them commercially.
- You MAY NOT present a modified build as "Caphlon", "Caphlon-2", or "Caphlon-Model".
- You MAY NOT claim that a redistribution is official, endorsed, sponsored, or
  reviewed by this project or its author.
- Factual statements such as "based on Caphlon-2" are fine and encouraged.

## What this project does NOT claim

- No warranty of fitness, accuracy, safety, or non-infringement (Apache-2.0,
  Sections 7–8).
- No guarantee that benchmark numbers reproduce on other hardware or with other
  sampling settings; all figures in the model card are measurements on the
  hardware and protocol stated there.
- No claim of affiliation with, or endorsement by, Alibaba Cloud / the Qwen team,
  PrismML, or UkisAI.

## Third-party components

`llama.cpp` and `PrismML-Eng/llama.cpp` (MIT License) are used for inference and evaluation;
our `PTQ1_SB` kernel patch (`patches/prismll_ptq1_sb_cuda_metal.patch`) applies cleanly to
`PrismML-Eng/llama.cpp` commit `6447906bf96c1a023980d3453fb59daa6af3bb23`.
Dataset references (GPQA-Diamond, ChemBench, HaluEval, TruthfulQA, OpenAI HumanEval, MMLU,
wiki.test.raw) remain under their own respective terms.
