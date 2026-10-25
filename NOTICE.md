# NOTICE

Caphlon-Model
Copyright 2026 Demir Alp Gündoğmuş (demiralpdev)

This repository's original contents (model card, documentation, scripts,
naming, and project files) are licensed under the Apache License, Version 2.0.
See LICENSE.

## Upstream and derivative-work notice

The model weights distributed under the name "Caphlon-Model" are a derivative
of **Qwen 3.8 27B**, published by the Qwen team. Qwen 3.8 27B is distributed
under the **Apache License, Version 2.0**. The upstream license governs the
weights and any redistribution of them; nothing in this repository narrows,
extends, or replaces those upstream terms.

Modification chain:

1. Qwen 3.8 27B (upstream, Apache-2.0) — https://huggingface.co/Qwen
2. Unsloth Dynamic IQ2_S quantization of the above (intermediate)
3. This project's changes:
   - vocabulary merge-table reduction (100k target, closure-preserving selection)
   - removal of layers 12–15 (4 layers, aligned to the hybrid layer interval)
   - repository packaging, naming, documentation, and evaluation scripts

## Required attributions

If you redistribute these weights or a modified build:

- retain this NOTICE and the upstream license text;
- credit **Qwen 3.8 27B** as the base model and the Qwen team as its authors;
- state clearly which changes you made;
- do not remove or alter copyright, patent, trademark, or attribution notices.

Apache-2.0 also requires that you do not use the names, logos, or marks of the
licensor to endorse your derivative. See the trademark section below.

## Trademark and endorsement

"**Caphlon**" and "Caphlon-Model" are project marks of Demir Alp Gündoğmuş.
The Apache License 2.0 grants **no trademark rights** (see Section 6 of the
license).

Therefore:

- You MAY redistribute the weights, modified or unmodified, under Apache-2.0.
- You MAY use them commercially.
- You MAY NOT present a modified build as "Caphlon" or "Caphlon-Model".
- You MAY NOT claim that a redistribution is official, endorsed, sponsored, or
  reviewed by this project or its author.
- Factual statements such as "based on Caphlon-Model" are fine and encouraged.

## What this project does NOT claim

- No warranty of fitness, accuracy, safety, or non-infringement (Apache-2.0,
  Sections 7–8).
- No guarantee that benchmark numbers reproduce on other hardware or with other
  sampling settings; all figures in the model card are measurements on the
  hardware and protocol stated there.
- No claim of affiliation with, or endorsement by, Alibaba Cloud or the Qwen
  team.

## Third-party components

llama.cpp (MIT License) is used for inference and evaluation but is not
redistributed here. Dataset references (openai/openai_humaneval, MMLU,
wiki.test.raw) remain under their own respective terms.
