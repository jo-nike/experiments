# Nemotron-3-Nano-30B KV Cache Quality Report

## q4_0 vs f16 KV Cache Comparison

**Model**: NVIDIA-Nemotron-3-Nano-30B-A3B-MXFP4_MOE
**GPU**: NVIDIA GeForce RTX 5080 (16GB VRAM)
**Test Config**: ncmoe=35, ctx-size=262144, flash-attn=on, temp=0.0 (deterministic)
**Date**: 2026-02-07

---

## Executive Summary

**q4_0 KV cache has zero measurable quality impact on this model.** Across 16 tests (math, code, knowledge, needle-in-a-haystack, instruction following, long-form coherence), both configs produced **identical or near-identical outputs** with matching correctness. This is expected given the architecture: only 6 out of 52 layers use KV cache.

---

## VRAM & Speed Comparison

| Metric | q4_0 KV | f16 KV (default) |
|--------|---------|-------------------|
| **Generation speed** | ~59.5 t/s | ~59.5 t/s |
| **Prompt processing** | ~100-130 t/s | ~100-130 t/s |
| **VRAM usage** | 9.0 GB | 10.1 GB |
| **VRAM saved** | **1.1 GB** | baseline |

Speed is identical. The 1.1 GB savings lets you lower ncmoe (more MoE on GPU = faster) or fit larger context.

---

## Test Results by Category

### 1. Reasoning / Math

| Test | q4_0 | f16 | Match? |
|------|------|-----|--------|
| 127 * 83 | 10541 (correct) | 10541 (correct) | Identical output |
| Sheep riddle | 9 (correct) | 9 (correct) | Identical output |
| Widget problem | 5 min (correct) | 5 min (correct) | Identical output |

Both configs produced correct answers with identical reasoning traces. The math_multiply outputs were character-for-character identical across runs.

### 2. Code Generation

| Test | q4_0 | f16 | Match? |
|------|------|-----|--------|
| LCS function | Correct DP impl, named `longest_common_subsequence` | Correct DP impl, named `lcs` | Both correct, minor style diff |
| Bash one-liner | `find /tmp -type f -size +100M -exec du -h {} + \| sort -hr` | Identical command | Identical |

Both produced working, well-documented code. The LCS function had a minor naming difference (`longest_common_subsequence` vs `lcs`) but identical algorithm and structure. The bash one-liner was character-for-character identical.

### 3. Factual Recall / Knowledge

| Test | q4_0 | f16 | Match? |
|------|------|-----|--------|
| TCP vs UDP | Comprehensive table + explanation | Identical structure, minor wording | Near-identical |
| SOLID principles | Correct table with all 5 | Identical structure, minor wording | Near-identical |

Outputs covered the same points with the same structure. Minor phrasing differences (e.g., "SYN, SYN-ACK, ACK" vs "SYN-SYN/ACK-ACK") but identical substance.

### 4. Needle in a Haystack (KEY TEST)

This is the most important test for KV cache quality, as retrieval from context is where quantization errors show up.

**Needle 1**: "The secret password is 'blue elephant 42'." (~2K words filler)

| Position | q4_0 Content | f16 Content | Both Found in Reasoning? |
|----------|-------------|-------------|--------------------------|
| Beginning | (empty - token budget) | (empty - token budget) | Yes |
| Middle | "blue elephant 42" | (empty - token budget) | Yes |
| End | "blue elephant 42" | "blue elephant 42" | Yes |

**Needle 2**: "The launch code for Project Epsilon is 'mercury-7-tango-19'." (~3K words filler)

| Position | q4_0 Content | f16 Content | Both Found in Reasoning? |
|----------|-------------|-------------|--------------------------|
| Beginning | (empty - token budget) | (empty - token budget) | Yes |
| Middle | (empty - token budget) | (empty - token budget) | Yes |
| End | (empty - token budget) | (empty - token budget) | Yes |

**Important note**: The empty content fields are NOT failures to find the needle. Both configs correctly identified the needle in their `reasoning_content` field every single time. The "empty" responses occurred when the model's reasoning phase consumed the entire 128 max_token budget before generating visible output. This is a test design limitation (max_tokens too low for reasoning models), not a quality difference.

**Conclusion**: Both configs found the needle 100% of the time at all positions in their reasoning. There is **no detectable difference** in retrieval quality between q4_0 and f16 KV cache at 2K-3K word context.

### 5. Instruction Following

| Test | q4_0 | f16 | Match? |
|------|------|-----|--------|
| 7 P-languages | Python, Perl, PHP, PowerShell, Pascal, Prolog, Pike | Identical list | Identical |
| Haiku | (stuck in reasoning, counting syllables) | (stuck in reasoning, counting syllables) | Identical behavior |

Both produced the exact same 7-language list. Both got stuck counting syllables for the haiku (another max_tokens issue with reasoning models).

### 6. Long-form Coherence

| Test | q4_0 | f16 | Match? |
|------|------|-----|--------|
| PostgreSQL vs MySQL | 9-row comparison table + summary | 9-row comparison table + summary | Near-identical |

Both generated comprehensive, well-structured comparisons with similar coverage. Minor wording differences but identical substance and structure.

---

## Why the Impact is So Low

This model has a **hybrid Mamba-2 + Transformer + MoE architecture**:

| Layer Type | Count | Uses KV Cache? |
|------------|-------|----------------|
| Mamba-2 | 23 | No (recurrent state) |
| MoE (feed-forward) | 23 | No |
| Attention | **6** | **Yes** |
| **Total** | **52** | **Only 6 (11.5%)** |

q4_0 KV cache quantization only affects the 6 attention layers. The other 46 layers (88.5% of the model) are completely unaffected. This is fundamentally different from a pure transformer where q4_0 would affect every single layer.

---

## Recommendations

1. **Always use `--cache-type-k q4_0 --cache-type-v q4_0`** for this model. There is no measurable quality penalty and it saves 1.1 GB of VRAM.

2. **The VRAM savings enable better configs**:
   - At 256K context: use ncmoe=20 instead of ~25 → **81.7 t/s vs ~65 t/s**
   - At 1M context: makes it possible to fit at all (ncmoe=27, 69.6 t/s)

3. **For pure transformers** (Gemma, Mistral, etc.), q4_0 KV cache would have a larger quality impact since it affects all layers. Test before using on those models, especially at long context.

4. **Longer context testing needed**: These tests used 2-3K word contexts. At 100K+ tokens, q4_0 KV cache *might* show degradation on the 6 attention layers. However, since Mamba-2 handles most of the sequence modeling, the impact would still be minimal.

---

## Optimal Configurations

| Use Case | Config | Speed | VRAM |
|----------|--------|-------|------|
| Max speed, 256K ctx | ngl=52, ncmoe=20, q4_0 KV | **81.7 t/s** | 13.5 GB |
| Max context, 1M ctx | ngl=52, ncmoe=27, q4_0 KV | **69.6 t/s** | 15.4 GB |

---

## Raw Data

Full test results with all outputs saved to:
- `/mnt/data/models/test_results/results_q4_0.json`
- `/mnt/data/models/test_results/results_f16.json`
- `/mnt/data/models/test_results/q4_0_log.txt`
- `/mnt/data/models/test_results/f16_log.txt`
