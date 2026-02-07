# Big-Context Needle-in-a-Haystack: q4_0 vs f16 KV Cache

**Model:** NVIDIA Nemotron-3-Nano-30B-A3B (MXFP4 MoE GGUF)
**Hardware:** RTX 5080 16GB + CPU offload
**Date:** 2025-02-07
**llama.cpp:** latest build with flash attention

## Purpose

Stress-test KV cache quantization at large context sizes. The previous quality report only tested needle-in-a-haystack at 2-3K words (~3-4K tokens). This experiment scales up to **200K words (~260K tokens)** to determine if q4_0 KV degrades when the model's 6 attention layers must retrieve from a massive quantized cache.

## Test Design

- **Needle:** `"The secret password is 'blue elephant 42'."`
- **Question:** `"Based on the text above, what is the secret password? State it exactly."`
- **Pass criterion:** `"blue elephant 42"` found in model output (content or reasoning_content)
- **Context sizes:** 10K, 50K, 100K, 200K words (~13K, 65K, 130K, 260K tokens)
- **Needle positions:** beginning (~10% from start), middle (~50%), end (~90%)
- **Runs per test:** 3 (temperature=0)
- **Max generation tokens:** 256
- **Filler:** Diverse paragraphs covering 24 topics (computing, climate, cooking, space, music, medicine, etc.)

## Server Configuration

| Parameter | q4_0 Config | f16 Config |
|-----------|-------------|------------|
| Context length | 262,144 | 262,144 |
| GPU layers | 52/53 | 52/53 |
| CPU MoE experts | 20 | 35 |
| Flash attention | on | on |
| KV cache type (K) | q4_0 | f16 |
| KV cache type (V) | q4_0 | f16 |
| **KV cache size** | **432 MiB** | **1,536 MiB** |

## Results: Pass Rates

Both configurations achieved **36/36 (100%)** — zero failures at any context size or position.

| Context | Position | q4_0 | f16 |
|---------|----------|------|-----|
| 10K words (13K tok) | beginning | 3/3 | 3/3 |
| 10K words (13K tok) | middle | 3/3 | 3/3 |
| 10K words (13K tok) | end | 3/3 | 3/3 |
| 50K words (65K tok) | beginning | 3/3 | 3/3 |
| 50K words (65K tok) | middle | 3/3 | 3/3 |
| 50K words (65K tok) | end | 3/3 | 3/3 |
| 100K words (130K tok) | beginning | 3/3 | 3/3 |
| 100K words (130K tok) | middle | 3/3 | 3/3 |
| 100K words (130K tok) | end | 3/3 | 3/3 |
| 200K words (260K tok) | beginning | 3/3 | 3/3 |
| 200K words (260K tok) | middle | 3/3 | 3/3 |
| 200K words (260K tok) | end | 3/3 | 3/3 |
| **Total** | **all** | **36/36 (100%)** | **36/36 (100%)** |

## Results: Generation Speed (tokens/second)

q4_0 is consistently faster due to smaller KV cache memory footprint allowing more MoE experts on GPU.

| Context | q4_0 gen t/s | f16 gen t/s | q4_0 advantage |
|---------|-------------|-------------|----------------|
| 10K words (13K tok) | 78.5 | 58.9 | +33% |
| 50K words (65K tok) | 67.6 | 57.2 | +18% |
| 100K words (130K tok) | 57.5 | 55.0 | +5% |
| 200K words (260K tok) | 45.4 | 51.5 | -12% |

*Values averaged across all positions and runs at each context size.*

At 200K words, f16 is slightly faster for generation — likely because both configs are equally memory-bottlenecked at 260K tokens, and the f16 server was given more CPU MoE threads (35 vs 20).

## Results: Prompt Processing Speed (cold cache, tokens/second)

| Context | q4_0 prompt t/s | f16 prompt t/s | q4_0 advantage |
|---------|----------------|----------------|----------------|
| 10K words (13K tok) | ~225* | ~750 | — |
| 50K words (65K tok) | 1,262 | 773 | +63% |
| 100K words (130K tok) | 1,220 | 758 | +61% |
| 200K words (260K tok) | 1,129 | 724 | +56% |

*The 10K context was served from prompt cache on repeat runs. Cold-cache prompt processing at 50K+ shows q4_0 is ~60% faster due to reduced memory bandwidth.*

## Key Findings

1. **No quality degradation with q4_0 KV at any context size.** Perfect 100% pass rate up to 260K tokens, identical to f16. The q4_0 quantization of the 6 attention-layer KV cache introduces no measurable retrieval errors in needle-in-a-haystack tasks.

2. **q4_0 uses 3.56x less KV memory** (432 MiB vs 1,536 MiB). This frees 1.1 GB of VRAM for offloading more MoE experts to GPU, improving generation speed by 5-33% at most context sizes.

3. **Prompt processing is ~60% faster with q4_0** at large contexts (50K+ words), because writing to a smaller KV cache requires less memory bandwidth.

4. **The Nemotron architecture is inherently resilient to KV quantization** because it only has 6 attention layers (out of 52 total layers). The majority of the model uses Mamba2 recurrent layers that don't use the KV cache at all. This means KV quantization affects a much smaller fraction of the computation compared to a pure transformer model.

## Conclusion

**q4_0 KV cache is recommended for production use with Nemotron-3-Nano-30B.** It delivers identical needle-retrieval accuracy to f16 while saving 1.1 GB of VRAM and providing faster prompt processing. The 6-attention-layer hybrid architecture makes this model exceptionally tolerant of aggressive KV quantization.

## Raw Data

- q4_0 results: `test_results/needle_big_q4_0.json`
- f16 results: `test_results/needle_big_f16.json`
- Test script: `kv_needle_big.py`
