[nemotron-q8-kv-cache-results.md](https://github.com/user-attachments/files/25154135/nemotron-q8-kv-cache-results.md)
# KV Cache Quantization Results: q4_0 vs q8_0 at 128K & 256K Context

**Model:** NVIDIA Nemotron-3-Nano-30B-A3B (MXFP4 MoE GGUF)
**Hardware:** RTX 5080 16GB + CPU offload
**Date:** 2025-02-07

All configurations use maximized VRAM — n-cpu-moe tuned to the lowest value that fits for each setup.

## Server Configurations

| | q4_0 @ 128K | q4_0 @ 256K | q8_0 @ 128K | q8_0 @ 256K |
|---|---|---|---|---|
| **Context length** | 131,072 | 262,144 | 131,072 | 262,144 |
| **KV cache** | 216 MiB | 432 MiB | 408 MiB | 816 MiB |
| **n-cpu-moe** | 9 | 11 | 12 | 15 |
| **GPU model** | 14,155 MiB | 13,508 MiB | 13,508 MiB | 12,861 MiB |
| **Compute buffer** | 664 MiB | 905 MiB | 664 MiB | 905 MiB |
| **VRAM free** | 12 MiB | 70 MiB | 468 MiB | 334 MiB |
| **Graph splits** | — | 15 | 15 | 17 |
| **Needle pass rate** | 27/27 (100%) | 36/36 (100%) | 27/27 (100%) | 36/36 (100%) |

## Generation Speed (tokens/second, averaged across positions)

| Context Fill | q4_0 @ 128K | q4_0 @ 256K | q8_0 @ 128K | q8_0 @ 256K |
|---|---|---|---|---|
| 10K words (13K tok) | **97.0** | 89.1 | 90.9 | 87.7 |
| 50K words (65K tok) | **81.5** | 72.9 | 79.0 | 75.3 |
| 95K words (123K tok) | **68.8** | — | 68.8 | — |
| 100K words (130K tok) | — | 62.3 | — | 64.2 |
| 200K words (260K tok) | — | 47.3 | — | 50.5 |

## Prompt Processing Speed (cold cache, tokens/second)

| Context Fill | q4_0 @ 128K | q4_0 @ 256K | q8_0 @ 128K | q8_0 @ 256K |
|---|---|---|---|---|
| 10K words (13K tok) | 2,008 | 1,724 | 1,717 | 1,536 |
| 50K words (65K tok) | 1,955 | 1,700 | 1,702 | 1,517 |
| 95K words (123K tok) | 1,851 | — | 1,625 | — |
| 100K words (130K tok) | — | 1,613 | — | 1,451 |
| 200K words (260K tok) | — | 1,446 | — | 1,314 |

## Key Findings

1. **q4_0 @ 128K is the fastest configuration overall** — 97 t/s generation at 10K context fill, thanks to the smallest KV cache (216 MiB) allowing n-cpu-moe=9 which packs 14,155 MiB of model weights on GPU. Only 12 MiB of VRAM left free.

2. **At 256K context, q8_0 edges out q4_0 for generation at large fills.** At 200K words q8_0 is 7% faster (50.5 vs 47.3 t/s), while q4_0 is faster at smaller fills within the same 256K context window. This crossover happens because at very large KV fills, the q8_0 cache's higher precision may require fewer attention recomputations.

3. **q4_0 dominates prompt processing speed** across all configurations — 10-17% faster than q8_0 at equivalent context sizes, because writing to the smaller KV cache requires less memory bandwidth.

4. **All configurations produce identical needle retrieval accuracy** — 100% pass rate at every context size, position, and KV type tested. Neither q4_0 nor q8_0 introduces any measurable retrieval errors up to 260K tokens.

5. **128K vs 256K tradeoff is clear:** if your workload fits in 128K tokens, you get ~8-10% faster generation by using the smaller context allocation (more model on GPU). If you need >128K context, the 256K configs still perform well.

## VRAM Budget Breakdown

```
RTX 5080 (16 GB = 15,808 MiB)

q4_0 @ 128K (n-cpu-moe=9) ─── FASTEST
├─ Model weights:  14,155 MiB (89.5%)  ████████████████████████████████████░
├─ KV cache:          216 MiB ( 1.4%)  █
├─ RS (recurrent):     46 MiB ( 0.3%)
├─ Compute buffer:    664 MiB ( 4.2%)  ██
├─ Overhead:          715 MiB ( 4.5%)  ██
└─ Free:               12 MiB ( 0.1%)

q4_0 @ 256K (n-cpu-moe=11)
├─ Model weights:  13,508 MiB (85.5%)  ████████████████████████████████████
├─ KV cache:          432 MiB ( 2.7%)  ██
├─ RS (recurrent):     46 MiB ( 0.3%)
├─ Compute buffer:    905 MiB ( 5.7%)  ███
├─ Overhead:          847 MiB ( 5.4%)  ██
└─ Free:               70 MiB ( 0.4%)

q8_0 @ 128K (n-cpu-moe=12)
├─ Model weights:  13,508 MiB (85.5%)  ████████████████████████████████████
├─ KV cache:          408 MiB ( 2.6%)  ██
├─ RS (recurrent):     46 MiB ( 0.3%)
├─ Compute buffer:    664 MiB ( 4.2%)  ██
├─ Overhead:          714 MiB ( 4.5%)  ██
└─ Free:              468 MiB ( 3.0%)  █

q8_0 @ 256K (n-cpu-moe=15)
├─ Model weights:  12,861 MiB (81.3%)  ██████████████████████████████████
├─ KV cache:          816 MiB ( 5.2%)  ███
├─ RS (recurrent):     46 MiB ( 0.3%)
├─ Compute buffer:    905 MiB ( 5.7%)  ███
├─ Overhead:          846 MiB ( 5.4%)  ██
└─ Free:              334 MiB ( 2.1%)  █
```

## Recommendation

- **128K context needed?** Use **q4_0 with n-cpu-moe=9** — fastest at 97 t/s, smallest KV footprint.
- **256K context needed?** Use **q4_0 with n-cpu-moe=11** for short-to-medium fills, or **q8_0 with n-cpu-moe=15** if you consistently fill >150K tokens.
- No quality difference between q4_0 and q8_0 was detected in any test.

## Raw Data

- q4_0 256K (optimized): `test_results/needle_big_q4_0_optimized.json`
- q4_0 128K: `test_results/needle_big_q4_0_128k.json`
- q8_0 256K: `test_results/needle_big_q8_0_256k.json`
- q8_0 128K: `test_results/needle_big_q8_0_128k.json`
- Test script: `kv_needle_big.py`
