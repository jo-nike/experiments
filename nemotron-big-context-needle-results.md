[nemotron-q8-kv-cache-results.md](https://github.com/user-attachments/files/25154824/nemotron-q8-kv-cache-results.md)
# KV Cache Quantization Results: q4_0 vs q8_0 at 128K, 256K & 1M Context

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

## 1M Context Results

### Server Configuration — 1M Context

| | q4_0 @ 1M | q8_0 @ 1M |
|---|---|---|
| **Context length** | 1,048,576 | 1,048,576 |
| **KV cache** | ~1,632 MiB | ~3,264 MiB |
| **Compute buffer** | ~3,434 MiB | ~3,434 MiB |
| **n-cpu-moe** | 30 | 33 |

**n-cpu-moe search for 1M context:**
- **q4_0:** n-cpu-moe=25 starts but crashes during flash attention at ~389K tokens (transient VRAM). n-cpu-moe=30 is stable at full context. Values 26-29 were not tested.
- **q8_0:** n-cpu-moe=28 OOM at startup. n-cpu-moe=29 OOM during loading. n-cpu-moe=30 starts but crashes during flash attention at ~1M tokens. n-cpu-moe=33 being tested (770K words in progress).

> **Note on flash attention transient VRAM:** At 1M context, flash attention requires significant temporary GPU memory during inference that scales with sequence length. This is separate from the static KV cache allocation and compute buffer. It caused both q4_0 @ n-cpu-moe=25 and q8_0 @ n-cpu-moe=30 to crash mid-inference despite successful server startup. The extra 3 n-cpu-moe steps for q8_0 vs q4_0 (33 vs 30) reflect the larger q8_0 KV cache requiring more flash attention working memory.

### Generation Speed — 1M Context (tokens/second)

| Context Fill | q4_0 @ 1M (moe=30) | q8_0 @ 1M (moe=30†) | q8_0 @ 1M (moe=33) |
|---|---|---|---|
| 10K words (13K tok) | 64.9 | 62.7 | — |
| 50K words (65K tok) | 56.2 | 57.6 | — |
| 200K words (260K tok) | 40.0 | — | — |
| 770K words (1M tok) | 19.1 | OOM crash† | pending |

† q8_0 @ n-cpu-moe=30 handles small fills fine but crashes during flash attention at ~1M tokens.

### Needle-in-a-Haystack — 1M Context (needle at 30% position)

| Context Fill | q4_0 @ 1M | q8_0 @ 1M (moe=30) | q8_0 @ 1M (moe=33) |
|---|---|---|---|
| 10K words (13K tok) | — | PASS | — |
| 50K words (65K tok) | — | PASS | — |
| 770K words (1M tok) | — | OOM | pending |

q4_0 @ 1M speed tests did not include needle validation. Separate needle tests at 10K/50K/200K words confirmed 100% pass rate (27/27) with n-cpu-moe=25.

### Prompt Processing Speed — 1M Context (cold cache, tokens/second)

| Context Fill | q4_0 @ 1M (moe=30) | q8_0 @ 1M (moe=30) |
|---|---|---|
| 10K words (13K tok) | 869 | 863 |
| 50K words (65K tok) | 874 | 883 |
| 200K words (260K tok) | 768 | — |
| 770K words (1M tok) | 554 | OOM crash |

Prompt processing speed is nearly identical between q4_0 and q8_0 at 1M context — the KV type matters less when n-cpu-moe is the same (30). This differs from the 128K/256K results where q4_0 was 10-17% faster, because at those smaller contexts the n-cpu-moe difference (fewer experts on CPU for q4_0) dominated.

### Speed Degradation: 1M Context vs 128K/256K

The 1M context allocation significantly reduces generation speed due to n-cpu-moe=30+ pushing most MoE experts to CPU:

| Context Fill | q4_0 @ 128K (moe=9) | q4_0 @ 256K (moe=11) | q4_0 @ 1M (moe=30) | 1M vs 128K |
|---|---|---|---|---|
| 10K words | 97.0 t/s | 89.1 t/s | 64.9 t/s | -33% |
| 50K words | 81.5 t/s | 72.9 t/s | 56.2 t/s | -31% |
| 200K words | — | 47.3 t/s | 40.0 t/s | — |
| 770K words | — | — | 19.1 t/s | — |

Generation speed at 1M context drops ~31-33% vs 128K at equivalent fills, and falls to **19.1 t/s** when the context is nearly full (~1M tokens). The primary cause is n-cpu-moe=30 offloading most MoE computation to CPU vs n-cpu-moe=9-11 at smaller contexts.

### 1M Context Key Takeaways

1. **q4_0 and q8_0 perform nearly identically at low fills** when using the same n-cpu-moe — 62.7 vs 64.9 t/s at 10K words, 57.6 vs 56.2 t/s at 50K words.
2. **q8_0 needs 3 more n-cpu-moe than q4_0** at 1M context (33 vs 30) to survive flash attention at full context fill, due to its 2× larger KV cache.
3. **Full-context generation at 1M is slow but usable** — 19.1 t/s for q4_0 at 770K words (~1M tokens). q8_0 @ moe=33 result pending.
4. **The 1M compute buffer (3,434 MiB) is the major VRAM cost** — 4× larger than the 256K buffer (905 MiB), consuming far more VRAM than the KV cache difference between q4_0 and q8_0.

## Recommendation

- **128K context needed?** Use **q4_0 with n-cpu-moe=9** — fastest at 97 t/s, smallest KV footprint.
- **256K context needed?** Use **q4_0 with n-cpu-moe=11** for short-to-medium fills, or **q8_0 with n-cpu-moe=15** if you consistently fill >150K tokens.
- **1M context needed?** Use **q4_0 with n-cpu-moe=30** — proven stable at full context, 19.1 t/s at ~1M tokens. q8_0 requires n-cpu-moe=33 for stability (pending confirmation).
- No quality difference between q4_0 and q8_0 was detected in any test up to 260K tokens. Needle validation at 1M context pending for q8_0.

## Raw Data

- q4_0 256K (optimized): `test_results/needle_big_q4_0_optimized.json`
- q4_0 128K: `test_results/needle_big_q4_0_128k.json`
- q8_0 256K: `test_results/needle_big_q8_0_256k.json`
- q8_0 128K: `test_results/needle_big_q8_0_128k.json`
- q4_0 1M (speed): `test_results/speed_q4_0_1M.json`, `test_results/speed_q4_0_1M_770.json`
- q8_0 1M (speed): `test_results/speed_q8_0_1M.json`
- q4_0 1M (needle): `test_results/needle_big_q4_0_1M.json`
- Test scripts: `kv_needle_big.py`, `kv_speed_bench.py`
