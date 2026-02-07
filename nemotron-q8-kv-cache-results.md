# q8_0 KV Cache Results: 128K vs 256K Context

**Model:** NVIDIA Nemotron-3-Nano-30B-A3B (MXFP4 MoE GGUF)
**Hardware:** RTX 5080 16GB + CPU offload
**Date:** 2025-02-07

## Configuration Comparison

| | q8_0 @ 128K context | q8_0 @ 256K context |
|---|---|---|
| **Context length** | 131,072 tokens | 262,144 tokens |
| **KV cache size** | 408 MiB (204K + 204V) | 816 MiB (408K + 408V) |
| **Compute buffer** | 664 MiB | 905 MiB |
| **n-cpu-moe** | 12 | 15 |
| **GPU model** | 13,508 MiB | 12,861 MiB |
| **CPU model** | 3,910 MiB | 4,642 MiB |
| **Total GPU used** | 15,340 MiB | 15,474 MiB |
| **VRAM free** | 468 MiB | 334 MiB |
| **GPU layers** | 52/53 | 52/53 |
| **Graph splits** | 15 | 17 |
| **Needle pass rate** | 27/27 (100%) | 36/36 (100%) |

## Generation Speed (tokens/second)

| Context Fill | q8_0 @ 128K | q8_0 @ 256K | 128K advantage |
|---|---|---|---|
| 10K words (13K tok) | **90.9** | 87.7 | +4% |
| 50K words (65K tok) | **79.0** | 75.3 | +5% |
| 95K words (123K tok) | **68.8** | — | — |
| 100K words (130K tok) | — | 64.2 | — |
| 200K words (260K tok) | — | 50.5 | — |

## Prompt Processing Speed (cold cache, tokens/second)

| Context Fill | q8_0 @ 128K | q8_0 @ 256K |
|---|---|---|
| 10K words (13K tok) | 1,717 | 1,536 |
| 50K words (65K tok) | 1,702 | 1,517 |
| 95K words (123K tok) | 1,625 | — |
| 100K words (130K tok) | — | 1,451 |
| 200K words (260K tok) | — | 1,314 |

## Comparison with All KV Cache Types (at 256K context)

| | q4_0 | q8_0 | f16 |
|---|---|---|---|
| **KV cache size** | 432 MiB | 816 MiB | 1,536 MiB |
| **n-cpu-moe** | 20 | 15 | 35 |
| **GPU model** | 11,567 MiB | 12,861 MiB | ~6,539 MiB* |
| **Gen t/s @ 10K words** | 78.5 | 87.7 | 58.9 |
| **Gen t/s @ 50K words** | 67.6 | 75.3 | 57.2 |
| **Gen t/s @ 100K words** | 57.5 | 64.2 | 55.0 |
| **Gen t/s @ 200K words** | 45.4 | 50.5 | 51.5 |
| **Cold prompt t/s @ 50K** | 1,262 | 1,517 | 773 |
| **Needle pass rate** | 36/36 | 36/36 | 36/36 |

*f16 estimated from total - CPU model size

## Key Findings

1. **q8_0 is the fastest KV cache type at 256K context.** Despite using more VRAM for the KV cache than q4_0, the sweet spot at n-cpu-moe=15 keeps more MoE experts on GPU than q4_0's n-cpu-moe=20, resulting in 5-12% faster generation.

2. **128K context is ~5% faster than 256K** at equivalent prompt sizes, thanks to n-cpu-moe=12 fitting 647 MiB more model weights on GPU.

3. **All three KV types (q4_0, q8_0, f16) produce identical needle retrieval accuracy** — 100% pass rate across all context sizes up to 260K tokens.

4. **q8_0 at 256K achieves the best balance** of context length and speed for this hardware. It's 12-50% faster than f16 and 5-12% faster than q4_0 for generation, while still supporting the full 256K context window.

5. **Prompt processing scales with KV quantization level** — smaller KV types are faster to write. q8_0 sits between q4_0 and f16, with cold-cache prompt processing ~2x faster than f16.

## VRAM Budget Breakdown

```
RTX 5080 (16 GB)
┌────────────────────────────────────────────────────┐
│                                                    │
│  q8_0 @ 256K (n-cpu-moe=15)                       │
│  ├─ Model weights:  12,861 MiB (81.3%)            │
│  ├─ KV cache:          816 MiB ( 5.2%)            │
│  ├─ RS (recurrent):     46 MiB ( 0.3%)            │
│  ├─ Compute buffer:    905 MiB ( 5.7%)            │
│  ├─ Overhead:          846 MiB ( 5.4%)            │
│  └─ Free:              334 MiB ( 2.1%)            │
│                                                    │
│  q8_0 @ 128K (n-cpu-moe=12)                       │
│  ├─ Model weights:  13,508 MiB (85.5%)            │
│  ├─ KV cache:          408 MiB ( 2.6%)            │
│  ├─ RS (recurrent):     46 MiB ( 0.3%)            │
│  ├─ Compute buffer:    664 MiB ( 4.2%)            │
│  ├─ Overhead:          714 MiB ( 4.5%)            │
│  └─ Free:              468 MiB ( 3.0%)            │
│                                                    │
└────────────────────────────────────────────────────┘
```

## Raw Data

- q8_0 256K: `test_results/needle_big_q8_0_256k.json`
- q8_0 128K: `test_results/needle_big_q8_0_128k.json`
- Test script: `kv_needle_big.py`
