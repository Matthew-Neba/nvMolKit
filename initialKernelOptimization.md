# Fused Butina Initial-Neighbor Kernel Optimization

## Goal and constraints

Optimize the CUDA fused-Butina initial-neighbor stage for `N >= 10,000`, using
the local RTX 2050 (compute capability 8.6) as a proxy while favoring designs
that scale to high-end GPUs. Raw Nsight reports and temporary benchmark helpers
are not committed; this file records the distilled measurements and decisions.

The supplied RTX 5080 comparison shows the CUDA implementation crossing from
parity with Triton at 10k to 5.76x slower at 500k:

| N | Triton | CUDA baseline | CUDA / Triton |
|---:|---:|---:|---:|
| 10k | 5.951 ms | 5.815 ms | 0.98x |
| 20k | 17.824 ms | 22.064 ms | 1.24x |
| 40k | 48.762 ms | 86.065 ms | 1.77x |
| 100k | 226.471 ms | 532.878 ms | 2.35x |
| 500k | 4.788 s | 27.581 s | 5.76x |

Exact initial Butina degrees require considering every fingerprint pair in the
general case, so the worst-case work remains quadratic. The optimization focus
is eliminating duplicated symmetric work and increasing reuse per global load.

## Reproducible proxy workload

- GPU: NVIDIA GeForce RTX 2050, 4 GiB, compute capability 8.6
- Input: seeded random contiguous `int32[N, 32]` fingerprints (1024 bits)
- Metric/cutoff: Tanimoto, `cutoff=0.35` (`threshold=0.65`)
- Timing: one warmup, seven CUDA-event samples, median reported
- Correctness gate: `build/tests/test_butina` and fused Python tests after each edit

## Iteration 0: one-row-per-block baseline

No code changes. Median end-to-end times:

| N | Time |
|---:|---:|
| 10k | 67.923 ms |
| 20k | 265.123 ms |
| 40k | 1063.864 ms |

The 3.90x and 4.01x increases for each doubling confirm quadratic scaling.

Nsight Compute, `initialNeighborCountKernel`, N=10k:

| Metric | Value |
|---|---:|
| Kernel duration | 66.37 ms |
| Compute throughput | 60.20% |
| L1/TEX throughput | 59.97% |
| DRAM throughput | 1.55% |
| L1/TEX hit rate | 91.66% |
| L2 hit rate | 90.26% |
| Registers/thread | 31 |
| Achieved occupancy | 98.96% |

Conclusion: neither DRAM bandwidth nor occupancy is the limiting factor. The
kernel assigns one block per row, evaluates both `(i,j)` and `(j,i)`, and does
not reuse a loaded fingerprint across multiple center rows. The first redesign
will use upper-triangular 2D tiles, evaluate each off-diagonal pair once, update
both endpoint degrees, and reuse each shared-memory fingerprint tile.

## Iteration 1: naive upper-triangular 32x32 tiles

Changed the initial count to evaluate upper-triangular 32x32 tiles, atomically
accumulate both endpoints of off-diagonal pairs, and cache both fingerprint
tiles in shared memory. Correctness passed (44/44 C++ and 22/22 fused Python).

| N | Baseline | Iteration 1 | Change |
|---:|---:|---:|---:|
| 10k | 67.923 ms | 78.633 ms | 15.77% slower |
| 20k | 265.123 ms | 307.985 ms | 16.17% slower |
| 40k | 1063.864 ms | 1210.929 ms | 13.82% slower |

The row-major shared layout has a stride of exactly 32 words. When a warp
compares one row against 32 candidates, its candidate loads therefore map to
the same shared-memory bank. Transposing the byte hit matrix for the second
endpoint reduction introduces another bank-conflicted access. The next edit
keeps symmetric tiling but pads fingerprint rows and represents each result row
as one warp ballot mask.

## Iteration 2: padded shared rows and ballot hit masks

Added one word of padding to each shared fingerprint row, so a warp's candidate
loads cycle through all 32 banks instead of repeatedly addressing one bank.
Replaced the 32x32 byte hit matrix with 32 `uint32_t` warp-ballot masks. Row
counts become one `popc(mask)` and transposed column counts read the masks via
shared-memory broadcasts.

Correctness passed (44/44 C++ and 22/22 fused Python). End-to-end proxy results:

| N | Baseline | Iteration 2 | Speedup |
|---:|---:|---:|---:|
| 10k | 67.923 ms | 12.398 ms | 5.48x |
| 20k | 265.123 ms | 43.225 ms | 6.13x |
| 40k | 1063.864 ms | 165.418 ms | 6.43x |

Nsight Compute, N=10k:

| Metric | Baseline | Iteration 2 |
|---|---:|---:|
| Kernel duration | 66.37 ms | 11.67 ms |
| Compute throughput | 60.20% | 98.58% |
| L1/TEX throughput | 59.97% | 98.60% |
| DRAM throughput | 1.55% | 5.53% |
| Registers/thread | 31 | 39 |
| Achieved occupancy | 98.96% | 82.08% |

The kernel itself is 5.69x faster and now saturates the SM/L1 execution path;
further gains require reducing executed pair/word work rather than increasing
occupancy. Against the historical Triton initial-count kernel on the same proxy
input, Triton takes 19.114 ms at 10k, 79.028 ms at 20k, and 317.672 ms at 40k.
Iteration 2 is respectively 1.64x, 1.83x, and 1.92x faster when comparing its
profiled/end-to-end times (the CUDA end-to-end number includes small additional
setup, argmax, and singleton-output costs).

## Iteration 3: population-count upper-bound rejection

Before intersecting fingerprint words, reject pairs whose bit populations make
the threshold mathematically unreachable. For Tanimoto the maximum possible
similarity is `min(popA,popB) / max(popA,popB)`; for cosine it is the square
root of that ratio. The same check is used by the initial, extraction, and
neighbor-update comparisons.

For the 9,999 valid molecules in `chembl_10k.smi`, the bound rejects 41.55% of
pairs at threshold 0.65, 66.60% at 0.8, and 83.75% at 0.9. End-to-end results:

| Cutoff | Iteration 2 | Iteration 3 | Improvement |
|---:|---:|---:|---:|
| 0.10 | 33.472 ms | 16.293 ms | 51.32% |
| 0.20 | 61.860 ms | 33.534 ms | 45.79% |
| 0.35 | 94.577 ms | 68.582 ms | 27.49% |

The initial kernel alone only falls from approximately 11.67 ms to 11.28 ms at
cutoff 0.35: mixed populations in a warp leave most word-loop instructions
active. Later one-thread-per-row comparisons benefit directly, explaining the
larger end-to-end gain. Correctness passed (44/44 C++ and 22/22 fused Python).

## Iteration 4: bitcount-sorted tile rejection

Radix-sort row indices by cached bit population on the GPU before the initial
count. In sorted order, all feasible pairs form a band around the diagonal, so
an entire 32x32 tile returns before loading fingerprints when its maximum left
population cannot reach the threshold against its minimum right population.
Neighbor counts are atomically accumulated at original row indices. If the
global minimum/maximum prove all pairs feasible, the kernel adaptively retains
the original contiguous traversal and omits the scalar bound checks.

On ChEMBL, the cutoff-0.35 initial kernel falls from 11.28 ms to 7.24 ms (35.82%
improvement). End-to-end results relative to iteration 3:

| Cutoff | Iteration 3 | Iteration 4 | Improvement |
|---:|---:|---:|---:|
| 0.10 | 16.293 ms | 9.031 ms | 44.57% |
| 0.20 | 33.534 ms | 28.958 ms | 13.65% |
| 0.35 | 68.582 ms | 65.761 ms | 4.11% |

Correctness again passed (44/44 C++ and 22/22 fused Python). Uniform-density
random inputs take the adaptive all-pairs-feasible path; their timings remain
noisy on the laptop GPU, while the profiler confirms the same full-tile kernel
work plus the small one-time radix-sort overhead.
