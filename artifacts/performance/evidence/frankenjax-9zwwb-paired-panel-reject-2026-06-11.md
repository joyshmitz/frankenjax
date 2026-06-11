# frankenjax-9zwwb paired-panel f32 GEMM rejection

- Date: 2026-06-11
- Bead: `frankenjax-9zwwb`
- Target: `fj-lax` `matmul_2d_packed_row_block_f32`
- Lever tested: compute two adjacent packed `F32_NR` panels per MR row tile, reusing the same four A broadcasts across two B panels.
- Outcome: rejected and removed. Score `0.0`.

## Baseline

Clean detached worktree:

```bash
RCH_FORCE_REMOTE=1 RCH_WORKER=vmi1227854 RCH_WORKERS=vmi1227854 \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,RCH_WORKER,RCH_WORKERS,RCH_FORCE_REMOTE \
rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-9zwwb-baseline-1500 \
  cargo bench -j 1 -p fj-lax --bench lax_baseline -- \
  linalg/f32_gemm_1024_packed --sample-size 10 --measurement-time 2 --warm-up-time 1 --noplot
```

Worker: `vmi1227854`

Result: `75.590 ms` mean, interval `[72.475 ms, 79.843 ms]`.

## Candidate

Shared worktree with only the paired-panel source hunk under test:

```bash
RCH_FORCE_REMOTE=1 RCH_WORKER=vmi1227854 RCH_WORKERS=vmi1227854 \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,RCH_WORKER,RCH_WORKERS,RCH_FORCE_REMOTE \
rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-9zwwb-candidate-1500 \
  cargo bench -j 1 -p fj-lax --bench lax_baseline -- \
  linalg/f32_gemm_1024_packed --sample-size 10 --measurement-time 2 --warm-up-time 1 --noplot
```

Worker: `vmi1227854`

Result: `86.356 ms` mean, interval `[84.201 ms, 88.170 ms]`.

Ratio: `75.590 / 86.356 = 0.875x`; regression.

## Isomorphism

The candidate preserved per-output ascending-`l` f32 accumulation, output row/column order, dtype conversion, and batch behavior. There is no tie-breaking or RNG surface. The rejection is performance-only.

No candidate source is kept. Remaining work is routed to `frankenjax-22hbx`: KC-blocked packed f32 GEMM with running-C panel reuse at larger `k*n` regimes.
