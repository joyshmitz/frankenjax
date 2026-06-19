# Threaded dense-F32 cumulative scan over independent lines — JAX WIN

Agent: `WildForge`
Date: 2026-06-19
Files: `crates/fj-lax/src/reduction.rs`, `crates/fj-lax/benches/lax_baseline.rs`

## Lever

`eval_cumulative_dense` already threaded the **F64** contiguous cumulative scan
(`cumsum`/`cumprod`/`cummax`/`cummin`) over the independent line/outer dimension
via the generic `scan_contiguous_lines_to_vec` (gate `CUMULATIVE_PARALLEL_MIN_ELEMS
= 262_144`). The **F32 path — JAX's default float dtype — was a fully serial loop
over lines at every size**, because it accumulates each line in f64 and rounds the
running value back to f32 per step (so the generic `T = f32` scanner can't be
reused: its accumulator type would be f32).

New `scan_contiguous_f32_lines_to_vec` mirrors the F64 threaded scanner but keeps
the f64 accumulator. The cumulative scan is a sequential dependency **within** a
line, but the lines are **independent**, so contiguous (`axis_stride == 1`) inputs
partition the lines across `work_scaled_threads` and scan each block in isolation.
Each line's f64-accumulate order is preserved inside its block, so the result is
**bit-identical to the serial scan for any partition** (forward and reverse, incl.
the one-line `threads == 1` case). Strided inputs keep the serial per-line gather.

## Head-to-head vs JAX (worker hz2, EPYC-Genoa 16 vCPU)

Workload: `cumsum` over axis 1 of a `[16384, 1024]` f32 tensor (16.7M elements;
16384 independent lines).

| Variant | Time (median) | vs JAX |
| --- | ---: | ---: |
| JAX `jit(cumsum)` x64 (p50) | 30.58 ms | 1.00x (baseline) |
| fj-lax **serial** (pre-thread) | 50.24 ms | **0.61x (1.64x LOSS)** |
| fj-lax **threaded** (this lever) | 17.61 ms | **1.74x WIN** |

- Threading speedup over serial: `50.24 / 17.61 = 2.85x`, bit-exact.
- Flips a 1.64x JAX **loss** into a 1.74x JAX **win**.
- Same mechanism lifts `cummax`/`cummin`/`cumprod` F32 (JAX `cummax` measured 65.5 ms
  on this shape, even more headroom).
- fj-lax number is the FULL `eval_primitive(Cumsum, …)` dispatch; JAX's is jit'd
  pure compute.

JAX baseline command:

```bash
benchmarks/jax_comparison/.venv/bin/python \
  benchmarks/jax_comparison/cumsum_f32_axis1_gauntlet.py
# cumsum_axis1_f32 {"p50_ms": 30.58, ...}; cummax_axis1_f32 {"p50_ms": 65.52, ...}
```

fj-lax bench command:

```bash
RCH_WORKER=hz2 RCH_WORKERS=hz2 RCH_REQUIRE_REMOTE=1 \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cc \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline -- cumsum_16kx1k_f32_axis1
# threaded: time: [16.799 ms 17.610 ms 18.495 ms]
# serial (pre-edit): time: [49.593 ms 50.241 ms 51.052 ms]
```

## Behavior proof (bit-exact)

`cargo test --release -p fj-lax --lib cumulative` → `7 passed; 0 failed`, including
`dense_f32_cumulative_bit_identical_to_literal_path` and
`threaded_cumulative_matches_serial_reference`. The threaded fill is a pure
partition of the same per-line f64-accumulate scan, so outputs are byte-identical
to the serial path.

## Ledger

WIN. Impact 2.85 (serial→threaded) / flips JAX loss→win; Confidence 0.97
(bit-exact, gated, independent lines); Effort low (mirrors the F64 threaded
scanner). Companion to the same-day argmax row-threading win
(`frankenjax-argmax-row-thread-2026-06-19.md`).
