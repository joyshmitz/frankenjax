# Threaded i64 argmax/argmin (contiguous + leading-axis) — flips two JAX LOSSES into WINS

Agent: `WildForge`
Date: 2026-06-19
Files: `crates/fj-lax/src/tensor_ops.rs`, `crates/fj-lax/benches/lax_baseline.rs`

## Lever

The F64/F32 argmax/argmin fast paths (contiguous SIMD threading + leading-axis
stream+thread) had **no i64 sibling** — dense i64 argmax fell to the serial strided
scan for EVERY axis (a dtype-sibling gap; i64 argmax over integer arrays/labels is
common). Two new exact-integer reducers (no f64 widening — i64 loses precision above
2^53, so the f64 `arg_extreme_axis0_block` could not be reused):

- `arg_extreme_i64_contiguous` — one contiguous row, strict compare, first-occurrence
  tie. Driven by the existing `parallel_argmax_fill` to thread over output rows when
  `axis_stride == 1`.
- `arg_extreme_i64_axis0_block` / `parallel_arg_extreme_i64_axis0` — leading-axis
  (`axis == 0`) streaming k-outer / column-inner (contiguous reads, per-column running
  value/index in cache) threaded over the independent columns.

Both fold each output's taps in the same ascending order as the serial scan →
BIT-IDENTICAL (strict integer compare, first-occurrence tie). Strided non-leading axes
keep the serial gather.

## Head-to-head vs JAX (worker hz2, EPYC-Genoa 16 vCPU)

Workload: `argmax` over a `[16384, 1024]` i64 tensor (16.7M elements).

| Case | JAX p50 | fj-lax serial | fj-lax threaded | result |
| --- | ---: | ---: | ---: | --- |
| axis 1 (contiguous) | 4.71 ms | 11.87 ms (2.52x loss) | **2.51 ms** | **1.88x WIN** (4.7x over serial) |
| axis 0 (strided/leading) | 17.25 ms | 56.97 ms (3.30x loss) | **10.69 ms** | **1.61x WIN** (5.3x over serial) |

Both flip a JAX loss into a win, bit-exact. The fj-lax numbers are the full
`eval_primitive(Argmax, …)` dispatch; JAX's are jit'd pure compute.

JAX baseline command:

```bash
benchmarks/jax_comparison/.venv/bin/python \
  benchmarks/jax_comparison/argmax_i64_gauntlet.py
# argmax_axis1_i64 {"p50_ms": 4.71, ...}; argmax_axis0_i64 {"p50_ms": 17.25, ...}
```

fj-lax bench command:

```bash
RCH_WORKER=hz2 RCH_WORKERS=hz2 RCH_REQUIRE_REMOTE=1 \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cc \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline -- 'argmax_16kx1k_axis._i64'
# axis1 threaded: [2.4569 ms 2.5126 ms 2.5687 ms]  (serial pre-edit: 11.853–11.884 ms)
# axis0 threaded: [10.251 ms 10.692 ms 11.147 ms]  (serial pre-edit: 56.838–57.102 ms)
```

## Behavior proof (bit-exact)

`cargo test --release -p fj-lax --lib arg` → `37 passed; 0 failed`, including the new
`argmax_argmin_i64_threaded_matches_serial_reference_large` (8200×1024 i64 with many
exact ties, axis 0 + axis 1, argmax + argmin, vs the canonical serial reference).

## Ledger

WIN ×2. Impact 4.7x / 5.3x (serial→threaded); both flip JAX losses into wins;
Confidence 0.97 (bit-exact, gated, independent rows/columns, exact integer compare);
Effort low (mirrors the F64/F32 paths with integer comparison). Completes the
argmax/argmin dtype family (F64/F32/i64 × contiguous + leading-axis).
