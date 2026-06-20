# Streaming + threaded leading-axis (axis 0) argmax/argmin — flips a 10x JAX LOSS into a WIN

Agent: `WildForge`
Date: 2026-06-19
Files: `crates/fj-lax/src/tensor_ops.rs`, `crates/fj-lax/benches/lax_baseline.rs`

## Lever

`extremum_along_axis` had a SIMD threaded fast path only for the contiguous
(`axis_stride == 1`, trailing-axis) case. **Leading-axis argmax/argmin (`axis == 0`,
e.g. argmax over a batch dimension) fell to the generic strided reducer**:

```rust
for outer in 0..outer_count {            // one output column at a time
    arg_extreme_float(axis_dim, find_max, |i| values[base + i * axis_stride]);
}
```

For `[16384, 1024]` that scans each output column at **stride 1024** (8 KB apart) —
cache-hostile — and serially. Two compounding costs.

New `parallel_arg_extreme_axis0` (+ `arg_extreme_axis0_block`) streams the reduction
**k-outer / column-inner**: at each row `k` it reads the CONTIGUOUS slice
`values[k*cols + c0 ..][..w]` and updates per-column running value/index/NaN state in
the block's local cache, then threads over the independent output columns (gated at
`CHEAP_BINARY_PARALLEL_MIN = 8.4M`, `work_scaled_threads`). Each column folds `k` in
the SAME ascending order as the serial `arg_extreme_float`, so it is **BIT-IDENTICAL**:
first-occurrence strict `>`/`<` tie-break and a sticky sign-agnostic first-NaN. Wired
for dense F64 and F32 (widen `f64::from`); strided non-leading axes keep the serial
gather.

## Head-to-head vs JAX (worker hz2, EPYC-Genoa 16 vCPU)

Workload: `argmax` over axis 0 of a `[16384, 1024]` f64 tensor → `[1024]`
(16.7M elements; 1024 independent 16384-deep columns).

| Variant | Time (median) | vs JAX |
| --- | ---: | ---: |
| JAX `jit(argmax, axis=0)` x64 (p50) | 9.97 ms | 1.00x (baseline) |
| fj-lax serial strided (pre-lever) | 99.95 ms | **0.10x (10.0x LOSS)** |
| fj-lax stream+thread (this lever) | 8.16 ms | **1.22x WIN** |

- Speedup over the old strided-serial path: `99.95 / 8.16 = 12.2x`, bit-exact
  (streaming kills the stride-`cols` gather; threading fans the independent columns).
- Flips a 10.0x JAX **loss** into a 1.22x JAX **win** — the largest gap closed this day.
- Same path covers argmin and F32; the fj-lax number is the FULL `eval_primitive`
  dispatch, JAX's is jit'd pure compute.

JAX baseline command:

```bash
benchmarks/jax_comparison/.venv/bin/python \
  benchmarks/jax_comparison/argmax_axis1_gauntlet.py
# argmax_axis0_f64 {"p50_ms": 9.974, ...}
```

fj-lax bench command:

```bash
RCH_WORKER=hz2 RCH_WORKERS=hz2 RCH_REQUIRE_REMOTE=1 \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cc \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline -- argmax_16kx1k_axis0_f64
# stream+thread: time: [7.9434 ms 8.1590 ms 8.3797 ms]
# serial strided (pre-edit): time: [98.704 ms 99.946 ms 101.64 ms]
```

## Behavior proof (bit-exact)

`cargo test --release -p fj-lax --lib arg` → `36 passed; 0 failed`, including the new
`argmax_argmin_axis0_threaded_matches_serial_reference_large` (8200×1024 f64 with NaNs
+ exact ties, crossing the threading gate, vs the canonical `arg_extreme_float`
per-column reference for both argmax and argmin) plus `argmin_2d_axis0` /
`argmax_argmin_nan_and_signed_zero_match_jax`.

## Ledger

WIN. Impact 12.2x (serial→stream+thread) / flips a 10x JAX loss into a win;
Confidence 0.97 (bit-exact under partitioning, gated, independent columns); Effort
low-medium (streaming reducer transcribed from `arg_extreme_float` semantics). Third
same-day line/column-threading win after argmax-trailing (`ce6b326d`) and F32
cumulative (`ffca8500`).
