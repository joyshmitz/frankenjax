# frankenjax-vrell BF16 eval fusion reroute

Date: 2026-06-13
Agent: BeigeMouse
Worker: vmi1152480 for baseline/profile, BF16 proof, rebench, check, and clippy allowlisted gate
Bead: frankenjax-vrell - perf(fj-interpreters): route BF16 eval fusion away from slower fused tape

## Profile-backed target

Command:

```bash
RCH_WORKER=vmi1152480 RCH_WORKERS=vmi1152480 RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo bench -j 1 -p fj-interpreters --bench eval_fusion_speed
```

Baseline/profile rows:

| Case | Unfused | eval_jaxpr | Speedup |
| --- | ---: | ---: | ---: |
| F64 | 13.178 ms | 3.198 ms | 4.12x |
| F32 | 8.603 ms | 1.844 ms | 4.67x |
| F32_CLAMP | 10.554 ms | 2.767 ms | 3.81x |
| F32_ROW_BROADCAST | 11.204 ms | 0.939 ms | 11.94x |
| F32_COL_BROADCAST | 10.151 ms | 1.190 ms | 8.53x |
| F64_ROW_BROADCAST | 8.806 ms | 3.077 ms | 2.86x |
| F64_COL_BROADCAST | 7.573 ms | 2.701 ms | 2.80x |
| I64 | 5.319 ms | 3.520 ms | 1.51x |
| BF16 | 12.199 ms | 53.538 ms | 0.23x |

BF16 was the profiler-evident regression: the half-fusion tape widened and
rounded every step, making `eval_jaxpr` 4.39x slower than the ordinary primitive
chain in the same benchmark.

## Lever

One production lever:

- `classify_half_fusion_operand` now rejects `DType::BF16`, so BF16 chains fall
  through to the ordinary per-equation primitive path.
- F16 remains eligible for the existing half-fusion path.

This is an algorithm-selection change, not a numeric-kernel rewrite. The next
BF16 primitive should be conversion-saving fusion rather than per-step
widen/round emulation.

## Isomorphism proof

- Ordering is preserved: BF16 equations execute through the existing generic
  interpreter path in original `jaxpr` equation order.
- Tie-breaking is unchanged: no comparisons or winner-selection logic changed.
- Floating-point behavior is preserved: each BF16 operation calls the same
  primitive evaluator used before fusion was selected; there is no reassociation,
  no changed rounding mode, and no mixed-dtype promotion change.
- RNG behavior is unchanged: the touched classifier has no RNG state and the
  fallback path does not reorder effects.
- Environment/liveness behavior is unchanged: the fallback path is the existing
  per-equation binding path; only BF16 fusion eligibility changes.
- F16 behavior is unchanged by construction and re-proved with its existing
  bit-for-bit golden test.

Golden outputs:

- BF16 golden SHA-256:
  `3132f039bc6e3cbc8f2654e641b4297f4163ecb2cb0b729835873079ae9339ff`
- F16 golden SHA-256:
  `50bd04003ca23bfb110a239a785969d8f4f5da9d3c9ab96f6a79a332d41a149c`

## Re-benchmark

Command:

```bash
RCH_WORKER=vmi1152480 RCH_WORKERS=vmi1152480 RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo bench -j 1 -p fj-interpreters --bench eval_fusion_speed
```

After rows:

| Case | Unfused | eval_jaxpr | Speedup |
| --- | ---: | ---: | ---: |
| F64 | 15.106 ms | 3.267 ms | 4.62x |
| F32 | 7.552 ms | 1.293 ms | 5.84x |
| F32_CLAMP | 9.102 ms | 2.068 ms | 4.40x |
| F32_ROW_BROADCAST | 10.049 ms | 1.116 ms | 9.00x |
| F32_COL_BROADCAST | 9.692 ms | 0.905 ms | 10.70x |
| F64_ROW_BROADCAST | 8.610 ms | 3.510 ms | 2.45x |
| F64_COL_BROADCAST | 7.694 ms | 3.305 ms | 2.33x |
| I64 | 5.390 ms | 3.073 ms | 1.75x |
| BF16 | 10.679 ms | 9.531 ms | 1.12x |

Target delta:

- BF16 `eval_jaxpr`: 53.538 ms -> 9.531 ms = 5.62x faster.
- BF16 now beats the same-run unfused chain: 10.679 ms -> 9.531 ms = 1.12x.

Score:

- Impact 5.62 x Confidence 0.95 / Effort 1 = 5.34.
- Keep threshold: >= 2.0.

## Validation

Passed:

```bash
rustfmt --edition 2024 --check crates/fj-interpreters/src/lib.rs
git diff --check -- crates/fj-interpreters/src/lib.rs
RCH_WORKER=vmi1152480 RCH_WORKERS=vmi1152480 RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo test -j 1 -p fj-interpreters --lib \
    fusion_bf16_chain_matches_reference_bit_for_bit -- --nocapture
RCH_WORKER=vmi1152480 RCH_WORKERS=vmi1152480 RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo test -j 1 -p fj-interpreters --lib \
    fusion_f16_maxmin_abs_chain_matches_reference_bit_for_bit -- --nocapture
RCH_WORKER=vmi1152480 RCH_WORKERS=vmi1152480 RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo check -j 1 -p fj-interpreters --lib
RCH_WORKER=vmi1152480 RCH_WORKERS=vmi1152480 RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo clippy -j 1 -p fj-interpreters --lib --no-deps -- \
    -D warnings -A clippy::question_mark
```

Notes:

- Plain `cargo clippy -p fj-interpreters --lib -- -D warnings` is blocked by
  pre-existing dependency lint debt in `fj-lax`.
- `cargo clippy -p fj-interpreters --lib --no-deps -- -D warnings` is blocked by
  a pre-existing unchanged `clippy::question_mark` lint in the f64 broadcast
  classifier at `crates/fj-interpreters/src/lib.rs:5301`; blame predates this
  lever.
- `ubs crates/fj-interpreters/src/lib.rs` exits nonzero on the repository's
  existing large-file inventory, but its fmt, clippy, cargo check, test-build,
  audit, and deny sections were clean for this run.
