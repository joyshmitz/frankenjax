# frankenjax-f16-axis-reduce-simd-pd3kl: F16 trailing-axis SIMD decode

Date: 2026-06-12
Agent: BeigeMouse
Crate: fj-lax

## Target

Profile-backed bead `frankenjax-f16-axis-reduce-simd-pd3kl` targeted F16 trailing-axis reductions shaped like softmax row reductions. The old dense path used `dense_f64_axis_reduce` with scalar `Literal::F16Bits(...).as_f64()` decode for every input element.

## Lever

Added a F16-only leading-prefix/trailing-reduce fast path:

- `fold_f16_axis_block` widens normal/zero F16 chunks through `f16_widen8`.
- Any chunk containing subnormal, inf, or NaN falls back to the scalar `Literal::F16Bits(...).as_f64()` decode.
- The fold remains a scalar f64 accumulator in ascending element order.
- The route is used only when kept axes are the leading prefix, so middle-axis and non-contiguous reductions keep the existing path.
- Work-scaled threading matches the existing large dense trailing-axis threshold.

## Behavior Proof

RCH proof:

```text
RCH_WORKER=vmi1152480 rch exec -- cargo test -j 1 -p fj-lax --lib dense_f16_trailing_axis_reduce_simd_decode_matches_boxed_edge_rows -- --nocapture
```

Result:

```text
test reduction::tests::dense_f16_trailing_axis_reduce_simd_decode_matches_boxed_edge_rows ... ok
```

The proof compares raw F16 output bits against the boxed `Literal::F16Bits` path for `ReduceSum`, `ReduceProd`, `ReduceMax`, and `ReduceMin` over edge-rich rows containing normal chunks, signed zero, subnormals, infinities, NaNs, and scalar tails. No NaN canonicalization is used in the assertion.

Isomorphism checklist:

- Output shape and dtype preserved.
- Per-row element order preserved.
- Floating-point operation order preserved.
- NaN payload behavior pinned against boxed scalar path.
- Signed-zero/tie behavior pinned against boxed scalar path.
- No RNG or nondeterministic tie-breaking introduced.

## Benchmark

Final release A/B on same worker and same test binary:

```text
RCH_WORKER=vmi1152480 rch exec -- cargo test --release -j 1 -p fj-lax --lib bench_f16_reduce_sum_axis_simd_decode_vs_scalar_dense -- --ignored --nocapture
```

Result:

```text
BENCH f16 reduce_sum axis1 [4096,1024]: scalar_dense=2.4290ms simd_decode=0.7383ms speedup=3.29x sha256=a321e41484cba20e931270f7c083710b843a9a031cb6c90830e82ab112739b4e
```

The ignored benchmark compares the old scalar dense helper directly with the new SIMD decode helper in the same binary, after asserting identical rounded F16 outputs. The SHA-256 fixture ID is now asserted in the benchmark.

Score: `3.29 impact x 0.95 confidence / 1.0 effort = 3.13`, keep.

## Validation

Passed:

```text
rustfmt --edition 2024 --check crates/fj-lax/src/reduction.rs
git diff --check -- crates/fj-lax/src/arithmetic.rs crates/fj-lax/src/reduction.rs
rch exec -- cargo check -j 1 -p fj-lax --lib
```

Known nonblocking debt observed:

- `rch exec -- cargo clippy -j 1 -p fj-lax --lib -- -D warnings` fails in pre-existing `crates/fj-lax/src/linalg.rs` lint debt (`doc_lazy_continuation` and `needless_range_loop`).
- `rustfmt --check` on `arithmetic.rs`/`linalg.rs` exposes broader pre-existing formatting drift; whole-file formatting was not run to avoid unrelated churn.
- `ubs crates/fj-lax/src/reduction.rs crates/fj-lax/src/arithmetic.rs` exits nonzero from pre-existing panic/unwrap/indexing inventory, while its built-in fmt, clippy, check, test-build, cargo-audit, and cargo-deny sections were clean.
