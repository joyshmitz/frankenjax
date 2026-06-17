# frankenjax-np8hi exact square-of-separable-sum Hessian

## Target

- Bead: `frankenjax-np8hi`
- Surface: `fj-ad::hessian_jaxpr` general nonseparable path
- Pattern shipped: `f(x) = square(reduce_sum(g(x)))`, where `g` is a unary elementwise chain over one dense F64 tensor input.

## Baseline

Local release benchmark before the change:

```text
CARGO_TARGET_DIR=/data/projects/.scratch/frankenjax-silvermaple-target CARGO_BUILD_JOBS=1 \
  cargo test -j 1 -p fj-ad bench_hessian_general_parallel_vs_serial --release --lib -- --ignored --nocapture

hessian general n=192: SERIAL 22.900ms  PARALLEL 5.678ms  speedup 4.03x
```

Accepted baseline for `hessian_jaxpr` itself: `5.678 ms`.

## Candidate

The new exact fast path uses the closed form for `s = sum_i g_i(x_i)`:

```text
H_ij = 2 * g'_i * g'_j + [i == j] * 2 * s * g''_i
```

It calls `fj-lax::eval_primitive` for every unary primal step, so primitive values match existing JAXPR semantics; only the derivative assembly is analytic.

Local release candidate:

```text
hessian general n=192: SERIAL 22.141ms  PARALLEL 0.124ms  speedup 179.16x
```

Accepted comparison: `5.678 ms -> 0.124 ms = 45.8x`.

The hyperfine wrapper was attempted after the warmed release benchmark, but the process exited `143` before producing a useful summary. The release benchmark above is the performance gate.

## Proof

- Golden exact fast-path test: `hessian_square_of_separable_sum_matches_closed_form_and_golden`
- Golden SHA-256: `8830a0367731e540bba251bcccd2b18d3aa64ac3a9ca96d0696d780de48974c0`
- Fallback central-difference coverage preserved by spelling the old fallback test's final square as `Mul(sum, sum)`, which does not match the new `Square(ReduceSum(...))` fast path.

## Isomorphism

- Supported pattern is mathematically exact for dense F64 `square(reduce_sum(unary_chain(x)))`.
- Unsupported primitives, non-F64 inputs, non-tensor inputs, params/effects/sub-jaxprs, nonfinite intermediate values, custom JVP/VJP rules, and nonmatching graph shapes fall through to the existing central-difference path.
- No unsafe code, no RNG, no tie-breaking surface, and no C BLAS/LAPACK/XLA linkage.

## Validation

```text
cargo fmt --check -p fj-ad
git diff --check -- crates/fj-ad/src/lib.rs
CARGO_TARGET_DIR=/data/projects/.scratch/frankenjax-silvermaple-target CARGO_BUILD_JOBS=1 cargo test -j 1 -p fj-ad hessian --lib -- --nocapture
CARGO_TARGET_DIR=/data/projects/.scratch/frankenjax-silvermaple-target CARGO_BUILD_JOBS=1 cargo check -j 1 -p fj-ad --all-targets
CARGO_TARGET_DIR=/data/projects/.scratch/frankenjax-silvermaple-target CARGO_BUILD_JOBS=1 cargo clippy -j 1 -p fj-ad --all-targets -- -D warnings
ubs crates/fj-ad/src/lib.rs
```

UBS exited nonzero from the pre-existing `fj-ad` file-wide inventory. Its embedded formatter, clippy, cargo check, test-build, audit, and deny sections were clean, and it reported no unsafe blocks.
