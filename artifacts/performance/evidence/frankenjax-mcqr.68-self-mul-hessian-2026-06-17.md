# frankenjax-mcqr.68: exact Hessian for self-mul separable sum

## Target

`frankenjax-mcqr.68` tracks the remaining exact Hessian gap after the
`Square(ReduceSum(unary_chain(x)))` fast path. This pass ships one narrow
profile-backed spelling:

```text
f(x) = mul(reduce_sum(g(x)), reduce_sum(g(x)))
```

where `g` is the same finite dense F64 unary elementwise chain already accepted
by the square fast path. Before this pass, this algebraically identical spelling
fell through to the central-difference Hessian fallback.

## Baseline

Local release benchmark while ts1/rch remote was offline:

```bash
CARGO_TARGET_DIR=/data/projects/.scratch/frankenjax-silvermaple-ad-target CARGO_BUILD_JOBS=1 cargo test -j 1 -p fj-ad bench_hessian_general_parallel_vs_serial --lib --release -- --ignored --nocapture
```

Result before the production recognizer change:

```text
hessian general n=192: SERIAL 20.472ms  PARALLEL 5.844ms  speedup 3.50x
```

Accepted baseline for `hessian_jaxpr`: `5.844ms`.

## Candidate

The exact Hessian recognizer now accepts the final self-multiply only when both
`Mul` inputs are the same `ReduceSum` output variable. Structurally different
forms still fall back to the existing central-difference path.

Same command after the change:

```text
hessian general n=192: SERIAL 22.910ms  PARALLEL 0.087ms  speedup 264.35x
```

Accepted comparison: `5.844ms -> 0.087ms` = `67.2x`.

Warmed hyperfine wrapper for the candidate command:

```text
Time (mean +/- sigma): 286.5 ms +/- 6.8 ms
Range: 282.6 ms ... 294.4 ms
```

Score: `4.5 = Impact 5.0 x Confidence 0.90 / Effort 1.0`.

## Proof

Focused Hessian release proof:

```bash
CARGO_TARGET_DIR=/data/projects/.scratch/frankenjax-silvermaple-ad-target CARGO_BUILD_JOBS=1 cargo test -j 1 -p fj-ad hessian --lib --release -- --nocapture
```

Result:

```text
6 passed; 0 failed; 2 ignored
```

The existing exact Hessian golden is preserved and is also asserted for the new
self-mul spelling:

```text
8830a0367731e540bba251bcccd2b18d3aa64ac3a9ca96d0696d780de48974c0
```

The central-difference fallback test now uses `sum * (sum + 0)` so it remains
outside the self-mul recognizer and still proves the fallback path.

## Isomorphism

- `Mul(sum, sum)` is algebraically identical to `Square(sum)` for the accepted
  dense finite F64 scalar-output pattern.
- The recognizer requires both final multiply inputs to be the exact same
  `ReduceSum` output variable; any distinct variable, params/effects/sub-jaxprs,
  unsupported primitive, custom JVP/VJP rule, non-F64 input, non-tensor input,
  or nonfinite intermediate falls through.
- Output shape and order remain row-major `[input_dim, input_dim]`.
- Existing separable diagonal golden SHA remains unchanged:
  `7a42b4e6a4b18cf77a7efcf248f694db80fe7b76ea40488f73210b4920e12764`.
- No tie-breaking surface, no RNG, no unsafe code, and no C BLAS/LAPACK/XLA
  linkage.

## Validation

Passed:

```text
cargo fmt --check -p fj-ad
git diff --check -- crates/fj-ad/src/lib.rs
cargo test -j 1 -p fj-ad hessian --lib --release -- --nocapture
cargo test -j 1 -p fj-ad bench_hessian_general_parallel_vs_serial --lib --release -- --ignored --nocapture
cargo check -j 1 -p fj-ad --all-targets
cargo clippy -j 1 -p fj-ad --all-targets -- -D warnings
hyperfine --warmup 1 --runs 3 'CARGO_TARGET_DIR=/data/projects/.scratch/frankenjax-silvermaple-ad-target CARGO_BUILD_JOBS=1 cargo test -j 1 -p fj-ad bench_hessian_general_parallel_vs_serial --lib --release -- --ignored --nocapture'
```

UBS on `crates/fj-ad/src/lib.rs` remained nonzero from the existing large-file
heuristic inventory. Its embedded formatter, clippy, cargo check, test-build,
audit, and deny sections were clean and it reported no unsafe blocks.
