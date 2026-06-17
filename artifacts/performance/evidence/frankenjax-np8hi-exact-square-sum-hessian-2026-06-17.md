# frankenjax-np8hi: exact square-of-separable-sum Hessian

## Target

`frankenjax-np8hi` tracks the general Hessian gap: the fallback path computes
central-difference columns of `grad_jaxpr`, which is approximate and costs two
gradient evaluations per input dimension. This pass keeps the existing
bit-pinned separable `sum(g(x_i))` path unchanged and adds a narrower exact
closed-form path for:

```text
f(x) = square(reduce_sum(g(x)))
```

where `g` is a single-input unary elementwise chain over a finite dense F64
tensor.

## Local Baseline

ts1/rch remote was offline, so this used local cargo with
`RCH_REQUIRE_REMOTE=0`.

Baseline worktree:
`/data/projects/.scratch/frankenjax-np8hi-baseline-20260617` at `HEAD`
`f20529af`.

Command:

```bash
RCH_REQUIRE_REMOTE=0 CARGO_TARGET_DIR=/data/projects/.scratch/frankenjax-np8hi-baseline-target cargo test -j 1 -p fj-ad bench_hessian_general_parallel_vs_serial --lib --release -- --ignored --nocapture
```

Result:

```text
hessian general n=192: SERIAL 20.324ms  PARALLEL 6.235ms  speedup 3.26x
```

Hyperfine wrapper, warmed target:

```text
Time (mean +/- sigma): 332.4 ms +/- 19.7 ms
Range: 307.1 ms ... 358.4 ms
```

## Candidate

Command:

```bash
RCH_REQUIRE_REMOTE=0 CARGO_TARGET_DIR=/data/projects/.scratch/frankenjax-silvermaple-ad-target cargo test -j 1 -p fj-ad bench_hessian_general_parallel_vs_serial --lib --release -- --ignored --nocapture
```

Result:

```text
hessian general n=192: SERIAL 19.460ms  PARALLEL 0.088ms  speedup 220.79x
```

Accepted comparison: old `hessian_jaxpr` path `6.235ms` to exact fast path
`0.088ms` = `70.9x`.

Hyperfine wrapper, warmed target:

```text
Time (mean +/- sigma): 292.3 ms +/- 14.7 ms
Range: 275.1 ms ... 310.3 ms
```

Command-level wrapper speedup is `1.14x`; the in-bench Hessian kernel speedup is
`70.9x`. Score: `4.5 = Impact 5.0 x Confidence 0.90 / Effort 1.0`.

## Proof

Focused Hessian proof suite:

```bash
RCH_REQUIRE_REMOTE=0 CARGO_TARGET_DIR=/data/projects/.scratch/frankenjax-silvermaple-ad-target cargo test -j 1 -p fj-ad hessian_ --lib --release -- --nocapture
```

Result:

```text
6 passed; 0 failed; 2 ignored
```

Covered checks:

- `hessian_square_of_separable_sum_matches_closed_form_and_golden`
- `hessian_general_parallel_matches_serial_central_difference`
- `hessian_separable_diagonal_matches_perbasis_and_golden`
- scalar/quadratic Hessian regression tests

Golden SHA-256 for the new exact path:

```text
8830a0367731e540bba251bcccd2b18d3aa64ac3a9ca96d0696d780de48974c0
```

Rejected sublever during this pass: an exact fast path for the already
bit-pinned separable `sum(g(x_i))` path changed the existing golden from
`7a42b4e6a4b18cf77a7efcf248f694db80fe7b76ea40488f73210b4920e12764` to
`eadb652b927b81142b9671ff9d09b08af425a08df149aab40ecc7e93cf94fe8f`, so it was
removed. The separable golden remains unchanged and passing.

## Isomorphism

- Scope is deliberately narrow: one dense finite F64 input, unary elementwise
  chain, full `ReduceSum`, final `Square`.
- Unsupported graph shapes, non-F64/non-tensor inputs, params/effects/sub-jaxprs,
  custom JVP/VJP rules, and nonfinite intermediates fall through to the existing
  central-difference path.
- Output shape and order remain row-major `[input_dim, input_dim]`.
- The accepted pattern intentionally changes the numerical contract from
  finite-difference approximate to exact closed form, matching the `np8hi`
  correctness target while leaving existing separable bit goldens unchanged.
- No tie-breaking surface, no RNG, no unsafe code, and no C BLAS/LAPACK/XLA
  linkage.

## Validation

Passed:

```text
cargo test -j 1 -p fj-ad hessian_ --lib --release -- --nocapture
cargo test -j 1 -p fj-ad bench_hessian_general_parallel_vs_serial --lib --release -- --ignored --nocapture
cargo check -j 1 -p fj-ad --all-targets
cargo clippy -j 1 -p fj-ad --all-targets --no-deps -- -D warnings
cargo fmt --check -p fj-ad
git diff --check -- crates/fj-ad/src/lib.rs
ubs crates/fj-ad/src/lib.rs
```

Full dependency clippy:

```text
cargo clippy -j 1 -p fj-ad --all-targets -- -D warnings
```

failed in unrelated dependency code:

```text
crates/fj-interpreters/src/partial_eval.rs:940:20:
cannot find function `det_scalar_dtype` in this scope
```

That file is outside this bead's reservation and was not modified by this pass.

UBS exited nonzero on pre-existing file-wide `fj-ad` inventory
(`unwrap`/`panic`/indexing/clone/string-allocation heuristics). Its embedded
fmt/clippy/check/test-build/audit/deny sections were clean and it reported no
unsafe blocks.
