# frankenjax-mcqr.72: exact Hessian for adjacent pair sum

## Target

`frankenjax-mcqr.72` tracks genuinely nonseparable scalar-output Hessians after
the separable square/self-mul/product-of-sums slices. This pass ships one
profile-backed local-interaction shape:

```text
f(x) = reduce_sum(slice(x, 0..n-1) * slice(x, 1..n))
```

Before this pass, the shape fell through to the central-difference Hessian path.

## Baseline

Local release benchmark while ts1/rch remote was offline. The detached baseline
worktree was `/data/projects/.scratch/frankenjax-mcqr72-clean-20260617` at
`HEAD` `256f4e38`, patched only to use the same local-pair benchmark workload
while leaving production code unchanged.

```bash
RCH_REQUIRE_REMOTE=0 CARGO_TARGET_DIR=/data/projects/.scratch/frankenjax-mcqr72-target cargo test -j 1 -p fj-ad bench_hessian_general_parallel_vs_serial --lib --release -- --ignored --nocapture
```

Result:

```text
hessian general n=512: SERIAL 15.405ms  PARALLEL 9.052ms  speedup 1.70x
```

Warmed hyperfine wrapper:

```text
Time (mean +/- sigma): 318.6 ms +/- 20.6 ms
Range: 303.9 ms ... 352.6 ms
```

Accepted baseline for `hessian_jaxpr`: `9.052ms`.

## Candidate

The candidate recognizes exactly a four-equation, single-input, dense F64 rank-1
JAXPR:

```text
left = slice(x, 0..n-1)
right = slice(x, 1..n)
pairwise = mul(left, right)
out = reduce_sum(pairwise)
```

It emits the analytic tridiagonal Hessian directly: `H[i, i+1] = 1` and
`H[i+1, i] = 1` for every adjacent pair; all other entries are `0`.

Same benchmark command in the candidate worktree:

```text
hessian local-pair n=512: SERIAL 14.817ms  PARALLEL 0.350ms  speedup 42.30x
```

Accepted comparison: `9.052ms -> 0.350ms` = `25.9x`.

Warmed hyperfine wrapper:

```text
Time (mean +/- sigma): 266.2 ms +/- 6.0 ms
Range: 260.1 ms ... 273.9 ms
```

Score: `4.5 = Impact 5.0 x Confidence 0.90 / Effort 1.0`.

## Proof

Focused Hessian release proof:

```bash
RCH_REQUIRE_REMOTE=0 CARGO_TARGET_DIR=/data/projects/.scratch/frankenjax-mcqr72-candidate-target cargo test -j 1 -p fj-ad hessian_ --lib --release -- --nocapture
```

Result:

```text
8 passed; 0 failed; 2 ignored
```

New adjacent-pair golden SHA:

```text
398f8f89b8579b1fc6de33dc44588af05bdd05ede91546372a5f9f25d3307cc6
```

Existing exact product-of-sums golden SHA remains pinned:

```text
dde9f0a7db4485d9dc9b6aaee09b7d179772c978ebbde7b024feff64deef46ad
```

Existing exact square/self-mul and separable diagonal SHAs remain pinned:

```text
8830a0367731e540bba251bcccd2b18d3aa64ac3a9ca96d0696d780de48974c0
7a42b4e6a4b18cf77a7efcf248f694db80fe7b76ea40488f73210b4920e12764
```

## Isomorphism

- For the accepted pattern, the prior finite-difference result is replaced by
  the exact analytic Hessian for the same scalar objective.
- Output shape and order remain row-major `[input_dim, input_dim]`.
- Floating-point output is deterministic constants only: `1.0` for adjacent
  off-diagonals, `0.0` elsewhere. There is no reduction-order, tie-breaking, or
  RNG surface.
- Custom JAXPR JVP/VJP, custom primitive JVP/VJP, params/effects/sub-JAXPRs,
  non-F64 inputs, non-rank-1 tensor inputs, nonfinite inputs, strides, malformed
  slice intervals, wrong equation order, extra equations, and partial matches all
  fall through.
- No unsafe code and no C BLAS/LAPACK/XLA linkage.

## Validation

Passed locally in the clean candidate worktree:

```text
cargo fmt --check -p fj-ad
git diff --check -- crates/fj-ad/src/lib.rs
cargo test -j 1 -p fj-ad hessian_ --lib --release -- --nocapture
cargo test -j 1 -p fj-ad bench_hessian_general_parallel_vs_serial --lib --release -- --ignored --nocapture
cargo check -j 1 -p fj-ad --all-targets
cargo clippy -j 1 -p fj-ad --all-targets --no-deps -- -D warnings
hyperfine --warmup 1 --runs 5 'RCH_REQUIRE_REMOTE=0 CARGO_TARGET_DIR=/data/projects/.scratch/frankenjax-mcqr72-candidate-target cargo test -j 1 -p fj-ad bench_hessian_general_parallel_vs_serial --lib --release -- --ignored --nocapture'
```

UBS on `crates/fj-ad/src/lib.rs` remained nonzero from the existing large-file
heuristic inventory. Its embedded formatter, clippy, cargo check, test-build,
audit, and deny sections were clean and it reported no unsafe blocks.
