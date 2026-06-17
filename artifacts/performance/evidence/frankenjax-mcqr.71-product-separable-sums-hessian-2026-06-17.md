# frankenjax-mcqr.71: exact Hessian for product of separable sums

## Target

`frankenjax-mcqr.71` tracks the remaining exact Hessian gap after the
`Square(ReduceSum(unary_chain(x)))` and `Mul(sum, sum)` exact slices. This pass
ships one profile-backed spelling:

```text
f(x) = reduce_sum(g(x)) * reduce_sum(h(x))
```

where `g` and `h` are independent finite dense F64 unary elementwise chains over
the same input tensor.

## Baseline

Local release benchmark while ts1/rch remote was offline. The detached baseline
worktree was `/data/projects/.scratch/frankenjax-mcqr71-prod-sums-baseline-20260617`
at `HEAD` `e9ee8022`, patched only to use the same two-reduction benchmark
workload while leaving production code unchanged.

```bash
RCH_REQUIRE_REMOTE=0 CARGO_TARGET_DIR=/data/projects/.scratch/frankenjax-mcqr71-prod-sums-baseline-target cargo test -j 1 -p fj-ad bench_hessian_general_parallel_vs_serial --lib --release -- --ignored --nocapture
```

Result:

```text
hessian general n=192: SERIAL 41.963ms  PARALLEL 8.198ms  speedup 5.12x
```

Warmed hyperfine wrapper:

```text
Time (mean +/- sigma): 470.7 ms +/- 3.5 ms
Range: 465.5 ms ... 474.4 ms
```

Accepted baseline for `hessian_jaxpr`: `8.198ms`.

## Candidate

The candidate recognizes only a final `Mul(left_sum, right_sum)` where each input
is produced by a distinct `ReduceSum` over a single-input unary elementwise F64
chain from the same input tensor, and every equation in the JAXPR is consumed by
one of the two chains plus the final multiply. Unsupported forms fall through to
the existing central-difference path.

Same benchmark command in the candidate tree:

```text
hessian general n=192: SERIAL 38.499ms  PARALLEL 0.162ms  speedup 237.91x
```

Accepted comparison: `8.198ms -> 0.162ms` = `50.6x`.

Warmed hyperfine wrapper:

```text
Time (mean +/- sigma): 428.1 ms +/- 13.5 ms
Range: 408.3 ms ... 441.3 ms
```

Score: `4.5 = Impact 5.0 x Confidence 0.90 / Effort 1.0`.

## Proof

Focused Hessian release proof:

```bash
RCH_REQUIRE_REMOTE=0 CARGO_TARGET_DIR=/data/projects/.scratch/frankenjax-silvermaple-ad-target cargo test -j 1 -p fj-ad hessian_ --lib --release -- --nocapture
```

Result:

```text
7 passed; 0 failed; 2 ignored
```

New product-of-sums golden SHA:

```text
dde9f0a7db4485d9dc9b6aaee09b7d179772c978ebbde7b024feff64deef46ad
```

Existing exact square/self-mul golden SHA remains pinned:

```text
8830a0367731e540bba251bcccd2b18d3aa64ac3a9ca96d0696d780de48974c0
```

Existing separable diagonal golden SHA remains pinned:

```text
7a42b4e6a4b18cf77a7efcf248f694db80fe7b76ea40488f73210b4920e12764
```

## Isomorphism

- For the accepted pattern, the new path computes the analytic exact Hessian:
  `H_ij = g'_i h'_j + h'_i g'_j + [i == j] * (sum(h) g''_i + sum(g) h''_i)`.
- Output shape and order remain row-major `[input_dim, input_dim]`.
- Floating-point operation order for each analytic entry is fixed by the formula
  above; no reduction order, tie-breaking, or RNG surface is introduced.
- Custom JAXPR JVP/VJP, custom primitive JVP/VJP, params, effects, sub-JAXPRs,
  non-F64 inputs, non-tensor inputs, nonfinite intermediates, shared reduce
  variables, unused equations, and partially matched graphs all fall through.
- No unsafe code and no C BLAS/LAPACK/XLA linkage.

## Validation

Passed locally:

```text
cargo fmt --check -p fj-ad
git diff --check -- crates/fj-ad/src/lib.rs
cargo test -j 1 -p fj-ad hessian_ --lib --release -- --nocapture
cargo test -j 1 -p fj-ad bench_hessian_general_parallel_vs_serial --lib --release -- --ignored --nocapture
cargo check -j 1 -p fj-ad --all-targets
cargo clippy -j 1 -p fj-ad --all-targets --no-deps -- -D warnings
hyperfine --warmup 1 --runs 5 'RCH_REQUIRE_REMOTE=0 CARGO_TARGET_DIR=/data/projects/.scratch/frankenjax-silvermaple-ad-target cargo test -j 1 -p fj-ad bench_hessian_general_parallel_vs_serial --lib --release -- --ignored --nocapture'
```

UBS on `crates/fj-ad/src/lib.rs` remained nonzero from the existing large-file
heuristic inventory. Its embedded formatter, clippy, cargo check, test-build,
audit, and deny sections were clean and it reported no unsafe blocks.
