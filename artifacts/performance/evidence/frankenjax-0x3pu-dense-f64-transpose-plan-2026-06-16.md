# frankenjax-0x3pu: dense F64 transpose typed-slot plan

Date: 2026-06-16
Bead: `frankenjax-0x3pu`
Crate: `fj-interpreters`
Touched source: `crates/fj-interpreters/src/lib.rs`

## Profile-backed target

After the reshape typed-slot plan, the same interpreter-dispatch frontier still
had no typed route for one-equation rank-2 dense F64 `Primitive::Transpose`
bodies. The repeated-body cost was dominated by re-entering `eval_primitive`,
including per-call `permutation` parsing and generic dispatch, before reaching
the already-fast fj-lax data-movement kernel.

Baseline was local because the 2026-06-16 user override marked `ts1` offline and
forbade waiting on remote RCH:

```text
cargo test -j 1 -p fj-interpreters bench_dense_f64_transpose_plan_overhead -- --ignored --nocapture
BENCH dense-f64 transpose [8,8]->[8,8] 1000000 evals:
GENERIC 6120.2ns/eval -> PLANNED 6201.0ns/eval = 0.99x
sha256=6aedc3b5cb517b847be5026fbfd3eb3f58215f415892d320849ff93a9b887713
```

The "PLANNED" arm above was the current dense-plan fallback route.

## One lever

Add `DenseF64TransposePlan` for exactly one effect-free, sub-jaxpr-free
`Primitive::Transpose` equation over one variable input whose output is the
Jaxpr output.

The plan pre-parses the optional `permutation` parameter once. Runtime handles
only dense F64 rank-2 `[1,0]` transposes, including the default rank-2 reversal,
and uses the same blocked row-major source-read / destination-write traversal as
the fj-lax fast path.

Unsupported cases fall through to the existing generic interpreter:

- scalar input,
- non-F64 tensors,
- non-rank-2 tensors,
- empty tensors,
- invalid or non-`[1,0]` permutations,
- effectful or multi-equation Jaxprs.

## Re-benchmark

Focused local post-lever benchmark:

```text
cargo test -j 1 -p fj-interpreters bench_dense_f64_transpose_plan_overhead -- --ignored --nocapture
BENCH dense-f64 transpose [8,8]->[8,8] 1000000 evals:
GENERIC 6358.4ns/eval -> PLANNED 3315.6ns/eval = 1.92x
sha256=6aedc3b5cb517b847be5026fbfd3eb3f58215f415892d320849ff93a9b887713
```

Accepted comparison: old planned fallback `6201.0ns/eval` to new planned route
`3315.6ns/eval` = `1.87x`.

Hyperfine wrapper timing for the focused local benchmark:

```text
hyperfine --warmup 0 --runs 3 \
  'CARGO_TARGET_DIR=/data/tmp/frankenjax-local-0x3pu-transpose cargo test -j 1 -p fj-interpreters bench_dense_f64_transpose_plan_overhead -- --ignored --nocapture'

Time (mean +- sigma): 58.693 s +- 1.881 s
Range: 56.547 s ... 60.057 s
```

Score: `2.3 = Impact 2.5 x Confidence 0.92 / Effort 1.0`.

## Isomorphism proof

Focused proof:

```text
cargo test -j 1 -p fj-interpreters dense_f64_transpose_plan_matches_generic_and_golden -- --nocapture
test tests::dense_f64_transpose_plan_matches_generic_and_golden ... ok
```

Golden output SHA-256:

```text
6aedc3b5cb517b847be5026fbfd3eb3f58215f415892d320849ff93a9b887713
```

Checklist:

- Equation order unchanged: the plan only accepts one transpose equation.
- Output ordering unchanged: the single equation output must equal the Jaxpr
  output list.
- Floating-point behavior unchanged: transpose performs no arithmetic, no
  reassociation, no FMA, and no rounding.
- Element order unchanged: each output cell reads exactly `src[i * cols + j]`
  into `dst[j * rows + i]`, matching fj-lax's rank-2 blocked transpose loop.
- Tie-breaking unchanged: no comparisons or tie choices exist.
- RNG unchanged: no randomness.
- Error behavior preserved: only valid dense F64 rank-2 `[1,0]` transposes take
  the planned route; invalid permutations, scalar input, empty tensors, wrong
  rank, wrong dtype, and malformed cases fall back to `eval_primitive`.
- Safety unchanged: safe Rust only; no `unsafe`.

## Validation

Passed locally with crate-scoped commands:

```text
cargo fmt -p fj-interpreters
cargo fmt --check -p fj-interpreters
cargo test -j 1 -p fj-interpreters dense_f64_transpose_plan_matches_generic_and_golden -- --nocapture
cargo test -j 1 -p fj-interpreters bench_dense_f64_transpose_plan_overhead -- --ignored --nocapture
hyperfine --warmup 0 --runs 3 'CARGO_TARGET_DIR=/data/tmp/frankenjax-local-0x3pu-transpose cargo test -j 1 -p fj-interpreters bench_dense_f64_transpose_plan_overhead -- --ignored --nocapture'
cargo check -j 1 -p fj-interpreters --all-targets
cargo clippy -j 1 -p fj-interpreters --all-targets -- -D warnings
git diff --check -- crates/fj-interpreters/src/lib.rs
```

`ubs crates/fj-interpreters/src/lib.rs` remained nonzero from the existing
file-wide heuristic inventory, including existing unwrap/panic/index warnings
and a false positive around a local variable named `decode`. Its embedded
fmt/clippy/check/test-build/audit/deny sections were clean, and it reported no
unsafe blocks.
