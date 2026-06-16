# frankenjax-0x3pu: dense F64 reduce_sum typed-slot plan

## Target

Profile-backed interpreter dispatch gap from `frankenjax-0x3pu`: small repeated
Jaxpr bodies with non-arithmetic primitives still fell through
`run_dense_env_into` and paid per-equation primitive dispatch. The focused slice
was a one-equation full `ReduceSum` over a dense F64 tensor.

## Baseline

RCH same-worker baseline before the lever, worker `vmi1152480`:

```text
cargo test -p fj-interpreters bench_dense_f64_reduce_sum_plan_overhead -- --ignored --nocapture
BENCH dense-f64 reduce_sum [64] 1000000 evals:
GENERIC 1780.6ns/eval -> PLANNED 1874.9ns/eval = 0.95x
sha256=561117f6fd0383063821dfcf3074491ba1e5f3943a828629e080c4fa7897c8cd
```

The "planned" route was slower because it failed to select a typed reduce plan
and then fell through to the generic interpreter.

## Lever

Added a narrow `DenseF64ReduceSumPlan` to `DenseEvalPlan`:

- compile-time shape: exactly one effect-free, sub-jaxpr-free `Primitive::ReduceSum`
  equation, one variable input, one output, no params, and the jaxpr output bound
  to that equation output.
- runtime shape: only dense `DType::F64` tensor storage is handled.
- fallback: scalar inputs, axes/params, non-F64 tensors, literal-backed tensors, and
  malformed/unsupported cases fall through to the existing generic interpreter.

The runner folds `values` in ascending backing-slice order with `acc = 0.0_f64`
and `acc += value`, matching `fj-lax` full dense F64 `reduce_sum`.

## After

RCH same-worker after run, worker `vmi1152480`:

```text
cargo test -p fj-interpreters bench_dense_f64_reduce_sum_plan_overhead -- --ignored --nocapture
BENCH dense-f64 reduce_sum [64] 1000000 evals:
GENERIC 2145.5ns/eval -> PLANNED 468.7ns/eval = 4.58x
sha256=561117f6fd0383063821dfcf3074491ba1e5f3943a828629e080c4fa7897c8cd
```

Comparable accepted delta: old planned fallback `1874.9ns/eval` to new planned
route `468.7ns/eval` = `4.00x`. Score `3.6 = Impact 4.0 x Confidence 0.90 /
Effort 1.0`.

## Isomorphism

- Ordering: one input equation and one output remain in the same order.
- Floating point: full dense F64 reduction keeps the exact prior seed `0.0_f64`
  and ascending element fold; no reassociation, chunking, SIMD, or threading.
- Tie/RNG: no comparisons or random state.
- Fallback: unsupported cases preserve existing scalar, axes, dtype, storage, and
  error behavior by falling through to `run_dense_env_into`.
- Safety: safe Rust only; no unsafe blocks.

## Proof And Gates

- RCH `cargo test -p fj-interpreters dense_f64_reduce_sum_plan_matches_generic_sha256 -- --nocapture`
  passed on `vmi1149989`.
- Golden output SHA-256 stayed
  `561117f6fd0383063821dfcf3074491ba1e5f3943a828629e080c4fa7897c8cd`.
- `cargo fmt --check -p fj-interpreters` passed.
- `git diff --check` passed.
- RCH `cargo check -p fj-interpreters --all-targets` passed on `vmi1156319`.
- RCH `cargo clippy -p fj-interpreters --all-targets --no-deps -- -D warnings`
  passed on `vmi1149989`.
- RCH `cargo clippy -p fj-interpreters --all-targets -- -D warnings` failed in
  committed `fj-lax` dependency code: `einsum.rs:1078` collapsible-if and
  `lib.rs:3894`/`lib.rs:4034` too-many-arguments. Follow-up bead:
  `frankenjax-ndyrj`.
- `ubs crates/fj-interpreters/src/lib.rs` remained nonzero from the existing
  file-wide heuristic inventory while its embedded formatting, clippy, cargo
  check, test-build, audit, and deny sections were clean.
