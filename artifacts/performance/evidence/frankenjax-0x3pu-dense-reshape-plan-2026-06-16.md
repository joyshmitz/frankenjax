# frankenjax-0x3pu: dense reshape typed-slot plan

Date: 2026-06-16
Bead: `frankenjax-0x3pu`
Crate: `fj-interpreters`
Touched source: `crates/fj-interpreters/src/lib.rs`

## Profile-backed target

`frankenjax-0x3pu` tracks the remaining dispatch tax in repeated Jaxpr bodies:
each unsupported primitive family falls through to `run_dense_env_into`, then
re-parses string params and dispatches through `eval_primitive` on every
iteration.

The fresh local dense-plan sweep after the gather/axis/arg-extremum closeout
showed that one-equation dense reshape bodies still had no typed-slot route:

```text
BENCH dense-f64 reshape [64]->[8,8] 1000000 evals:
GENERIC 3355.6ns/eval -> PLANNED 3360.2ns/eval = 1.00x
sha256=ebe11fbcbab2dff9893ae1a4a4f90c0eeb1881d8d4c5ce14f447e73bad18a7e3
```

The "PLANNED" arm above was the pre-existing dense-plan fallback route, not a
reshape fast path. The user override for 2026-06-16 said `ts1` was offline, so
this pass used local `cargo` and `hyperfine` rather than waiting on remote RCH.

## One lever

Add `DenseReshapePlan` for exactly one effect-free, sub-jaxpr-free
`Primitive::Reshape` equation whose input and output are variables and whose
single output is the Jaxpr output.

The plan pre-parses the static `new_shape` string once during plan construction
and records the target element count. Runtime accepts only tensor input with
matching element count, then constructs the same metadata-only tensor reshape
via `TensorValue::new_with_literal_buffer` and a clone of the existing literal
buffer.

Unsupported cases fall through to the existing generic interpreter:

- scalar input,
- inferred negative dimensions,
- malformed shape params,
- dynamic shape mismatches,
- effectful or multi-equation Jaxprs.

## Re-benchmark

Focused local post-lever benchmark:

```text
cargo test -j 1 -p fj-interpreters bench_dense_f64_reshape_plan_overhead -- --ignored --nocapture
BENCH dense-f64 reshape [64]->[8,8] 1000000 evals:
GENERIC 3030.5ns/eval -> PLANNED 952.0ns/eval = 3.18x
sha256=ebe11fbcbab2dff9893ae1a4a4f90c0eeb1881d8d4c5ce14f447e73bad18a7e3
```

Accepted comparison: old planned fallback `3360.2ns/eval` to new planned route
`952.0ns/eval` = `3.53x`.

Hyperfine wrapper timing for the focused local benchmark:

```text
hyperfine --warmup 0 --runs 3 \
  'CARGO_TARGET_DIR=/data/tmp/frankenjax-local-0x3pu-benches cargo test -j 1 -p fj-interpreters bench_dense_f64_reshape_plan_overhead -- --ignored --nocapture'

Time (mean +- sigma): 24.387 s +- 0.264 s
Range: 24.110 s .. 24.635 s
```

Score: `3.8 = Impact 4.0 x Confidence 0.95 / Effort 1.0`.

## Isomorphism proof

Focused proof:

```text
cargo test -j 1 -p fj-interpreters dense_reshape_plan_matches_generic_and_golden -- --nocapture
test tests::dense_reshape_plan_matches_generic_and_golden ... ok
```

Golden output SHA-256:

```text
ebe11fbcbab2dff9893ae1a4a4f90c0eeb1881d8d4c5ce14f447e73bad18a7e3
```

Checklist:

- Equation order unchanged: the plan only accepts one reshape equation.
- Output ordering unchanged: the single equation output must equal the Jaxpr
  output list.
- Floating-point behavior unchanged: reshape performs no arithmetic, no
  reassociation, no FMA, and no rounding.
- Element order unchanged: the existing literal buffer is cloned in its current
  row-major element order and only the shape tag changes.
- Tie-breaking unchanged: no comparisons or tie choices exist.
- RNG unchanged: no randomness.
- Error behavior preserved: static valid tensor reshapes take the planned route;
  scalar, inferred-dim, malformed, and shape-mismatch cases fall back to
  `eval_primitive`.
- Safety unchanged: safe Rust only; no `unsafe`.

## Validation

Passed locally with crate-scoped commands:

```text
cargo fmt -p fj-interpreters
cargo fmt --check -p fj-interpreters
cargo test -j 1 -p fj-interpreters dense_reshape_plan_matches_generic_and_golden -- --nocapture
cargo test -j 1 -p fj-interpreters bench_dense_f64_reshape_plan_overhead -- --ignored --nocapture
hyperfine --warmup 0 --runs 3 'CARGO_TARGET_DIR=/data/tmp/frankenjax-local-0x3pu-benches cargo test -j 1 -p fj-interpreters bench_dense_f64_reshape_plan_overhead -- --ignored --nocapture'
cargo check -j 1 -p fj-interpreters --all-targets
cargo clippy -j 1 -p fj-interpreters --all-targets -- -D warnings
git diff --check -- crates/fj-interpreters/src/lib.rs
```

`ubs crates/fj-interpreters/src/lib.rs` remained nonzero from the existing
file-wide heuristic inventory, including existing unwrap/panic/index warnings
and a false positive around a local variable named `decode`. Its embedded
fmt/clippy/check/test-build/audit/deny sections were clean, and it reported no
unsafe blocks.
