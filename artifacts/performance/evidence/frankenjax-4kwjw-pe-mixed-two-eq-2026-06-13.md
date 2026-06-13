# frankenjax-4kwjw: two-equation mixed partial-eval shortcut

Date: 2026-06-13
Agent: BeigeMouse
Worker: vmi1227854
Crate: fj-interpreters

## Target

Profile-backed bead `frankenjax-4kwjw` targeted the mixed partial-eval benchmark:

```text
partial_eval/mixed/neg_mul [350.00 ns 360.43 ns 371.53 ns]
```

The hotspot was the generic mixed partial-eval classifier for the common two-equation residual shape:

```text
known input, unknown input
residual = known_eqn(known)
out = unknown_eqn(residual, unknown)
```

## Lever

One lever shipped: a guarded two-equation shortcut in `partial_eval_jaxpr_typed_with_consts`.

The shortcut is only enabled when all of these are true:

- no external const values
- no Jaxpr constvars
- no supplied input abstract values
- exactly two invars, two equations, one output
- exactly one known input and one unknown input
- the first equation consumes only the known input or literals and emits one residual
- the second equation emits the final output and consumes both the residual and unknown input

All other shapes fall through to the existing generic mixed partial-eval path.

## Benchmark

Command:

```bash
RCH_WORKERS=vmi1227854 RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -j 1 -p fj-interpreters --bench pe_baseline -- partial_eval/mixed/neg_mul
```

Before:

```text
partial_eval/mixed/neg_mul [350.00 ns 360.43 ns 371.53 ns]
```

After:

```text
partial_eval/mixed/neg_mul [221.51 ns 225.14 ns 229.45 ns]
```

Mean ratio: `360.43 / 225.14 = 1.60x`.

Score: `Impact 1.60 * Confidence 0.95 / Effort 0.50 = 3.04`.

## Behavior Proof

The shortcut is a syntactic specialization of the existing generic mixed path. For the accepted shape it constructs the same result the fallback produces:

- known Jaxpr input order remains the original known input order
- known Jaxpr equation order remains the first original equation
- known Jaxpr output order remains original known outputs followed by residuals; for this shape that is the first equation output
- unknown Jaxpr input order remains residuals first, then original unknown inputs
- unknown Jaxpr equation order remains the second original equation
- unknown output order remains the original unknown output order
- `known_consts` remains empty because constvars and external const values are rejected
- `out_unknowns` remains `[true]`
- residual abstract value remains the generic fallback default: `F64` scalar when `in_avals` is absent
- no floating-point operations are executed by this transform, so FP ordering, NaN/tie behavior, and rounding are unchanged
- no RNG is used

Focused golden proof:

```bash
RCH_WORKERS=vmi1227854 RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -j 1 -p fj-interpreters --lib test_pe_two_eq_mixed_residual_fast_path_golden -- --nocapture
```

Result:

```text
test partial_eval::tests::test_pe_two_eq_mixed_residual_fast_path_golden ... ok
two-equation mixed partial-eval golden digest: f51e1a62763e23c83ec7a1433ef7e3ec3e1e9122a78edcc2559f1e5d4f97e88d
```

Golden SHA-256:

```text
f51e1a62763e23c83ec7a1433ef7e3ec3e1e9122a78edcc2559f1e5d4f97e88d
```

## Validation

Passed:

```bash
rustfmt --edition 2024 --check crates/fj-interpreters/src/partial_eval.rs
git diff --check
RCH_WORKERS=vmi1227854 RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -j 1 -p fj-interpreters --lib
```

`cargo check` repeated pre-existing dependency warnings outside this lever:

- `crates/fj-lax/src/lib.rs:3623`: `eval_reduce_window_iN_sum_sat` non-snake-case
- `crates/fj-trace/src/lib.rs:1808`: unused `num_spatial`

Blocked by pre-existing debt outside this lever:

```bash
RCH_WORKERS=vmi1227854 RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -j 1 -p fj-interpreters --lib --no-deps -- -D warnings
```

Failure:

```text
crates/fj-interpreters/src/lib.rs:5466: clippy::question_mark
```

`ubs crates/fj-interpreters/src/partial_eval.rs` exited nonzero from existing file-wide inventory. Its built-in formatting, clippy, cargo check, and test-build sections were clean.

## Decision

Keep and close `frankenjax-4kwjw`. The measured same-worker gain is real, the score is above the `2.0` threshold, and the proof surface is a strict guarded specialization of existing mixed partial-eval semantics.
