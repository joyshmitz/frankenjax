# frankenjax-kg68b - Direct Lazy Grad Tensor Construction

Date: 2026-06-13
Agent: BeigeMouse
Bead: frankenjax-kg68b
Crate: fj-ad

## Target

Post-`frankenjax-zos8i` RCH reprofile on `vmi1152480`:

- Command: `cargo bench -p fj-ad --bench ad_baseline -- 'ad/(grad_sum_x2_plus_x_1k|grad_sin_cos_mul|grad_exp_log|jvp_sin_cos_mul|value_and_grad_poly)' --sample-size 20 --measurement-time 2 --warm-up-time 1 --noplot`
- Top row: `ad/grad_sum_x2_plus_x_1k [168.68 ns 173.20 ns 177.03 ns]`.
- Other rows: `grad_exp_log [150.03 ns 158.82 ns 166.57 ns]`, `value_and_grad_poly [145.63 ns 154.06 ns 167.94 ns]`, `jvp_sin_cos_mul [146.53 ns 159.06 ns 171.05 ns]`, `grad_sin_cos_mul [123.86 ns 137.96 ns 154.77 ns]`.

The vector-fill cost was removed by `frankenjax-zos8i`; the residual on this target is tensor output construction.

## Lever

One lever kept:

- In the exact dense F64 `grad_jaxpr(sum(x*x + x))` fast path, construct the output `TensorValue` directly after the guard proves the input tensor and lazy gradient buffer share the same element count.
- This bypasses only the redundant `TensorValue::new_with_literal_buffer` validation/error conversion in this exact path.
- All generic, custom-VJP, non-F64, non-dense, or nonmatching Jaxpr cases still fall back.

Alien-graveyard mapping:

- Certified rewrite / equality saturation: the exact Jaxpr guard proves the specialized output expression.
- Adaptive runtime specialization: direct construction applies only to the already-specialized dense F64 grad-only path.
- Deforestation: the path now skips both eager gradient materialization and redundant reconstruction validation.

## Baseline And Rebench

Route baseline:

- Worker: `vmi1152480`
- Result: `ad/grad_sum_x2_plus_x_1k [168.68 ns 173.20 ns 177.03 ns]`.

Same-worker acceptance baseline:

- Worker: `vmi1149989`
- Parent commit: `ac2f255b`
- Source: previous pass candidate rebench before this lever.
- Result: `ad/grad_sum_x2_plus_x_1k [148.22 ns 151.49 ns 154.69 ns]`.

Candidate:

- Requested worker: `vmi1152480`
- RCH-selected worker: `vmi1149989`
- Command: `cargo bench -p fj-ad --bench ad_baseline -- 'ad/grad_sum_x2_plus_x_1k' --sample-size 20 --measurement-time 2 --warm-up-time 1 --noplot`
- Result: `ad/grad_sum_x2_plus_x_1k [128.03 ns 130.31 ns 133.52 ns]`.

Win against same-worker parent:

- Midpoint speedup: `151.49 ns / 130.31 ns = 1.16x`.
- Conservative speedup: `148.22 ns / 133.52 ns = 1.11x`.
- Intervals are non-overlapping.
- Score: `Impact 2 x Confidence 5 / Effort 1 = 10.0`.

## Behavior Proof

Guard and construction invariant:

- The direct construction is reached only after `dense_f64_square_plus_linear_reducesum_input` accepts the exact three-equation Jaxpr and `dense_f64_add_mul_reducesum_fast_path_uses_builtin_vjp` confirms built-in VJPs.
- The input `TensorValue` was already constructed through normal validation, so `tensor.shape.element_count() == tensor.elements.len()`.
- `f64_values_arc()` returns the dense F64 storage backing that same tensor; for lazy F64 storage it first materializes a vector with the same base length.
- `LiteralBuffer::from_f64_one_plus_x_plus_x(input)` reports `len() == input.len()`, so the output shape and lazy buffer length match by construction.

Ordering and shape:

- Output arity, dtype, shape, and row-major element order are unchanged.
- The fast path still returns one tensor gradient in the same position.

Floating point:

- This lever does not change any FP operation.
- Lazy materialization still uses the exact prior per-element order: `1.0_f64`, then `+= x`, then `+= x`.
- Signed zeros, infinities, and NaN payload propagation remain covered by the existing generic bit-equivalence test.

Tie-breaking and RNG:

- No comparisons, tie-breaking, or RNG state are introduced.

Golden output:

- Test: `grad_sum_x2_plus_x_1k_golden_sha256`.
- SHA-256: `5282853e2bd187c1c1bfdfa612bd74776fb403e6b767eb0a8bf0c8bcd2fe2a19`.

## Validation

- `cargo fmt --package fj-ad -- --check` passed.
- `git diff --check` passed.
- RCH `cargo test -j 1 -p fj-ad grad_sum_x2_plus_x_1k_golden_sha256 -- --nocapture` passed on `vmi1227854`.
- RCH `cargo test -j 1 -p fj-ad dense_f64_square_plus_linear_reducesum_grad_matches_generic_bits -- --nocapture` passed on `vmi1227854`.
- RCH `cargo check -j 1 -p fj-ad --all-targets` passed on `vmi1152480`.
- RCH `cargo clippy -j 1 -p fj-ad --all-targets --no-deps -- -D warnings` passed on `vmi1152480`.
- Known background warning observed during dependency compilation: pre-existing `fj-trace` unused variable `num_spatial`.
- UBS on touched files exited 1 because of broad pre-existing `fj-ad` inventory: unwrap/expect in tests and helpers, panic macros in tests, direct-indexing heuristics, clone/allocation inventories, and existing format/allocation warnings. No new actionable defect was found in the direct `TensorValue` construction hunk.
