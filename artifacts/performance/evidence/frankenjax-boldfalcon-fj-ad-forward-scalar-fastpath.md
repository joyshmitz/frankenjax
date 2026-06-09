# fj-ad Forward Tape Scalar F64 Fast Path

Pass: BoldFalcon profile-backed optimization pass 3
Commit base: 0161f9a9
Target: `value_and_grad_runtime/shared/deep_100_nodes`

## Baseline

Command:

```bash
rch exec -- cargo bench -p fj-api --bench api_overhead -- value_and_grad_runtime/shared/deep_100_nodes --noplot
```

Worktree: `/data/projects/.scratch/frankenjax-boldfalcon-scalar-push-20260609`
Execution: RCH local fail-open

Result:

```text
time: [78.099 us 79.513 us 81.113 us]
```

## Candidate

Command:

```bash
rch exec -- cargo bench -p fj-api --bench api_overhead -- value_and_grad_runtime/shared/deep_100_nodes --noplot
```

Worktree: `/data/projects/.scratch/frankenjax-boldfalcon-scalar-push-20260609`
Execution: RCH local fail-open

Result:

```text
time: [72.004 us 72.845 us 73.786 us]
```

Median speedup: `79.513 / 72.845 = 1.092x`.

Score: `Impact 1.092 * Confidence 0.95 / Effort 0.50 = 2.07`.

## Lever

`forward_with_tape` now bypasses `fj-lax::eval_primitive_multi` for
parameter-free scalar F64 Add/Mul equations. It returns the exact same single
scalar output and keeps all other primitives, arities, dtypes, tensors, and
parameterized equations on the existing evaluator path.

## Isomorphism Proof

- Ordering: unchanged. Each matched equation still executes exactly one scalar
  Add or Mul at the same forward-tape position.
- Tape shape: unchanged for Add/Mul. These primitives already omit
  `output_values` from the VJP tape via `single_output_vjp_ignores_outputs`.
- Tie-breaking: not applicable.
- Floating-point: the fast path computes `f64::from_bits(lhs) op
  f64::from_bits(rhs)` and re-wraps with `Literal::from_f64`, matching the
  generic evaluator's scalar F64 path bit-for-bit.
- RNG: no RNG surface.
- Golden SHA: `grad_sum_x2_plus_x_1k_golden_sha256 =
  5282853e2bd187c1c1bfdfa612bd74776fb403e6b767eb0a8bf0c8bcd2fe2a19`.

## Validation

```bash
rch exec -- cargo test -p fj-ad scalar_f64_forward_tape_matches_eval_primitive_bits -- --nocapture
cargo fmt -p fj-ad -- --check
git diff --check -- crates/fj-ad/src/lib.rs artifacts/performance/evidence/frankenjax-boldfalcon-fj-ad-forward-scalar-fastpath.md
rch exec -- cargo test -p fj-ad scalar_f64_ad_arithmetic_matches_eval_primitive_bits -- --nocapture
rch exec -- cargo test -p fj-ad grad_sum_x2_plus_x_1k_golden_sha256 -- --nocapture
rch exec -- cargo check -p fj-ad --all-targets
rch exec -- cargo test -p fj-ad --lib
rch exec -- cargo clippy -p fj-ad --all-targets -- -D warnings -A unused-variables -A clippy::too_many_arguments -A clippy::manual_is_multiple_of -A clippy::needless_range_loop -A clippy::useless_vec
ubs crates/fj-ad/src/lib.rs artifacts/performance/evidence/frankenjax-boldfalcon-fj-ad-forward-scalar-fastpath.md
```

Focused forward-tape bit/tape-shape test passed. The scalar arithmetic
isomorphism test passed. The 1k gradient golden SHA test passed. `cargo check
-p fj-ad --all-targets`, `cargo test -p fj-ad --lib` (362 tests), fmt, diff
check, and adjusted strict clippy all passed. RCH executed these locally when
the worker pool reported critical pressure.

UBS exits 1 on the longstanding `fj-ad/src/lib.rs` panic/unwrap/direct-indexing
inventory. Its lint/style and build-health sections are clean, dependency
checks are clean, and the new scalar fast path no longer contributes the
checked-indexing sample after rewriting it to a slice-pattern match.
