# fj-ad grad-only dense reduce fast path

Date: 2026-06-10
Worker: `vmi1227854`
Crate: `fj-ad`
Lever: exact-pattern `grad_jaxpr` fast path for dense F64 `reduce_sum(x * x + x)`

## Profile target

Full `fj-ad` reprofile via RCH/criterion on `vmi1227854` identified:

- `ad/grad_sum_x2_plus_x_1k`: 2.3995 us
- `ad/grad_sin_cos_mul`: 1.0042 us
- `ad/grad_exp_log`: 680.84 ns
- `ad/jvp_sin_cos_mul`: 831.41 ns

The top row computes `grad_jaxpr(sum(x*x + x))` over a dense F64 vector of
length 1024. The pre-existing value-and-grad fast path computed both the scalar
primal output and the gradient; `grad_jaxpr` then discarded the primal output.

## Change

Add `try_dense_f64_square_plus_linear_reducesum_grad`, called only from
`grad_jaxpr`, for the exact same pure Jaxpr shape as the existing
value-and-grad fast path:

1. `Mul(input, input)`.
2. `Add(square, input)`.
3. `ReduceSum(shifted)`.

The path requires:

- one input var, one output var, no const vars, params, effects, or sub-Jaxprs
- dense F64 tensor input
- no custom Add, Mul, ReduceSum, or whole-Jaxpr VJP

The gradient loop preserves the generic reverse-mode element expression order:

```text
cotangent = 1.0
cotangent += x
cotangent += x
```

`value_and_grad_jaxpr` is unchanged and still computes the scalar primal output.

## Benchmark

Baseline command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 RCH_WORKERS=vmi1227854 \
  rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-rch-peachlion-fjad-pass91-profile \
  cargo bench -j 1 -p fj-ad --bench ad_baseline -- \
  --sample-size 20 --measurement-time 2 --warm-up-time 1 --noplot
```

Candidate command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 RCH_WORKERS=vmi1227854 \
  rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-rch-peachlion-fjad-pass91-candidate \
  cargo bench -j 1 -p fj-ad --bench ad_baseline -- \
  'ad/grad_(sum_x2_plus_x_1k|sin_cos_mul|exp_log)' \
  --sample-size 20 --measurement-time 2 --warm-up-time 1 --noplot
```

Results:

| Row | Baseline mean | Candidate mean | Speedup |
| --- | ---: | ---: | ---: |
| `ad/grad_sum_x2_plus_x_1k` | 2.3995 us | 1.2874 us | 1.86x |
| `ad/grad_sin_cos_mul` | 1.0042 us | 939.00 ns | 1.07x |
| `ad/grad_exp_log` | 680.84 ns | 661.84 ns | 1.03x |

Criterion intervals:

- `ad/grad_sum_x2_plus_x_1k`: [1.2430 us, 1.2874 us, 1.3335 us]
- `ad/grad_sin_cos_mul`: [920.94 ns, 939.00 ns, 957.30 ns]
- `ad/grad_exp_log`: [644.89 ns, 661.84 ns, 676.91 ns]

Conservative target-row speedup using baseline mean / candidate high:
2.3995 / 1.3335 = 1.80x.

ICE score: Impact 4 x Confidence 3 / Effort 2 = 6.0.

## Isomorphism proof

- Ordering: the exact Jaxpr pattern preserves input order, output order, and
  row-major tensor order.
- Floating point: gradient per element is still `1.0 + x + x`; the fast path
  does not compute or expose the discarded scalar primal output.
- Tie-breaking: no sort, map, or nondeterministic reduction is introduced.
- RNG: no RNG path is involved.
- Custom rules: custom primitive VJPs and whole-Jaxpr VJPs disable the fast path.
- Fallbacks: non-dense, non-F64, malformed, effectful, param-bearing, or
  sub-Jaxpr-bearing cases route to the existing generic path.

Golden digest remains:

```text
grad_sum_x2_plus_x_1k_golden_sha256 =
5282853e2bd187c1c1bfdfa612bd74776fb403e6b767eb0a8bf0c8bcd2fe2a19
```

Focused bitwise test:

```text
dense_f64_square_plus_linear_reducesum_grad_matches_generic_bits
```

The test compares `grad_jaxpr` against the forced generic route via
`grad_jaxpr_with_custom_vjp_key(..., "force-generic")` on signed zero, finite
values, infinity, and a NaN payload.

## Validation

- `rustfmt --edition 2024 crates/fj-ad/src/lib.rs`
- `rustfmt --edition 2024 --check crates/fj-ad/src/lib.rs`
- `git diff --check`
- RCH `cargo check -j 1 -p fj-ad --all-targets`
- RCH `cargo clippy -j 1 -p fj-ad --all-targets --no-deps -- -D warnings`
- RCH `cargo test -j 1 -p fj-ad --lib grad_sum_x2_plus_x_1k_golden_sha256 -- --nocapture`
- RCH `cargo test -j 1 -p fj-ad --lib dense_f64_square_plus_linear_reducesum -- --nocapture`

UBS command:

```bash
ubs crates/fj-ad/src/lib.rs \
  artifacts/performance/evidence/frankenjax-peachlion-fj-ad-grad-only-dense-reduce-2026-06-10.md
```

UBS exited 1 on the pre-existing whole-file `fj-ad` inventory:

- 21 critical panic-macro findings, all in existing test regions outside this diff
- 3455 unwrap/expect findings, all existing after the new proof test was rewritten to use `Result`
- 1111 direct-indexing findings, with examples before this diff
- UBS subchecks passed: formatting, clippy, cargo check, test build, cargo-audit, cargo-deny
