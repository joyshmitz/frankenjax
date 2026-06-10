# fj-ad vector reduce split-loop pass

## Target

- Directive: `frankenjax-mcqr` no-gaps root.
- Ready perf beads remained occupied or policy-blocked: `frankenjax-lu4yw` was reserved by BoldFalcon, `frankenjax-mcqr.30`/`frankenjax-p1vbf` were assigned to IcyGlacier, and `frankenjax-cz0g0` requires a policy decision.
- Profile-backed lane: fj-ad Criterion `ad_baseline`, current top row after the dense JVP pass.
- Lever: split `try_dense_f64_square_plus_linear_reducesum_value_and_grad` into an output-reduction loop and a gradient-materialization loop. The output loop keeps the same sequential `x*x + x` accumulation order; the gradient loop keeps the same `(1.0 + x) + x` arithmetic per element while removing the scalar reduction dependency from the dense write stream.

## Baseline

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_QUEUE_WHEN_BUSY=1 RCH_PRIORITY=normal RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,RCH_REQUIRE_REMOTE,RCH_QUEUE_WHEN_BUSY,RCH_PRIORITY rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-rch-peachlion-fjad-pass2-baseline cargo bench -j 1 -p fj-ad --bench ad_baseline -- --sample-size 20 --measurement-time 2 --warm-up-time 1 --noplot
```

Worker: `vmi1227854`

| Benchmark | Baseline mean |
| --- | ---: |
| `ad/grad_sum_x2_plus_x_1k` | `3.3379 us` |

Full baseline context from the same run:

| Benchmark | Mean |
| --- | ---: |
| `ad/grad_square` | `106.88 ns` |
| `ad/grad_sum_x2_plus_x_1k` | `3.3379 us` |
| `ad/grad_poly_x3+x2+x` | `134.01 ns` |
| `ad/grad_sin_cos_mul` | `956.20 ns` |
| `ad/grad_exp_log` | `692.60 ns` |
| `ad/value_and_grad_poly` | `125.31 ns` |
| `ad/jvp_square` | `398.50 ns` |
| `ad/jvp_poly_x3+x2+x` | `1.0652 us` |
| `ad/jvp_sin_cos_mul` | `929.25 ns` |

## Candidate

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 RCH_WORKERS=vmi1227854 RCH_QUEUE_WHEN_BUSY=1 RCH_PRIORITY=normal RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,RCH_REQUIRE_REMOTE,RCH_WORKER,RCH_WORKERS,RCH_QUEUE_WHEN_BUSY,RCH_PRIORITY rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-rch-peachlion-fjad-pass2-candidate cargo bench -j 1 -p fj-ad --bench ad_baseline -- 'ad/grad_sum_x2_plus_x_1k' --sample-size 20 --measurement-time 2 --warm-up-time 1 --noplot
```

Worker: `vmi1227854`

| Benchmark | Candidate mean | Speedup |
| --- | ---: | ---: |
| `ad/grad_sum_x2_plus_x_1k` | `2.3974 us` | `1.39x` |

Score: Impact `3` x Confidence `4` / Effort `1` = `12.0`.

## Isomorphism Proof

- Ordering: output reduction order is unchanged: the pass still visits `input` in row-major slice order and applies `output += x*x + x` once per element.
- Tie-breaking: not applicable; no comparator or selection changed.
- Floating point: output arithmetic remains `let squared = x * x; let shifted = squared + x; output += shifted;`. Gradient arithmetic remains `let mut cotangent = 1.0; cotangent += x; cotangent += x;`, so signed zero, NaN payload propagation through these operations, and rounding points are unchanged.
- RNG: not applicable; this AD path has no RNG state.
- Resource envelope: one additional linear pass over the input; allocation count unchanged for the output gradient vector.
- Golden digest: existing `grad_sum_x2_plus_x_1k_golden_sha256` pins `5282853e2bd187c1c1bfdfa612bd74776fb403e6b767eb0a8bf0c8bcd2fe2a19`; existing `dense_f64_square_plus_linear_reducesum_matches_generic_bits` compares the specialized path to the generic tape-backed path on signed zero, infinities, and NaN.

## Validation

- `rustfmt --edition 2024 --check crates/fj-ad/src/lib.rs`
- `git diff --check -- crates/fj-ad/src/lib.rs`
- `RCH ... cargo check -p fj-ad --all-targets` on `vmi1227854`: pass
- `RCH ... cargo clippy -p fj-ad --all-targets --no-deps -- -D warnings` on `vmi1227854`: pass
- `RCH ... cargo test -p fj-ad grad_sum_x2_plus_x_1k_golden_sha256 --lib -- --nocapture` on `vmi1227854`: pass
- `RCH ... cargo test -p fj-ad dense_f64_square_plus_linear_reducesum_matches_generic_bits --lib -- --nocapture` on `vmi1227854`: pass

Known unrelated warnings during remote validation: `fj-trace` unused `num_spatial`; `fj-lax` unused `cell_f64_reference`.
