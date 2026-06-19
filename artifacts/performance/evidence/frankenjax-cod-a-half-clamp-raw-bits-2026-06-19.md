# frankenjax-mcqr.108 - Raw Half-Bits Clamp

Date: 2026-06-19
Agent: cod-a / WildForge
Decision: keep

## Lever

Replaced same-half BF16/F16 dense clamp loops that called `clamp_literal` per
lane with a raw `u16` two-pass composition of the existing bit-proven half
Max/Min kernels:

- `tmp = max(lo, x)` with JAX operand order preserving `x` on signed-zero ties.
- `out = min(hi, tmp)` with reversed operand order so `clamp` keeps `tmp` on
  signed-zero ties.
- Mixed scalar dtypes still fall back to the old `clamp_literal` path.

## Commands

- `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo check -p fj-lax --bench clamp_gauntlet`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo bench -p fj-lax --bench clamp_gauntlet -- 'bf16_mixed_scalar_tensor_1m|f16_mixed_scalar_tensor_1m|bf16_tensor_tensor_tensor_1m|f16_tensor_tensor_tensor_1m' --sample-size 20 --warm-up-time 1 --measurement-time 3 --save-baseline frankenjax-mcqr-108-before`
- `RCH_WORKER=ovh-a CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo bench -p fj-lax --bench clamp_gauntlet -- 'bf16_mixed_scalar_tensor_1m|f16_mixed_scalar_tensor_1m|bf16_tensor_tensor_tensor_1m|f16_tensor_tensor_tensor_1m' --sample-size 20 --warm-up-time 1 --measurement-time 3 --save-baseline frankenjax-mcqr-108-after`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a cargo bench -p fj-lax --bench clamp_gauntlet -- 'bf16_mixed_scalar_tensor_1m|f16_mixed_scalar_tensor_1m|bf16_tensor_tensor_tensor_1m|f16_tensor_tensor_tensor_1m' --sample-size 20 --warm-up-time 1 --measurement-time 3 --save-baseline frankenjax-mcqr-108-local-after`
- `benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/clamp_gauntlet.py --runs 50 --warmup 10 --inner-loops 100 --output /tmp/frankenjax_mcqr_108_half_clamp_jax_raw.json`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo test -p fj-lax half_clamp --lib`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo test -p fj-lax clamp --lib`
- `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo clippy -p fj-lax --lib -- -D warnings`
- `cargo fmt --check`

## Results

RCH before/after rows are same-worker on `ovh-a`. Local rows are same-host with
the JAX CPU oracle, matching the scorecard comparison rule.

| Workload | RCH before dense | RCH after dense | RCH speedup | Local after dense | Local boxed ref | Dense/boxed | JAX mean | Local Rust/JAX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bf16_mixed_scalar_tensor_1m` | 21.783 ms | 2.534 ms | 8.60x | 3.616 ms | 45.091 ms | 12.47x faster | 122.705 us | 29.47x slower |
| `f16_mixed_scalar_tensor_1m` | 28.986 ms | 3.448 ms | 8.41x | 3.521 ms | 37.471 ms | 10.64x faster | 319.088 us | 11.03x slower |
| `bf16_tensor_tensor_tensor_1m` | 22.412 ms | 2.934 ms | 7.64x | 2.993 ms | 32.126 ms | 10.73x faster | 148.870 us | 20.10x slower |
| `f16_tensor_tensor_tensor_1m` | 29.748 ms | 4.038 ms | 7.37x | 3.653 ms | 32.192 ms | 8.81x faster | 196.938 us | 18.55x slower |

## Verification

- `cargo check -p fj-lax --bench clamp_gauntlet` passed on rch.
- Focused tests passed: 3 passed, 0 failed, 2 ignored benchmark tests for
  `cargo test -p fj-lax half_clamp --lib`.
- Broader clamp tests passed: 26 passed, 0 failed, 6 ignored benchmark tests for
  `cargo test -p fj-lax clamp --lib`.
- New signed-zero regression covers BF16/F16 scalar-bound and tensor-bound
  forms where `clamp` must keep `x`'s `-0.0` when `hi` is `+0.0`.
- `cargo clippy -p fj-lax --lib -- -D warnings` is blocked by pre-existing
  unrelated findings in `tensor_ops.rs` and `tree_util.rs`; the new
  `arithmetic.rs` change no longer reports a clippy finding.
- `cargo fmt --check` is blocked by pre-existing formatting drift across
  unrelated workspace files.

## Follow-Up

Keep the lever, but do not claim JAX domination. The next half-clamp frontier is
no longer boxed literal overhead; it is JAX's fused/vectorized CPU kernel floor.
Further attempts should attack allocation and the two-pass temp vector directly,
or use a one-pass half clamp kernel with the same signed-zero and NaN proof.
