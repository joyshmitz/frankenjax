# frankenjax-5l5x4: native-f32 conv2d accumulation

Date: 2026-06-11
Bead: frankenjax-5l5x4
Crate: fj-lax

## Target

Profile-backed follow-up to `frankenjax-cz0g0`: dense f32 matmul now matches the XLA/JAX policy by accumulating f32 inputs in f32 with safe Rust portable SIMD. f32 conv2d still widened f32 operands to f64 for the im2col GEMM path, making the default ML conv path slower and more precise than the upstream parity target.

Graveyard mapping: vectorized execution / cache-sized typed kernels from `alien_cs_graveyard.md` section 8.2, compiled as one safe-Rust native-f32 conv primitive using the existing `batched_matmul_2d_f32_in` microkernel.

## Baseline

Command:

```bash
rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-5l5x4-baseline-target cargo test -j 1 -p fj-lax --release bench_f32_conv2d -- --ignored --nocapture
```

Worker: `vmi1227854`

Rows:

| Case | Baseline |
| --- | ---: |
| `[8,32,32,16] * [3,3,16,32]` | 3.5237 ms |
| `[4,28,28,32] * [3,3,32,64]` | 3.0493 ms |

## Lever

One production lever:

- `eval_conv_2d` recognizes `DType::F32` output with f32 lhs/rhs operands.
- Large convs build f32 im2col and call `batched_matmul_2d_f32_in`.
- Small convs use the same ascending `(kh, kw, ci)` direct f32 fold.
- Dense f32 buffers are borrowed when present; boxed `Literal::F32Bits` inputs unpack once.
- Grouped/depthwise, non-f32, complex, malformed, and unsupported parameter paths fall back unchanged.

## Rebench

Command:

```bash
rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-5l5x4-proofbench-target cargo test -j 1 -p fj-lax --release conv2d -- --include-ignored --nocapture
```

Worker: `vmi1227854`

Rows:

| Case | Baseline | Candidate | Speedup |
| --- | ---: | ---: | ---: |
| `[8,32,32,16] * [3,3,16,32]` | 3.5237 ms | 2.0783 ms | 1.70x |
| `[4,28,28,32] * [3,3,32,64]` | 3.0493 ms | 2.4600 ms | 1.24x |

Score: Impact 3.0 x Confidence 5.0 / Effort 2.0 = 7.5.

## Isomorphism Proof

Ordering: output remains row-major `[batch, out_h, out_w, c_out]`. Large im2col rows are generated in the same `(n, oh, ow, kh, kw, ci)` order as the f64 path; small conv keeps the existing loop order.

Tie-breaking: not applicable.

Floating point: this intentionally changes f32 conv accumulation from f64-promote to native f32, matching the accepted `cz0g0` XLA-parity policy for f32 matmul. Within the native-f32 path, each output element accumulates products in ascending `(kh, kw, ci)` order. The large path uses SIMD lanes across output columns, not within one dot product.

RNG: not applicable.

Fallback surface: grouped/depthwise, non-f32, complex, non-tensor, shape mismatch, unsupported params, and overflow errors are unchanged.

Golden output digest:

```text
4642006de6ba3f3a608d30fb5a7904647f37a9a8d0277894fb7c45b1c8491490
```

Proof tests from the rebench command:

- `conv2d_f32_native_accum_golden_sha256`
- `conv2d_f32_im2col_gemm_bit_identical_to_reference`
- `f32_conv2d_emits_dense_f32_storage`
- existing conv2d grouped/depthwise/rhs-dilation/complex/half tests

Result: 11 conv2d tests passed.

Current-HEAD proof after the later conv1d and half-conv follow-ups:

```bash
rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-5l5x4-current-proof-target cargo test -j 1 -p fj-lax --release conv2d_f32 -- --nocapture
```

Worker: `vmi1227854`

Result: `2 passed; 0 failed`:

- `conv2d_f32_im2col_gemm_bit_identical_to_reference`
- `conv2d_f32_native_accum_golden_sha256`

## Validation

Passing:

- `git diff --check`
- `rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-5l5x4-check-target cargo check -j 1 -p fj-lax --all-targets`
- `ubs crates/fj-lax/src/tensor_ops.rs crates/fj-lax/benches/lax_baseline.rs` embedded fmt/clippy/check/test-build/audit/deny subchecks

Caveats:

- Direct `rustfmt --edition 2024 --check crates/fj-lax/src/tensor_ops.rs crates/fj-lax/benches/lax_baseline.rs` still reports pre-existing formatting drift elsewhere in these large files.
- UBS exits nonzero on broad pre-existing panic/direct-index/security-pattern inventory in the touched files; its build and lint subchecks were clean.
- Direct strict `rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-5l5x4-clippy-target cargo clippy -j 1 -p fj-lax --all-targets -- -D warnings` still fails on existing fj-lax lint debt outside the f32 conv2d lever (`too_many_arguments` in arithmetic/linalg helpers, `manual_option_zip` in reductions, simd-exp doc/precision lints, threefry range-loop lints, and a conv1d follow-up `manual_checked_ops` lint).
