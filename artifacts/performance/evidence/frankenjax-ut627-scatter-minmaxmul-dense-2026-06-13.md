# frankenjax-ut627: dense scatter Mul/Min/Max combiners

Date: 2026-06-13
Owner: BlackThrush
Worker: ovh-a (rch)
Crate: fj-lax (tensor_ops.rs)

## Lever

`eval_scatter_dense` only accelerated `Overwrite` + `Add` (the embedding
lookup/gradient cases); `Mul`/`Min`/`Max` returned `None` and fell to the generic
per-`Literal` loop (`scatter_combine_literal` → `binary_literal_op`) for ALL
dtypes. Those combiners back `jax.lax.scatter_mul/min/max`, `segment_max/min/prod`,
and max-pool gradients.

Generalize the dense typed path (f64/f32/i64/bf16/f16): the `scatter_typed!` macro
now takes a combine closure (Overwrite = contiguous copy; the four reduce
combiners apply it per element). Each dtype's closure matches
`scatter_combine_literal` exactly:
- f64: `+` / `*` / `jax_min_f64` / `jax_max_f64`.
- f32: promote→f64, op, round to f32 (same as `binary_literal_op`'s f32 branch).
- i64: `wrapping_add` / `wrapping_mul` / `i64::min` / `i64::max`.
- bf16/f16: route through `binary_literal_op` with the matching primitive (widen
  u16→f64, op, round to half).

## Parity (bit-identical)

The dense-vs-generic parity sweeps (`dense_scatter_matches_literal_path`,
`dense_half_float_scatter_matches_literal_path`, and the F32 sweep) now iterate
`overwrite/add/mul/min/max × fill_or_drop/clip` and assert the dense output bits
equal the boxed-`Literal` generic path's. 26 scatter lib tests pass.

## Result (same-invocation A/B, dense vs generic per-Literal)

```text
rch exec -- cargo test -j 1 -p fj-lax --lib bench_f32_scatter_max_dense_vs_generic --release -- --ignored --nocapture

BENCH f32 scatter-max [50000,256] x 8192: generic(per-Literal)=266.4522ms dense=29.0176ms speedup=9.18x
```

(f32 `TensorValue::new` with `F32Bits` literals stays boxed → genuine generic
path; the analogous f32 scatter-ADD bench is 7.18x.) Keep: **9.18x**.
Score: 9.18 × 0.95 / 1 = 8.7.
