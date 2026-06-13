# frankenjax-qb3r6: dense non-contiguous (strided) gather path

Date: 2026-06-13
Owner: BlackThrush
Worker: ovh-a (rch)
Crate: fj-lax (tensor_ops.rs)

## Lever

The gather dense fast path only covered the **contiguous** case
(`trailing_slice_is_contiguous`: full trailing slice = whole rows). The
**non-contiguous** case (partial trailing slice — `slice_sizes[1..] < dims[1..]`,
i.e. partial-column / sub-block gather, common in advanced indexing) ran the
generic per-`Literal` odometer loop for ALL dtypes (`operand.elements.get(flat)`
reconstructs a 24-byte `Literal` per access).

Add `dense_strided_gather!`: the SAME odometer offset math (checked, identical to
the generic loop) but reading typed slices (f64/f32/i64/bf16/f16) into dense typed
output, with OOB fill = `gather_fill_literal` bits. Other dtypes fall to the
generic Literal loop.

## Parity (bit-identical)

Same resolved indices, same per-axis offset math, same row-major slice order,
same OOB fill. `dense_gather_strided_matches_literal_path` asserts the dense
(typed storage) output bits equal the boxed-`Literal` generic path's for
f64/i64/bf16 across clip + fill_or_drop, on a partial-trailing slice
(`slice_sizes=[1,13]` of `[24,40]`). 15 gather lib tests pass.

## Result (same-invocation A/B, dense vs generic per-Literal)

```text
rch exec -- cargo test -j 1 -p fj-lax --lib bench_f32_strided_gather_dense_vs_generic --release -- --ignored --nocapture

BENCH f32 strided gather [50000,256]->[8192,96]: generic=11.1737ms dense=4.4280ms speedup=2.52x
```

(f32 boxed via `TensorValue::new` with `F32Bits` literals stays generic → genuine
comparison.) Keep: **2.52x**. Score: 2.52 × 0.95 / 1 = 2.4.
