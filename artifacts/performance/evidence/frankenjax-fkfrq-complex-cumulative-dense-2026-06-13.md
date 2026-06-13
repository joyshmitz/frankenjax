# frankenjax-fkfrq: dense complex cumulative (cumsum/cumprod/cummax/cummin)

Date: 2026-06-13
Owner: BlackThrush
Worker: ovh-a (rch)
Crate: fj-lax (reduction.rs)

## Lever

Complex (Complex64/Complex128) cumulative ops ran the generic per-`Literal` scan;
`eval_cumulative_dense` covered f64/f32/i64/bf16/f16 but returned `None` for
complex. Add a dense complex branch: thread the primitive into the dense fn, read
the contiguous `(re, im)` backing (`as_complex_slice`), and scan each line in
place.

## Parity (bit-identical)

Mirrors the generic complex path exactly: same seeds (cumprod=(1,0),
cummax/cummin=(`float_init`,`float_init`), cumsum=(0,0)), same complex-multiply,
the same lexicographic `complex_lex_cmp` for max/min, the same component-add for
sum — in the same per-line order. `new_complex_values` rounds Complex64 to f32
exactly as `complex_literal_from_parts` does, so the f32-rounded-per-step Complex64
output matches; Complex128 stays full f64. Reading `out[fi]` yields the original
input (each position is read once before its own write).

`complex_cumulative_dense_matches_boxed` asserts the dense (Complex storage)
output's `(re,im)` bits equal the boxed-`Literal` construction for
cumsum/cumprod/cummax/cummin × forward/reverse. 41 cum lib tests pass.

## Result (same-invocation A/B, dense vs generic per-Literal)

```text
rch exec -- cargo test -j 1 -p fj-lax --lib bench_complex_cumsum_dense_vs_generic --release -- --ignored --nocapture

BENCH complex128 cumsum axis1 [4096,1024]: generic=93.3678ms dense=39.8428ms speedup=2.34x
```

Keep: **2.34x**. Score: 2.34 × 0.95 / 1 = 2.2.

## cumulative dtype family — now COMPLETE

f64/f32/i64 (+ threaded multi-line), bf16/f16, and now complex all have dense
fast paths; integer/real small cases and the generic path remain as fallbacks.
