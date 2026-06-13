# frankenjax-zm4p9: dense bool reduce_window fast path (logical pooling)

Date: 2026-06-13
Owner: BlackThrush
Worker: ovh-a (rch)
Crate: fj-lax

## Lever

Completes the reduce_window dense-path family. Float (any-rank) and i64
(frankenjax-01x1q) had dense fast paths; **bool input still ran the generic
per-`Literal` gather + string-dispatched `reduce_window_accumulate_literal`
loop**. Boolean windowed reductions are real (binary morphology
dilation/erosion, validity-mask any/all pooling).

`eval_reduce_window_dense_bool` mirrors `eval_reduce_window_dense_i64`: read the
dense `bool` slice (`as_bool_slice`), hoist the op once over the same row-major
tap-offset stencil with the same interior/border split. `min` => logical AND
(init `true`), every other op (`max`/`sum`) => logical OR (init `false`) —
exactly the Bool arm of `reduce_window_accumulate_literal`. Wired after the i64
block, gated `no_base_dilation && no_window_dilation && dtype==Bool &&
(max|min|sum)`. Bit-packed `BoolWords` storage exposes no `bool` slice → stays
on the generic path.

## Baseline + Result (same worker, same invocation A/B)

```text
rch exec -- cargo test -j 1 -p fj-lax --lib bench_reduce_window_dense_bool_vs_generic --release -- --ignored --nocapture

BENCH reduce_window bool or([512,512],win=3x3,stride=1): generic=14.8310ms dense=3.2485ms speedup=4.57x
```

Fair A/B: both build an output buffer; the generic reference calls the SAME
`reduce_window_initial_accumulator` / `reduce_window_accumulate_literal` /
`reduce_window_accumulator_literal` helpers the production generic loop uses.
Checksums (count of `true`) match (`assert_eq!(d_gen, d_dense)`).

Keep: **4.57x**. Score: Impact 4.57 x Confidence 0.95 / Effort 1 = 4.3.

## Isomorphism Proof

- Same row-major tap order / stencil construction as the i64 + float paths.
- `min` => `&` (init `true`), `max`/`sum` => `|` (init `false`) — the exact Bool
  arm of `reduce_window_accumulate_literal`.
- OOB/border taps contribute `init`, the identity for each op
  (`acc & true == acc`, `acc | false == acc`) — same as the generic `pad_literal`.
- Output is `Literal::Bool` (`new_bool_values`), matching
  `reduce_window_accumulator_literal` for Bool.
- AND/OR are associative + idempotent → result is order-independent → bit-identical
  regardless of interior/border traversal.

Behavior proof: all 37 `reduce_window` lib tests pass unchanged (incl.
`reduce_window_sum_preserves_bool_literal_dtype_as_or`,
`reduce_window_min_preserves_bool_literal_dtype_as_and`); the new bench's
checksum matches the generic reference byte-for-byte.

```text
rch exec -- cargo test -p fj-lax --lib reduce_window --release
=> test result: ok. 37 passed; 0 failed
```
