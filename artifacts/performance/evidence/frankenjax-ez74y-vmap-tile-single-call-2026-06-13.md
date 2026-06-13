# frankenjax-ez74y: single-call vmap rule for Tile

Date: 2026-06-13
Owner: BlackThrush
Worker: ovh-a (rch)
Crate: fj-dispatch (batching.rs)

## Lever

`apply_batch_rule` routed `Tile` through `batch_passthrough_leading` — per-slice
eval+stack (B dispatches + B tiled-result allocs + a `stack_axis0`). `tile`
applies the same `"reps"` (one positive rep per axis, output keeps rank) to every
batch slice, so vmap = tile the batch-front tensor with a **leading rep of 1**
for the batch axis, in ONE `eval_primitive` call.

`batch_tile`: move batch dim to front, prepend `"1,"` to the `"reps"` param, eval
once. The batch axis (rep 1) passes through untouched; every per-element axis
keeps its original rep, so the result `[B, tiled…]` equals the stack of per-slice
tiles (tile is a deterministic block copy). The rank-0 per-element case (scalar
tile, which changes rank 0→1 and has no batch-front-prepend equivalent) defers to
the per-slice path.

## Parity

`batch_tile_matches_per_slice_fallback` asserts the single-call output
(batch_dim + shape + f64 values) is element-identical to
`batch_passthrough_leading` across reps `1,1 / 2,1 / 1,3 / 2,3`, per-element rank
2, and batch_dim at 0 and 1.

## Result (same-invocation A/B, single-call vs per-slice)

```text
rch exec -- cargo test -j 1 -p fj-dispatch --lib bench_batch_tile_single_call_vs_per_slice --release -- --ignored --nocapture

BENCH vmap(tile) [262144,4] reps=8: per-slice=181.8122ms single-call=57.0763ms speedup=3.19x
```

The realistic broadcast-via-tile pattern (large batch of small per-element
tensors) is dominated by the B dispatches + B allocs the single call eliminates.
Keep: **3.19x**. Score: 3.19 × 0.95 / 1 = 3.03.

Behavior proof: 293 fj-dispatch lib tests pass (incl. the new parity test); the
bench also asserts identical output element count before timing.
