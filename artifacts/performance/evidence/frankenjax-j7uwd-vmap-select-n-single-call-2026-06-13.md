# frankenjax-j7uwd: single-call vmap rule for SelectN

Date: 2026-06-13
Owner: BlackThrush
Worker: ovh-a (rch)
Crate: fj-dispatch (batching.rs)

## Lever

`apply_batch_rule` routed `SelectN` (`select_n(index, case0, case1, …)`, an
elementwise pick-by-index among same-shape cases) through
`batch_passthrough_leading` — per-slice eval+stack, and each slice additionally
slices all N case operands. `batch_select_n` harmonizes the index + every case to
a common batch-front shape (move-to-front / broadcast-unbatched) and evals ONCE.

SAFETY GATE: the fast path is taken only when the harmonized index shape equals
the cases' shape — the true elementwise contract. The scalar-index-per-slice form
(a rank-0 index that picks a whole case; after vmap the index is `[B]` while cases
are `[B,…]`) does not satisfy that and falls back to the correct per-slice path.

## Parity

`batch_select_n_matches_per_slice_fallback` asserts the single-call output
(batch_dim + shape + f64 bits) equals `batch_passthrough_leading` for (1) an
elementwise batched index+cases and (2) a shared unbatched index broadcast across
the batch.

## Result (same-invocation A/B, single-call vs per-slice)

```text
rch exec -- cargo test -j 1 -p fj-dispatch --lib bench_batch_select_n_single_call_vs_per_slice --release -- --ignored --nocapture

BENCH vmap(select_n) [262144,4] 3 cases: per-slice=208.8375ms single-call=19.6368ms speedup=10.64x
```

The per-slice path pays B dispatches AND slices all 3 case operands per element
(3B+ slices + result/stack allocs); the single call is one elementwise pass.
Keep: **10.64x**. Score: 10.64 × 0.95 / 1 = 10.1.

Behavior proof: 295 fj-dispatch lib tests pass (incl. the new parity test); the
bench asserts identical output element count before timing.
