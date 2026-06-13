# frankenjax-mqueo: single-call vmap rule for TopK

Date: 2026-06-13
Owner: BlackThrush
Worker: ovh-a (rch)
Crate: fj-dispatch (batching.rs)

## Lever

`apply_batch_rule_multi` routed `TopK` (multi-output: values + indices) through
`batch_passthrough_leading_multi` — B per-slice top_k evals + a stack of BOTH
outputs. `top_k` always operates on the operand's LAST axis and treats every
leading dim as an independent slice, so a vmap batch axis is just another leading
slice dim. `batch_top_k_multi` moves the batch dim to front and evals ONCE on
`[B, …, N]` — the eval's threaded radix multi-slice path handles all `B·…` slices
in one call. Non-tensor / rank-<2 inputs defer to the per-slice multi rule.

## Parity

The batch axis is prepended (top_k axis stays last) and top_k is deterministic
per slice, so both outputs `[B, …, k]` equal the per-slice stack; `k` passes
through. `batch_top_k_multi_matches_per_slice_fallback` asserts identical
batch_dim+shape+f64 bits for BOTH outputs vs `batch_passthrough_leading_multi`
across k=1/2/3, ranks `[B,5]` and `[B,3,4]`, and batch_dim 0 and 1.

## Result (same-invocation A/B, single-call vs per-slice)

```text
rch exec -- cargo test -j 1 -p fj-dispatch --lib bench_batch_top_k_single_call_vs_per_slice --release -- --ignored --nocapture

BENCH vmap(top_k) [131072,16] k=4: per-slice=339.9006ms single-call=62.3127ms speedup=5.45x
```

Keep: **5.45x**. Score: 5.45 × 0.95 / 1 = 5.18.

Behavior proof: 297 fj-dispatch lib tests pass (incl. the new parity test); the
bench asserts identical total output element count before timing.
