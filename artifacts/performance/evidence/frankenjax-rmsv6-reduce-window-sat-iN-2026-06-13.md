# frankenjax-rmsv6: general-rank summed-area-table i64 sum reduce_window

Date: 2026-06-13
Owner: BlackThrush
Worker: ovh-a (rch)
Crate: fj-lax (lib.rs)

## Lever

Generalizes the rank-2 SAT (frankenjax-wjty3) to arbitrary rank. Build an N-dim
integral image once (prefix-sum along every axis), then each output window sum is
a `2^rank`-corner inclusion-exclusion: O(rank·input + 2^rank·output),
**window-independent**. Covers the ranks the rank-2 path doesn't:

- rank 1 — sliding-window integer sums (box filter / moving sum over a sequence),
- rank 3/4 — NHWC integer sum-pooling (spatial window, unit window on N/C).

`eval_reduce_window_iN_sum_sat`: scatter input into the +1-shifted interior of a
`∏(n_d+1)` padded array, prefix-sum along each axis in-place (ascending flat
order ⇒ the axis-`d` predecessor is already accumulated this pass), then per
output sum `Σ_corner (−1)^(#low picks) sat[corner]` over the clamped in-bounds
box. Gated `rank ∈ {1}∪{3..6}` (rank 2 keeps its specialized path), `∏window ≥ 16`,
i64, sum, no dilation. `rank ≤ 6` bounds the corner loop (≤ 64).

## Parity (bit-exact for integers)

i64 `wrapping_add` is associative+commutative in ℤ/2⁶⁴ and `wrapping_sub` inverts
it, so the box sum equals the per-window ascending wrapping tap sum bit-for-bit
even under overflow; OOB/pad taps contribute 0 in both (= the clamped in-bounds
box). `reduce_window_iN_sum_sat_matches_generic` asserts equality vs an
independent per-window reference for rank-1 (win 16/20, VALID/SAME, stride 1/3),
rank-3 (`9×9×3`, win `4×4×1`), and rank-4 (`2×8×8×3`, win `1×4×4×1`, stride
`1×2×2×1`).

## Result (same-invocation A/B, SAT vs per-window dense)

```text
rch exec -- cargo test -j 1 -p fj-lax --lib bench_reduce_window_iN_sum_sat_vs_dense --release -- --ignored --nocapture

BENCH reduce_window i64 sum 1-D([1048576],win=4096,s=1): per-window=181.3649ms SAT=19.7881ms speedup=9.17x
```

Keep: **9.17x** at win=4096 (1-D), and the ratio grows with window (SAT is
window-independent; per-window is O(∏window)). Score: 9.17 × 0.95 / 1 = 8.7.

Behavior proof: 39 reduce_window lib tests pass (incl. the new general-rank parity
test); the bench asserts identical checksum vs the per-window reference.
