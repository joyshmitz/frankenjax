# frankenjax-mv6p6: separable monotonic-deque i64 max/min reduce_window

Date: 2026-06-13
Owner: BlackThrush
Worker: ovh-a (rch)
Crate: fj-lax (lib.rs)

## Lever

The window-independent monotonic-deque separable max/min path
(`reduce_window_separable_maxmin`, O(input+output) per axis) was gated
float-only (F64/F32). Integer (i64) large-window max/min pooling fell to the
per-window dense path: O(output·∏window). Add `reduce_window_separable_maxmin_i64`
— the integer sibling, reducing each window axis with an independent
monotonic-deque 1D pass. Simpler than the float version: i64 max/min is a total
order, so no NaN tracking or signed-zero handling is needed. Gated
`i64 && max/min && ∏window > 2·∑window` (same threshold as float; small windows
stay on the per-window dense path).

## Parity (bit-exact)

i64 max/min is a unique total order, so the window extremum VALUE is unique and
the deque result equals the ascending per-window `i64::max`/`i64::min` fold
regardless of order; OOB taps contribute the init (`i64::MIN`/`i64::MAX`), a
no-op. `reduce_window_i64_maxmin_separable_matches_per_window` asserts equality
vs an independent per-window reference for BOTH max and min across rank-1
(VALID/SAME, stride 1/3), rank-2 (`7×5` s2 VALID, `6×6` SAME), and rank-3
(`4×4×1`).

## Result (same-invocation A/B, separable vs per-window dense)

```text
rch exec -- cargo test -j 1 -p fj-lax --lib bench_reduce_window_i64_maxmin_separable_vs_dense --release -- --ignored --nocapture

BENCH reduce_window i64 max([512,512],win=15x15,s=1): per-window=70.5180ms separable=8.3946ms speedup=8.40x
```

Keep: **8.40x** at 15×15, and the ratio grows with window (separable is
window-independent; per-window is O(∏window)). Score: 8.40 × 0.95 / 1 = 7.98.

Behavior proof: 40 reduce_window lib tests pass (incl. the new parity test); the
bench asserts identical checksum vs the per-window reference.

## reduce_window family status

Integer windowed reductions are now all window-independent for large windows:
sum via summed-area table (wjty3 rank-2, rmsv6 rank 1/3-6), max/min via
monotonic-deque (this, mirroring the existing float deque).
