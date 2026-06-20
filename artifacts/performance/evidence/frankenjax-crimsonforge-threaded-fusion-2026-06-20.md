# Threaded elementwise-fusion chunk driver (eager eval_jaxpr) — KEPT

- Date: 2026-06-20
- Agent: CrimsonForge (claude-code, opus-4-8)
- Crate: `fj-interpreters`
- Commits: `c5ce4988` (driver), `fe0985bc` (A/B knob), `755e6b9c` (gate raise to 1Mi)
- Target gap: the eager `eval_jaxpr` elementwise-FUSION path
  (`try_fuse_elementwise_chain_{f64,f32,i64,half}`) drove its `FUSION_CHUNK`
  chunks on ONE core. Measured a deep 1M-elem 8-op f64 chain at ~6.3 GB/s
  (3.794 ms) — far below DRAM saturation — vs JAX/XLA's all-core ~0.172 ms.

## Lever

The chunk loop in each dtype builder writes disjoint `values[s..e]` and reads
`tape`/`ext` read-only, with the global `base` passed through for broadcast
row/col indexing. New `drive_fusion_chunks` fans the chunks out across a
`std::thread::scope` over contiguous, `FUSION_CHUNK`-aligned segments. Every
element is still produced by the SAME ordered scalar-op sequence — only which
worker owns which contiguous run changes — so the result is **bit-identical**
(no reassociation, no cross-chunk reduction). Work-scaled worker count
(`FUSION_ELEMS_PER_THREAD = 256Ki`, clamped to hardware threads), gated at
`FUSION_THREAD_MIN_ELEMS = 1Mi`.

`forbid(unsafe_code)` kept: `vec![0.0; n]` stays (it is `alloc_zeroed`/calloc =
lazy zero pages, faulted in by the per-chunk seed, so not a real serial cost).

## Same-invocation A/B (the only worker-variance-immune signal)

`FUSION_THREAD_CAP_OVERRIDE` (bench-only, default 0) caps the worker count so
`run_f64_thread_ab` times SERIAL (cap=1) vs THREADED over identical data in ONE
process. Absolute ms drifts 4-10x across separate rch invocations (the unfused
control jumped 18.8 ms -> 71.3 ms on a contended worker between two runs), so
cross-run comparison is meaningless; the in-process ratio is not.

| n | serial | threaded | speedup | verdict |
| --- | ---: | ---: | ---: | --- |
| 1M (L3-resident) | 12.157 ms | 9.937 ms | 1.22x | KEEP |
| 4M | 75.841 ms | 59.493 ms | 1.27x | KEEP |
| 16M (DRAM-bound) | 305.924 ms | 231.395 ms | 1.32x | KEEP |

3 wins / 0 losses / 0 neutral. The win grows with n. Modest scaling (not Nx on
N cores) is Amdahl: the threaded fusion is only part of `eval_jaxpr`'s per-call
cost (arg clone, `TensorValue` construction, liveness). The deeper interpreter
overhead and the per-core ~6 GB/s inner loop (no SIMD single-pass register
reuse) remain the next, larger lever toward XLA parity — separate work.

## Honest framing vs JAX

This NARROWS but does not flip the large-fused-chain JAX loss: even threaded, the
interpreter path is far from XLA's fused all-core SIMD kernel. It is a real,
bit-exact, downside-free internal speedup on the fastest existing chain path
(cod-a's ledger notes the eager fusion path already beats the compiled runner).

## Validation

- `cargo test -p fj-interpreters --lib`: 202 passed, 8 failed. The 8 failures are
  PRE-EXISTING golden-hash serialization drift (reshape/transpose/broadcast/scan/
  partial_eval/staging) — confirmed identical on baseline `aff2ee5d` in a clean
  worktree (`0fd7ec…` vs committed `ebe11f…`), unrelated to this change. All 12
  fusion tests + the 2 new threaded-path bit-exact guards pass.
- `cargo clippy -p fj-interpreters --lib --no-deps -- -D warnings`: clean.
- rustfmt: additions clean (file has pre-existing full-file drift, not touched).
- Bit-exactness: `fusion_threaded_f64_chain_matches_reference_bit_for_bit` and
  `fusion_threaded_f64_row_broadcast_matches_reference_bit_for_bit` (cols=257
  splitting rows across thread/chunk seams) assert serial==threaded==reference.
- Target dir: `/data/projects/.rch-targets/frankenjax-cc`; JAX 0.10.1 x64 CPU.

## CORRECTION (same day): contention re-measurement → gate raised to 8.4Mi

The A/B table above was on idle workers. A run on a CONTENDED shared rch worker
showed the SAME code regressing at L3-resident sizes (f64 1M 0.42x, f32 1M 0.79x,
i64 1M 0.42x) while >= 16M stayed positive (f64 1.21x, f32 1.33x, i64 1.07x).
Same-invocation A/B is NOT contention-immune for a THREADING lever (the threaded
arm oversubscribes under load while serial is robust; the ratio swings with load).

FINAL: `FUSION_THREAD_MIN_ELEMS` raised 1Mi -> 8.4Mi (commit c23c5a0c), matching the
established single-op gate. The lever is KEPT but narrowed to >= 8.4M-element
f64/f32/i64 elementwise chains (robust 1.07-1.33x across idle + contended); half
stays serial (measured neutral 0.86-1.01x). Below 8.4M = serial, no regression.
Per-dtype A/B harnesses: `run_{f64,f32,i64,bf16}_thread_ab` in eval_fusion_speed.

## Same-host head-to-head vs JAX/XLA (gauntlet completion)

Both Rust (local release bench, NOT rch) and JAX 0.10.1 x64 run on the SAME host,
so the ratio is honest (no rch-worker ~1.45x penalty). 8-op cheap chain.

| workload | Rust serial | Rust threaded | JAX (XLA) | threaded vs JAX |
| --- | ---: | ---: | ---: | --- |
| f64 16M | 268.4 ms | 194.5 ms | 25.1 ms | **JAX 7.75x faster** |
| f32 16M | 134.5 ms | 103.6 ms | 13.6 ms | **JAX 7.6x faster** |

VERDICT: a LARGE remaining JAX loss. Threading recovered ~1.3-1.4x (f64 16M loss
10.7x -> 7.75x) but XLA still wins ~7-8x. Root cause is NOT parallelism anymore —
it is per-element interpreter inefficiency: (1) step-major chunk apply reloads the
64KB chunk from L2 once PER op (8 passes) instead of XLA's single load-compute-store
in registers; (2) per-call eval_jaxpr overhead (arg clone, TensorValue construction,
liveness); vs XLA's one fused all-core SIMD kernel. The next levers (SIMD single-pass
register-resident chunk eval; cut per-call construction) are larger and partly
blocked: single-pass register fusion was REJECTED in the compiled path (stash@{2},
f32 1.74x/i64 1.34x REGRESS) and SIMD-poly needs the policy-gated +fma flag. The 1M
ratios were local-contention-noisy (concurrent processes) and are omitted.
