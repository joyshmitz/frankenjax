# frankenjax-mcqr: dense SelectN wide-storage pass

Date: 2026-06-18
Agent: cod-a / RedBeaver
Status: in_progress, code-first batch-test pending

## Lever

Extend the existing tensor-index `SelectN` dense path beyond f64/f32/i64/i32/half
to cover U32, U64, Bool, Complex64, and Complex128 case tensors.

`SelectN` is a pure per-position copy from one of N same-shaped case tensors.
For dense operands, the selected element can be copied directly from the typed
backing while preserving the existing index validation, out-of-bounds errors,
and output element order. This avoids the generic per-`Literal` case read and
boxed output for unsigned routing tables, nested boolean masks, and complex
switch/case pipelines.

## Alien-source mapping

- `/alien-graveyard`: vectorized execution / flat data layout guidance favors
  dense contiguous buffers and segment-aware operations over boxed element
  walks.
- `/alien-artifact-coding`: preserve the semantic witness: `SelectN` is
  isomorphic to "decode index, copy case[index][i]".
- `/extreme-software-optimization`: one lever, one benchmark target, behavior
  proof before any keep claim.

## Correctness guard

Updated `dense_select_n_matches_literal_path_and_stays_dense` in
`crates/fj-lax/src/arithmetic.rs` to compare dense outputs against forced
boxed-`Literal` operands for:

- U32
- U64
- Bool
- Complex64
- Complex128

The test also asserts the result stays in the matching dense backing.

## Benchmark guard

Added criterion rows in `crates/fj-lax/benches/lax_baseline.rs`:

- `eval/select_n_64k_u32_vec`
- `eval/select_n_64k_u32_literal_ref`

These quantify the dense unsigned case-tensor switch against the literal
reference path in the next batch run.

## Negative-evidence ledger

- `tile` U32/U64/Complex was investigated first and rejected as a code target:
  current `eval_tile` already has dense U32/U64/Bool/Complex arms. No retry
  without a fresh profile showing a different tile bottleneck.
- FMA/SIMD-exp/GEMM remains maintainer-gated under `frankenjax-cntiy`; no retry
  until the FMA policy changes.
- Cumsum axis-specialization remains a prior rejected/non-comparable family; no
  retry without fresh same-worker evidence.
- This commit does not claim a speedup yet. It only lands the code path,
  conformance guard, and criterion rows per the code-first directive.

## Local validation

Passed:

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a cargo check -p fj-lax
```
