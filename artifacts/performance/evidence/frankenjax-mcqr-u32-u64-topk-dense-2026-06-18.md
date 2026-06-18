# frankenjax-mcqr: U32/U64 dense TopK fast path

Date: 2026-06-18
Agent: cod-a / RedBeaver
Status: in_progress, code-first batch-test pending

## Attempt

Lever: specialize `lax.top_k` for dense `U32` and `U64` tensor storage.

Real-world target: token/id/rank selection, Threefry-like RNG key ranking,
hash-bucket pruning, categorical feature scoring, and packed unsigned score
pipelines where data is already in typed dense storage.

Source ideas:

- Running-the-gauntlet negative ledger: every attempt gets a keep/reject rule.
- Alien-graveyard adaptive specialization: guarded typed fast path with exact fallback.
- Extreme optimization: avoid `Vec<Literal>` materialization and boxed scalar walks.

## Behavior Contract

Dense U32 key: `!u64::from(value)`.
Dense U64 key: `!value`.

Both use the existing `(complement_key, original_index)` ordering and
`order_top_k_pairs`, so the result order is the same as the generic comparator:
descending unsigned value, ascending index for ties. Shapes, dtypes, and i64
indices are unchanged. Literal-backed U32/U64 tensors still fall through to the
generic literal-key path.

## Guards Added

- `top_k_unsigned_dense_matches_literal` compares dense U32/U64 tensors against
  forced Literal-backed tensors for k = 1, 32, 257 with max values, zero, high-bit
  values, and duplicate ties.
- Criterion hooks:
  - `eval/topk_64k_k128_u32_vec`
  - `eval/topk_64k_k128_u32_literal_ref`
  - `eval/topk_64k_k128_u64_vec`
  - `eval/topk_64k_k128_u64_literal_ref`

## Negative-Evidence Ledger

Decision: pending.

Keep only if same-worker criterion shows a material dense-vs-literal speedup for
U32 or U64 TopK and conformance remains green.

Reject if the dense path is slower, neutral under same-worker measurement, or
requires semantic exceptions. If rejected, do not retry unsigned TopK dense
specialization unless a fresh profile shows U32/U64 TopK as a top realistic
workload hotspot with changed storage or workload shape.

## Verification

Requested verification scope for this code-first batch:

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a cargo check -p fj-lax
```

Tests, criterion benchmarks, rch, and conformance are intentionally not run in
this batch per campaign instruction.
