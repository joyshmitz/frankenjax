# frankenjax-mcqr: U32/U64 dense Sort fast path

Date: 2026-06-18
Agent: cod-a / RedBeaver
Status: in_progress, code-first batch-test pending

## Attempt

Lever: specialize `lax.sort` and `lax.argsort` for dense `U32` and `U64`
tensor storage.

Real-world target: ranking token ids, RNG/key streams, hash buckets, categorical
scores, packed feature ids, and integer ordering workloads where upstream ops
already produce typed dense unsigned tensors.

Source ideas:

- Running-the-gauntlet negative ledger: every code-first candidate gets a
  decision rule before batch benchmarking.
- Alien-graveyard Swiss-table/SIMD-layout pattern: keep metadata/keys packed and
  avoid cache-hostile boxed scalar walks.
- Extreme optimization: one typed-layout lever, preserving the same stable radix
  ordering and fallback semantics.

## Behavior Contract

Dense U32 key: `u64::from(value)`.
Dense U64 key: `value`.
Descending order uses the existing complement-key rule. The radix pair stream is
created in input-index order and the radix sort is stable, so equal unsigned
values retain ascending original-index order. Literal-backed U32/U64 tensors
still fall through to `sort_along_axis_literal_radix`.

Shapes and dtypes are unchanged:

- `sort(U32)` emits dense U32.
- `sort(U64)` emits dense U64.
- `argsort` emits dense I64 indices.

## Guards Added

- `dense_unsigned_sort_matches_literal_radix` compares dense U32/U64 tensors
  against forced Literal-backed tensors for ascending and descending sort and
  argsort, including zero, max, high-bit values, and duplicate ties.
- Criterion hooks:
  - `eval/sort_64k_u32`
  - `eval/sort_64k_u32_literal_ref`
  - `eval/sort_64k_u64`
  - `eval/sort_64k_u64_literal_ref`

## Negative-Evidence Ledger

Decision: pending.

Keep only if same-worker criterion shows material dense-vs-literal speedup for
U32 or U64 sort/argsort and conformance remains green.

Reject if the dense path is slower, neutral under same-worker measurement, or
needs a semantic exception. If rejected, do not retry unsigned sort dense
specialization unless a fresh profile shows U32/U64 sort as a top realistic
workload hotspot with changed storage or workload shape.

## Verification

Requested verification scope for this code-first batch:

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a cargo check -p fj-lax
```

Tests, criterion benchmarks, rch, and conformance are intentionally not run in
this batch per campaign instruction.
