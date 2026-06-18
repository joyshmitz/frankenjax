# frankenjax-cod-b-dense-scalar-repeat-axis0-zb2b7: dense scalar repeat_axis0 packing

Status: code-first batch-test pending
Owner: cod-b
Date: 2026-06-18

## Lever

Specialize `TensorValue::repeat_axis0` for scalar repeats and route directly to
dense typed constructors. This removes the boxed `Vec<Literal>` construction for
float, bool, half, and complex scalar broadcasting, and makes integer dense
construction explicit instead of relying on the generic constructor to discover
it afterward.

This targets realistic `vmap` scalar-broadcast workloads and scan/batching
helpers that repeat a scalar across a leading axis.

## Negative-Evidence Ledger

- Do not touch the current `fj-lax` U32/U64 sort work in `tensor_ops.rs` or
  `lax_baseline.rs`; those files were dirty before this lever and appear to be a
  peer-owned in-flight radix-sort batch.
- Do not retry the maintainer-gated FMA/SIMD lane until `frankenjax-cntiy`
  receives a decision or an audited per-function target-feature plan.
- Do not retry rejected GEMM packing, QR/SVD preprocessing, cumsum prefix, or
  eager concat materialization families without a fresh profile showing a
  different bottleneck and a concrete retry predicate.

## Behavior Preservation

- Ordering: unchanged; every output slot repeats the same scalar literal.
- Tie-breaking: not applicable.
- Floating point: bit-identical; f32/f64 paths repeat stored bit patterns,
  including signed zero and NaN payloads.
- RNG: unchanged/not applicable.
- Fallback: tensor repeats still use the existing tensor path.

## Validation

- Required by directive for this code-first batch: local
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo check -p fj-core`.
- Criterion and full conformance remain pending for batch-test; do not cite this
  artifact as benchmark proof until measured deltas are added.

## Retry Predicate

Retry only if benchmark evidence attributes material time or allocation pressure
to scalar repeat/broadcast packing. If the batch-test phase finds a slowdown,
revert the storage fast path and route to a deeper dispatch/batching primitive
instead of repeating scalar materialization variants.
