# frankenjax-cod-b-dense-scalar-stack-axis0-tobpl: dense scalar stack_axis0 packing

Status: code-first batch-test pending
Owner: cod-b
Date: 2026-06-18

## Lever

Specialize `TensorValue::stack_axis0` for homogeneous scalar outputs and route
directly to dense typed constructors. This avoids first building a boxed
`Vec<Literal>` for vmap loop-and-stack scalar outputs, repeated scalar stacking,
and any other scalar-output batching path that funnels through `stack_axis0`.

The fast path is storage-only. It preserves the old mixed-scalar fallback and
materializes bit-identical literals for homogeneous integer, bool, half, float,
and complex scalar families.

## Negative-Evidence Ledger

- Do not retry the abandoned `tensor_ops.rs` integer `iota` fast path while
  RedBeaver owns that file reservation.
- Do not retry prior rejected GEMM packing, QR/SVD preprocessing, cumsum prefix,
  or eager concat materialization families without a new profile showing a
  different bottleneck and a concrete retry predicate.
- If this attempt regresses batch benchmarks, keep the dense scalar-stack tests
  only if they expose a correctness gap; otherwise revert the storage fast path.

## Validation

- Required by directive for this code-first batch: local
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo check -p fj-core`.
- Criterion and full conformance are intentionally pending for the batch-test
  phase; this artifact must not be cited as benchmark proof until those numbers
  are added.

## Retry Predicate

Retry only if profiling attributes material vmap/stack overhead to scalar-output
packing or if criterion shows a neutral-to-positive result with no conformance
regression. If benchmarks show a slowdown, record the exact workload here and
route to a deeper primitive rather than repeating scalar packing variants.
