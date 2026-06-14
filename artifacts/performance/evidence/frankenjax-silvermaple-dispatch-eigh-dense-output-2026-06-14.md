# fj-dispatch batched Eigh F64 output storage

Bead: frankenjax-p3qej
Agent: SilverMaple
Date: 2026-06-14
Worker: vmi1293453
Commit base: aafebd3a4d4b44be4c87ba5ae33b67243517cf1d

## Profile-backed target

Baseline command:

```bash
RCH_WORKER=vmi1293453 RCH_WORKERS=vmi1293453 RCH_REQUIRE_REMOTE=1 \
rch exec -- cargo bench -j 1 -p fj-dispatch --bench dispatch_baseline -- \
  vmap_ --noplot --sample-size 10 --measurement-time 1 --warm-up-time 1
```

Top measured `vmap_` dispatch row:

```text
vmap_eigh/batched_matrix_3x3: [94.904 us 97.201 us 99.744 us]
```

The target was selected because `vmap_eigh/batched_matrix_3x3` was the slowest
measured dispatch vmap row in this crate-scoped profile.

## One lever

`batch_eigh_multi` now accumulates eigenvalues and eigenvectors as dense
`Vec<f64>` until final tensor construction. For F64 outputs it builds the
tensor with `LiteralBuffer::from_f64_values`; non-F64 outputs keep the prior
`Literal::from_f64` materialization path.

This removes per-element `Literal` construction for the F64 hot path without
changing the eigensolver, value ordering, sorting, sign convention, RNG state,
or any arithmetic operation sequence.

## Behavior proof

Focused golden command:

```bash
RCH_WORKER=vmi1293453 RCH_WORKERS=vmi1293453 RCH_REQUIRE_REMOTE=1 \
rch exec -- cargo test -j 1 -p fj-dispatch --lib \
  test_batch_trace_eigh_multi_rank3_golden_sha256 -- --nocapture
```

Result: passed.

Golden SHA256:

```text
rank3: 9c8554df967d304b2570460fc5db4fca86602577232fcf8b01177fcd41cd365f
```

Broader Eigh command:

```bash
RCH_WORKER=vmi1293453 RCH_WORKERS=vmi1293453 RCH_REQUIRE_REMOTE=1 \
rch exec -- cargo test -j 1 -p fj-dispatch --lib \
  test_batch_trace_eigh_multi -- --nocapture
```

Result: 6 passed, 0 failed.

Covered cases:

```text
test_batch_trace_eigh_multi_non_square_batched_matrix_rejects
test_batch_trace_eigh_multi_matrix_rank_error_preserved
test_batch_trace_eigh_multi_rank3_golden_sha256
test_batch_trace_eigh_multi_leading_batch_dim
test_batch_trace_eigh_multi_leading_batch_dim_golden_sha256
test_batch_trace_eigh_multi_nonleading_batch_dim
```

Leading-batch golden SHA256 remains:

```text
leading_batch: de40295687095bc622bd73074d24337004f440bdd2cc65d8a8759dfb5cf0b106
```

Isomorphism notes:

```text
ordering: unchanged; values still come from sorted Eigh scratch buffers
tie-breaking: unchanged; no comparator or sorting code changed
floating point: unchanged; no arithmetic operation sequence changed
RNG: unchanged; no random source is read
F64 bits: preserved by LiteralBuffer::from_f64_values
non-F64: preserves prior Literal::from_f64 path
```

## Re-benchmark

After command:

```bash
RCH_WORKER=vmi1293453 RCH_WORKERS=vmi1293453 RCH_REQUIRE_REMOTE=1 \
rch exec -- cargo bench -j 1 -p fj-dispatch --bench dispatch_baseline -- \
  vmap_eigh/batched_matrix_3x3 --noplot --sample-size 10 \
  --measurement-time 1 --warm-up-time 1
```

Result:

```text
before: [94.904 us 97.201 us 99.744 us]
after:  [91.390 us 92.041 us 93.123 us]
p50 speedup: 1.056x
```

Score:

```text
Impact 1.06 * Confidence 0.90 / Effort 0.35 = 2.72
Decision: keep
```

## Gates

Passed:

```text
git diff --check
RCH focused golden test
RCH Eigh filtered test set
RCH same-worker benchmark
```

Inherited blockers observed while gating:

```text
rustfmt --edition 2024 --check crates/fj-dispatch/src/batching.rs
  failed on broad pre-existing formatting drift outside this lever

RCH cargo clippy -j 1 -p fj-dispatch --lib -- -D warnings
  failed before this crate due pre-existing fj-lax linalg lint debt

ubs crates/fj-dispatch/src/batching.rs
  exited 1 on inherited file-wide panic/indexing/cast policy findings
```
