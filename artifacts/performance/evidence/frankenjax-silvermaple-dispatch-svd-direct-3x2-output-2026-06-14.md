# fj-dispatch batched SVD 3x2 direct output

Bead: frankenjax-nldtv
Agent: SilverMaple
Date: 2026-06-14
Worker: vmi1293453
Commit base: b2733679c3e0a8e15b2ba4fac7d4a471d6b2fd8f9

## Profile-backed target

Post-Eigh re-profile command:

```bash
RCH_WORKER=vmi1293453 RCH_WORKERS=vmi1293453 RCH_REQUIRE_REMOTE=1 \
rch exec -- cargo bench -j 1 -p fj-dispatch --bench dispatch_baseline -- \
  vmap_ --noplot --sample-size 10 --measurement-time 1 --warm-up-time 1
```

Shifted profile row:

```text
vmap_svd/batched_matrix_3x2: [43.078 us 43.523 us 43.959 us]
```

Focused baseline command:

```bash
RCH_WORKER=vmi1293453 RCH_WORKERS=vmi1293453 RCH_REQUIRE_REMOTE=1 \
rch exec -- cargo bench -j 1 -p fj-dispatch --bench dispatch_baseline -- \
  vmap_svd/batched_matrix_3x2 --noplot --sample-size 10 \
  --measurement-time 1 --warm-up-time 1
```

Focused baseline:

```text
before: [42.132 us 43.080 us 43.998 us]
```

## One lever

The 512 x 3 x 2 F64 benchmark already uses the specialized thin 3x2 SVD
arithmetic. The lever keeps that exact scalar arithmetic but writes U, S, and
Vt directly into the batched output slices for 3x2 thin SVD.

Before this change, every slice wrote through `SvdScratch` vectors and then
copied `scratch.u_out`, `scratch.sigma`, and `scratch.vt` into the batched
output buffers. The new helper writes the same values into caller-provided
fixed-size slices, and the scratch-populating wrapper remains test-only for
bit-identity coverage.

## Behavior proof

Full SVD-filtered command before the test-only wrapper annotation:

```bash
RCH_WORKER=vmi1293453 RCH_WORKERS=vmi1293453 RCH_REQUIRE_REMOTE=1 \
rch exec -- cargo test -j 1 -p fj-dispatch --lib svd -- --nocapture
```

Result:

```text
8 passed, 0 failed, 1 ignored
```

Covered checks:

```text
svd_3x2_thin_fast_path_matches_generic_bits
test_batch_trace_svd_multi_full_matrices
test_batch_trace_svd_multi_leading_batch_dim
test_batch_trace_svd_multi_leading_batch_dim_golden_sha256
test_batch_trace_svd_multi_matrix_rank_error_preserved
test_batch_trace_svd_multi_nonleading_batch_dim
batch_svd_f32_parallel_path_is_bit_identical_to_serial
batch_svd_parallel_path_is_bit_identical_to_serial
```

Post-annotation focused check:

```bash
RCH_WORKER=vmi1293453 RCH_WORKERS=vmi1293453 RCH_REQUIRE_REMOTE=1 \
rch exec -- cargo test -j 1 -p fj-dispatch --lib \
  svd_3x2_thin_fast_path_matches_generic_bits -- --nocapture
```

Result:

```text
1 passed, 0 failed
```

Golden SHA256:

```text
leading_batch: 165205f8b8911fcc1d544aeb134f92fddf303cebe7ef7770c7718a80735eabbe
```

Isomorphism notes:

```text
ordering: unchanged; output offsets and batch order are unchanged
tie-breaking: unchanged; total_cmp column ordering is unchanged
floating point: unchanged; the 3x2 scalar arithmetic sequence is unchanged
RNG: unchanged; no random source is read
sign convention: unchanged; V sorting and U/Vt assignment formulas are unchanged
errors: unchanged; rank/type/full_matrices routing remains unchanged
```

## Re-benchmark

After command:

```bash
RCH_WORKER=vmi1293453 RCH_WORKERS=vmi1293453 RCH_REQUIRE_REMOTE=1 \
rch exec -- cargo bench -j 1 -p fj-dispatch --bench dispatch_baseline -- \
  vmap_svd/batched_matrix_3x2 --noplot --sample-size 10 \
  --measurement-time 1 --warm-up-time 1
```

Result:

```text
before: [42.132 us 43.080 us 43.998 us]
after:  [36.197 us 37.043 us 37.933 us]
p50 speedup: 1.163x
```

Score:

```text
Impact 1.16 * Confidence 0.95 / Effort 0.35 = 3.15
Decision: keep
```

## Gates

Passed:

```text
RCH SVD-filtered test set
RCH post-annotation direct wrapper test
RCH same-worker focused benchmark
```

Inherited warnings observed during RCH builds:

```text
fj-ad/src/lib.rs: unused variable lhs_rank
fj-dispatch/src/batching.rs: pre-existing unused_mut in triangular_solve helper
```
