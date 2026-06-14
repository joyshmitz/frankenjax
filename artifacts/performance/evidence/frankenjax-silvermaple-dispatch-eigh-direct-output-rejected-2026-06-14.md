# fj-dispatch batched Eigh 3x3 direct output rejection

Bead: frankenjax-es1d1
Agent: SilverMaple
Date: 2026-06-14
Worker: vmi1293453
Base: 011d975313b080bdaae7f873868ac04f97377d28

## Profile-backed target

Post-SVD re-profile command:

```bash
RCH_WORKER=vmi1293453 RCH_WORKERS=vmi1293453 RCH_REQUIRE_REMOTE=1 \
rch exec -- cargo bench -j 1 -p fj-dispatch --bench dispatch_baseline -- \
  vmap_ --noplot --sample-size 10 --measurement-time 1 --warm-up-time 1
```

Target row:

```text
vmap_eigh/batched_matrix_3x3: [92.763 us 94.349 us 96.684 us]
```

## Attempted lever

The attempted change pre-sized the 3x3 Eigh output buffers and wrote each
batch's analytic Eigh outputs into fixed slices instead of appending into
capacity-reserved vectors.

This preserved the analytic Eigh arithmetic, ordering, tie-breaking, RNG
absence, and output offsets, but it did not improve the measured hot row.

## Behavior proof

Command:

```bash
RCH_WORKER=vmi1293453 RCH_WORKERS=vmi1293453 RCH_REQUIRE_REMOTE=1 \
rch exec -- cargo test -j 1 -p fj-dispatch --lib \
  test_batch_trace_eigh_multi -- --nocapture
```

Result:

```text
6 passed, 0 failed
```

Covered golden tests:

```text
test_batch_trace_eigh_multi_rank3_golden_sha256
test_batch_trace_eigh_multi_leading_batch_dim_golden_sha256
```

Known golden SHA256 values remained:

```text
rank3: 9c8554df967d304b2570460fc5db4fca86602577232fcf8b01177fcd41cd365f
leading_batch: de40295687095bc622bd73074d24337004f440bdd2cc65d8a8759dfb5cf0b106
```

## Re-benchmark

Command:

```bash
RCH_WORKER=vmi1293453 RCH_WORKERS=vmi1293453 RCH_REQUIRE_REMOTE=1 \
rch exec -- cargo bench -j 1 -p fj-dispatch --bench dispatch_baseline -- \
  vmap_eigh/batched_matrix_3x3 --noplot --sample-size 10 \
  --measurement-time 1 --warm-up-time 1
```

Result:

```text
before: [92.763 us 94.349 us 96.684 us]
after:  [95.070 us 98.034 us 101.28 us]
p50 speedup: 0.962x
Decision: reject, source reverted
```

## Next routing

This rejection means the remaining Eigh cost is not output append overhead.
The next Eigh pass should attack the analytic 3x3 kernel itself or a deeper
algorithmic primitive, not repeat output-buffer microstructure.
