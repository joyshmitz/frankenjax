# frankenjax-0iqdb -- structural cache-probe rejection

Date: 2026-06-14
Agent: BeigeMouse
Bead: frankenjax-0iqdb
Crate: fj-interpreters
Decision: rejected, no source hunk kept.

## Profile-backed target

Post-`frankenjax-aip01` RCH profile on worker `vmi1152480` ranked this as the
largest remaining `fj-interpreters` row:

```text
staging/full_pipeline/chain_100eq [1.6313 us 1.7492 us 1.8690 us]
```

The benchmark stages a 100-equation `Add(var, 1)` chain with the single input
known, then executes the all-known staged program.

## Baselines

Same-worker acceptance baseline from the post-landing profile:

```text
RCH worker: vmi1152480
staging/full_pipeline/chain_100eq [1.6313 us 1.7492 us 1.8690 us]
```

Focused pre-change baseline on another RCH worker:

```text
RCH worker: vmi1153651
staging/full_pipeline/chain_100eq [2.4287 us 2.6198 us 2.8848 us]
```

The rejection decision uses the comparable `vmi1152480` row.

## Attempted lever

Temporary source change only: remove `Jaxpr::canonical_fingerprint()`
recomputation from the staged-program cache hit path and use exact structural
`Jaxpr` equality plus const value, unknown mask, and known value equality.

A focused mutation-safety assertion was added temporarily to the existing golden
test to verify that mutating a cloned Jaxpr after a cache hit missed the cache
and produced the new result.

## Proof result

Focused RCH proof passed on worker `vmi1227854`:

```text
rch exec -- cargo test -j 1 -p fj-interpreters --lib test_staging_single_unknown_equation_fast_path_golden -- --nocapture
```

Golden SHA remained unchanged:

```text
a3cb705ac10423c13f45917bfb71b6daeae347b7a42765da666800bf6e8f48af
```

The proof was clean, but performance failed the keep gate.

## After

Final after-run on the same worker as the profile-backed baseline:

```text
RCH worker: vmi1152480
staging/full_pipeline/chain_100eq [1.6685 us 1.8003 us 1.9276 us]
```

Median ratio: `1.7492 / 1.8003 = 0.97x`

Score: `0.0` because the candidate regressed the comparable timing row.

## Rejection rationale

The cache-hit cost is not just fingerprint recomputation. For the all-known
100-equation chain, the remaining cost is exact structural comparison plus
cloning a `StagedProgram` containing the 100-equation known Jaxpr. Removing the
fingerprint probe did not reduce the row and slightly regressed it.

## State kept

No production or test source hunk was kept. The working tree was manually
restored to the already-landed `frankenjax-aip01` staging implementation.

## Next primitive

Do not repeat cache-probe micro-tuning. The next attack should change the
representation used for repeated staged programs, for example a compact
all-known staged-result plan or shared/interior representation that avoids
cloning the full 100-equation known Jaxpr on every cache hit while preserving
the public `StagedProgram` fields and exact ordering semantics.
