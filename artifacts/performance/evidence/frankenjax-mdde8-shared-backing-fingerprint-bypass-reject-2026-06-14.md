# frankenjax-mdde8: shared-backing fingerprint bypass reject

## Target

- Bead: `frankenjax-mdde8`
- Crate: `fj-interpreters`
- File tested: `crates/fj-interpreters/src/staging.rs`
- Profile-backed row: `staging/full_pipeline/chain_100eq`
- Route: after `frankenjax-yuve3`, full RCH Criterion on `vmi1227854` still ranked this row highest in `fj-interpreters`.

## Baseline

Command:

```text
RCH_WORKER=vmi1227854 rch exec -- cargo bench -j 1 -p fj-interpreters --bench pe_baseline -- --noplot
```

Pre-edit row from the full post-`frankenjax-yuve3` profile:

```text
staging/full_pipeline/chain_100eq time: [909.44 ns 953.31 ns 999.93 ns]
```

## Rejected Lever

The probe added a same-backing cache-hit path in `cached_staged_program`: if the cached and candidate `Jaxpr` shared the same `EquationList` backing and top-level fields matched, it returned the cached staged program before comparing the long canonical-fingerprint string. All other cases fell back to the existing fingerprint plus structural equality path.

This was behavior-preserving in the focused proof, but the extra branch and helper did not beat the existing branch layout on the target benchmark.

## Proof

Passed:

```text
rustfmt --edition 2024 --check crates/fj-interpreters/src/staging.rs
git diff --check
RCH_WORKER=vmi1227854 rch exec -- cargo test -j 1 -p fj-interpreters --lib test_staging_single_unknown_equation_fast_path_golden -- --nocapture
```

Golden SHA remained:

```text
a3cb705ac10423c13f45917bfb71b6daeae347b7a42765da666800bf6e8f48af
```

The existing mutation check still proved that changing a cached Jaxpr misses instead of returning stale staged output.

## Rejection Gate

Command:

```text
RCH_WORKER=vmi1227854 rch exec -- cargo bench -j 1 -p fj-interpreters --bench pe_baseline -- --noplot staging/full_pipeline/chain_100eq
```

Candidate:

```text
staging/full_pipeline/chain_100eq time: [1.0690 us 1.1208 us 1.1701 us]
```

Midpoint ratio: `953.31 ns / 1.1208 us = 0.85x`

Conservative interval ratio: `909.44 ns / 1.1701 us = 0.78x`

Score: `0.0`, reject.

## Decision

No production source hunk was kept.

The row is no longer bottlenecked by the fingerprint string compare in a way this branch layout can exploit. The next attempt should avoid returning an owned full `StagedProgram` on hot all-known cache hits, or split the public value construction from the execution result path with a representation-level plan that preserves public `StagedProgram` fields only when callers actually need them.
