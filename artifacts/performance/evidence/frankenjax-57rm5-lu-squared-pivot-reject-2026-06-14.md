# frankenjax-57rm5 -- LU squared pivot magnitude probe

Status: rejected; no production or test source hunk kept.

## Target

- Parent bead: `frankenjax-p1vbf`
- Hot row: `linalg/lu_1024x1024_f64`
- Baseline/profile evidence: RCH Criterion on `vmi1227854` ranked the selected linalg LU row at `[105.48 ms 120.29 ms 136.57 ms]`, above the other sampled linalg candidates in the same run.
- Alien-graveyard routing: communication-avoiding and blocked linear algebra remain the right primitive family for this parent. This probe tested only the narrow pivot-scan magnitude check before moving deeper.

## Probe

The temporary hunk replaced blocked real-LU panel pivot comparisons of `sqrt(x * x)` with squared-magnitude comparisons, and similarly compared the diagonal threshold in squared space. It did not change pivot ordering for finite values, row swaps, triangular solves, Schur updates, floating-point arithmetic in the factorization update, tie-breaking, or RNG behavior.

## Proof

Focused RCH tests passed before benchmarking the candidate:

```text
rch exec -- cargo test -j 1 -p fj-lax lu_blocked_path -- --nocapture
worker: vmi1227854
tests:
  linalg::tests::lu_blocked_path_golden_output_digest
  linalg::tests::lu_blocked_path_reconstructs_and_matches_scalar
golden sha256: 4015f89e43b02bad7dc3f84df97617fd1d93332a81682e3bada8da779af55a91
```

## Benchmark Decision

The first candidate run on `vmi1227854` measured `[53.875 ms 59.711 ms 66.301 ms]`, which looked faster than the earlier broad profile row. The narrower same-worker check rejected the probe:

```text
worker: vmi1153651
baseline, original sqrt form:   [113.33 ms 123.12 ms 133.86 ms]
candidate, squared comparison:  [175.47 ms 189.34 ms 204.32 ms]
ratio: 0.65x by median, regression
score: 0.0
```

Because the stricter single-row same-worker pair showed a regression, the source hunk was restored and is not shipped.

## Next Primitive

Continue `frankenjax-p1vbf` from a structural linalg lever, not another pivot-scan microprobe. The next target should be a blocked/communication-avoiding primitive that attacks the dominant matrix-update work directly.
