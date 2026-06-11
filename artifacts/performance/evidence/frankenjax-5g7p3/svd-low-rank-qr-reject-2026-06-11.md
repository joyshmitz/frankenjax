# frankenjax-5g7p3: low-rank QR SVD candidate rejection

Date: 2026-06-11
Agent: BeigeMouse
Target crate: fj-lax
Target bench: `linalg/svd_48x48_f64`
Worker: `vmi1227854`

## Target

`frankenjax-5g7p3` followed the `frankenjax-dn4le` slice-index win. The profile
fixture is numerically low rank: the pass225 counter proof reports 44 near-zero
singular columns after cyclic one-sided Jacobi. The candidate tested whether a
rank-revealing QR certificate could compress the matrix before SVD.

One temporary lever was tested:

- build a deterministic pivoted modified-Gram-Schmidt QR certificate;
- accept only when unresolved residual energy was below `1.0e-20` of initial
  column energy;
- run the existing one-sided Jacobi SVD on the compressed `R` factor;
- lift `U` back through `Q`, return zero tail singular columns, and leave all
  other inputs on the existing fallback.

No source hunk was kept.

## Baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- \
  cargo bench -p fj-lax --bench lax_baseline -- \
  linalg/svd_48x48_f64 --sample-size 20 --warm-up-time 1 --measurement-time 3
```

Clean baseline at `cf1dae66`:

```text
linalg/svd_48x48_f64 [1.0427 ms 1.0602 ms 1.0831 ms]
```

## Proof gate

Focused proof command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- \
  cargo test -p fj-lax --lib 48x48 -- --nocapture
```

Result: rejected. The profile counter test still passed, but the public 48x48
golden digest changed:

```text
expected 6f1b0069586dda5b23d377bbb171a18ac0e24b6e0309dabc4ad0e0d2d1864d90
got      216b56ec9646bad87f7300d4dc864b058b1568ba34e459c15f11f0dca1c58ced
```

That fails the exact-output gate for this pass.

## Benchmark

The rejected candidate was measured once on the same worker for routing value:

```text
linalg/svd_48x48_f64 [2.2950 ms 2.3150 ms 2.3339 ms]
```

Compared with the clean baseline midpoint, this is `0.458x` as fast. Score:
`0.0`; proof failed and performance regressed.

## Decision

Reject the low-rank-QR plus wide-Jacobi construction. It pays QR setup cost,
still performs an `n`-wide Jacobi/V rotation problem on the compressed `R`, and
changes the pinned public output digest.

The source tree was restored to the shipped `cf1dae66` SVD implementation before
this evidence was committed.

## Next primitive

Do not repeat this QR-plus-wide-Jacobi shape. The next deeper route should be a
compressed SVD primitive that avoids the `n`-wide Jacobi core entirely, for
example:

- pivoted QR certificate to rank `r`;
- form the small `r x r` matrix `R R^T`;
- diagonalize that small problem;
- construct right singular vectors as `R^T U_r / sigma`;
- define and prove the SVD output-contract surface explicitly before any
  production routing.
