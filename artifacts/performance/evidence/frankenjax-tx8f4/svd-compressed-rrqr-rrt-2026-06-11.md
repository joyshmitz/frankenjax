# frankenjax-tx8f4: compressed RRQR SVD via small RRT eigensolve

Date: 2026-06-11
Agent: BeigeMouse
Target crate: fj-lax
Target bench: `linalg/svd_48x48_f64`
Worker: `vmi1227854`

## Target

The 48x48 SVD profile fixture is numerically low rank. Pass225 measured 44
near-zero singular columns after the cyclic one-sided Jacobi path. Pass228
rejected a low-rank QR route that still ran a wide one-sided Jacobi SVD on `R`:
it changed the digest and regressed to `2.3150 ms`.

This pass keeps the QR certificate idea but removes the wide-Jacobi core:

1. Build a deterministic pivoted modified-Gram-Schmidt certificate.
2. Accept only when unresolved residual energy is at most `1.0e-20` of the
   original column energy.
3. Form the small `r x r` matrix `R R^T`.
4. Diagonalize that small symmetric problem.
5. Construct right singular vectors as `R^T U_r / sigma`.
6. Extend the right-singular basis with deterministic Gram-Schmidt.

Inputs outside the conservative gate fall back to the existing one-sided Jacobi
SVD path.

## Baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- \
  cargo bench -p fj-lax --bench lax_baseline -- \
  linalg/svd_48x48_f64 --sample-size 20 --warm-up-time 1 --measurement-time 3
```

Clean baseline after `cf1dae66` / evidence-only `db162499`:

```text
linalg/svd_48x48_f64 [1.0427 ms 1.0602 ms 1.0831 ms]
```

## Candidate

Same command and worker:

```text
linalg/svd_48x48_f64 [90.955 us 95.436 us 100.69 us]
```

Median ratio: `11.11x`.
Conservative lower/upper ratio: `10.36x`.

Score: `14.8 = Impact 11.1 * Confidence 4 / Effort 3`.

## Contract proof

This is not a bit-isomorphic singular-vector change. The compressed route chooses
a different deterministic basis inside the SVD's non-unique singular-vector
surface for this numerically low-rank matrix. That is already the documented
real-SVD contract in `eval_svd_real`: reconstruction and spectrum parity, not
bit identity of U/V.

The profile proof was strengthened to pin the new deterministic output and
verify the observable SVD contract:

- output digest: `5d9fc335e204bea56ea3e086abf97a7e271c9d70aaad4352503d8de8457f197f`;
- singular values are descending;
- `U diag(S) V^T` reconstructs the 48x48 profile matrix within `1e-9`;
- fallback remains unchanged for `m < n`, `n < 32`, non-finite data, failed QR
  residual certificate, or rank not compressed below `n`;
- pivot and eigenvalue ordering ties are deterministic by index;
- there is no RNG.

The old exact digest
`6f1b0069586dda5b23d377bbb171a18ac0e24b6e0309dabc4ad0e0d2d1864d90` is replaced
only for this explicitly contract-proven SVD route.

## Validation

Passed:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- \
  cargo test -p fj-lax --lib 48x48 -- --nocapture
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- \
  cargo test -p fj-lax --lib svd -- --nocapture
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- \
  cargo check -p fj-lax --lib
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- \
  cargo clippy -p fj-lax --lib -- -D warnings
rustfmt --edition 2024 --check crates/fj-lax/src/linalg.rs
git diff --check
```

`ubs crates/fj-lax/src/linalg.rs` remains nonzero from pre-existing file-wide
panic/unwrap/direct-indexing inventory. Its built-in fmt, clippy, check,
test-build, cargo-audit, and cargo-deny sections were clean.

## Next route

Reprofile after landing. The SVD profile target should no longer be the top
SVD/Jacobi hotspot; do not repeat low-rank QR plus wide-Jacobi.
