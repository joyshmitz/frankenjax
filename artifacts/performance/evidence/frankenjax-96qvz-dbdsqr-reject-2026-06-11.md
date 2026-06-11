# frankenjax-96qvz DBDSQR Square-SVD Candidate Rejection

## Target

- Bead: `frankenjax-96qvz`
- Hotspot: `fj-lax` real thin SVD, Criterion `linalg/svd_48x48_f64`
- Profile-backed problem: current one-sided Jacobi SVD remains the dominant `fj-lax` real-SVD route at the 48x48 benchmark shape.
- Candidate primitive: Golub-Kahan bidiagonalization followed by a DBDSQR-style implicit-shift bidiagonal QR loop with U/V accumulation, gated to square `n >= 32`.

## Baseline

Baseline was taken from a clean detached worktree at `a73c3cd0` because the shared checkout already contained an uncommitted candidate hunk when this pass resumed.

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1153651 rch exec -- cargo bench -j 1 -p fj-lax --bench lax_baseline -- linalg/svd_48x48_f64
```

Result on `vmi1153651`:

```text
linalg/svd_48x48_f64 time: [2.6364 ms 2.8001 ms 3.0064 ms]
```

## Candidate

Command:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -j 1 -p fj-lax --bench lax_baseline -- linalg/svd_48x48_f64
```

Result on `vmi1153651`:

```text
linalg/svd_48x48_f64 time: [2.3853 ms 2.4282 ms 2.4717 ms]
```

Same-worker delta:

- Median speedup: `2.8001 / 2.4282 = 1.15x`
- Conservative bound: `2.6364 / 2.4717 = 1.07x`

## Behavior Proof

Focused SVD tests passed for the candidate:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1153651 rch exec -- cargo test -j 1 -p fj-lax --lib svd -- --nocapture
```

RCH selected `vmi1152480` for the correctness run; the run passed `10` SVD-filtered tests.

Proof gap: the existing SVD-filtered tests do not exercise the candidate's new `n >= 32` DBDSQR route or provide a golden-output digest for that route. The candidate also intentionally changes the floating-point algorithm and accumulation path for square real SVD, so tolerance reconstruction alone is not enough proof for production routing.

The final source state keeps no candidate code. `crates/fj-lax/src/linalg.rs` has no diff from `HEAD`, so production behavior is unchanged by this closeout artifact.

## Rejection Gate

Score:

```text
Impact 2 x Confidence 2 / Effort 4 = 1.0
```

Decision: rejected. The measured same-worker improvement is below the `Score >= 2.0` keep gate for a large floating-point algorithm replacement, and the proof is incomplete for the newly gated path.

## Next Route

Do not repeat AtA/normal-equations, cached-column-norm Jacobi, bidiagonalize-then-Jacobi, augmented symmetric-eigen, or scalar DBDSQR prototypes for this profile shape.

Next profile-backed primitive should be structurally different:

- blocked/cache-resident one-sided Jacobi sweep with deterministic column-pair tile scheduling, or
- a smaller certified DBDSQR proof harness and golden digest before another production routing attempt.

The immediate target ratio for the blocked Jacobi route is at least `2.0x` on `linalg/svd_48x48_f64` while preserving singular-value ordering, deterministic sign normalization, no RNG, and the current reconstruction tolerance contract.
