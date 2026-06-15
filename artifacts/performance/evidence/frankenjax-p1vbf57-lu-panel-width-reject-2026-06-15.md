# frankenjax-p1vbf.57 -- LU1024 blocked panel-width A/B

Status: rejected

Parent: `frankenjax-p1vbf`

## Target

Fresh post-`4037d77b` RCH linalg routing on `ovh-a` put the absolute top rows in
f32/bf16 GEMM territory, overlapping SageAnchor's `frankenjax-9zwwb` lane. The
highest non-overlapping real-linalg row was:

- `linalg/lu_1024x1024_f64`: `[29.111 ms 29.324 ms 29.538 ms]`

A focused pre-edit baseline on the same worker measured:

- `LU_BLOCK_SIZE = 128`: `[27.299 ms 27.484 ms 27.674 ms]`

## Lever Probed

Change only the cache-blocked right-looking f64 LU panel width:

- current: `LU_BLOCK_SIZE = 128`
- candidate A: `LU_BLOCK_SIZE = 256`
- candidate B: `LU_BLOCK_SIZE = 64`

The probe did not change pivot search order, row-swap behavior, panel
factorization arithmetic, Schur update algorithm, dtype/shape handling, or any
f32/GEMM assigned lane.

## Results

Same-worker RCH command:

```bash
rch exec -- cargo bench -j 1 -p fj-lax --bench lax_baseline -- linalg/lu_1024x1024_f64 --noplot
```

Candidate A:

- `LU_BLOCK_SIZE = 256`: `[48.359 ms 49.380 ms 50.344 ms]`
- Decision: reject, median regression `27.484 / 49.380 = 0.56x`

Candidate B:

- `LU_BLOCK_SIZE = 64`: `[25.523 ms 25.688 ms 25.854 ms]`
- Timing alone: median ratio `27.484 / 25.688 = 1.070x`
- Conservative interval: `27.299 / 25.854 = 1.056x`

## Proof Failure

Command:

```bash
rch exec -- cargo test -j 1 -p fj-lax --lib lu_blocked_path -- --nocapture
```

`LU_BLOCK_SIZE = 64` failed the pinned LU golden digest:

- observed: `fe6750928341fb13c65a7597f5e416fb2478e3fe84df63862f6a8cf3d3ddfd6d`
- expected: `4015f89e43b02bad7dc3f84df97617fd1d93332a81682e3bada8da779af55a91`

The reconstruction/tolerance test still passed, so this is a bit-level ordering
change from the different panel partitioning rather than a gross numerical
failure. The user directive requires golden SHA preservation for this campaign,
so the faster `64` candidate is not shippable.

## Final Decision

No production source hunk kept. `LU_BLOCK_SIZE` was restored to `128`.

The next LU attack should avoid changing panel boundaries that feed the golden
digest unless the proof target is explicitly updated with an oracle-backed
tolerance contract. Better next routes:

- reduce scratch/copy overhead around `L21`, `U12`, and `prod` without changing
  panel boundaries;
- fuse Schur product subtraction without changing per-output GEMM accumulation
  order;
- add instrumentation to split LU1024 time into panel factorization, U12 solve,
  copy, GEMM, and subtraction before another algorithmic change.
