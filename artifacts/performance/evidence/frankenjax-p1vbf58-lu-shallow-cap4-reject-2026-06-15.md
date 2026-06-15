# frankenjax-p1vbf.58 -- LU shallow-panel f64 GEMM cap-4 probe

Status: rejected

Parent: `frankenjax-p1vbf`

## Target

After `4037d77b`, a focused RCH baseline on `ovh-a` for the current cap-8
implementation measured:

- `linalg/lu_1024x1024_f64`: `[27.299 ms 27.484 ms 27.674 ms]`

`frankenjax-p1vbf.57` showed that changing `LU_BLOCK_SIZE` could improve timing
but changed the pinned golden digest, so this probe kept the 128-column panel
boundary and only changed row-worker fanout for f64 shallow-k GEMMs.

## Lever Probed

Change only:

```rust
const SHALLOW_K_MAX_THREADS: usize = 8;
```

to:

```rust
const SHALLOW_K_MAX_THREADS: usize = 4;
```

The intended isomorphism was the same as the shipped cap-8 pass: output rows
remain disjoint, each output element keeps the same ascending-k fold, B packing
and result writeback are unchanged, and panel boundaries are unchanged.

## Benchmark

Command:

```bash
rch exec -- cargo bench -j 1 -p fj-lax --bench lax_baseline -- linalg/lu_1024x1024_f64 --noplot
```

Same-worker `ovh-a` result:

- baseline cap 8: `[27.299 ms 27.484 ms 27.674 ms]`
- candidate cap 4: `[32.403 ms 32.755 ms 33.103 ms]`

Median ratio: `27.484 / 32.755 = 0.839x`, regression.

## Decision

Rejected before running the proof suite. No production source hunk kept; the cap
and its unit-test expectation were restored to 8.

Next route: stop retuning shallow-k worker count. The LU row now needs either
component-level timing around panel factorization/copy/GEMM/subtract or a
non-boundary-changing structural lever such as fusing Schur subtraction into the
existing product path while preserving each output's GEMM accumulation order.
