# frankenjax-p1vbf.56 -- LU shallow-k f64 GEMM thread cap

Status: shipped

Parent: `frankenjax-p1vbf`

## Target

The active fj-lax linalg route ranked `linalg/lu_1024x1024_f64` as the current
hot row. Blocked LU reaches the f64 `matmul_2d` Schur update with panel depth
`k = 128`; the generic f64 selector allowed up to 16 scoped row workers there.

Prior route notes rejected more exact-order GEMM microkernel/layout probes. A
same-process route check also rejected Strassen for this lane:

- `linalg/strassen_ab_1024_matmul2d`: `[45.019 ms 46.482 ms 48.222 ms]`
- `linalg/strassen_ab_1024_strassen`: `[190.55 ms 216.20 ms 249.60 ms]`

## Lever

Add a shape-aware f64 `matmul_2d_thread_count(m, k, n)` wrapper around the
existing operation-count selector:

- `k <= 128` caps fanout at 8 workers.
- Deeper f64 GEMM shapes use the existing selector unchanged.
- f32, i64, complex, batched matmul, packing policy, row ownership, and kernel
  arithmetic are unchanged.

The cap only changes how many disjoint output-row chunks are assigned to scoped
workers. Each output element still runs the same ascending-k fold over the same
packed-B values and writes the same C slot once.

## Isomorphism Proof

- Output row partitioning remains disjoint; no worker writes another worker's
  rows.
- Per-output FP operation order is unchanged: the inner fold and accumulator
  update sequence are identical.
- B packing, C initialization, result writeback, shape checks, dtype checks, and
  error paths are unchanged.
- Tie-breaking and RNG surfaces are absent for this code path.
- f32 and non-f64 linalg selectors are untouched.

Golden output proof:

- `rch exec -- cargo test -j 1 -p fj-lax --lib matmul_2d -- --nocapture`
  - worker: `vmi1227854`
  - result: 22 passed, 1 ignored, 0 failed
- `rch exec -- cargo test -j 1 -p fj-lax --lib lu_blocked_path -- --nocapture`
  - worker: `vmi1153651`
  - result: 2 passed, 0 failed
  - unchanged LU golden SHA:
    `4015f89e43b02bad7dc3f84df97617fd1d93332a81682e3bada8da779af55a91`

## Benchmark

Command:

```bash
rch exec -- cargo bench -j 1 -p fj-lax --bench lax_baseline -- linalg/lu_1024x1024_f64 --noplot
```

Acceptance uses a same-worker restored-baseline pair on `vmi1227854`.

Baseline with production call sites temporarily restored to the old selector:

- `[51.471 ms 57.060 ms 64.643 ms]`

Candidate with shallow-k cap:

- `[36.916 ms 38.202 ms 39.582 ms]`

Ratios:

- median: `57.060 / 38.202 = 1.493x`
- conservative interval: `51.471 / 39.582 = 1.300x`

Score: `Impact 4 x Confidence 4 / Effort 2 = 8.0`, keep.

## Validation

- `git diff --check`: passed.
- `rch exec -- cargo check -j 1 -p fj-lax --lib`: passed on `ovh-a`.
- `ubs crates/fj-lax/src/tensor_contraction.rs`: nonzero from pre-existing
  file-wide heuristic inventory, including false-positive JWT decode matches on
  local numeric `decode` closures and older unwrap/indexing/test findings. Its
  built-in cargo/rustfmt/clippy/check/test-build/audit/deny sections were clean.
- `rch exec -- cargo clippy -j 1 -p fj-lax --lib -- -D warnings`: blocked by
  pre-existing `crates/fj-lax/src/linalg.rs` lint debt outside this lever:
  `clippy::doc_lazy_continuation` at lines 680-693 and
  `clippy::needless_range_loop` at lines 727, 779, 3563, 6229, and 6529.
- `cargo fmt --check -p fj-lax`: blocked by pre-existing crate-wide formatting
  drift across unrelated fj-lax files.
