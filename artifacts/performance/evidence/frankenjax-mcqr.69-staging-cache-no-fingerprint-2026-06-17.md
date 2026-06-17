# frankenjax-mcqr.69: staging cache hit skips redundant fingerprint

## Target

Profile-backed local Criterion row while ts1/rch remote was offline:

```text
staging/full_pipeline/chain_100eq
```

The hot path was `stage_jaxpr_with_consts` hitting the one-entry
`StagedProgram` cache. Before this pass, every cache lookup computed
`jaxpr.canonical_fingerprint()` before also checking full structural equality.

## Baseline

Command:

```bash
CARGO_TARGET_DIR=/data/projects/.scratch/frankenjax-silvermaple-target CARGO_BUILD_JOBS=1 cargo bench -j 1 -p fj-interpreters --bench pe_baseline staging/full_pipeline/chain_100eq -- --warm-up-time 1 --measurement-time 2 --sample-size 10 --noplot
```

Result:

```text
time: [1.0608 us 1.0996 us 1.1374 us]
```

## Candidate

The cache entry no longer stores or compares the canonical fingerprint. Cache
hits are still guarded by const values, unknown mask, known values, and full
JAXPR structural equality over inputs, constvars, outputs, effects, and
equations.

Command:

```bash
CARGO_TARGET_DIR=/data/projects/.scratch/frankenjax-silvermaple-target CARGO_BUILD_JOBS=1 cargo bench -j 1 -p fj-interpreters --bench pe_baseline staging/full_pipeline/chain_100eq -- --warm-up-time 1 --measurement-time 2 --sample-size 10 --noplot
```

Result:

```text
time: [915.48 ns 933.19 ns 946.58 ns]
change: [-20.701% -17.401% -14.017%] (p = 0.00 < 0.05)
```

Accepted comparison: median `1.0996 us -> 0.93319 us` = `1.18x`.
Score: `2.7 = Impact 3.0 x Confidence 0.90 / Effort 1.0`.

## Proof

Focused staging proof:

```bash
CARGO_TARGET_DIR=/data/projects/.scratch/frankenjax-silvermaple-target CARGO_BUILD_JOBS=1 cargo test -j 1 -p fj-interpreters staging --lib -- --nocapture
```

Result:

```text
24 passed; 0 failed; 0 ignored; 206 filtered out
```

Golden cache-hit/mutated-miss digest:

```text
a3cb705ac10423c13f45917bfb71b6daeae347b7a42765da666800bf6e8f48af
```

## Isomorphism

- The old predicate required both fingerprint equality and structural equality.
- The new predicate keeps structural equality, so it cannot introduce a cache hit
  for a structurally different JAXPR.
- Const values, unknown masks, known values, output order, equation order,
  effects, dtype/shape/error behavior, tie surfaces, floating-point arithmetic,
  and RNG absence are unchanged.
- No unsafe code and no C BLAS/LAPACK/XLA linkage.

## Validation

Passed:

```text
cargo fmt --check -p fj-interpreters
git diff --check -- crates/fj-interpreters/src/staging.rs
cargo test -j 1 -p fj-interpreters staging --lib -- --nocapture
cargo bench -j 1 -p fj-interpreters --bench pe_baseline staging/full_pipeline/chain_100eq -- --warm-up-time 1 --measurement-time 2 --sample-size 10 --noplot
cargo check -j 1 -p fj-interpreters --all-targets
cargo clippy -j 1 -p fj-interpreters --all-targets -- -D warnings
```

UBS on `crates/fj-interpreters/src/staging.rs` remained nonzero from existing
test-heavy file-wide heuristic inventory. Its embedded formatter, clippy, cargo
check, test-build, audit, and deny sections were clean and it reported no unsafe
blocks.
