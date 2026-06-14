# frankenjax-yuve3: staged cache equation-slice equality guard

## Target

- Bead: `frankenjax-yuve3`
- Crate: `fj-interpreters`
- File: `crates/fj-interpreters/src/staging.rs`
- Profile-backed row: `staging/full_pipeline/chain_100eq`
- Parent route: successor to rejected `frankenjax-0iqdb`; the rejected structural-cache probe showed the row was still dominated by cache-hit comparison plus staged-program cloning, not fingerprint recomputation alone.

## Lever

`cached_staged_program` still requires the canonical fingerprint, const values, unknown mask, and known values to match. The single production lever is replacing the derived `Jaxpr` equality call with `cached_jaxpr_eq`, which compares the cheap top-level fields directly and short-circuits equation equality when the cached and candidate equation slices are the same backing slice before falling back to full equation equality.

This is a cache-hit comparison shortcut only. It does not change staging, partial evaluation, execution, cache insertion, or public `StagedProgram` fields.

## Benchmark Gate

Command:

```text
RCH_WORKER=vmi1227854 rch exec -- cargo bench -j 1 -p fj-interpreters --bench pe_baseline -- --noplot staging/full_pipeline/chain_100eq
```

Same-worker clean baseline at `bd0432dd` in `/data/projects/.scratch/frankenjax-yuve3-baseline-20260614T0225`:

```text
staging/full_pipeline/chain_100eq time: [1.1493 us 1.1806 us 1.2065 us]
```

Same-worker candidate in `/data/projects/.scratch/frankenjax-beigemouse-pass98-20260613T122136`:

```text
staging/full_pipeline/chain_100eq time: [977.00 ns 1.0093 us 1.0454 us]
```

Midpoint speedup: `1.1806 / 1.0093 = 1.170x`

Conservative interval speedup: `1.1493 / 1.0454 = 1.099x`

Score: `Impact 2 * Confidence 4 / Effort 2 = 4.0`, keep.

## Isomorphism Proof

- Ordering preserved: yes. Cache hits return the same cloned `StagedProgram`; misses still run the original staging path.
- Tie-breaking unchanged: yes. The comparison only decides whether a cached staged program is structurally identical; equal equations preserve their existing order.
- Floating-point behavior: unchanged. No primitive evaluation or arithmetic changed.
- RNG behavior: unchanged. No RNG surface.
- Error behavior: unchanged. A nonmatching Jaxpr still misses the cache and runs full staging.
- Golden output: focused proof test preserved SHA `a3cb705ac10423c13f45917bfb71b6daeae347b7a42765da666800bf6e8f48af`.

Mutation-safety proof: the focused test mutates the first cached Jaxpr equation from `Neg` to `Abs`; the cache correctly misses and produces the new result instead of returning the old staged output.

## Validation

Passed:

```text
git diff --check
rustfmt --edition 2024 --check crates/fj-interpreters/src/staging.rs
RCH_WORKER=vmi1227854 rch exec -- cargo test -j 1 -p fj-interpreters --lib test_staging_single_unknown_equation_fast_path_golden -- --nocapture
RCH_WORKER=vmi1227854 rch exec -- cargo check -j 1 -p fj-interpreters --lib
```

`cargo clippy -j 1 -p fj-interpreters --lib -- -D warnings` remains blocked before reaching `fj-interpreters` by pre-existing `fj-lax/src/linalg.rs` lints tracked separately (`doc_lazy_continuation`, `needless_range_loop`).

`ubs crates/fj-interpreters/src/staging.rs` remains nonzero from existing file-wide heuristic inventory and a false positive on `Instant::now`; its built-in fmt/clippy/check/test-build/audit/deny sections were clean.

## Next Route

Reprofile after landing. If `staging/full_pipeline/chain_100eq` remains hot, the next primitive is a representation-level staged plan that avoids cloning the known Jaxpr payload on cache hits, not another fingerprint/equality micro-probe.
