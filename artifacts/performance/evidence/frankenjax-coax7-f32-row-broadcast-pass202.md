# frankenjax-coax7 pass202: f32 row-broadcast eval_jaxpr fusion

Commit: this commit
Bead: `frankenjax-coax7`
Follow-up bead for remaining scope: `frankenjax-lcozh`

## Target

Extend the dense F32 eval_jaxpr fusion path from same-shape operands to the
common rank-2 row-bias broadcast shape `[rows, cols] op [cols]`. This pass is
one lever only: no half-float, I64, column broadcast, general NumPy broadcast,
or Max/Min semantics.

## Baseline And Benchmark

Pre-fast-path benchmark-only baseline on RCH worker `ovh-a`:

```text
EVAL_FUSION_SPEED_F32_ROW_BROADCAST rows=1024 cols=1024 ops=8 unfused=9.920ms fused=9.063ms speedup=1.09x
```

Post-fast-path same-worker RCH result on `ovh-a`:

```text
EVAL_FUSION_SPEED_F32_ROW_BROADCAST rows=1024 cols=1024 ops=8 unfused=8.935ms fused=6.875ms speedup=1.30x
```

The scored same-worker delta is the eval_jaxpr row-broadcast path dropping from
`9.063ms` to `6.875ms`, a `1.32x` speedup. A second RCH worker (`vmi1227854`)
measured `unfused=9.519ms fused=7.478ms speedup=1.27x`; it is recorded as
supporting evidence only, not the scored comparison.

Score: `6.0` (`Impact=3`, `Confidence=4`, `Effort=2`).

Rejected intermediate: a naive I64 same-shape fusion route was measured and
removed before commit. Its RCH `vmi1227854` baseline was `unfused=8.074ms
fused=7.018ms speedup=1.15x`; the candidate regressed to `unfused=6.989ms
fused=8.633ms speedup=0.81x`. The follow-up bead should use a different I64
primitive, not repeat that route.

## Isomorphism

- Ordering preserved: each fused row-broadcast step is applied in original
  equation order for each output element.
- Tie-breaking unchanged: no comparison or tie surface exists in this pass.
- Floating point preserved: every F32 step widens operands to F64, applies the
  primitive, and rounds back to F32 after that primitive, matching the same
  contract used by the same-shape F32 fusion path.
- Broadcast indexing preserved: row operands gather by row-major column index
  `linear_index % cols`; only `[rows, cols]` with `[cols]` is accepted.
- RNG unchanged: no RNG surface.
- Fallback preserved: non-F32, non-dense, non-row-broadcast, params, effects,
  sub-jaxpr, multi-output, half, I64, Max, and Min cases stay on the normal
  interpreter path.

Golden output SHA:

```text
1f742aad15797ada82394f8d78c5b2d488ac650c272e8a81330a694621a64494
```

## Validation

```text
rustfmt --edition 2024 --check crates/fj-interpreters/src/lib.rs crates/fj-interpreters/benches/eval_fusion_speed.rs
rch exec -- cargo test -p fj-interpreters fusion_f32_row_broadcast_chain_matches_reference_bit_for_bit -- --nocapture
rch exec -- cargo check -p fj-interpreters --all-targets
rch exec -- cargo test -p fj-interpreters
rch exec -- cargo clippy -p fj-interpreters --all-targets --no-deps -- -D warnings
git diff --check -- crates/fj-interpreters/src/lib.rs crates/fj-interpreters/benches/eval_fusion_speed.rs
```

Results: format passed, focused golden proof passed, crate check passed, full
`fj-interpreters` tests passed (`148 passed`; doc-tests `0`), strict
crate-scoped clippy passed remotely, and diff whitespace check passed. UBS over
the two touched files remains nonzero on broad pre-existing test/bench
unwrap/panic/indexing inventory; its formatting, clippy, check, test-build,
audit, and deny subchecks were clean.
