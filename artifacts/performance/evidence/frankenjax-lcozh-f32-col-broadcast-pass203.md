# frankenjax-lcozh pass203: f32 column-broadcast eval_jaxpr fusion

## Target

- Bead: `frankenjax-lcozh`
- Crate: `fj-interpreters`
- Hot row: dense rank-2 F32 `eval_jaxpr` fusion with a column-bias operand shaped `[rows, 1]` against a full `[rows, cols]` chain.
- Constraint: one lever only. This pass handles F32 column broadcast. Half, I64, general broadcast, scalar-tensor broadcast, and Max/Min remain out of scope.

## Baseline

RCH same-worker baseline on `ovh-a`, before adding the column-broadcast fast path but after adding the benchmark-only column row:

```text
EVAL_FUSION_SPEED_F32_COL_BROADCAST rows=1024 cols=1024 ops=8 unfused=9.054ms fused=8.677ms speedup=1.04x
```

The scored path is the existing `eval_jaxpr` fused dispatcher row, `8.677ms`.

## Candidate

RCH same-worker candidate on `ovh-a` after adding `F32Operand::ColBroadcast`:

```text
EVAL_FUSION_SPEED_F32_COL_BROADCAST rows=1024 cols=1024 ops=8 unfused=8.958ms fused=6.940ms speedup=1.29x
```

Scored `eval_jaxpr` path delta: `8.677ms -> 6.940ms`, a `1.25x` speedup.

Score: `Impact 3 * Confidence 4 / Effort 2 = 6.0`, keep.

## Lever

`F32Operand::ColBroadcast { idx, cols }` classifies a dense F32 operand shaped `[rows, 1]` against a full chain shape `[rows, cols]`. The chunk evaluator gathers the column operand by row-major row index:

```text
row = linear_index / cols
```

All unsupported shapes, dtypes, params, effects, sub-jaxprs, multi-output equations, half/I64, Max/Min, and general broadcast patterns keep the existing fallback path.

## Isomorphism Proof

- Equation order is unchanged.
- Per-element primitive order is unchanged.
- F32 primitive semantics preserve the existing contract: widen to F64, apply Add/Sub/Mul/Div/Neg, round back to F32 after each primitive.
- Broadcast ordering is row-major and matches the forced-unfused interpreter: an operand shaped `[rows, 1]` contributes element `linear_index / cols`.
- Signed zero, infinities, subnormals, and NaN payload coverage are included in the focused proof test.
- No RNG, tie-breaking, associative reassociation, dtype change, shape change, or error-surface change is introduced.

Golden output SHA256 from the forced-unfused reference bits:

```text
5762f3ec4614f491d21407cbb09c5cd92915840f65d145070f8d8b5e8c7c5e3a
```

## Validation

- `rustfmt --edition 2024 --check crates/fj-interpreters/src/lib.rs crates/fj-interpreters/benches/eval_fusion_speed.rs`
- Focused proof: `fusion_f32_col_broadcast_chain_matches_reference_bit_for_bit`
- `rch exec -- cargo check -p fj-interpreters --all-targets`
- `rch exec -- cargo test -p fj-interpreters`
- `rch exec -- cargo clippy -p fj-interpreters --all-targets --no-deps -- -D warnings`
- Scoped `git diff --check`
- `ubs crates/fj-interpreters/src/lib.rs crates/fj-interpreters/benches/eval_fusion_speed.rs` remains nonzero on pre-existing broad test/bench inventory, while its fmt/check/clippy/test-build/audit/deny sections were clean.

## Next Primitive

Reprofile after landing. The remaining `frankenjax-lcozh` families are split to a follow-up bead: half/BF16 fusion, I64 only with a different primitive than the rejected direct route, and broader broadcast patterns after a fresh baseline.
