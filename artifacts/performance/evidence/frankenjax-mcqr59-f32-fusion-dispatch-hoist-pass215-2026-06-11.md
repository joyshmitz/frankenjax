# frankenjax-mcqr.59 - F32 Fusion Dispatch Hoist

## Scope

- Bead: `frankenjax-mcqr.59`
- Pass: 215
- Worktree: `/data/projects/.scratch/frankenjax-peachlion-pass215-20260611T011031Z`
- Base: `439539bb`
- Lever: hoist `CheapOp` dispatch out of the F32 eval-jaxpr fusion inner loops by specializing the loop body for each binary primitive.

## Profile-Backed Target

Baseline command:

```bash
rch exec -- cargo bench -j 1 -p fj-interpreters --bench eval_fusion_speed
```

RCH selected `vmi1227854`.

Pre-lever rows:

| Row | Unfused | Fused | Speedup |
| --- | ---: | ---: | ---: |
| `EVAL_FUSION_SPEED_F32` | 10.388 ms | 7.435 ms | 1.40x |
| `EVAL_FUSION_SPEED_F32_CLAMP` | 12.022 ms | 8.598 ms | 1.40x |
| `EVAL_FUSION_SPEED_F32_ROW_BROADCAST` | 15.415 ms | 8.685 ms | 1.77x |
| `EVAL_FUSION_SPEED_F32_COL_BROADCAST` | 11.565 ms | 2.991 ms | 3.87x |

The profile target was the fused F32 tape itself: same-shape and row-broadcast F32 chains were still spending most of the candidate time in a per-element `match CheapOp` helper, despite previous broadcast-index strength reduction.

## Implementation

`crates/fj-interpreters/src/lib.rs` previously routed each F32 binary element through `f32_fused_binary(op, left, right)`. This pass removes that helper and expands the dispatch once per step helper:

- scalar-other F32 chains
- same-shape external operand chains
- row-broadcast operand chains
- column-broadcast operand chains
- chain/chain operations

Each specialized arm still performs the exact previous arithmetic expression: widen each F32 operand to F64, apply the primitive, then round the primitive result back to F32. `Max` and `Min` still call `fused_jax_max` / `fused_jax_min`.

## Benchmark Result

Rebench command:

```bash
rch exec -- cargo bench -j 1 -p fj-interpreters --bench eval_fusion_speed
```

RCH again selected `vmi1227854`.

Post-lever rows:

| Row | Unfused | Fused | Speedup | Fused delta |
| --- | ---: | ---: | ---: | ---: |
| `EVAL_FUSION_SPEED_F32` | 7.183 ms | 1.004 ms | 7.15x | 7.435 ms -> 1.004 ms, 7.41x |
| `EVAL_FUSION_SPEED_F32_CLAMP` | 8.558 ms | 2.708 ms | 3.16x | 8.598 ms -> 2.708 ms, 3.18x |
| `EVAL_FUSION_SPEED_F32_ROW_BROADCAST` | 9.190 ms | 1.001 ms | 9.18x | 8.685 ms -> 1.001 ms, 8.68x |
| `EVAL_FUSION_SPEED_F32_COL_BROADCAST` | 8.843 ms | 0.797 ms | 11.09x | 2.991 ms -> 0.797 ms, 3.75x |

Score: `12.5` (`Impact 5 * Confidence 5 / Effort 2`).

## Isomorphism Proof

- Ordering: fused tape step order, per-element row-major order, row-broadcast span partitioning, and column-broadcast span partitioning are unchanged.
- Tie-breaking: no sort, top-k, or branch priority surface is introduced; `Max` and `Min` still use the same NaN-propagating helpers.
- Floating point: every binary primitive still executes `f64::from(left) op f64::from(right)` and rounds that primitive result to `f32`. `Sub` and `Div` preserve operand orientation for chain-left and chain-right cases. Chain/chain `x - x` and `x / x` still execute, preserving NaN behavior.
- RNG: no RNG surface exists in this path.
- Fallback/error behavior: classification gates, unsupported dtype/shape fallback, arity checks, and env lookup behavior are unchanged.
- Output container: output dtype and storage class remain the same F32 tensor path.

Golden output SHA-256 values exercised by the bit-for-bit proof tests:

- Same-shape F32: `fef28624a52e5647abc35f0d388072b443cf081e5941243c6c58a8bd91f40a84`
- Row-broadcast F32: `1f742aad15797ada82394f8d78c5b2d488ac650c272e8a81330a694621a64494`
- Column-broadcast F32: `5762f3ec4614f491d21407cbb09c5cd92915840f65d145070f8d8b5e8c7c5e3a`

## Validation

```bash
rch exec -- cargo test -j 1 -p fj-interpreters --lib fusion_f32 -- --nocapture
cargo fmt -p fj-interpreters --check
git diff --check
rch exec -- cargo check -j 1 -p fj-interpreters --all-targets
rch exec -- cargo clippy -j 1 -p fj-interpreters --all-targets --no-deps -- -D warnings
rch exec -- cargo test -j 1 -p fj-interpreters
ubs crates/fj-interpreters/src/lib.rs crates/fj-interpreters/benches/eval_fusion_speed.rs
```

Results:

- Focused remote F32 fusion tests passed on `vmi1227854` (`3 passed`).
- Format and diff whitespace checks passed.
- Crate-scoped remote check passed on `vmi1227854`.
- Strict crate-scoped remote clippy passed on `vmi1227854`.
- Full `fj-interpreters` crate test suite passed on `vmi1227854`: `153 passed`, doc-tests `0 passed`.
- UBS returned nonzero on broad pre-existing inventory in the touched interpreter source and benchmark file: test/bench panic/unwrap/indexing/style heuristics. Its formatting, clippy, build, test-build, audit, and deny sections were clean; no hunk-specific defect was identified.

Dependency warnings observed during remote check/clippy/test were pre-existing warnings in `fj-trace` and `fj-lax`, outside the touched crate surface.

## Coordination

Agent Mail registration succeeded as `PeachLion`, but later Agent Mail reservation/message calls failed with a local MCP HTTP transport error. The bead was claimed directly with `br update frankenjax-mcqr.59 --status in_progress --assignee PeachLion`.
