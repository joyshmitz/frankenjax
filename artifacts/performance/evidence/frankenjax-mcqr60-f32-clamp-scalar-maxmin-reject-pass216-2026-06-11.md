# frankenjax-mcqr.60 - F32 Clamp Scalar Max/Min Reject

## Scope

- Bead: `frankenjax-mcqr.60`
- Pass: 216
- Worktree: `/data/projects/.scratch/frankenjax-peachlion-pass215-20260611T011031Z`
- Base: `6c974e10`
- Candidate: specialize F32 scalar `Max`/`Min` fusion when the scalar operand is non-NaN, hoisting the scalar `is_nan()` branch out of the element loop.

## Baseline

Command:

```bash
rch exec -- cargo bench -j 1 -p fj-interpreters --bench eval_fusion_speed
```

RCH selected `vmi1227854`.

Relevant baseline rows:

| Row | Unfused | Fused | Speedup |
| --- | ---: | ---: | ---: |
| `EVAL_FUSION_SPEED_F32` | 7.724 ms | 1.200 ms | 6.44x |
| `EVAL_FUSION_SPEED_F32_CLAMP` | 9.849 ms | 2.700 ms | 3.65x |
| `EVAL_FUSION_SPEED_F32_ROW_BROADCAST` | 11.803 ms | 1.312 ms | 8.99x |
| `EVAL_FUSION_SPEED_F32_COL_BROADCAST` | 11.599 ms | 1.207 ms | 9.61x |

## Candidate

The candidate added `f32_fused_max_scalar` and `f32_fused_min_scalar` helpers and changed only the `F32Operand::Scalar` arms for `CheapOp::Max` and `CheapOp::Min`.

Behavior proof passed:

```bash
rch exec -- cargo test -j 1 -p fj-interpreters --lib fusion -- --nocapture
```

Result on `vmi1227854`: `9 passed`, including `fusion_max_min_abs_chain_matches_reference_bit_for_bit` and the F32 same-shape/row/column broadcast golden tests.

Isomorphism:

- Fused tape order, row-major order, and fallback gates were unchanged.
- The candidate preserved `f64::from(value)` then `max`/`min` then round to F32.
- Scalar-NaN behavior filled canonical F32 NaN, matching the old `fused_jax_max/min(..., scalar_nan) as f32` result in the proof tests.
- Chain-operand NaN behavior remained NaN-propagating.
- No RNG or tie-breaking surface was introduced.

## Rebench And Decision

Rebench command:

```bash
rch exec -- cargo bench -j 1 -p fj-interpreters --bench eval_fusion_speed
```

RCH selected `vmi1227854` again.

Candidate rows:

| Row | Baseline fused | Candidate fused | Delta |
| --- | ---: | ---: | ---: |
| `EVAL_FUSION_SPEED_F32` | 1.200 ms | 1.619 ms | 0.74x |
| `EVAL_FUSION_SPEED_F32_CLAMP` | 2.700 ms | 3.243 ms | 0.83x |
| `EVAL_FUSION_SPEED_F32_ROW_BROADCAST` | 1.312 ms | 1.204 ms | 1.09x |
| `EVAL_FUSION_SPEED_F32_COL_BROADCAST` | 1.207 ms | 0.922 ms | 1.31x |

Target row regressed, so Score `0.0`. The production source hunk was restored before commit.

## Next Route

Do not repeat scalar `Max`/`Min` NaN-hoisting for F32 clamp. The next useful interpreter target should come from a fresh profile and likely a different primitive family: I64 fusion throughput, F64 broadcast fusion, or an algorithmic/data-layout route if those remain dominant.
