# V1 Parity Report

Mode: `strict`

FrankenJAX: `0.1.0` | Oracle: `jax-0.9.0.1`

Audit: **2026-03-04** (cargo test --workspace: 1532 tests, 0 failures)

## Summary

| Metric | Value |
|---|---|
| Total Conformance Cases | 242 |
| Matched | 242 |
| Mismatched | 0 |
| Pass Rate | 100.00% |
| Gate | **pass** |
| RNG Determinism Cases | 20+ |
| RNG Determinism Pass Rate | 100.00% |
| Total Workspace Tests | 1532 |

## Per-Family Breakdown

| Family | Total | Matched | Mismatched |
|---|---|---|---|
| jit | 49 | 49 | 0 |
| grad | 19 | 19 | 0 |
| vmap | 9 | 9 | 0 |
| lax | 165 | 165 | 0 |
| random | 20+ | 20+ | 0 |

## Per-Primitive Breakdown

| Primitive | Total | Matched | Mismatched |
|---|---|---|---|
| Add2 | 8 | 8 | 0 |
| AddOne | 12 | 12 | 0 |
| CosX | 10 | 10 | 0 |
| Dot3 | 4 | 4 | 0 |
| LaxAbs | 6 | 6 | 0 |
| LaxAcos | 5 | 5 | 0 |
| LaxAsin | 5 | 5 | 0 |
| LaxAtan | 6 | 6 | 0 |
| LaxAtan2 | 4 | 4 | 0 |
| LaxCeil | 5 | 5 | 0 |
| LaxClamp | 4 | 4 | 0 |
| LaxCosh | 4 | 4 | 0 |
| LaxDiv | 4 | 4 | 0 |
| LaxErf | 5 | 5 | 0 |
| LaxErfc | 5 | 5 | 0 |
| LaxExp | 4 | 4 | 0 |
| LaxExpm1 | 5 | 5 | 0 |
| LaxFloor | 5 | 5 | 0 |
| LaxLog | 5 | 5 | 0 |
| LaxLog1p | 5 | 5 | 0 |
| LaxLogistic | 5 | 5 | 0 |
| LaxMax | 4 | 4 | 0 |
| LaxMin | 4 | 4 | 0 |
| LaxMul | 4 | 4 | 0 |
| LaxNeg | 6 | 6 | 0 |
| LaxPow | 3 | 3 | 0 |
| LaxReciprocal | 5 | 5 | 0 |
| LaxReduceMax | 3 | 3 | 0 |
| LaxReduceMin | 3 | 3 | 0 |
| LaxReduceProd | 3 | 3 | 0 |
| LaxRem | 4 | 4 | 0 |
| LaxRound | 5 | 5 | 0 |
| LaxRsqrt | 5 | 5 | 0 |
| LaxSign | 5 | 5 | 0 |
| LaxSinh | 4 | 4 | 0 |
| LaxSqrt | 5 | 5 | 0 |
| LaxSquare | 6 | 6 | 0 |
| LaxSub | 4 | 4 | 0 |
| LaxTan | 5 | 5 | 0 |
| LaxTanh | 5 | 5 | 0 |
| ReduceSumVec | 4 | 4 | 0 |
| SinX | 13 | 13 | 0 |
| Square | 16 | 16 | 0 |
| SquarePlusLinear | 10 | 10 | 0 |

## V2 Feature Completion Status

| Feature | Status | Evidence |
|---|---|---|
| DType system (11 types) | parity_green | BF16/F16/F32/F64/I32/I64/U32/U64/Bool/Complex64/Complex128 + promotion rules |
| AD engine (68/77 VJP+JVP) | parity_green | Tape-based reverse mode, custom_vjp/jvp, Jacobian/Hessian, value_and_grad |
| Vmap BatchTrace | parity_green | Per-primitive batching rules, in_axes/out_axes, axis resolution |
| RNG ThreeFry2x32 | parity_green | key/split/fold_in/uniform/normal/bernoulli/categorical, fixture-backed |
| Control flow | parity_green | cond/scan/while_loop/fori_loop/switch with sub_jaxprs |
| Tracing | parity_green | make_jaxpr/make_jaxpr_fallible, nested trace contexts |
| E-graph optimizer | parity_green | 80+ rewrite rules, equality saturation, wired into dispatch |
| Effects system | parity_green | Effects on Jaxpr+Equation, EffectContext token threading |
| CPU parallel backend | parity_green | Dependency-wave parallel executor |
| Linalg/FFT stubs | parity_gap | Cholesky/Qr/Svd/TriangularSolve/Eigh/Fft/Ifft/Rfft/Irfft (eval stubs only) |

## Coverage Exceptions

- `linalg/fft`: 9 operations have eval stubs but no VJP/JVP rules (tracked in bd-3w2j)

## Parity Exceptions

None.
