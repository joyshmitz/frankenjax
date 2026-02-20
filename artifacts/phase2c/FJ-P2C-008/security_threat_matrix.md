# FJ-P2C-008 Security & Compatibility Threat Matrix: LAX Primitive First Wave

## Packet Scope

LAX primitive evaluation — the core `eval_primitive` dispatch in `fj-lax` that evaluates 35 primitive operations (25 fully implemented, 2 unsupported: Gather, Scatter). This packet covers arithmetic, transcendental, comparison, logical, reduction, and shape manipulation primitives. All code is `#![forbid(unsafe_code)]` — no memory safety concerns, but semantic correctness threats can cause silent data corruption.

---

## Threat Matrix

| # | Threat | Severity | Likelihood | Residual Risk | Mitigation | Anchor Ref |
|---|--------|----------|------------|---------------|------------|------------|
| T1 | **Type confusion in binary ops**: applying integer operation to float inputs or vice versa | High | Medium | Low | `infer_dtype()` inspects actual element values via `Literal` variant matching. Binary ops dispatch to int or float path based on inferred dtype. Type mismatch returns EvalError. | P2C008-A28 |
| T2 | **Silent NaN propagation failure**: arithmetic or reduction suppresses NaN instead of propagating | High | Low | Negligible | Rust `f64` operations follow IEEE 754 by specification. No custom NaN filtering code exists. All arithmetic (add, sub, mul, div, pow) and transcendental (exp, log, sqrt) ops propagate NaN through standard f64 semantics. | P2C008-A30 |
| T3 | **Integer overflow in reduction**: ReduceSum/ReduceProd of large i64 values wraps silently | Medium | Medium | Medium | V1: Rust i64 arithmetic wraps (2's complement) matching JAX numpy behavior on 64-bit integers. Hardened mode logs overflow events. No panic or error — wrapping is the correct behavior for numpy-compatible semantics. | P2C008-A25, P2C008-A28 |
| T4 | **Shape mismatch in binary ops**: tensors with different shapes passed to elementwise op | High | Medium | Low | `eval_binary_elementwise` validates shape match for Tensor+Tensor case. Mismatch returns `EvalError::ShapeMismatch`. Only scalar-tensor broadcast is supported — no implicit rank expansion. | P2C008-A29 |
| T5 | **Reshape to incompatible size**: product(new_shape) != product(old_shape) | High | Medium | Low | `eval_reshape` validates total element count before proceeding. Mismatch returns `EvalError` with old and new shapes in message. No partial reshape possible. | P2C008-A20 |
| T6 | **Transpose with invalid permutation**: permutation length != rank or contains duplicates/out-of-range indices | High | Low | Low | `eval_transpose` validates permutation length matches input rank. Out-of-range indices cause index-out-of-bounds error. Duplicates cause wrong output shape (detected by downstream shape checks). | P2C008-A21 |
| T7 | **Reduction over invalid axis**: axis index >= rank of input tensor | High | Low | Low | Axis validation in `eval_reduce`. Out-of-range axis causes index-out-of-bounds panic (converted to EvalError in interpreter). | P2C008-A25 |
| T8 | **Division by zero**: integer division (0/0) or modular division by zero | Medium | Medium | Low | Float: 0.0/0.0 → NaN, x/0.0 → +/-Inf (IEEE 754). Integer: Rust i64 division by zero panics (caught by interpreter error handling). | P2C008-A30 |
| T9 | **Transcendental domain errors**: log(0), log(negative), sqrt(negative) | Low | Medium | Negligible | All produce correct IEEE 754 results: log(0)=-Inf, log(-x)=NaN, sqrt(-x)=NaN. Hardened mode logs warnings. No panic. | P2C008-A08, P2C008-A09 |
| T10 | **Wrong arity**: calling unary op with 2 inputs or binary op with 1 input | Medium | Low | Negligible | Arity check at top of each eval function. Returns `EvalError::ArityMismatch { primitive, expected, actual }`. | P2C008-A28 |
| T11 | **Adversarial NaN injection**: crafted NaN payload in significand bits affects downstream logic | Low | Low | Negligible | All NaN values are equivalent in Rust f64 comparison semantics. No NaN payload inspection occurs. f64::is_nan() returns true for all NaN variants. | P2C008-A30 |
| T12 | **Dot product dimension mismatch**: inner dimensions of matrix multiply don't match | High | Medium | Low | `eval_dot` validates that contraction dimensions match. Mismatch returns EvalError. | P2C008-A19 |

---

## Compatibility Envelope

| Row | Feature | JAX Behavior | FrankenJAX V1 | Divergence | Risk |
|-----|---------|-------------|---------------|------------|------|
| CE1 | add_p | Elementwise add, full broadcast | Elementwise add, scalar-tensor broadcast only | No rank expansion | Low |
| CE2 | sub_p | Elementwise sub, full broadcast | Elementwise sub, scalar-tensor broadcast only | No rank expansion | Low |
| CE3 | mul_p | Elementwise mul, full broadcast | Elementwise mul, scalar-tensor broadcast only | No rank expansion | Low |
| CE4 | div_p | Elementwise div, full broadcast | Elementwise div, scalar-tensor broadcast only | No rank expansion | Low |
| CE5 | neg_p | Unary negate | Unary negate | Identical | None |
| CE6 | abs_p | Unary absolute value | Unary absolute value | Identical | None |
| CE7 | max_p | Elementwise max, broadcast | Elementwise max, scalar-tensor only | No rank expansion | Low |
| CE8 | min_p | Elementwise min, broadcast | Elementwise min, scalar-tensor only | No rank expansion | Low |
| CE9 | pow_p | Elementwise power | Elementwise power | Identical | None |
| CE10 | exp_p | Elementwise e^x | Elementwise e^x | Identical | None |
| CE11 | log_p | Elementwise ln(x) | Elementwise ln(x) | Identical | None |
| CE12 | sqrt_p | Elementwise sqrt | Elementwise sqrt | Identical | None |
| CE13 | sin_p / cos_p / tanh_p | Trig and hyperbolic | Trig and hyperbolic | Identical | None |
| CE14 | floor_p / ceil_p / round_p | Rounding ops | Rounding ops | Identical | None |
| CE15 | sign_p | Sign function | Sign function | Identical | None |
| CE16 | eq_p / ne_p / lt_p / le_p / gt_p / ge_p | Comparison → bool | Comparison → bool | Identical | None |
| CE17 | not_p / and_p / or_p | Logical ops | Logical ops | Identical | None |
| CE18 | select_p | Conditional select | Conditional select | Identical | None |
| CE19 | dot_general_p | General dot with contraction dims | dot_p with vector/matrix multiply | Subset (no batch dims, no arbitrary contraction) | Medium |
| CE20 | reshape_p | Arbitrary reshape | Reshape with -1 inference | Identical | None |
| CE21 | transpose_p | Arbitrary permutation | Arbitrary permutation | Identical | None |
| CE22 | broadcast_in_dim_p | Broadcast with dim mapping | broadcast_p with scalar broadcast only | Subset (no general broadcast_in_dim) | Medium |
| CE23 | reduce_sum_p | Sum over axes | Sum over single axis | Subset (no multi-axis reduction) | Low |
| CE24 | reduce_prod_p | Product over axes | Product over single axis | Subset (no multi-axis reduction) | Low |
| CE25 | reduce_max_p / reduce_min_p | Max/min over axes | Max/min over single axis | Subset (no multi-axis reduction) | Low |
| CE26 | gather | General gather (slicing, indexing) | Not supported (Unsupported error) | Missing — JAX programs using gather will error | High |
| CE27 | scatter | General scatter (updates, adds) | Not supported (Unsupported error) | Missing — JAX programs using scatter will error | High |
| CE28 | convert_element_type_p | Arbitrary dtype casts | convert_p for F64/I64/Bool only | Subset (3 dtypes only) | Medium |
| CE29 | concatenate_p | Concat along axis | concatenate_p | Identical | None |
| CE30 | Type promotion | numpy-style lattice (many dtypes) | I64+I64→I64, any-float→F64, Bool→partner | Strict subset (3 dtypes only) | Low |

---

## Fail-Closed Rules

| # | Rule | Trigger | Action |
|---|------|---------|--------|
| FC1 | Wrong arity | eval_primitive called with wrong input count | Return EvalError::ArityMismatch |
| FC2 | Shape mismatch in binary ops | Tensor+Tensor with different shapes | Return EvalError::ShapeMismatch |
| FC3 | Reshape size mismatch | product(new_shape) != product(old_shape) | Return EvalError with shape details |
| FC4 | Unsupported primitive | Gather or Scatter encountered | Return EvalError::Unsupported |
| FC5 | Unsupported dtype | Convert to type not in {F64, I64, Bool} | Return EvalError |
| FC6 | Dot dimension mismatch | Inner dimensions don't match for matrix multiply | Return EvalError |
| FC7 | Transpose invalid permutation | Permutation length != input rank | Return EvalError |
| FC8 | Reduction invalid axis | Axis >= input rank | Return EvalError (index out of bounds) |
| FC9 | Integer division by zero | i64 / 0 or i64 % 0 | Rust panic caught by interpreter |
| FC10 | Empty tensor reduction | Reduce over axis with 0 elements | Return identity element (sum→0, prod→1, max→MIN, min→MAX) |

---

## Safe Code Audit Notes

Unlike FJ-P2C-007 (FFI), this packet contains **zero unsafe code**. All primitives are implemented using safe Rust:

1. **`#![forbid(unsafe_code)]`** on fj-lax crate — compiler-enforced
2. All arithmetic uses Rust's safe f64/i64 operators
3. All tensor indexing uses bounds-checked Vec indexing
4. No raw pointers, no transmute, no manual memory management
5. Type dispatch via Rust enum matching — exhaustive and compiler-checked
6. Shape computation is pure function with no mutable state

### Security-Relevant Design Decisions

- **No implicit broadcast**: Only scalar-tensor broadcast is supported. Tensor-tensor operations require exact shape match. This eliminates a class of silent shape-mismatch bugs.
- **No implicit type widening**: Types are inferred from actual values, not declared. Binary ops on mixed types use explicit promotion rules (I64+F64→F64).
- **Fail-closed on unknown primitives**: The `Primitive` enum is exhaustive. Any match arm not handled is a compile error.
- **IEEE 754 compliance**: All floating-point edge cases (NaN, Inf, -0.0) follow IEEE 754 through Rust's f64 implementation.
