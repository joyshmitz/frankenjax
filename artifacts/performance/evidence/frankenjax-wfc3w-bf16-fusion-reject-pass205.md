# frankenjax-wfc3w pass205: BF16/F16 packed-u16 eval_jaxpr fusion rejected

## Target

- Bead: `frankenjax-wfc3w`
- Crate: `fj-interpreters`
- Hotspot family: remaining `eval_jaxpr` cheap-elementwise fusion primitives after F32 same-shape, row-broadcast, and column-broadcast shipped.
- Candidate primitive: same-shape BF16/F16 chains over packed `u16` tensor storage.

## Baseline

Benchmark-only BF16 row was added before any half-fusion source change and run through RCH on `ovh-a`:

```text
EVAL_FUSION_SPEED_BF16 n=1048576 ops=8 unfused=44.743ms fused=43.336ms speedup=1.03x
```

The `fused` label is the normal `eval_jaxpr` path before a half fast path; it is effectively still per-equation for BF16.

## Lever Tried

The rejected candidate added a strict same-shape half-float sibling to the existing f64/f32 fusion scanner:

- accepted only dense BF16 or dense F16 tensors
- required one half dtype throughout the run
- rejected BF16/F16 mixing and all non-half operands
- rejected shape mismatch, params, effects, sub-jaxprs, multi-output equations, and unsupported primitives
- widened each packed `u16` through `Literal::{BF16,F16}Bits(...).as_f64()`
- applied Add/Sub/Mul/Div/Neg in original equation order
- rounded after every primitive through `Literal::from_{bf16,f16}_f64`
- materialized the final output with `TensorValue::new_half_float_values`

## Isomorphism Proof

The focused proof compared fused `eval_jaxpr` against forced-unfused `eval_jaxpr_hashed_env` for BF16 and F16 chains with signed zero, infinities, NaN payloads, and subnormal coverage.

RCH proof command:

```text
rch exec -- cargo test -p fj-interpreters fusion_half_chain_matches_reference_bit_for_bit -- --nocapture
```

Result on `ovh-a`:

```text
test tests::fusion_half_chain_matches_reference_bit_for_bit ... ok
```

Golden output digests:

- BF16: `46926fd8cdda6d8c45a18cc3178dbb5ff3fa2fdd2021e2d985199e0f01218cf6`
- F16: `3dceb054f3551c13f4650094c80289ed81187534813220a0dbcc7ba34475efab`

Ordering/tie/RNG/floating-point notes:

- equation order unchanged
- per-element primitive order unchanged
- half widening and rounding matched the existing `fj-lax` half arithmetic contract
- no RNG surface introduced
- no tie-breaking surface introduced
- fallback/error behavior unchanged for all guarded-out cases

## Rebenchmark

Candidate rebench used the same worker as baseline (`ovh-a`):

```text
EVAL_FUSION_SPEED_BF16 n=1048576 ops=8 unfused=44.075ms fused=57.110ms speedup=0.77x
```

Scored path delta:

- baseline normal `eval_jaxpr`: `43.336ms`
- candidate half-fused `eval_jaxpr`: `57.110ms`
- speed ratio: `0.76x`
- Score: `0.0`

## Decision

Rejected. Source, test, and benchmark scaffolding were removed before commit.

Do not repeat the per-step packed-half widen/apply/round fusion route. The correctness proof is clean, but the primitive is the wrong shape for performance because it replaces the existing vectorized per-primitive half loops with a scalarized per-element tape that performs repeated half widen/round work inside one pass.

Next pass should attack a different primitive: either a profile-backed non-half eval-fusion residual, or a fundamentally different half strategy that reduces conversion count rather than merely moving the existing half arithmetic into the eval_jaxpr fusion tape.
