# frankenjax-zhxk0 compiled dense value_and_grad evidence

Date: 2026-06-15
Agent: SilverMaple
Bead: frankenjax-zhxk0

## Target

Repeated public `value_and_grad` calls for the dense F64 `sum(x*x + x)` tensor
Jaxpr fell back through the generic AD/dispatch path even though the direct AD
core already had a dense value-and-gradient fast path. The target was a narrow
compiled reverse plan for this exact effect-free three-equation graph.

## Lever

`CompiledValueAndGradJaxpr` now has two variants:

- existing scalar-F64 reverse plan, unchanged for pass258 scalar programs
- dense F64 `Mul(input,input) -> Add(square,input) -> ReduceSum` plan

The dense plan keeps the original forward equation order:

1. `squared = x * x`
2. `shifted = squared + x`
3. `output += shifted`

The returned gradient uses the already-proven lazy dense buffer
`LiteralBuffer::from_f64_one_plus_x_plus_x`, which materializes each element as
`1.0; += x; += x`. Runtime guards require one packed F64 tensor argument; other
representations return `None` and use the existing fallback path. Dynamic
custom-VJP checks still run after cache warm for the function fingerprint and
for `Add`, `Mul`, and `ReduceSum`.

## Baseline

Pre-edit RCH context:

- `fj-api value_and_grad_runtime/shared/square_plus_linear`: `[2.1340us 2.1356us 2.1422us]`
- `fj-api value_and_grad_runtime/shared/deep_100_nodes`: `[5.0631us 5.0641us 5.0643us]`
- `fj-ad ad/grad_sum_x2_plus_x_1k`: `[132.10ns 133.44ns 133.77ns]`
- scalar compiled reverse baseline on `vmi1152480`:
  - `tape/value_and_grad_trig`: `[1.1490us 1.1621us 1.1654us]`
  - `compiled/value_and_grad_trig`: `[171.60ns 171.97ns 173.45ns]`
  - `api_warmed/value_and_grad_trig`: `[334.52ns 337.73ns 350.57ns]`

The exact tensor API row did not exist before this pass; the same-binary
post-edit benchmark includes the old direct AD route as the baseline arm.

## Post-change benchmark

Command:

```bash
rch exec -- cargo bench -j 1 -p fj-api --bench api_overhead -- ad_compiled_reverse_plan --quick --noplot
```

Worker: `ovh-a`

Results:

- `direct/value_and_grad_sum_x2_plus_x_1k`: `[1.9131us 1.9537us 1.9639us]`
- `compiled/value_and_grad_sum_x2_plus_x_1k`: `[758.34ns 758.65ns 758.73ns]`
- `api_warmed/value_and_grad_sum_x2_plus_x_1k`: `[844.92ns 845.00ns 845.35ns]`

Public API midpoint ratio: `1.9537us / 845.00ns = 2.31x`.
Conservative interval ratio: `1.9131us / 845.35ns = 2.26x`.

Score: `Impact 2.26 * Confidence 0.95 / Effort 0.75 = 2.86`.

## Proof

Focused RCH tests:

```bash
rch exec -- cargo test -j 1 -p fj-ad dense_f64_square_plus_linear_reducesum -- --nocapture
rch exec -- cargo test -j 1 -p fj-ad compiled_scalar_f64_reverse_plan_matches_generic_bits -- --nocapture
```

Results:

- dense suite: 3 passed
- scalar compiled regression: 1 passed

Dense proof covers:

- compiled dense value-and-grad equals forced generic bits for `0.0`, `-0.0`,
  finite values, `inf`, and a NaN payload
- gradient dtype, shape, element order, and bits match generic
- dynamic shape is carried from the runtime tensor; a length-3 call returns a
  length-3 gradient after the length-1024 call
- scalar, non-F64 tensor, and non-packed F64 tensor inputs return `None`
- warmed compiled plan defers after registering a custom `ReduceSum` VJP
- 1024-element gradient golden SHA remains
  `5282853e2bd187c1c1bfdfa612bd74776fb403e6b767eb0a8bf0c8bcd2fe2a19`

Isomorphism:

- equation order is unchanged
- reverse derivative formula is the existing `1 + x + x`
- FP operation order is preserved for value and gradient materialization
- tensor dtype and runtime shape are preserved
- no tie-breaking, ordering, RNG, or backend behavior surface changes
- unsupported values and custom derivative state fall back to the existing path

## Validation

```bash
cargo fmt --check
rch exec -- cargo check -j 1 -p fj-ad --all-targets
rch exec -- cargo check -j 1 -p fj-api --all-targets
rch exec -- cargo clippy -j 1 -p fj-ad --all-targets -- -D warnings
rch exec -- cargo clippy -j 1 -p fj-api --all-targets -- -D warnings
```

All passed.
