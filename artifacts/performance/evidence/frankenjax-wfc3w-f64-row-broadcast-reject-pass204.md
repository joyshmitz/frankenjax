# frankenjax-wfc3w pass204: f64 row-broadcast direct fusion rejected

## Target

- Bead: `frankenjax-wfc3w`
- Crate: `fj-interpreters`
- Candidate: extend the existing F64 eval_jaxpr elementwise fusion scanner to accept rank-2 row broadcast operands shaped `[cols]` against full operands shaped `[rows, cols]`.

## Baseline

Benchmark-only row, before adding any F64 row-broadcast source fast path:

```text
RCH worker: ovh-a
EVAL_FUSION_SPEED_F64_ROW_BROADCAST rows=1024 cols=1024 ops=8 unfused=7.626ms fused=5.462ms speedup=1.40x
```

## Candidate

The source candidate was bit-clean under the focused forced-unfused proof, with golden SHA:

```text
ea879236c0bd2823d6161300f66a8a1b0e7122bc6ebc4ad41fa21c2d3bc950e5
```

RCH ignored the `ovh-a` hint for the after benchmark and selected `vmi1227854`. That row was not comparable to the `ovh-a` baseline and was also directionally bad:

```text
RCH worker: vmi1227854
EVAL_FUSION_SPEED_F64_ROW_BROADCAST rows=1024 cols=1024 ops=8 unfused=6.523ms fused=8.687ms speedup=0.75x
```

## Decision

Rejected. Score `0.0`.

The direct per-element row gather (`(base + offset) % row.len()`) is not a keepable primitive on the available evidence. The source and benchmark changes were removed before commit.

## Next Primitive

Do not repeat the same F64 row-gather micro-route. Continue `frankenjax-wfc3w` with a different primitive: likely packed-u16 BF16/F16 same-shape fusion, or a broadcast plan that removes the per-element modulus rather than adding it to the hot loop.
