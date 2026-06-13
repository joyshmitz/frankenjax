# frankenjax-sltet: FMA GEMM Policy Gate

Date: 2026-06-13
Agent: BeigeMouse
Worker: `vmi1152480`
Bead: `frankenjax-sltet`
Result: rejected/deferred; no source lever kept

## Target

`frankenjax-sltet` proposed a separate FMA-contracted f64 GEMM variant for
tolerance-checked linalg trailing updates. The production GEMM path is
deliberately FMA-free because user-facing dot/conv paths pin bit identity to
the scalar `a * b` then `+` fold.

This pass only rechecked whether the committed FMA evidence harness is
shippable under the current project build policy. It did not edit
`tensor_contraction.rs`, `linalg.rs`, `.cargo/config.toml`, or any production
source.

## Benchmark Gate

Command:

```text
rch exec -- cargo test -j 1 -p fj-lax --lib bench_fma_vs_nonfma_matmul --release -- --ignored --nocapture
```

RCH selected worker `vmi1152480`.

Output:

```text
BENCH matmul 256x256 f64: non_fma=1.126ms (29.8 GFLOP/s) fma=47.309ms (0.7 GFLOP/s) speedup=0.02x
BENCH matmul 512x512 f64: non_fma=12.227ms (22.0 GFLOP/s) fma=397.447ms (0.7 GFLOP/s) speedup=0.03x
test cz0g0_fma_evidence::tests::bench_fma_vs_nonfma_matmul ... ok
```

Score: `0.0` because the candidate is a 33x to 42x regression under the
current default build, far below the `Score >= 2.0` keep gate.

## Behavior Proof

Command:

```text
rch exec -- cargo test -j 1 -p fj-lax --lib fma_kernel_within_tolerance --release -- --nocapture
```

RCH selected the same worker, `vmi1152480`.

Output:

```text
test cz0g0_fma_evidence::tests::fma_kernel_within_tolerance ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1449 filtered out
```

## Isomorphism

No production source was changed. The existing production GEMM still uses
separate multiply and add operations in the existing row-major traversal, so
per-output accumulation order, tie-breaking surface, floating-point rounding
surface, dtype/shape/error behavior, and RNG absence are unchanged.

The FMA evidence harness proves the alternate FMA arithmetic stays within the
existing JAX/XLA tolerance contract, but it also documents that FMA changes bit
rounding. Under the current `.cargo/config.toml` policy (`+avx2`, no `+fma`),
`mul_add` does not become a useful production FMA kernel and instead regresses
catastrophically.

Golden evidence payload SHA-256:

```text
e5715cfb1cf53a254d8fb382bd40b5f06c32294e745f382692eeae35e305ad89
```

Payload:

```text
frankenjax-sltet fma policy gate 2026-06-13 vmi1152480
bench release fj-lax cz0g0_fma_evidence::bench_fma_vs_nonfma_matmul
256 non_fma=1.126ms fma=47.309ms speedup=0.02x
512 non_fma=12.227ms fma=397.447ms speedup=0.03x
proof fma_kernel_within_tolerance ok
source unchanged; .cargo/config.toml remains avx2-only no +fma
```

## Decision

Close `frankenjax-sltet` as rejected/deferred. The only route to this specific
FMA lever is a maintainer-level global `+fma` policy change plus full
conformance revalidation. Until then, the next performance work should attack a
different algorithmic primitive rather than repeating `mul_add` under the
current build flags.
