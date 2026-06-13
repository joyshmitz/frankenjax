# frankenjax-lu86j direct-threaded 3x3 f32 conv2d rejection

Date: 2026-06-13
Agent: BeigeMouse
Crate: fj-lax

## Target

`frankenjax-lu86j` tracks the 3x3 stride-1 conv2d gap. A prior F(2x2,3x3)
Winograd implementation was reverted after regressing because it split one large
packed im2col GEMM into sixteen small unpacked GEMMs plus serial transforms.

This pass tested a different non-Winograd subroute: bypass im2col for ordinary
valid 3x3 same-dtype f32-family conv2d and split independent output elements
across `conv_morsel_threads`, preserving each output element's scalar
`kh, kw, ci` accumulation order.

## Baseline

Command:

```text
RCH_WORKER=hz1 RCH_WORKERS=hz1 RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo test -j 1 -p fj-lax --lib bench_f32_conv2d --release -- --ignored --nocapture
```

RCH selected worker `vmi1152480`.

Rows:

- `[8,32,32,16] * [3,3,16,32]`: `2.1209ms`
- `[4,28,28,32] * [3,3,32,64]`: `1.8446ms`

## Candidate

The candidate compiled and passed the focused behavior proof before the
performance gate:

```text
RCH_WORKER=vmi1152480 RCH_WORKERS=vmi1152480 RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo test -j 1 -p fj-lax --lib \
  conv2d_f32_valid_3x3_threaded_direct_matches_reference_and_golden -- --nocapture
```

Proof result: passed.

Golden digest from the candidate proof:

```text
e0deb36c857ef890b58a2049b039cf74834414aed9180ccece8523746cb5542f
```

Isomorphism notes:

- Ordering: each output element used the same `kh, kw, ci` order as the scalar
  f32 reference and the existing f32 im2col/GEMM contract.
- Tie-breaking/RNG: no comparisons or RNG.
- Floating point: no FMA or reassociation was introduced; each output kept the
  exact scalar f32 multiply/add sequence. Threading only partitioned independent
  output elements.

## Rebench

Command:

```text
RCH_WORKER=vmi1152480 RCH_WORKERS=vmi1152480 RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo test -j 1 -p fj-lax --lib bench_f32_conv2d --release -- --ignored --nocapture
```

Rows:

- `[8,32,32,16] * [3,3,16,32]`: `9.0729ms` (`0.234x` vs baseline)
- `[4,28,28,32] * [3,3,32,64]`: `12.4393ms` (`0.148x` vs baseline)

Score: `0.0`, reject.

## Decision

No source hunk or proof test was kept. The result confirms the current packed
im2col GEMM is much stronger than per-output direct threading on these CPU
shapes.

Next route for `frankenjax-lu86j`: do not repeat direct-threaded scalar output
loops or the reverted F(2,3) implementation. The next candidate needs a deeper
primitive: F(4x4,3x3) with fused/packed transform GEMM, persistent transform
parallelism, or a design that keeps the sixteen Winograd position matrices
packed enough to beat the single im2col GEMM.
