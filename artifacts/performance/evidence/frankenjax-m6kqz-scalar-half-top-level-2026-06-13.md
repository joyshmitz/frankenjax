# frankenjax-m6kqz: top-level scalar half arithmetic fast path

## Target

- Bead: `frankenjax-m6kqz`
- Crate: `fj-interpreters`
- Profile-backed target: top current `fj-interpreters` eval rows after pass240
- Baseline worker: `vmi1227854`
- Baseline command:

```bash
RCH_WORKERS=vmi1227854 RCH_REQUIRE_REMOTE=1 rch exec -- \
  cargo bench -j 1 -p fj-interpreters --bench pe_baseline -- eval/
```

Current post-pass240 profile:

```text
eval/scalar_bf16_half_arith_body
  [433.55 ns 447.95 ns 462.12 ns]
eval/scalar_f16_half_arith_body
  [456.55 ns 467.51 ns 478.67 ns]
```

## One Lever

Recognize only the top-level scalar half arithmetic body:

```text
neg = Neg(x)
abs = Abs(neg)
prod = Mul(abs, y)
sum = Add(prod, half_literal)
quot = Div(sum, y)
out = Max(quot, x)
```

The fast path runs before dense-plan construction and delegates every operation to the existing `apply_scalar_half_op` half rounding chokepoint.

## Isomorphism Proof

- Equation order: unchanged; the direct path executes `Neg`, `Abs`, `Mul`, `Add`, `Div`, `Max` in the same order.
- Boundary ordering: unchanged; the guard requires exactly two input vars and one output var, and the final `Max` output must equal the Jaxpr output.
- Dtype: unchanged; the guard accepts only BF16 scalar inputs with a BF16 literal or F16 scalar inputs with an F16 literal.
- Half rounding: unchanged; each intermediate result is produced by the same `apply_scalar_half_op` path as the scalar half arena.
- Floating point behavior: the same half widen, op, and round sequence is used at every equation boundary.
- Tie/NaN behavior: unchanged because `Max`/`Abs`/arithmetic are delegated to the existing half op helper.
- RNG: unchanged; accepted Jaxprs must be effect-free and this path has no RNG source.
- Fallback surface: any consts, effects, params, sub-Jaxprs, arity mismatch, graph mismatch, non-half literal, or runtime dtype mismatch returns `None` and uses the previous interpreter path.

## Golden Output

Focused proof compares the new top-level fast path and `eval_jaxpr` against forced hash-map generic evaluation for BF16 and F16 edge bit patterns:

```text
(0x3f80, 0x4000)
(0x8000, 0x3f80)
(0x7f80, 0x4000)
(0xff80, 0x4000)
(0x7fc1, 0x4000)
(0x0001, 0x4000)
```

Golden digest:

```text
fdd1466145016f889175119a82d8a56655c636ca3beb53b5229865ab35ccaf1b
```

Proof command:

```bash
RCH_WORKERS=vmi1227854 RCH_REQUIRE_REMOTE=1 rch exec -- \
  cargo test -j 1 -p fj-interpreters --lib \
  top_level_scalar_half_arith_fast_path_matches_generic_and_golden -- --nocapture
```

Result: passed, 1 test.

## Benchmark Gate

Same-worker after command:

```bash
RCH_WORKERS=vmi1227854 RCH_REQUIRE_REMOTE=1 rch exec -- \
  cargo bench -j 1 -p fj-interpreters --bench pe_baseline -- eval/scalar_
```

```text
bf16 baseline: [433.55 ns 447.95 ns 462.12 ns]
bf16 after:    [136.31 ns 141.21 ns 146.89 ns]
bf16 median:   447.95 ns -> 141.21 ns
bf16 ratio:    3.17x

f16 baseline:  [456.55 ns 467.51 ns 478.67 ns]
f16 after:     [125.79 ns 128.47 ns 131.23 ns]
f16 median:    467.51 ns -> 128.47 ns
f16 ratio:     3.64x

score:         6.03 = 3.17 conservative impact * 0.95 confidence / 0.50 effort
```

Decision: keep.

## Validation

Passed:

```bash
rustfmt --edition 2024 --check crates/fj-interpreters/src/lib.rs
git diff --check
RCH_WORKERS=vmi1227854 RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -j 1 -p fj-interpreters --lib
```

RCH clippy remains blocked by pre-existing lint debt outside this pass:

```text
crates/fj-interpreters/src/lib.rs:5466 clippy::question_mark
```

`ubs crates/fj-interpreters/src/lib.rs` remains nonzero from pre-existing file-wide inventory. Its built-in formatting, clippy, cargo check, test-build, audit, and deny checks reported clean.
