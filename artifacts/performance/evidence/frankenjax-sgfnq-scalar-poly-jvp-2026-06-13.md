# frankenjax-sgfnq: scalar polynomial JVP fast path

## Target

Profile-backed target from the post-`frankenjax-06zxm` AD profile:

```text
ad/jvp_poly_x3+x2+x time: [190.08 ns 204.76 ns 220.20 ns]
```

The target Jaxpr is the exact scalar F64 graph:

```text
x2 = x * x
x3 = x2 * x
s = x3 + x2
out = s + x
```

The lever adds a direct JVP fast path for this graph before the generic scalar Add/Mul slot loop.

## Isomorphism

The direct path preserves the existing generic scalar Add/Mul JVP operation order:

```text
x2 = x * x
x2_t = (dx * x) + (x * dx)
x3 = x2 * x
x3_t = (x2_t * x) + (x2 * dx)
s = x3 + x2
s_t = x3_t + x2_t
out = s + x
out_t = s_t + dx
```

No ordering, tie-breaking, random state, or error behavior changes:

- The recognizer requires the exact existing `scalar_f64_cubic_plus_square_plus_linear_input` graph.
- It only handles scalar F64 primals and tangents; all other dtypes/shapes fall through.
- Registered Jaxpr custom JVP rules still run before any fast path.
- Registered primitive Add/Mul custom JVP rules disable the fast path and fall through to the generic interpreter.
- The direct proof compares direct bits against the previous generic scalar Add/Mul fast path and public `jvp`.

Golden output SHA:

```text
1161a25bcc839668145518ffa39be3cdae32e63ae95c433fc722d7c8ab1ee62f
```

## Baseline

```text
Parent commit: 0b65807b
RCH_WORKER=vmi1149989 rch exec -- cargo bench -p fj-ad --bench ad_baseline -- 'ad/jvp_poly_x3\+x2\+x' --sample-size 30 --measurement-time 3 --warm-up-time 1 --noplot
Worker: vmi1149989
ad/jvp_poly_x3+x2+x time: [190.08 ns 204.76 ns 220.20 ns]
```

## Candidate

```text
RCH_WORKER=vmi1149989 rch exec -- cargo bench -p fj-ad --bench ad_baseline -- 'ad/jvp_poly_x3\+x2\+x' --sample-size 30 --measurement-time 3 --warm-up-time 1 --noplot
Worker: vmi1149989
ad/jvp_poly_x3+x2+x time: [58.737 ns 59.610 ns 60.720 ns]
```

Midpoint speedup: `204.76 / 59.610 = 3.44x`
Conservative speedup: `190.08 / 60.720 = 3.13x`

Score: `Impact 8 * Confidence 5 / Effort 2 = 20.0`
Decision: keep.

## Validation

```text
RCH_WORKERS=vmi1149989 rch exec -- cargo test -j 1 -p fj-ad scalar_f64_polynomial_jvp_matches_generic_order_bits -- --nocapture
Expected first run: failed only on placeholder SHA and printed 1161a25bcc839668145518ffa39be3cdae32e63ae95c433fc722d7c8ab1ee62f
```

```text
RCH_WORKERS=vmi1149989 rch exec -- cargo test -j 1 -p fj-ad scalar_f64_polynomial_jvp -- --nocapture
Worker: vmi1153651
Result: 2 passed, 0 failed
```

```text
cargo fmt --package fj-ad -- --check
Result: pass
```

```text
git diff --check
Result: pass
```

```text
RCH_WORKER=vmi1149989 rch exec -- cargo check -j 1 -p fj-ad --all-targets
Result: pass
```

```text
RCH_WORKER=vmi1149989 rch exec -- cargo clippy -j 1 -p fj-ad --all-targets --no-deps -- -D warnings
Result: pass
```

```text
ubs crates/fj-ad/src/lib.rs
Result: exit 1 with broad pre-existing fj-ad inventory; UBS subchecks for formatting, clippy, cargo check, tests build, unsafe, hardcoded secrets, and TODO/FIXME/HACK markers were clean.
```

Ambient warning observed during RCH builds, unchanged and outside this bead:

- `crates/fj-trace/src/lib.rs:1808` unused variable `num_spatial`.
