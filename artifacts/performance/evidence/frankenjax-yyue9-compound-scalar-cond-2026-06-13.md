# frankenjax-yyue9: compound scalar condition plan

Date: 2026-06-13
Agent: BeigeMouse
Bead: frankenjax-yyue9
Crate: fj-interpreters

## Target

Profile-backed follow-up to the scalar predicate plan: a single `var CMP val`
condition already uses a monomorphic scalar compare plan, but a range predicate
such as `0 < x && x < 10` lowers to two comparisons plus a boolean bitwise op.
That multi-equation predicate missed the single-comparison guard and returned to
the generic dense interpreter on every loop iteration.

One production lever:

- build `ScalarCompoundComparePlan` for side-effect-free predicate jaxprs
- accept comparison steps over scalar operands and boolean logic steps over
  prior boolean slots
- reuse the existing scalar compare semantics for Eq/Ne/Lt/Le/Gt/Ge
- execute BitwiseAnd/BitwiseOr/BitwiseXor with eager bool operations
- fall back unchanged on tensors, unsupported dtypes, arity mismatch, effects,
  params, sub-jaxprs, shifts, or malformed boolean dependencies

## Baseline

Command:

```bash
RCH_WORKER=vmi1152480 rch exec -- cargo test -j 1 -p fj-interpreters --lib bench_compound_scalar_compare_cond_overhead --release -- --ignored --nocapture
```

Pre-production same-worker baseline on `vmi1152480`:

```text
BENCH compound scalar cond `0<x&&x<10` 4000000 evals: GENERIC 219.5ns/eval -> PLANNED 222.2ns/eval = 0.99x
```

## Final Benchmark

Final same-worker release rebench on `vmi1152480`:

```text
BENCH compound scalar cond `0<x&&x<10` 4000000 evals: GENERIC 240.2ns/eval -> PLANNED 22.1ns/eval = 10.88x
```

The earlier post-change same-worker run before the final clippy cleanup measured
`218.0ns/eval -> 21.9ns/eval = 9.94x`, confirming the result was stable across
two release test invocations.

## Isomorphism Proof

Ordering: the compiled path executes accepted equations in original jaxpr order.
It writes each comparison or logic result into the original output variable slot
and reads the final value from `jaxpr.outvars[0]`, including non-max output slot
layouts.

Tie-breaking: not applicable for scalar comparisons. Boolean logic has no
tie-breaking surface.

Floating point: f32 inputs and literals follow the existing scalar comparison
policy by widening to f64 before comparison; f64 stays f64; integer comparisons
stay integer. NaN behavior is the existing compare behavior: ordered comparisons
are false and `Ne` is true. The plan does not reassociate or reorder arithmetic.

RNG: not applicable.

Short-circuiting: the logic steps use eager safe-Rust bool `&`, `|`, and `^`,
matching the existing `apply_bitwise_binary_bool` primitive behavior. No branch
short-circuiting is introduced.

Fallback surface: any non-covered runtime value returns `None` and delegates to
the existing dense interpreter. Unsupported tensor/scalar dtypes, shape changes,
effects, params, sub-jaxprs, bad arity, and logical inputs that are not prior
bool slots are rejected by construction or runtime guard miss.

Golden output digest from the focused i64 parity test:

```text
b4d2e55ea3321f3774d7b08216eb26e5fab867a5c8b3d5d35a86b7d826c505cb
```

Focused proof:

```bash
RCH_WORKER=vmi1152480 rch exec -- cargo test -j 1 -p fj-interpreters --lib scalar_compare -- --nocapture
```

Result: `10 passed; 0 failed; 2 ignored`; the ignored entries are benchmark
tests. The run emitted the preexisting dependency warning in `fj-trace` for
`num_spatial`.

## Validation

Passing:

- `rustfmt --edition 2024 --check crates/fj-interpreters/src/lib.rs`
- `git diff --check -- crates/fj-interpreters/src/lib.rs`
- `RCH_WORKER=vmi1152480 rch exec -- cargo test -j 1 -p fj-interpreters --lib compound_scalar_compare -- --nocapture`
- `RCH_WORKER=vmi1152480 rch exec -- cargo test -j 1 -p fj-interpreters --lib scalar_compare -- --nocapture`
- `RCH_WORKER=vmi1152480 rch exec -- cargo check -j 1 -p fj-interpreters --lib`
- `RCH_WORKER=vmi1152480 rch exec -- cargo clippy -j 1 -p fj-interpreters --lib --no-deps -- -D warnings`
- `ubs crates/fj-interpreters/src/lib.rs`
- final same-worker release benchmark above

Caveats:

- rch crate-scoped runs still print the existing `fj-trace` dependency warning
  for `num_spatial`.
- UBS exited 1 on the preexisting broad panic/unwrap/indexing inventory in
  `crates/fj-interpreters/src/lib.rs`; its embedded formatting, clippy, cargo
  check, test-build, cargo-audit, and cargo-deny subchecks were clean.

## Decision

Keep.

Score: Impact 5 x Confidence 5 / Effort 2 = 12.5.

The final same-worker A/B is 10.88x on the profiled compound scalar predicate
miss, with bit-equivalent compare/logic semantics and guarded fallback for
programs outside the accepted predicate subset.
