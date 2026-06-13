# frankenjax-krb67 Scalar Trig JVP Evidence

Date: 2026-06-13
Agent: BeigeMouse
Bead: frankenjax-krb67
Crate: fj-ad

## Target

Profile-backed target from the post-frankenjax-gbtzl scalar AD reprofile:
`ad/jvp_sin_cos_mul` measured 765.37 ns / 766.36 ns / 767.23 ns on `ovh-a`
and was the remaining scalar JVP row in the slice after the scalar `Exp -> Log`
reverse-mode keep. Live linalg/GEMM perf beads were already claimed by other
agents, so this target avoided overlap.

Focused pre-edit baseline was rerun via RCH before changing code:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ovh-a RCH_WORKERS=ovh-a rch exec -- \
  cargo bench -p fj-ad --bench ad_baseline -- 'ad/jvp_sin_cos_mul' \
  --sample-size 40 --measurement-time 3 --warm-up-time 1 --noplot
```

RCH selected `vmi1264463`; accepted baseline:

```text
ad/jvp_sin_cos_mul [1.8327 us 1.9398 us 2.0781 us]
```

## Alien Primitive

Graveyard mapping:

- Section 6.6, Equality Saturation / E-Graphs: use a certified local rewrite
  card for an exact algebraic trace shape rather than a broad optimizer pass.
- Section 6.17, Adaptive Compilation and Runtime Specialization: specialize a
  hot, fully identified scalar JVP trace after profiling.

Alien-artifact mapping:

- Certified rewrite pipeline: exact shape guard, proof obligations, forced
  generic replay, and a golden SHA over observable output bits.

## One Lever

Add a guarded `fj-ad::jvp_inner` fast path for exactly:

```text
Sin(x) -> s
Cos(x) -> c
Mul(s, c) -> y
```

The fast path accepts only scalar F64 primal/tangent inputs, built-in `Sin`,
`Cos`, and `Mul` JVP rules, no custom JVP key, no params, no effects, no
sub-Jaxprs, no constvars, one input, and one output. Every other case falls
back to the generic JVP interpreter.

## Isomorphism Proof

Ordering:

- The guard requires the exact three-equation order above and returns the same
  single primal/tangent output order as generic JVP.
- It does not reorder user-visible outputs or change arity/error behavior.

Tie-breaking:

- Not applicable. The lever has no comparisons, sorting, hashing, or
  iteration-dependent tie selection.

Floating point:

- The fast path preserves the generic arithmetic grouping:
  `sin_tangent = cos(x) * dx`, `cos_tangent = (-sin(x)) * dx`,
  `primal_out = sin(x) * cos(x)`, and `tangent_out =
  (sin_tangent * cos(x)) + (sin(x) * cos_tangent)`.
- It intentionally recomputes `sin(x)` and `cos(x)` in the same places the
  generic primitive JVP rules do; it does not use algebraic identities such as
  `cos(2x)` or `1 - 2sin^2(x)`.
- Signed zero, infinities, and NaN payload behavior are covered by a
  bit-for-bit forced-generic comparison test.

RNG:

- Not applicable. The trace has no random primitive and the fast path performs
  no random operation.

Custom rules and fallback:

- A custom JVP key disables this path.
- Registered custom JVP rules for `Sin`, `Cos`, or `Mul` disable this path.
- Non-F64 values, tensors, params, effects, sub-Jaxprs, constvars, arity
  mismatches, and non-matching equation graphs use the existing generic path.

Golden output SHA-256:

```text
e5aecb887d2833d453dc40e5a4ddaa2dfe94ef3e045fcda9743f2d545eb0ca03
```

The golden digest is over primal/tangent output bit pairs for:
`0.0`, `-0.0`, finite positives/negatives, `FRAC_PI_4` with `-0.0` tangent,
positive infinity, negative infinity, a payloaded NaN primal, and a payloaded
NaN tangent.

## Benchmark Result

Post-change RCH rebench, pinned to the same selected worker:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1264463 RCH_WORKERS=vmi1264463 rch exec -- \
  cargo bench -p fj-ad --bench ad_baseline -- 'ad/jvp_sin_cos_mul' \
  --sample-size 40 --measurement-time 3 --warm-up-time 1 --noplot
```

Result on `vmi1264463`:

```text
ad/jvp_sin_cos_mul [153.11 ns 156.68 ns 160.68 ns]
```

Delta:

- Midpoint: 1.9398 us -> 156.68 ns, 12.38x faster.
- Conservative: 1.8327 us lower bound -> 160.68 ns upper bound, 11.41x faster.
- Score: Impact 5 * Confidence 5 / Effort 1 = 25.0.

## Validation

```text
cargo fmt --package fj-ad -- --check
PASS

git diff --check
PASS

RCH_REQUIRE_REMOTE=1 rch exec -- \
  cargo test -j 1 -p fj-ad scalar_f64_sin_cos_mul_jvp_matches_generic_bits -- --nocapture
PASS on vmi1264463
Final rerun after test-side error-propagation cleanup also passed on
vmi1264463: 1 passed, 370 filtered out.

RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -j 1 -p fj-ad
PASS on vmi1227854: 371 tests passed; doctests passed

RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -j 1 -p fj-ad --all-targets
PASS on vmi1227854

RCH_REQUIRE_REMOTE=1 rch exec -- \
  cargo clippy -j 1 -p fj-ad --all-targets --no-deps -- -D warnings
PASS on vmi1152480
```

The `cargo check` and clippy transcripts still print the pre-existing
`fj-trace` warning for `num_spatial`; it did not fail the fj-ad crate-scoped
gate and is outside this lever.

UBS:

```text
ubs crates/fj-ad/src/lib.rs \
  artifacts/performance/evidence/frankenjax-krb67-scalar-trig-jvp-2026-06-13.md \
  .skill-loop-progress.md .beads/issues.jsonl
EXIT 1
```

UBS scanned the full `crates/fj-ad/src/lib.rs` file and reported the
pre-existing broad inventory: 21 critical panic macro findings, 5384 warning
findings, and 943 info items. Its internal `cargo fmt`, clippy, cargo check,
test-build, cargo-audit, and cargo-deny gates were clean. The new proof test
uses `Result<(), String>` error propagation for the added digest path; the
remaining UBS inventory is outside this one-lever perf change.
