# frankenjax-ulkot: dense u32/u64 structural ops

Date: 2026-06-14
Agent: SilverMaple
Bead: frankenjax-ulkot
Crate: fj-lax

## Target

`frankenjax-ulkot` tracks dense u32/u64 storage and operator routing. Dense u32/u64
backing already existed for several producer/operator families; this pass closes the
remaining structural data-movement gap in `crates/fj-lax/src/tensor_ops.rs`.

## Lever

One lever was kept: route dense `as_u32_slice()` / `as_u64_slice()` through the
existing typed kernels for:

- `transpose`
- `broadcast_in_dim`
- `pad`
- contiguous and strided `slice`
- `dynamic_slice`
- `dynamic_update_slice`
- `rev`
- `tile`

No arithmetic, comparison, tie-breaking, RNG, or floating-point behavior changed.
The lever only avoids boxing each unsigned element as `Literal::U32` / `Literal::U64`
when the operation is pure structural data movement.

## Benchmark

Focused ignored benchmark:

```text
rch exec -- cargo test -p fj-lax tensor_ops::tests::bench_u32_structural_dense_vs_generic -- --ignored --nocapture
```

Remote worker: `vmi1227854`

```text
BENCH u32 slice-contig [2048x2048]->[1024,2048]: generic=45.15ms dense=0.23ms speedup=194.26x
```

The generic baseline is the boxed `LiteralBuffer::new(...)` path in the same test
binary; the dense candidate is `TensorValue::new(... Literal::U32 ...)`, which now
stays on the packed u32 backing. Score: impact very high, confidence high from
same-test generic-vs-dense comparison and golden proof, effort low; conservative
Score >= 8.0.

## Isomorphism Proof

Focused golden test:

```text
rch exec -- cargo test -p fj-lax tensor_ops::tests::u32_u64_structural_ops_dense_match_generic_and_preserve_dtype -- --nocapture
```

Remote worker: `vmi1152480`

```text
test tensor_ops::tests::u32_u64_structural_ops_dense_match_generic_and_preserve_dtype ... ok
```

Golden SHA-256 enforced in code:

```text
0ff54f1e6c29ea74d3ae1999fef679dc08bfb9e1913e92bd0387fa9030b94beb
```

Fixture payload hashed by `fj_test_utils::fixture_id_from_json` includes:

- dtype label (`U32` / `U64`)
- operation label
- output shape
- output words as unsigned integers

The test compares dense outputs against boxed generic outputs for all covered ops,
asserts that dtype remains exactly `U32` or `U64`, and uses values above the signed
boundaries (`2^31`, `2^63`) so no signed reinterpretation can pass accidentally.

Preserved surfaces:

- Ordering: output linearization is the same typed gather/copy order as existing
  dense structural kernels and is checked against boxed generic outputs.
- Tie-breaking: not applicable; no comparison or reduction is performed.
- Floating point: not applicable; the fixture is unsigned integer only.
- RNG: not applicable; no random primitive or state is touched.
- Errors: shape/rank/parameter validation remains before the typed fast paths and
  falls through unchanged for unsupported layouts.

## Validation

Passed:

- `git diff --check`
- `rch exec -- cargo test -p fj-lax tensor_ops::tests::u32_u64_structural_ops_dense_match_generic_and_preserve_dtype -- --nocapture`
- `rch exec -- cargo test -p fj-lax tensor_ops::tests::bench_u32_structural_dense_vs_generic -- --ignored --nocapture`
- `rch exec -- cargo check -p fj-lax --lib` (earlier in this closeout; final code change after that was test-only and covered by the focused test)

Blocked by pre-existing project/file debt outside this lever:

- `cargo clippy -p fj-lax --lib -- -D warnings` fails in `crates/fj-lax/src/linalg.rs`
  on existing `doc_lazy_continuation` and `needless_range_loop` debt.
- direct touched-file `rustfmt --check` reports broad existing formatting drift in
  `crates/fj-lax/src/tensor_ops.rs`; this pass manually formatted only its new
  additions and `git diff --check` is clean.
- `ubs crates/fj-lax/src/tensor_ops.rs` exits nonzero from the existing file-wide
  heuristic inventory. Its built-in fmt, clippy, cargo check, test-build, audit,
  and deny sections were clean.

## Decision

Kept. The lever is behavior-preserving, has an enforced golden SHA, and the focused
benchmark shows a 194.26x dense structural materialization win on the retained
same-worker RCH run.
