# frankenjax-mcqr.30.38 dense BF16/F16 sort proof

Bead: `frankenjax-mcqr.30.38`
Production lever: `83dd3e8e perf(fj-lax): dense bf16/f16 sort along axis - 3.16x`
Proof hardening commit: staged separately from the production lever.

## Benchmark evidence

Same-worker timing row:

Command:
`RCH_WORKER=vmi1293453 rch exec -- cargo test --release -p fj-lax bench_bf16_sort_dense_vs_literal_radix --lib -- --ignored --nocapture`

Worker: `vmi1293453`

Result:
`BENCH bf16 sort [2048x256]: literal-radix=9.87ms dense=3.36ms speedup=2.94x`

Scoring:
Impact 4, Confidence 5, Effort 2 => Score 10.0.

## Behavior proof

Focused proof command:
`rch exec -- cargo test -p fj-lax dense_half_sort_matches_literal_radix_path --lib -- --nocapture`

Result:
`dense_half_sort_matches_literal_radix_path ... ok`

Follow-up proof hardening:

- Asserted the dense input actually uses `as_half_float_slice`.
- Asserted the boxed baseline does not use `as_half_float_slice`.
- Added a rank-2 `[256, 3]` case sorted along `dimension=0`, forcing `axis_stride != 1`.
- Covered BF16 and F16, sort and argsort, ascending and descending paths.

## Isomorphism checklist

- Ordering: both dense and boxed paths use the same decoded half values and sort-order key family; the new test compares full output bits.
- Tie-breaking: radix payload includes original index, preserving stable original-index tie behavior for argsort and duplicate values.
- Floating point: BF16/F16 bit payloads are decoded through the same half literal semantics as the boxed path; NaN, infinities, signed zero, and duplicate values are in the proof corpus.
- RNG: no RNG is used.
- Shape and dtype: dense sort returns the same shape and half dtype; argsort returns the same i64 index vector.
- Error behavior: only the dense fast path is selected when dense half backing exists; boxed fallback remains the baseline and is still tested.

Warnings observed during proof were pre-existing fj-lax warnings:

- `crates/fj-lax/src/linalg.rs:10539` duplicate `#[test]` attribute.
- `crates/fj-lax/src/lib.rs:15088` unused variable `w`.

## Gate notes

- `git diff --cached --check`: pass.
- `rustfmt --edition 2024 --check crates/fj-lax/src/tensor_ops.rs`: pass.
- `rch exec -- cargo check -p fj-lax --lib`: pass on `vmi1156319`.
- `rch exec -- cargo clippy -p fj-lax --lib -- -D warnings`: blocked by unrelated `crates/fj-lax/src/linalg.rs` `doc_lazy_continuation` and `needless_range_loop` findings.
- `ubs crates/fj-lax/src/tensor_ops.rs`: nonzero on broad pre-existing `tensor_ops.rs` inventories; its internal formatter, clippy, cargo check, and test-build sections were clean.
