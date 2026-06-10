# frankenjax-lu4yw: word-native Bool mask pipeline

Date: 2026-06-10
Agent: PeachLion
Bead: frankenjax-lu4yw

## Change

One lever: add a packed `LiteralBufferStorage::BoolWords` representation and wire the profile-backed Bool-mask pipeline so dense same-shape F64 comparisons emit `u64` predicate words directly, while Bool reductions can consume those words without materializing `Literal::Bool` values.

Mapped primitives:
- Graveyard vectorized execution: SIMD compare produces a bitmask instead of one tuple/value at a time.
- Graveyard succinct bit-vectors / SWAR: store Bool masks as `Vec<u64>` with canonical tail bits and consume via word tests / popcount.

## Benchmark Evidence

Criterion via RCH, crate-scoped, remote worker `vmi1227854`.

Fresh current-main baseline (`31ea9db9`):

| row | baseline interval | baseline midpoint |
| --- | ---: | ---: |
| `eval/select_1024x1024_f64` | `[650.03 us, 696.08 us]` | `668.35 us` |
| `eval/lt_same_shape_1024x1024_f64` | `[370.55 us, 396.33 us]` | `380.52 us` |
| `eval/reduce_and_64k_bool_vec` | `[964.16 ns, 1.0655 us]` | `1.0262 us` |
| `eval/reduce_and_256_axis1_bool_vec` | `[1.3747 us, 1.5221 us]` | `1.4597 us` |

Candidate run:

| row | candidate interval | candidate midpoint | midpoint delta |
| --- | ---: | ---: | ---: |
| `eval/select_1024x1024_f64` | `[677.53 us, 758.28 us]` | `703.89 us` | `0.95x` |
| `eval/lt_same_shape_1024x1024_f64` | `[259.37 us, 282.67 us]` | `272.27 us` | `1.40x` |
| `eval/reduce_and_64k_bool_vec` | `[1.0476 us, 1.1031 us]` | `1.0674 us` | `0.96x` |
| `eval/reduce_and_256_axis1_bool_vec` | `[1.1298 us, 1.3861 us]` | `1.2480 us` | `1.17x` |

Acceptance target: `lt_same_shape_1024x1024_f64`, because the bead is profile-backed on comparison output materialization. Its candidate interval is strictly below the fresh baseline interval, with midpoint `380.52 us -> 272.27 us` (`1.40x`) and conservative edge `370.55 us / 282.67 us = 1.31x`.

Discarded noisy confirmation: a later candidate confirmation measured `lt_same_shape_1024x1024_f64` at `1.3491 ms`, and a later baseline measured it at `1.0212 ms`, while RCH reported multiple concurrent jobs on `vmi1227854` and load above core count. Those runs are environmental context only, not keep/reject evidence.

## Score

Impact `3.0` x Confidence `4.0` / Effort `3.0` = `4.0`.

Keep threshold: `>= 2.0`. Verdict: keep.

## Isomorphism Proof

- Ordering preserved: yes. F64 same-shape comparison still evaluates logical row-major positions in ascending order; SIMD chunks only pack the eight adjacent predicate results into the corresponding bit positions.
- Tie-breaking unchanged: yes. Comparisons produce Bool values only; no ordering ties are introduced. Bool reduction axis paths keep the same output-index odometer order.
- Floating-point preserved: yes. The changed comparison path performs IEEE comparison predicates only, with no FP arithmetic. NaN, infinities, and signed-zero comparison semantics match the old `float_cmp` fallback; scalar tail elements still call the same closure.
- RNG unchanged: N/A. No RNG path touched.
- Tail bits canonicalized: yes. `LiteralBuffer::from_bool_words` clears unused tail bits, so whole-word `any` and `popcount` cannot observe padding.
- Fallback semantics preserved: yes. `as_slice`, `to_vec`, `IntoIterator`, serde, COW mutation, and generic literal consumers materialize the same `Literal::Bool` sequence lazily.

## Golden Verification

Golden transcript:

```text
bool_word_literal_buffer_preserves_literal_api_and_tail_canonicalization: ok
f64_compare_word_masks_match_literal_path_at_word_boundaries: ok
bool_word_reduce_bit_identical_to_literal_path: ok
dense_bool_reduce_bit_identical_to_literal_path: ok
dense_bool_axis_reduce_bit_identical_to_literal_path: ok
```

SHA-256:

```text
be1abf63838788669eb4c443d22ae3c2a704dc676b9d4e2ece1a47da2ad869d2
```

Verification commands:

```text
RCH remote vmi1227854: cargo test -p fj-lax word -- --nocapture
RCH remote vmi1227854: cargo test -p fj-core bool_word_literal_buffer_preserves_literal_api_and_tail_canonicalization -- --nocapture
RCH remote vmi1227854: cargo test -p fj-lax dense_bool -- --nocapture
RCH remote vmi1227854: cargo check -p fj-lax --all-targets
```

`cargo check -p fj-lax --all-targets` passed with one pre-existing warning in `crates/fj-lax/src/cz0g0_f32accum_evidence.rs`.
