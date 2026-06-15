# frankenjax-mcqr.30.39 -- dense f64 atan2 scalar-broadcast serial route

Date: 2026-06-15
Agent: SilverMaple
Bead: frankenjax-mcqr.30.39

## Target

Profile-backed bead from RCH `vmi1152480` showed dense f64 scalar-broadcast
`atan2` regressed by scoped thread fan-out:

- `eval/atan2_scalar_1m_f64_vec`: 140.92 ms median
- `eval/atan2_scalar_1m_f64_literal_ref`: 67.712 ms median

The hot path was `atan2(tensor, scalar)` / `atan2(scalar, tensor)` where the
per-lane libm call did not amortize the thread-spawn overhead on current
workers.

## One Lever

Route dense f64 scalar-broadcast `Primitive::Atan2` away from
`eval_f64_scalar_expensive_parallel` and through the serial dense
`f64_scalar_broadcast_fn` path.

No other primitive or dtype route changed.

## RCH Rebench

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1152480 RCH_WORKERS=vmi1152480 rch exec -- cargo bench -p fj-lax --bench lax_baseline -- 'eval/atan2_scalar_1m_f64_(vec|literal_ref)' --warm-up-time 1 --measurement-time 3 --sample-size 10
```

RCH selected `vmi1227854`; treat this as a large-margin comparison against the
profile-backed bead baseline rather than a strict same-worker A/B.

Results:

- `eval/atan2_scalar_1m_f64_vec`: [12.685 ms 13.890 ms 15.001 ms]
- `eval/atan2_scalar_1m_f64_literal_ref`: [13.203 ms 13.765 ms 14.448 ms]

Speedup versus bead baseline median: 140.92 / 13.890 = 10.15x.
Conservative interval ratio: 140.92 / 15.001 = 9.39x.

Score: 25.0 (Impact 5 * Confidence 5 / Effort 1).

## Isomorphism Proof

- Ordering preserved: yes. The serial dense path maps the input slice in
  row-major order and writes the same output order as the generic broadcast path.
- Tie-breaking unchanged: N/A. `atan2` has no tie-breaking policy.
- Floating-point preserved: yes. Each lane calls the same `f64::atan2` with the
  same operand order. The removed thread fan-out never changed the per-lane
  operation; it only changed scheduling.
- RNG seeds: N/A. The path is deterministic and has no RNG.
- DType/shape/error behavior: unchanged. The route still requires dense f64
  scalar broadcast and falls back for other dtypes or malformed tensors.
- Golden output: focused test digest
  `2d89e8c1aeb21c2033b4ba82dfa30d2cf90767131bc896454cc8289a6e020896`.

## Validation

- `cargo fmt --check --package fj-lax`: pass.
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p fj-lax dense_f64_scalar_atan2_serial_route_preserves_bits_and_golden -- --nocapture`: pass on `ovh-a`.
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p fj-lax --all-targets`: pass on `ovh-a`.
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p fj-lax --all-targets -- -D warnings`: pass on `vmi1227854`.
- `ubs crates/fj-lax/src/arithmetic.rs`: exit 1 from existing file-wide
  panic/unwrap/indexing heuristic inventory; embedded fmt/clippy/check/test-build
  sections were clean and no new targeted finding blocks this lever.

## Keep Decision

Keep. The route preserves behavior and clears the Score >= 2.0 gate by a wide
margin.

## Next Primitive

Reprofile after landing. If dense scalar transcendentals remain hot, attack the
next per-primitive scheduling mismatch or a deeper dense transcendental kernel
with a separate profile-backed bead.
