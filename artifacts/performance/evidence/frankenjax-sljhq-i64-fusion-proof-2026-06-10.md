# frankenjax-sljhq i64 fusion proof

## Target

- Bead: frankenjax-sljhq
- Lever: same-shape dense `I64` elementwise chain fusion in `fj-interpreters`
- Scope: `Add`, `Sub`, `Mul`, `Div`, and `Neg` with dense same-shape tensor operands plus `I64` scalar literals

## Benchmark Evidence

- Previous shipped A/B: `eval_fusion_speed` 1M elements, 8 integer ops, forced-unfused `38.433ms`, fused `3.488ms`, speedup `11.02x`
- This verification pass: forced-unfused `46.750ms`, fused `4.057ms`, speedup `11.52x`
- Hyperfine artifact: `artifacts/performance/evidence/frankenjax-i64-fusion-hyperfine-2026-06-10.json`
- Score: Impact `5` x Confidence `5` / Effort `2` = `12.5`

## Isomorphism Proof

- Ordering: the fused runner evaluates the same chain order and materializes one dense row-major `I64` tensor.
- Tie-breaking: not applicable; no reductions or comparisons participate in this fusion path.
- Floating point: not applicable; the fused path is `I64` only and cannot promote dtype.
- RNG: not applicable; the participating primitives are deterministic arithmetic primitives.
- Integer semantics: fused `Add`, `Sub`, `Mul`, `Div`, and `Neg` use the same closures as the scalar interpreter path: `wrapping_add`, `wrapping_sub`, `wrapping_mul`, `checked_div(...).unwrap_or(0)`, and `wrapping_neg`.
- Fallback: unsupported dtype, shape mismatch, broadcast, multi-output, effectful equations, or liveness boundary falls back to the existing per-equation interpreter path.

## Golden Output

- Test: `cargo test -p fj-interpreters fusion_i64_chain_matches_reference_bit_for_bit --lib -- --nocapture`
- Golden digest: `7f7e34d693a2f0e9f63a0d4575b03db01882e16d74746eb5b3978d8a6d25b297`
- Digest source: manual independent fold over 4096 elements covering `i64::MAX`, `i64::MIN`, wrapping overflow, division by zero, and `MIN / -1`.

## Validation

- `rustfmt --edition 2024 --check crates/fj-interpreters/src/lib.rs crates/fj-interpreters/benches/eval_fusion_speed.rs`
- `git diff --check -- crates/fj-interpreters/src/lib.rs crates/fj-interpreters/benches/eval_fusion_speed.rs`
- `CARGO_TARGET_DIR=/data/tmp/cargo-target-frankenjax-peachlion-i64 rch exec -- cargo test -p fj-interpreters fusion_i64_chain_matches_reference_bit_for_bit --lib -- --nocapture`
- `rch exec -- cargo check -p fj-interpreters --all-targets`
- `rch exec -- cargo clippy -p fj-interpreters --all-targets --no-deps -- -D warnings`

Full workspace `cargo fmt --check` and full dependency-inclusive clippy are blocked by pre-existing unrelated formatting/lint debt outside this lever.
