# Invariant Checklist: Dispatch/AD/Effects Runtime (FJ-P2C-004)

## Checked Invariants

- [x] **Reverse-mode AD matches finite-diff within tolerance for all test functions**
  - Analytical vs VJP vs JVP three-way comparison: 0 max error (`oracle_grad_*_three_way`)
  - Finite-diff cross-check within 1e-3 tolerance (`metamorphic_ad_matches_finite_diff`)
  - Property test with 256+ random inputs (`prop_grad_is_2x_for_square`)

- [x] **Effect tokens consumed in declaration order**
  - Single-transform dispatch: 1 token (`effect_tokens_in_dispatch_single_transform`)
  - Triple-transform dispatch: 3 tokens in correct order (`effect_tokens_in_dispatch_triple_transform`)
  - Insertion-order Vec preserves ordering (`effect_context_tracks_transform_tokens`)

- [x] **Transform stack depth does not cause stack overflow**
  - 1000-dispatch load test with latency tracking (`e2e_p2c004_dispatch_under_load`)
  - Iterative Jit-skip eliminates recursive frames for Jit chains
  - Composition proof bounds depth (at-most-one Grad, at-most-one Vmap)

- [x] **Dispatch routing is deterministic for same input**
  - Property test: 256+ random inputs produce identical outputs (`prop_dispatch_deterministic`)
  - Cache key stability: same request = same key (`prop_cache_key_stability`)
  - Integration test: two identical requests compared (`dispatch_cache_hit_miss_determinism`)

- [x] **Gradient of constant is exactly zero**
  - Metamorphic test: grad(constant_function) = 0.0 (`metamorphic_grad_constant_is_zero`)

- [x] **Forward-mode JVP matches reverse-mode VJP**
  - Property test: 256+ random inputs, tolerance 1e-6 (`prop_jvp_matches_vjp`)
  - Direct comparison: same function, same input (`jvp_matches_reverse_mode`)
  - Polynomial, trig, exponential functions tested (`jvp_x_squared_at_3`, `jvp_sin_at_zero`)

## Verification Summary

All 6 invariants verified. 70 tests covering oracle (5), metamorphic (4), adversarial (7), e2e (6), unit (11), integration (31), and JVP (6) validation. Three-way gradient comparison confirms AD correctness across 4 function families.
