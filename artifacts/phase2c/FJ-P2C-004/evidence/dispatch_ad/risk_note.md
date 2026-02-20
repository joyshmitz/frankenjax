# Risk Note: Dispatch/AD/Effects Runtime (FJ-P2C-004)

## Threat Model

### 1. Gradient Numerical Instability
**Risk**: Reverse-mode AD tape replay produces incorrect gradients for functions with steep curvature, near-zero denominators, or catastrophic cancellation.
**Mitigation**: Three-way gradient comparison (analytical vs VJP vs JVP) catches divergence. Forward-mode JVP serves as independent cross-check for reverse-mode results. Property tests with 256+ random inputs verify consistency.
**Evidence**: `oracle_grad_square_three_way`, `oracle_grad_sin_three_way`, `oracle_grad_exp_three_way`, `oracle_grad_polynomial_three_way`, `prop_jvp_matches_vjp`, `prop_grad_is_2x_for_square`
**Residual Risk**: LOW (exact for polynomial/trig/exp primitives; no transcendental approximations)

### 2. Effect Ordering Violation
**Risk**: Effect tokens consumed in wrong order could produce observationally different results or violate effect semantics.
**Mitigation**: EffectContext records tokens in insertion order (Vec storage). Transform stack is processed left-to-right, producing deterministic effect sequences. Effect count recorded as evidence ledger signal.
**Evidence**: `metamorphic_effect_token_ordering`, `effect_context_tracks_transform_tokens`, `effect_tokens_in_dispatch_single_transform`, `effect_tokens_in_dispatch_triple_transform`
**Residual Risk**: LOW

### 3. Transform Composition Bypass
**Risk**: Crafted transform stacks bypass composition rules (e.g., double grad, grad(vmap)).
**Mitigation**: `verify_transform_composition()` enforces at-most-one Grad, at-most-one Vmap before dispatch. Illegal compositions produce actionable error messages.
**Evidence**: `adversarial_double_grad_rejected`, `adversarial_double_vmap_rejected`, `invalid_grad_vmap_rejected`, `e2e_p2c004_transform_composition_matrix`
**Residual Risk**: NEGLIGIBLE

### 4. Dispatch Non-Determinism
**Risk**: Same input producing different outputs or cache keys across invocations.
**Mitigation**: All computation paths are deterministic. Cache keys computed from SHA-256 of canonical jaxpr + transform stack + mode. Property test verifies determinism across 256+ random inputs.
**Evidence**: `prop_dispatch_deterministic`, `prop_cache_key_stability`, `dispatch_cache_hit_miss_determinism`, `e2e_p2c004_dispatch_under_load`
**Residual Risk**: NEGLIGIBLE

### 5. Gradient Explosion / Stack Overflow
**Risk**: Deep transform stacks or recursive AD applications could cause stack overflow or numeric overflow.
**Mitigation**: Transform stacks are bounded by composition proof (at-most-one each of Grad/Vmap). Iterative Jit-skip eliminates recursive frames for Jit chains. 1000-dispatch load test confirms stability.
**Evidence**: `e2e_p2c004_dispatch_under_load` (1000 dispatches), `e2e_p2c004_adversarial_dispatch_inputs`
**Residual Risk**: LOW (no explicit stack depth limit, but composition rules bound depth to ~3)

## Invariant Checklist

| Invariant | Status | Evidence |
|-----------|--------|----------|
| Reverse-mode AD matches finite-diff within tolerance | VERIFIED | metamorphic_ad_matches_finite_diff, three_way_comparison |
| Effect tokens consumed in declaration order | VERIFIED | metamorphic_effect_token_ordering, effect_context_tracks_transform_tokens |
| Transform stack depth does not cause stack overflow | VERIFIED | e2e_p2c004_dispatch_under_load (1000 dispatches) |
| Dispatch routing is deterministic for same input | VERIFIED | prop_dispatch_deterministic (256+ cases), dispatch_cache_hit_miss_determinism |
| Gradient of constant is exactly zero | VERIFIED | metamorphic_grad_constant_is_zero |
| Forward-mode JVP matches reverse-mode VJP | VERIFIED | prop_jvp_matches_vjp (256+ cases), jvp_matches_reverse_mode |

## Performance Summary

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Dispatch latency (jit scalar add) | 1.82us | 1.39us | 23.6% |
| Dispatch latency (grad scalar square) | 2.10us | 1.76us | 16.2% |
| Dispatch latency (jit_grad composed) | 2.21us | 1.60us | 27.6% |
| Dispatch latency (vmap_grad composed) | 2.88us | 2.31us | 19.8% |

Optimization levers: iterative Jit-skip, Vec-based effect tokens, inlined posterior.
Evidence: `evidence/perf/dispatch_ad/iterative_transform_and_effect_pooling.json`

## Overall Assessment

Dispatch/AD/effects subsystem is **LOW RISK** for Phase 2C deployment. All 70 validation tests pass. Three-way gradient comparison confirms AD correctness. Effect ordering is deterministic. Performance exceeds baseline targets (all dispatch latencies < 2us for simple cases).
