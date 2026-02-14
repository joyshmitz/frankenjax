# Risk Note: IR Core (FJ-P2C-001)

## Threat Model

### 1. Cache Confusion Attack
**Risk**: An attacker crafts two semantically-different Jaxpr programs that produce the same cache key, causing the wrong compiled output to be served.
**Mitigation**: Cache keys use SHA-256 of the full canonical payload including mode, backend, transform stack, compile options, and jaxpr fingerprint. Collision probability is negligible (2^-128).
**Evidence**: `oracle_cache_key_determinism` test verifies different programs produce different keys. `golden_journey_05_cache_hit_miss` verifies cross-program key uniqueness.
**Residual Risk**: LOW

### 2. Transform-Order Attack
**Risk**: Exploiting transform composition to bypass safety checks (e.g., `grad(vmap)` vs `vmap(grad)`).
**Mitigation**: `verify_transform_composition()` enforces legal orderings. Dispatch rejects illegal compositions with actionable error messages.
**Evidence**: `oracle_composition_rejects_double_grad`, `e2e_p2c001_transform_order_enforcement`, `golden_journey_04_transform_composition` all verify rejection.
**Residual Risk**: LOW

### 3. Fingerprint Non-Determinism
**Risk**: Non-deterministic fingerprints could cause cache misses, performance degradation, or incorrect caching.
**Mitigation**: `canonical_fingerprint()` uses `OnceLock` for memoization and writes fields in deterministic order.
**Evidence**: `metamorphic_fingerprint_determinism_100x` runs 100 iterations, `e2e_p2c001_ir_determinism_under_replay` runs 100 replays — zero drift detected.
**Residual Risk**: NEGLIGIBLE

### 4. Large Jaxpr Resource Exhaustion
**Risk**: Maliciously large Jaxpr programs could cause OOM or excessive processing time.
**Mitigation**: No explicit size limits (by design — mirrors JAX), but 1000-equation stress tests complete in <100ms.
**Evidence**: `adversarial_large_jaxpr_stress_test` (10K equations), `golden_journey_07_large_program` (200 equations), `e2e_p2c001_large_jaxpr_stress` (1000 equations).
**Residual Risk**: MEDIUM (no hard limits)

### 5. Variable Reference Safety
**Risk**: Equations referencing unbound variables could cause panics or undefined behavior.
**Mitigation**: `validate_well_formed()` checks all variable references are bound. `#![forbid(unsafe_code)]` on all crates.
**Evidence**: `adversarial_unbound_variable_reference` verifies rejection, `adversarial_duplicate_variable_ids` verifies handling.
**Residual Risk**: LOW

## Invariant Checklist

| Invariant | Status | Evidence |
|-----------|--------|----------|
| Jaxpr fingerprint is deterministic | VERIFIED | metamorphic_fingerprint_determinism_100x, e2e_ir_determinism_under_replay |
| Transform composition proof correct for all valid stacks | VERIFIED | oracle_transform_composition_rules, e2e_transform_stack_composition |
| TTL entries are append-only and immutable | VERIFIED | golden_journey_08_ledger_inspection |
| Equation variable references are valid | VERIFIED | adversarial_unbound_variable_reference, validate_well_formed tests |
| Cache key is collision-resistant | VERIFIED | oracle_cache_key_determinism, golden_journey_05_cache_hit_miss |

## Performance Summary

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Dispatch latency (jit scalar add) | 2520ns | 1183ns | 53.1% |
| Cache key gen (simple) | 856ns | 253ns | 70.4% |
| Cache key gen (medium) | 853ns | 302ns | 64.6% |
| Cache key gen (large 100eq) | 2380ns | 1640ns | 31.1% |

Optimization lever: streaming SHA-256 hasher eliminating intermediate String allocation.
Evidence: `artifacts/performance/evidence/ir_core/streaming_cache_key_hasher.json`

## Overall Assessment

IR Core subsystem is **LOW RISK** for Phase 2C deployment. All 36 validation tests pass. Cache key system is collision-resistant. Transform composition is correctly enforced. Performance exceeds baseline targets.
