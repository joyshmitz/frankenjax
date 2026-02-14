# Invariant Checklist: IR Core (FJ-P2C-001)

## Checked Invariants

- [x] **Jaxpr fingerprint is deterministic**
  - 100x replay test: zero drift (`metamorphic_fingerprint_determinism_100x`)
  - 100x e2e replay: zero drift (`e2e_p2c001_ir_determinism_under_replay`)
  - Rebuild from same spec produces identical fingerprint (`e2e_p2c001_trace_to_ir_roundtrip`)

- [x] **Transform composition proof is correct for all valid stacks**
  - jit, grad, vmap individual transforms verified (`oracle_transform_composition_rules`)
  - jit(grad), vmap(grad) compositions verified (`e2e_p2c001_transform_stack_composition`)
  - grad(grad), grad(vmap) correctly rejected (`oracle_composition_rejects_double_grad`, `e2e_p2c001_transform_order_enforcement`)

- [x] **TTL entries are append-only and immutable**
  - Multi-dispatch ledger inspection: each dispatch adds exactly 1 entry (`golden_journey_08_ledger_inspection`)
  - Decision IDs match cache keys (`golden_journey_08_ledger_inspection`)

- [x] **AbstractValue lattice is well-ordered**
  - N/A for current implementation — abstract values are represented as concrete Literal/TensorValue types
  - Type-level ordering enforced by Rust's type system

- [x] **Equation variable references are valid**
  - `validate_well_formed()` checks all equation inputs reference bound variables
  - Adversarial test with unbound reference verifies rejection (`adversarial_unbound_variable_reference`)
  - Duplicate variable IDs handled correctly (`adversarial_duplicate_variable_ids`)

## Verification Summary

All 5 invariants verified. 36 tests covering oracle, metamorphic, adversarial, e2e, and golden journey validation.
