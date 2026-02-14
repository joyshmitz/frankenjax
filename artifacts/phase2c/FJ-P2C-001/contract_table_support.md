# FJ-P2C-001 Contract Support Table (bd-3dl.12.2)

This table provides the explicit column mapping requested for packet contracts.

| invariant_id | description | mode | enforcement | test_ref | legacy_anchor | violation_error |
|---|---|---|---|---|---|---|
| `p2c001.strict.inv001` | Every equation input `VarId` must already be bound (input/const/previous output) | strict | fail-closed | `crates/fj-interpreters/src/lib.rs` (`input_arity_mismatch_is_reported`); planned: `bd-3dl.12.5::p2c001_strict_inv001_unbound_var_fail_closed` | `jax/jax/_src/core.py:3251` (`check_jaxpr`) | `InterpreterError::MissingVariable` |
| `p2c001.strict.inv002` | Equation outputs must not shadow existing `VarId`s | strict | fail-closed | planned: `bd-3dl.12.5::p2c001_strict_inv002_output_shadow_rejected` | `jax/jax/_src/core.py:3251` (`check_jaxpr`) | `strict.var_output_shadow` |
| `p2c001.strict.inv003` | `outvars` must reference values defined by Jaxpr body | strict | fail-closed | planned: `bd-3dl.12.5::p2c001_strict_inv003_outvar_unresolved_rejected` | `jax/jax/_src/core.py:3251` (`check_jaxpr`) | `InterpreterError::MissingVariable` |
| `p2c001.strict.inv004` | `canonical_fingerprint()` must be deterministic for equivalent Jaxpr | strict | fail-closed | `crates/fj-core/src/proptest_strategies.rs` (`prop_ir_fingerprint_determinism`) | `jax/jax/_src/core.py:96` (`Jaxpr`) | `consistency.fingerprint_mismatch` |
| `p2c001.strict.inv005` | Composition-signature FNV hash must be stable across runs | strict | fail-closed | `crates/fj-core/src/proptest_strategies.rs` (`prop_ttl_composition_signature_determinism`) | `jax/jax/_src/core.py:3763` (`JaxprPpContext`) | `consistency.stack_hash_mismatch` |
| `p2c001.strict.inv006` | Transform composition permits at most one `Grad`, one `Vmap`, and equal transform/evidence counts | strict | fail-closed | `crates/fj-core/src/lib.rs` (`unsupported_sequence_rejected`) | `jax/jax/_src/core.py:910` (`Tracer`) + `jax/jax/_src/interpreters/partial_eval.py:2465` (`trace_to_jaxpr_dynamic`) | `TransformCompositionError` |
| `p2c001.hardened.inv007` | Malformed `VarId` refs log evidence and return deterministic default (no panic) | hardened | allowlisted-repair | planned: `bd-3dl.12.5::p2c001_hardened_inv007_repair_logs`; planned differential: `bd-3dl.12.6::p2c001_hardened_inv007_malformed_var_ref_replay` | `jax/jax/_src/core.py:3251` (`check_jaxpr`) | `hardened.var_ref_repair` |
| `p2c001.hardened.inv008` | Empty Jaxpr is accepted and logged as benign | hardened | warn-and-continue | planned: `bd-3dl.12.5::p2c001_hardened_inv008_empty_jaxpr_logged` | `jax/jax/_src/core.py:96` (`Jaxpr`) | `hardened.empty_jaxpr` |
| `p2c001.hardened.inv009` | Very large equation count logs pressure warning but continues deterministically | hardened | warn-and-continue | planned: `bd-3dl.12.8::p2c001_hardened_inv009_large_eqn_pressure_budget` | `jax/jax/_src/core.py:729` (`eval_jaxpr`) | `hardened.eqn_count_warning` |

## Mode Boundary Law (Explicit)

| Boundary case | strict mode | hardened mode |
|---|---|---|
| unknown incompatible IR/cache metadata | fail-closed | allowlisted-repair with deterministic audit event |
| malformed variable references | fail-closed | allowlisted-repair (`hardened.var_ref_repair`) |
| empty Jaxpr | deterministic success if valid; fail on malformed graph refs | warn-and-continue (`hardened.empty_jaxpr`) |
| large equation-count pressure | deterministic execution without semantic rewrite | warn-and-continue (`hardened.eqn_count_warning`) |

Fail-closed invariant:
- No branch in either mode may silently alter transform composition semantics.
