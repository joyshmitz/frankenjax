# FJ-P2C-001 Behavior Extraction Ledger (IR Core / Tracer)

Scope: packet `FJ-P2C-001` (`fj-core`) with legacy anchors from `jax/jax/_src/core.py` and `jax/jax/_src/interpreters/partial_eval.py`.

## Normal Pathways

| Pathway | Legacy expectation | Current Rust realization | Status |
|---|---|---|---|
| Jaxpr construction | Stable container for const/input/output vars + equation list + effects metadata | `fj_core::Jaxpr` + deterministic fingerprinting + equation ordering | implemented (subset) |
| Equation construction | Source/context tagged equation creation via `new_jaxpr_eqn` | `Equation::new` style assembly and primitive enum constraints in `fj-core` | implemented (subset) |
| Transform stack evidence | Deterministic transform composition and replayability | `TraceTransformLedger` + `verify_transform_composition` with hash proof | implemented |
| Jaxpr evaluation contract | `eval_jaxpr` executes in equation order with env writes/reads | Interpreter path in `fj-interpreters` consumes `fj-core` IR consistently | implemented (single-output subset) |

## Edge Pathways

| Edge case | Legacy behavior | Current Rust behavior | Risk |
|---|---|---|---|
| Unsupported transform sequence (`grad(grad)`, double `vmap`) | Rejected by type/rule checks | Rejected by `TransformCompositionError` | low |
| Empty/missing transform evidence | Invalid trace artifact | Rejected by composition verification | low |
| Wrong tensor element count | Invalid abstract/value consistency | Rejected by `TensorValue::new` | low |
| Multi-output equation flow | Supported in parts of legacy ecosystem | Rejected/unsupported in current interpreter path | high |

## Adversarial Pathways

| Adversarial class | Legacy anchor | Current handling | Gap |
|---|---|---|---|
| Malformed var/effect declarations | `check_jaxpr` rejects invalid bindings/effect subsets | Composition and interpreter checks catch subsets of malformed traces | full `check_jaxpr` parity missing |
| Tracer leakage outside transformation scope | `ensure_no_leaks` detects leaked tracers | No direct tracer lifecycle equivalent yet | tracer protocol missing |
| Dynamic partial-eval misuse | `trace_to_jaxpr_dynamic` validates traced jaxpr | No packet-complete partial-eval/staging implementation yet | staging missing |

## Hidden Assumptions (Must Stay Explicit)

- `TraceTransformLedger` evidence entries are non-empty and aligned 1:1 with transform stack.
- Transform composition currently assumes at most one `grad` and one `vmap` in scoped v1.
- Packet currently models effect-free IR core; effectful jaxpr parity is deferred.

## Undefined/Deferred Zones

- Tracer API parity (`Tracer`, `full_raise`, `pure/sublift`, `process_primitive`) is not yet implemented in Rust.
- Full abstract-value lattice parity (e.g., richer `AbstractValue` hierarchy) is missing.
- Partial-eval dynamic tracing semantics are deferred to staging packet work.

## Resolution Policy

1. Any unresolved legacy behavior stays explicit as deferred with packet-local risk note linkage.
2. Strict mode remains fail-closed on incompatible unknowns; no silent repair.
3. Behavior may be promoted from deferred to guaranteed only with conformance + invariant evidence.
