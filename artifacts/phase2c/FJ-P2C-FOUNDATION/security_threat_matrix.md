# FJ-P2C-FOUNDATION Security/Compatibility Threat Matrix (bd-3dl.2)

## Scope

This artifact pins threat classes, mitigations, and residual risk for foundation packet families:

- `trace+ir` (`crates/fj-core`)
- `cache` (`crates/fj-cache`)
- `dispatch` (`crates/fj-dispatch`)
- `ledger` (`crates/fj-ledger`)
- `conformance+durability` (`crates/fj-conformance`)
- `dtype+numeric semantics` (`crates/fj-core`, `crates/fj-lax`)
- `rng determinism` (`crates/fj-lax/src/threefry.rs`)
- `control flow` (`crates/fj-lax`, `crates/fj-dispatch`)
- `ad/custom derivative surface` (`crates/fj-ad`, `crates/fj-api`)

## Hardened-Mode Deviation Allowlist

Only the following hardened deviations are allowed.

| Deviation ID | Category | Status | Notes |
|---|---|---|---|
| `HD-0` | `fail-closed-parity` | implemented | Hardened behaves the same as strict (default posture). |
| `HD-1` | `allowlisted_unknown_feature_hashing` | implemented | Unknown incompatible features are included in cache-key payload instead of rejected (`fj-cache` hardened mode only). |
| `HD-2` | `bounded_diagnostic_recovery` | planned | Explicitly deferred until adversarial fixture suite lands; no implicit recovery paths allowed today. |

Any hardened behavior outside `HD-0..HD-2` is out of policy.

## Threat Matrix

| Threat class | Packet family | Subsystem | Strict mitigation | Hardened mitigation | Residual risk | Evidence |
|---|---|---|---|---|---|---|
| Cache confusion / cache poisoning | foundation | `fj-cache`, `fj-dispatch` | Reject unknown incompatible features (`fail-closed`), canonical key includes mode+backend+transform stack+JAXPR fingerprint | `HD-1` allowlisted hashing with deterministic key and ledger trace | Medium: broaden adversarial corpus for key collisions and malformed compile options | `crates/fj-cache/src/lib.rs` (`build_cache_key`, `build_cache_key_ref`), `crates/fj-dispatch/src/lib.rs` (`dispatch`) |
| Transform-order confusion | foundation | `fj-core`, `fj-dispatch` | `verify_transform_composition` rejects unsupported stacks/evidence mismatch before execution | `HD-0` (same fail-closed behavior) | Low-Medium: composition rules still intentionally narrow | `crates/fj-core/src/lib.rs` (`verify_transform_composition`), `crates/fj-dispatch/src/lib.rs` (`transform_order_is_explicit`) |
| Effect-token misuse / sequencing bypass | foundation | `fj-core`, `fj-dispatch` | Effect metadata is carried in canonical fingerprints and dispatch records effect-token counts in ledger signals | `HD-0` | Medium: token system currently tracks/records ordering signals but does not enforce full runtime side-effect sequencing | `crates/fj-core/src/lib.rs` (`effects` + fingerprint tests), `crates/fj-dispatch/src/lib.rs` (`EffectContext`, `effect_token_count` tests) |
| Malformed graph injection | foundation | `fj-core`, `fj-interpreters` | Type/shape/arity invariants and interpreter errors fail execution | `HD-0` | Medium: no full malformed-graph fuzz corpus yet | `crates/fj-core/src/lib.rs`, `crates/fj-interpreters/src/lib.rs` |
| Shape signature spoofing / rank abuse | foundation | `fj-dispatch`, `fj-lax` | Explicit rank/arity checks in grad/vmap and primitive eval paths | `HD-0` | Medium-High: rank>1 and adversarial shape corpus deferred | `crates/fj-dispatch/src/lib.rs` (`TransformExecutionError::*`), `crates/fj-lax/src/lib.rs` |
| Complex numeric edge-case abuse (NaN/Inf propagation, zero-divisor, invalid comparisons) | v2 | `fj-core`, `fj-lax` | Complex dtype is explicit and comparison semantics remain fail-closed for unsupported ops; divide/log/sqrt paths surface domain errors instead of silent coercion | `HD-0` | Medium: add adversarial complex fixture corpus for NaN/Inf payload behavior and comparator misuse | `crates/fj-core/src/lib.rs` (`DType::{Complex64,Complex128}`), `crates/fj-lax/src/lib.rs` (complex eval + compare op handling) |
| RNG key lifecycle misuse (reused keys, weak seeds, non-deterministic output drift) | v2 | `fj-lax` ThreeFry module | Deterministic key/split/fold_in and sampler tests enforce stable bitstreams for same key+counter | `HD-0` | Medium: no JAX-oracle random fixture family yet; key-reuse policy enforcement still advisory | `crates/fj-lax/src/threefry.rs` (key/split/fold_in/sampler tests), `FEATURE_PARITY.md` RNG rows |
| Control-flow safety hazards (non-boolean condition, unbounded iteration, deep nesting pressure) | v2 | `fj-lax`, `fj-dispatch`, `fj-ad` | Condition type checks and max-iteration limits in loop evaluators; transform execution rejects unsupported stacks early | `HD-0` | Medium-High: broaden adversarial loops with pathological nesting and non-convergent bodies | `crates/fj-lax/src/lib.rs` (`while`/`scan` eval + guard rails), `crates/fj-dispatch/src/lib.rs` (transform constraints) |
| Custom-derivative rule abuse (malicious gradients, recursion, type mismatch) | v2 | `fj-ad`, `fj-api` | Current system remains default-rule-only; custom-rule registration path is not yet enabled in core runtime | `HD-0` | Medium: once custom JVP/VJP lands, add recursion guard + shape/type contract checks + audit logs as mandatory gate | `bd-2oap` task tracking custom rule registration, `crates/fj-ad/src/lib.rs` current VJP/JVP path |
| Evidence ledger tampering | foundation | `fj-ledger`, `fj-dispatch` | Decision records include mode, timestamp, and signal detail; append-only ledger model | `HD-0` | Medium: cryptographic signing not yet implemented | `crates/fj-ledger/src/lib.rs`, `crates/fj-dispatch/src/lib.rs` |
| RaptorQ sidecar corruption | foundation | `fj-conformance` durability pipeline | Scrub + decode-proof gates detect corruption and recovery capability | `HD-0` | Medium: nightly scrub automation still pending | `crates/fj-conformance/src/durability.rs`, `crates/fj-conformance/src/bin/fj_durability.rs`, `artifacts/durability/*.json` |

## Compatibility Envelope (Per Subsystem)

| Subsystem | JAX-observable behavior in scope | Strict mode | Hardened mode | Not guaranteed yet |
|---|---|---|---|---|
| `fj-core` | Canonical IR determinism and transform-stack composition invariants | guaranteed | guaranteed (`HD-0`) | multi-grad/vmap composition families outside current scope |
| `fj-cache` | Cache-key semantics bind mode/backend/transform stack/JAXPR fingerprint | guaranteed | guarded-recovery via `HD-1` | unknown-feature semantic repair beyond allowlisted hashing |
| `fj-dispatch` | Explicit transform execution semantics (`jit`, `grad`, `vmap`) with deterministic failures | guaranteed | guaranteed (`HD-0`) | higher-rank gradient/vmap semantics beyond current scalar/vector subset |
| `fj-ledger` | Deterministic decision/audit record shape | guaranteed | guaranteed (`HD-0`) | signed/tamper-evident cryptographic attestations |
| `fj-conformance` | Differential parity classification + machine-readable reports | guaranteed | out-of-scope (hardened-specific divergence policy not yet implemented) | full legacy-oracle breadth for all future packets |
| `fj-conformance` durability | Sidecar/scrub/decode-proof workflows | guaranteed | guaranteed (`HD-0`) | complete CI/nightly automation for every protected artifact category |
| `fj-lax` RNG | Deterministic ThreeFry key/split/fold_in and sampler behavior for identical inputs | in_progress | guaranteed (`HD-0`) | oracle-backed random family parity + key-reuse misuse defenses |
| `fj-lax` control flow | Functional `cond`/`scan`/`while` semantics with bounded execution guards | in_progress | guaranteed (`HD-0`) | full control-flow transform parity (especially grad-through-loop edge cases) |
| `fj-ad` custom derivatives | Safe custom JVP/VJP registration and execution contracts | not guaranteed yet | not guaranteed yet | recursion guards, type-shape contract checks, and audit trail for user rules |

## Adversarial Input Classes (Required Test Expansion Set)

| Class ID | Malicious input family | Target subsystem(s) | Current coverage | Required next test hook |
|---|---|---|---|---|
| `ADV-001` | Unknown incompatible feature tokens | `fj-cache`, `fj-dispatch` | unit coverage exists | expand to property/fuzz over feature-token space |
| `ADV-002` | Invalid transform stacks (order/count/evidence mismatch) | `fj-core`, `fj-dispatch` | unit coverage exists | randomized transform-sequence property suite |
| `ADV-003` | Malformed JAXPR graphs (missing vars / invalid equations) | `fj-core`, `fj-interpreters` | partial | dedicated parser/interpreter fuzz target |
| `ADV-004` | Shape spoofing and rank abuse | `fj-lax`, `fj-dispatch` | partial | adversarial tensor-shape corpus + proptest harness |
| `ADV-005` | Cache payload ambiguity (compile-option canonicalization attacks) | `fj-cache` | partial | differential key-collision property tests |
| `ADV-006` | Durability sidecar corruption (bit flips/truncation) | `fj-conformance` durability | covered for synthetic corruption | broaden decode-proof corruption matrix |
| `ADV-007` | Complex numeric adversaries (`NaN`, `Inf`, signed zeros, zero divisors) | `fj-lax`, `fj-ad` | partial | complex-focused oracle/adversarial fixtures with explicit failure-mode assertions |
| `ADV-008` | RNG misuse (key reuse, low-entropy seeds, deterministic drift) | `fj-lax`, `fj-conformance` | partial | random-family oracle fixtures + deterministic replay checks in CI |
| `ADV-009` | Control-flow non-termination / malformed predicates | `fj-lax`, `fj-dispatch` | partial | adversarial loop fixture family with bounded-iteration and type-check assertions |
| `ADV-010` | Custom derivative abuse (malicious cotangent maps, recursive rule graphs) | `fj-ad`, `fj-api` | not started | mandatory validation suite when `custom_jvp/custom_vjp` registration lands |
| `ADV-011` | Effect-token spoofing / unexpected sequencing graphs | `fj-core`, `fj-dispatch` | partial | adversarial dispatch traces that assert token-order evidence consistency and fail-closed policy on malformed effect sets |

## Deferred Items

- `HD-2` bounded diagnostic recovery is intentionally deferred until adversarial fixtures are in place.
- Hardened-mode divergence reporting is not yet implemented as a first-class conformance output channel.
- Random-family oracle capture and replay gate are not yet part of the default conformance run.
- Custom-derivative threat controls are deferred until `bd-2oap` is implemented.
