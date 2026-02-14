# FJ-P2C-FOUNDATION Security/Compatibility Threat Matrix (bd-3dl.2)

## Scope

This artifact pins threat classes, mitigations, and residual risk for foundation packet families:

- `trace+ir` (`crates/fj-core`)
- `cache` (`crates/fj-cache`)
- `dispatch` (`crates/fj-dispatch`)
- `ledger` (`crates/fj-ledger`)
- `conformance+durability` (`crates/fj-conformance`)

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
| Malformed graph injection | foundation | `fj-core`, `fj-interpreters` | Type/shape/arity invariants and interpreter errors fail execution | `HD-0` | Medium: no full malformed-graph fuzz corpus yet | `crates/fj-core/src/lib.rs`, `crates/fj-interpreters/src/lib.rs` |
| Shape signature spoofing / rank abuse | foundation | `fj-dispatch`, `fj-lax` | Explicit rank/arity checks in grad/vmap and primitive eval paths | `HD-0` | Medium-High: rank>1 and adversarial shape corpus deferred | `crates/fj-dispatch/src/lib.rs` (`TransformExecutionError::*`), `crates/fj-lax/src/lib.rs` |
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

## Adversarial Input Classes (Required Test Expansion Set)

| Class ID | Malicious input family | Target subsystem(s) | Current coverage | Required next test hook |
|---|---|---|---|---|
| `ADV-001` | Unknown incompatible feature tokens | `fj-cache`, `fj-dispatch` | unit coverage exists | expand to property/fuzz over feature-token space |
| `ADV-002` | Invalid transform stacks (order/count/evidence mismatch) | `fj-core`, `fj-dispatch` | unit coverage exists | randomized transform-sequence property suite |
| `ADV-003` | Malformed JAXPR graphs (missing vars / invalid equations) | `fj-core`, `fj-interpreters` | partial | dedicated parser/interpreter fuzz target |
| `ADV-004` | Shape spoofing and rank abuse | `fj-lax`, `fj-dispatch` | partial | adversarial tensor-shape corpus + proptest harness |
| `ADV-005` | Cache payload ambiguity (compile-option canonicalization attacks) | `fj-cache` | partial | differential key-collision property tests |
| `ADV-006` | Durability sidecar corruption (bit flips/truncation) | `fj-conformance` durability | covered for synthetic corruption | broaden decode-proof corruption matrix |

## Deferred Items

- `HD-2` bounded diagnostic recovery is intentionally deferred until adversarial fixtures are in place.
- Hardened-mode divergence reporting is not yet implemented as a first-class conformance output channel.
