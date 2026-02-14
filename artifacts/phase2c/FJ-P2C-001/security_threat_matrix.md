# FJ-P2C-001 Security + Compatibility Threat Matrix (bd-3dl.12.3)

## Scope

Packet boundary: IR core (`Jaxpr`/`Tracer`) and immediate execution boundary.

Primary subsystems:
- `fj-core`
- `fj-interpreters`
- `fj-dispatch`
- `fj-cache` (for fingerprint/cache confusion boundary coupling)

## Threat Matrix

| Threat class | Attack vector | Strict mitigation | Hardened mitigation | Fail-closed boundary | Residual risk | Evidence |
|---|---|---|---|---|---|---|
| Malformed graph injection | crafted Jaxpr with dangling refs, invalid outvars, or impossible equation wiring | reject at invariant/interpretation checks; no successful eval path | bounded diagnostic context only; no semantic repair without allowlist | unresolved var/outvar references terminate request | Medium: dedicated malformed-graph fuzz corpus still pending | `crates/fj-interpreters/src/lib.rs`, `artifacts/phase2c/FJ-P2C-001/contract_table.v1.json` |
| Canonical fingerprint collision | semantically distinct IR crafted to collide under weak fingerprinting assumptions | canonical fingerprint determinism checks and contract gate on mismatch | same as strict for fingerprint mismatch | mismatch blocks compatibility-safe routing | Medium: collision-adversarial corpus incomplete | `crates/fj-core/src/proptest_strategies.rs`, `artifacts/phase2c/FJ-P2C-001/contract_table_support.md` |
| Transform composition bypass | TTL crafted to evade composition cardinality/order checks | `verify_transform_composition` rejects unsupported stacks/evidence mismatch | bounded fallback only where explicitly allowlisted; no silent bypass | composition violation blocks dispatch | Low-Medium: composition families are intentionally scoped | `crates/fj-core/src/lib.rs`, `crates/fj-dispatch/src/lib.rs` |
| Shape/signature spoofing | inconsistent dtype/shape metadata supplied to IR/primitive boundaries | strict validation at primitive/interpreter boundary; deterministic error on mismatch | same semantic boundary; enriched diagnostics only | shape/type mismatch blocks execution | Medium: high-rank shape adversarial cases deferred | `crates/fj-lax/src/lib.rs`, `crates/fj-dispatch/src/lib.rs` |
| Cache confusion at IR boundary | non-equivalent IR routed to same cached path via metadata ambiguity | strict mode rejects unknown incompatible metadata (`fail_closed`) | allowlisted hashed unknown-feature path only, with explicit audit trail | unknown incompatible metadata cannot silently pass strict gate | Medium: broader cache-key collision tests pending | `crates/fj-cache/src/lib.rs`, `crates/fj-dispatch/src/lib.rs` |

## Compatibility Envelope

| JAX behavior | FrankenJAX status | strict mode | hardened mode | evidence |
|---|---|---|---|---|
| Jaxpr pretty-print byte formatting | not guaranteed | out-of-scope | out-of-scope | `artifacts/phase2c/FJ-P2C-001/legacy_anchor_map.v1.json` |
| Variable naming identity | not guaranteed (`VarId(u32)` model) | out-of-scope | out-of-scope | `crates/fj-core/src/lib.rs` |
| Equation ordering (topological/semantic order preservation) | guaranteed (scoped packet) | guaranteed | guaranteed | `crates/fj-core/src/lib.rs`, `crates/fj-interpreters/src/lib.rs` |
| Transform composition rejection semantics | guaranteed (scoped packet) | guaranteed | guaranteed except explicit allowlisted hardened deviations | `crates/fj-core/src/lib.rs`, `artifacts/phase2c/FJ-P2C-001/contract_table.v1.json` |
| Unknown incompatible metadata handling at packet boundary | guaranteed strict fail-closed | guaranteed (`fail_closed`) | guarded-recovery (`allowlisted_repair`) | `crates/fj-cache/src/lib.rs`, `artifacts/phase2c/global/compatibility_matrix.v1.json` |

## Explicit Fail-Closed Rules

1. Unknown incompatible features are rejected in strict mode before successful packet execution.
2. Transform-order/cardinality violations are rejected before dispatch.
3. Invalid IR variable references do not produce a successful output pathway.
4. Hardened mode may only deviate via explicit allowlisted repair behavior; all other unknown behaviors are fail-closed.
