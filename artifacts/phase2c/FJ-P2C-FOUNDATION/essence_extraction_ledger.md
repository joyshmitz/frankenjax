# FJ-P2C-FOUNDATION Essence Extraction Ledger (bd-3dl.1)

## 1) Per-Packet Extraction Ledger

| Packet family | Primary crate(s) | Invariants / behavioral boundaries | Hidden assumptions | Undefined-behavior / unsupported zone | Explicit non-goals |
|---|---|---|---|---|---|
| IR core + TTL | `crates/fj-core/src/lib.rs` | Canonical transform stack signature and composition proof are deterministic; only one `grad` and one `vmap` currently allowed; tensor constructors reject element-count mismatch | Evidence strings are assumed non-empty before composition verification; IR is assumed single-output friendly downstream | Unsupported transform stacks fail closed (`TransformCompositionError`) | Nested `grad(grad(...))` and multi-`vmap` stacks are out of scope in v1 |
| LAX primitive eval | `crates/fj-lax/src/lib.rs` | Primitive semantics are deterministic for `{add,mul,dot,sin,cos,reduce_sum}`; dtype inference is explicit; shape checks gate execution | Dot/reduce_sum paths assume rank-1 vectors and scalar-like reductions | Unsupported primitives error immediately (`EvalError::Unsupported`) | Full LAX surface, broadcasting generality, and high-rank tensor ops are deferred |
| Jaxpr interpreter | `crates/fj-interpreters/src/lib.rs` | Environment resolution must bind every var before read; equation execution order is strict | Every equation is implicitly expected to have exactly one output | Multi-output equations are rejected (`UnexpectedOutputArity`) | General tuple/multi-result equation support is deferred |
| Cache keying + compatibility gating | `crates/fj-cache/src/lib.rs` | Cache keys bind mode/backend/hook/compile-options/transform-stack/JAXPR fingerprint deterministically | `BTreeMap` ordering and canonical payload stability are assumed for key reproducibility | Strict mode rejects unknown incompatible features (fail-closed) | Heuristic repair of unknown features in strict mode is explicitly disallowed |
| Dispatch transform orchestration | `crates/fj-dispatch/src/lib.rs` | Dispatch path always verifies composition then cache key then execution; errors are explicit and typed | Grad assumes scalar first input/output; vmap assumes rank-1 mapped dimension and scalar inner outputs | Vector-Jacobian and higher-rank vmap gradients are rejected | Full AD semantics and higher-rank batching are deferred |
| Evidence/decision ledger | `crates/fj-ledger/src/lib.rs` | Decision records and evidence signals are structured and deterministic; loss-matrix decisions are stable | Runtime admission quality assumes posterior calibration quality | Unsafely calibrated inputs can degrade quality but do not fail closed automatically | Cryptographic signing/tamper-proof attestations are deferred |
| Runtime admission | `crates/fj-runtime/src/lib.rs` | Admission is governed by shared loss matrix and posterior thresholding | Heuristic posterior from dispatch is assumed reasonable until conformal calibration is supplied | Admission confidence can be under-informative on novel workloads | Advanced policy controllers and richer risk controls are deferred |
| Conformance + durability | `crates/fj-conformance/src/lib.rs`, `crates/fj-conformance/src/durability.rs` | Differential parity and drift taxonomy are deterministic; sidecar/scrub/decode-proof logic is explicit | Batch runner assumes timeout-based triage is sufficient for fixture-level gating | Long-running fixtures are timeout-classified; durability auth signatures are not enforced | Full adversarial fixture corpus and cryptographic durability attestations are deferred |

## 2) Existing-Code Audit Table

| Subsystem | Exists now | Stubbed / partial | Missing |
|---|---|---|---|
| IR / Jaxpr core | `Jaxpr`, `Equation`, `VarId`, canonical fingerprinting, transform composition proof, TTL | No `ClosedJaxpr` equivalent in Rust public model; `constvars` behavior not fully modeled | Tracer protocol, abstract-value lattice parity, effect-aware IR forms |
| Primitive execution | First-wave primitive set and deterministic evaluator | Rank support and shape polymorphism are narrow | Wider primitive set and richer shape semantics |
| Interpreter | Deterministic eval with env and typed errors | Single-output equation restriction | Multi-output equation and nested-jaxpr execution parity |
| AD + transforms | `grad` path + finite-difference fallback, `jit` pass-through, `vmap` slice-stack behavior | AD coverage is scalar-centric; transform composition space intentionally small | Proper reverse-mode/autodiff parity and higher-order transform support |
| Cache / compatibility | Strict fail-closed unknown-feature handling, deterministic keys, hardened allowlisted hashing | Hardened behavior mostly parity-with-strict except explicit allowlist | Rich compatibility drift gate automation across all packet surfaces |
| Ledger / runtime | Bayesian-ish decision records, conformal calibration primitives, runtime admission model | Calibration pathways can fall back to heuristic behavior | Full threat-driven policy controls and signed audit logs |
| Conformance / oracle | Expanded fixture capture, comparator taxonomy, drift classes, report emitters, batch + timeout | Hardened divergence reporting not first-class; adversarial fixture depth limited | Full packet-wide adversarial/fuzz and scenario orchestration |
| Durability | RaptorQ sidecar/scrub/decode-proof pipeline and CLI | Batch/nightly coverage not comprehensive yet | Complete automation across all long-lived artifact categories |

## 3) Gap Ranking By Downstream Impact

| Rank | Gap | Impact class | Why this blocks downstream work |
|---|---|---|---|
| 1 | Missing Rust Tracer protocol + abstract-value lattice parity | Critical | Blocks packet `FJ-P2C-001` parity claims and higher packet work that depends on faithful tracing semantics (front-door, partial-eval, cache semantics). |
| 2 | Partial-evaluation / staging parity not implemented | Critical | Blocks packet `FJ-P2C-003` and all transform/lowering paths needing staged Jaxpr behavior. |
| 3 | Proper AD parity (beyond scalar + finite-diff fallback) | High | Blocks robust `grad`/composition conformance and oracle parity for nontrivial program families. |
| 4 | Higher-rank tensor and shape-polymorphic behavior | High | Blocks real batched workloads, richer `vmap`, and large subsets of JAX-observable behavior. |
| 5 | Effect-system parity (ordered/unordered/input effects) | High | Blocks safe modeling of effectful jaxprs and correctness/security around effect propagation. |
| 6 | FFI boundary and callback lifecycle model | Medium-High | Blocks hardened runtime posture and compatibility for host/device extension paths. |
| 7 | Backend abstraction (dispatch backend currently string-based) | Medium | Blocks multi-backend compatibility envelope and cache-key policy hardening for backend drift. |

## 4) Legacy Ambiguities Called Out

- JAX has effect-tracking richness (`jaxpr_effects_test.py`) not yet represented in current Rust IR; effect drift is currently an explicit non-goal.
- JAX tracer lifecycle leak detection (`ensure_no_leaks`) has no direct Rust equivalent yet.
- JAX partial-eval dynamic tracing (`trace_to_jaxpr_dynamic`) semantics are only partially reflected by current dispatch/interpreter flow.

## 5) Ambiguity Resolution Policy

1. If an observed legacy behavior is not implemented in the scoped v1 surface, strict mode fails closed and the behavior is recorded as deferred.
2. Hardened mode may only diverge via explicit allowlisted categories (`HD-*`) with an auditable note; no implicit repairs.
3. Any low-confidence anchor mapping must be backed by differential oracle fixtures before it can be promoted to a guaranteed invariant.
4. When legacy behavior and current implementation conflict, packet-level risk notes take precedence and must cite concrete mitigation/defer decisions.
