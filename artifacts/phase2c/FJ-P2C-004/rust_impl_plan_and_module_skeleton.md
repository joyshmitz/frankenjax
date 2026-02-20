# FJ-P2C-004 Rust Implementation Plan + Module Boundary Skeleton (bd-3dl.15.4)

Date: 2026-02-20
Packet: `FJ-P2C-004` (Dispatch/effects runtime)

## 1. Architecture Decision

Decision: **Extend existing crates** rather than creating new ones.

- `fj-dispatch` already contains the full dispatch pipeline (`dispatch()`, `execute_with_transforms()`, per-transform handlers). Extend in-place.
- `fj-ad` already implements reverse-mode AD (`forward_with_tape`, `backward`, `vjp` rules for 24+ primitives). Extend with forward-mode JVP.
- No new crates needed. Effects modeling integrates into `fj-dispatch` (effect threading) and `fj-core` (effect types).

Why extend-in-place:
- `bd-3dl.15.2` contracts and `bd-3dl.15.3` threat model anchor boundaries around current crate semantics.
- All 351 workspace tests pass on current implementation. New modules add to existing crates without breaking interfaces.
- Composition proof and cache key generation are already correctly wired; no restructuring needed.

## 2. Current State Assessment

### 2.1 `fj-dispatch` (existing — fully functional)

| Component | Status | Location |
|---|---|---|
| `dispatch()` entry point | Complete | `lib.rs:135-186` |
| `DispatchRequest`/`DispatchResponse` | Complete | `lib.rs:15-30` |
| `execute_with_transforms()` recursive router | Complete | `lib.rs:188-202` |
| `execute_grad()` with symbolic AD | Complete | `lib.rs:204-229` |
| `execute_grad_finite_diff()` fallback | Complete | `lib.rs:231-260` |
| `execute_vmap()` slice-stack | Complete | `lib.rs:262-352` |
| `heuristic_posterior_abandoned()` | Complete | `lib.rs:354-359` |
| `calibrated_posterior_abandoned()` | Complete | `lib.rs:362-375` |
| Effect threading | NOT IMPLEMENTED | — |
| Transform module extraction | NOT IMPLEMENTED (all in lib.rs) | — |

### 2.2 `fj-ad` (existing — reverse-mode complete)

| Component | Status | Location |
|---|---|---|
| `TapeEntry` / `Tape` | Complete | `lib.rs:34-45` |
| `forward_with_tape()` | Complete | `lib.rs:49-109` |
| `backward()` | Complete | `lib.rs:111-143` |
| `vjp()` per-primitive rules | Complete (24 primitives) | `lib.rs:145-244` |
| `grad_jaxpr()` / `grad_first()` | Complete | `lib.rs:256-274` |
| Forward-mode JVP | NOT IMPLEMENTED | — |
| `ad.Zero` / `UndefinedPrimal` sentinels | NOT IMPLEMENTED | — |

### 2.3 `fj-core` (existing — transform model complete)

| Component | Status | Location |
|---|---|---|
| `Transform` enum (Jit, Grad, Vmap) | Complete | `lib.rs` |
| `TraceTransformLedger` | Complete | `lib.rs` |
| `verify_transform_composition()` | Complete | `lib.rs:947-997` |
| `TransformCompositionProof` | Complete | `lib.rs` |
| Effect types (Effect, OrderedEffect, Token) | NOT IMPLEMENTED | — |

## 3. Module Boundary Skeleton

### 3.1 `fj-dispatch` Extensions

```rust
// ---- transforms.rs (extract from lib.rs) ----

/// Per-transform execution handlers, extracted from lib.rs for clarity.
/// Public interface unchanged; dispatch() still calls execute_with_transforms().

pub(crate) fn execute_with_transforms(
    root_jaxpr: &Jaxpr,
    transforms: &[Transform],
    args: &[Value],
) -> Result<Vec<Value>, DispatchError>;

pub(crate) fn execute_grad(
    root_jaxpr: &Jaxpr,
    tail: &[Transform],
    args: &[Value],
) -> Result<Vec<Value>, DispatchError>;

pub(crate) fn execute_grad_finite_diff(
    root_jaxpr: &Jaxpr,
    tail: &[Transform],
    args: &[Value],
) -> Result<Vec<Value>, DispatchError>;

pub(crate) fn execute_vmap(
    root_jaxpr: &Jaxpr,
    tail: &[Transform],
    args: &[Value],
) -> Result<Vec<Value>, DispatchError>;


// ---- effects.rs (new module) ----

/// Effect token threading for ordered side-effects.
/// Models JAX's OrderedEffect token system via EvidenceLedger recording.
///
/// V1 scope: effect tracking only (no actual runtime token threading).
/// Effect types are logged in the evidence ledger; no execution ordering is enforced.

pub struct EffectToken {
    pub effect_name: String,
    pub sequence_number: u64,
}

pub struct EffectContext {
    tokens: BTreeMap<String, EffectToken>,
    next_sequence: u64,
}

impl EffectContext {
    pub fn new() -> Self;
    pub fn thread_token(&mut self, effect_name: &str) -> EffectToken;
    pub fn finalize(self) -> Vec<EffectToken>;
}

/// Integration point: dispatch() creates EffectContext, threads through
/// execute_with_transforms, and records finalized tokens in evidence ledger.
```

### 3.2 `fj-ad` Extensions

```rust
// ---- forward.rs (new module) ----

/// Forward-mode JVP (Jacobian-Vector Product) implementation.
/// Propagates (primal, tangent) pairs through the computation graph.

pub struct JvpResult {
    pub primals: Vec<Value>,
    pub tangents: Vec<Value>,
}

/// Compute JVP: (primals, tangents) -> (out_primals, out_tangents)
pub fn jvp(jaxpr: &Jaxpr, primals: &[Value], tangents: &[Value]) -> Result<JvpResult, AdError>;

/// Per-primitive JVP rule: given (primal_inputs, tangent_inputs) -> tangent_output
fn jvp_rule(primitive: Primitive, primals: &[Value], tangents: &[f64]) -> Result<f64, AdError>;


// ---- sentinels.rs (new module) ----

/// Sparse gradient sentinels for efficiency.
/// Avoids unnecessary computation when gradients are known to be zero.

#[derive(Debug, Clone, PartialEq)]
pub enum Tangent {
    /// Concrete tangent value
    Value(f64),
    /// Zero tangent — skip computation
    Zero,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PrimalStatus {
    /// Primal is available for backward pass
    Known(Value),
    /// Primal not needed — backward pass can skip
    Undefined,
}
```

### 3.3 `fj-core` Extensions

```rust
// ---- effects.rs (new module in fj-core) ----

/// Effect type system — models JAX's Effect/OrderedEffect hierarchy.
/// V1 scope: type definitions only, no runtime enforcement.

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EffectKind {
    Ordered,
    Unordered,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Effect {
    pub name: String,
    pub kind: EffectKind,
}

/// Effect set for a Jaxpr equation (parallel to JAX's JaxprEqn.effects)
pub type EffectSet = BTreeSet<Effect>;
```

## 4. Implementation Sequence

| Step | Description | Crate | Blocks |
|---|---|---|---|
| 1 | Extract `execute_*` functions to `fj-dispatch/src/transforms.rs` | fj-dispatch | Steps 2-3 |
| 2 | Add `EffectContext` and `EffectToken` types to `fj-dispatch/src/effects.rs` | fj-dispatch | Step 5 |
| 3 | Wire `EffectContext` into `dispatch()` — create, thread, record in ledger | fj-dispatch | Step 5 |
| 4 | Add forward-mode JVP to `fj-ad/src/forward.rs` for all 24 supported primitives | fj-ad | None |
| 5 | Add `Tangent::Zero` / `PrimalStatus::Undefined` sentinels | fj-ad | None |
| 6 | Add `Effect`/`EffectKind`/`EffectSet` types to `fj-core/src/effects.rs` | fj-core | Step 2 |
| 7 | Update `Equation` to carry optional `EffectSet` field | fj-core | Step 3 |

Note: Steps 1-3 and 4-5 can proceed in parallel. Step 6-7 should come first if effect types need to be shared across crates.

## 5. Migration Plan: Finite-Diff → Symbolic AD

Current state: `execute_grad()` in fj-dispatch delegates to `fj_ad::grad_first()` for innermost grad, falls back to finite-diff when grad has tail transforms.

Migration path:
1. **Keep finite-diff as validation oracle** — rename to `execute_grad_finite_diff_oracle()`
2. **Add JVP-based grad for tail transforms** — when forward-mode JVP is available, use `jvp(jaxpr, primals, tangents)` instead of finite-diff for nested transforms
3. **Cross-validate** — for each dispatch, optionally run both symbolic and finite-diff paths and compare results (gated by test/debug flag)
4. **Remove finite-diff production path** — once JVP is validated across all transform compositions

## 6. Risk Register and Controls

| Risk | Impact | Control |
|---|---|---|
| Transform extraction breaks existing tests | regression | Run full workspace tests after each extraction step; no behavior change |
| Effect types coupled too tightly to JAX semantics | architectural debt | V1 effects are tracking-only; no execution ordering enforced |
| Forward-mode JVP rule bugs | incorrect gradients | Cross-validate with existing reverse-mode and finite-diff |
| Module boundary changes break downstream crates | compilation failures | All new types are additive; no existing signatures removed |
| Performance regression from effect tracking overhead | latency increase | Effect tracking is O(1) per dispatch; benchmark after integration |

## 7. Contract and Security Alignment

This plan is aligned with:
- `artifacts/phase2c/FJ-P2C-004/contract_table.v1.json` (9 strict + 4 hardened invariants)
- `artifacts/phase2c/FJ-P2C-004/security_threat_matrix.md` (9 threats, 16 compatibility rows)
- `artifacts/phase2c/FJ-P2C-004/legacy_anchor_map.v1.json` (30 anchors)

Boundary law:
- Strict mode remains fail-closed for composition violations and unknown features.
- Hardened mode deviates only through explicit allowlisted paths.
- Effect tracking is observability-only in V1; no semantic changes to dispatch.
- All new code maintains `#![forbid(unsafe_code)]`.

## 8. Definition of Done for bd-3dl.15.4

1. Architecture decision is explicit and justified (extend-in-place).
2. Current state assessment documents what exists vs what's needed.
3. Module boundary skeleton defines all new types and functions with signatures.
4. Implementation sequence is ordered and parallelizable where possible.
5. Migration plan for finite-diff → symbolic AD is documented.
6. Risk register covers extraction, coupling, correctness, and performance risks.
7. Follow-on bead (`bd-3dl.15.10`) has a clear handoff for implementation.
