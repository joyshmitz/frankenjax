# FJ-P2C-001 Rust Implementation Plan + Module Boundary Skeleton (bd-3dl.12.4)

Date: 2026-02-14
Packet: `FJ-P2C-001` (IR core: `Jaxpr` / `Tracer`)

## 1. Architecture Decision

Decision: **Option C** from `bd-3dl.12.4` design notes.

- Keep `fj-core` as the current canonical IR/value/TTL owner to minimize rework in already-passing paths.
- Add `fj-trace` as a focused tracing boundary crate for tracer abstractions and trace-to-JAXPR conversion contracts.
- Defer full monolith split (`fj-types`, `fj-ir`, `fj-transforms`) until post-packet hardening to avoid destabilizing parity-critical paths.

Why this is least-risk now:
- `bd-3dl.12.2` strict/hardened contracts and `bd-3dl.12.3` threat model already anchor packet boundaries around current `fj-core` semantics.
- `bd-3dl.12.10` can now implement tracing behavior behind explicit interfaces without changing existing evaluator/dispatch invariants first.

## 2. Module Boundary Skeleton

### 2.1 `fj-core` (retain as-is for packet)

Ownership retained in `fj-core`:
- canonical IR: `Jaxpr`, `Equation`, `Atom`, `VarId`
- value model: `Value`, `TensorValue`, `Shape`, `DType`
- transform model: `Transform`, TTL, composition proof checks
- existing determinism and composition invariants

### 2.2 `fj-trace` (new crate; signatures only)

Added signature-only boundary in `crates/fj-trace/src/lib.rs`:
- abstract value model contracts: `AbstractValue`, `ShapedArray`, `ConcreteArray`
- tracer identity contracts: `TracerId`, `Tracer`
- trace context and conversion contracts: `TraceContext`, `TraceToJaxpr`
- trace artifacts: `JaxprTrace`, `ClosedJaxpr`
- boundary errors: `TraceError`

Design law:
- this crate contains only interface signatures in this bead,
- behavioral bodies are deferred to `bd-3dl.12.10`.

## 3. Implementation Sequence (Unblocks `bd-3dl.12.10`)

1. Implement `AbstractValue` semantics for shaped/concrete array forms.
2. Implement `Tracer` concrete type(s) and `TraceContext` state machine.
3. Implement `TraceToJaxpr::trace_to_jaxpr` with deterministic `ClosedJaxpr` emission.
4. Add structural validation hook before returning `ClosedJaxpr`:
   - unbound input refs rejected,
   - outvar derivability enforced,
   - output shadowing rejected.
5. Integrate with existing TTL construction path in `fj-core`/`fj-dispatch` without changing transform-order semantics.
6. Add tests for trace determinism and malformed trace rejection.

## 4. Risk Register and Controls

| Risk | Impact | Control |
|---|---|---|
| Tracer dynamic dispatch mismatch vs Python semantics | wrong trace graph lowering | keep interface-first boundary; implement via trait objects in follow-on bead with strict fixture replay |
| Hidden coupling between trace and dispatch | semantic drift in transform composition | keep composition checks in `fj-core` authoritative; trace crate may not bypass them |
| Premature crate split churn | delays and regressions | defer full split until packet completion and parity evidence |
| cache-key behavior drift from trace shape metadata | compatibility regressions | require cache-key regression checks in follow-on implementation bead |

## 5. Contract and Security Alignment

This plan is aligned with:
- `artifacts/phase2c/FJ-P2C-001/contract_table.v1.json`
- `artifacts/phase2c/FJ-P2C-001/contract_table_support.md`
- `artifacts/phase2c/FJ-P2C-001/security_threat_matrix.md`
- `artifacts/phase2c/FJ-P2C-001/compatibility_matrix.v1.json`

Boundary law:
- strict mode remains fail-closed for unknown incompatible behavior,
- hardened mode may only diverge through explicit allowlisted repair paths,
- no trace implementation may bypass transform-composition validation.

## 6. Definition of Done for `bd-3dl.12.4`

1. Crate/module layout decision is explicit and justified (Option C).
2. Signature-only trace boundary skeleton exists and compiles.
3. Implementation sequence is explicit and ordered for low semantic risk.
4. Follow-on bead (`bd-3dl.12.10`) has a clear contract-level handoff.
