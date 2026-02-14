# M0 Method-Stack Artifacts (2026-02-13)

## 1. alien-artifact-coding artifacts

Produced:

- Decision-theoretic loss matrix and expected-loss action model in `crates/fj-ledger/src/lib.rs`.
- Evidence signal schema and ledger records in `crates/fj-ledger/src/lib.rs`.
- Dispatch-generated evidence entries including transform-stack proof signal in `crates/fj-dispatch/src/lib.rs`.
- Transform composition proof checker in `crates/fj-core/src/lib.rs`.
- Essence extraction ledger + legacy anchor map artifacts:
  - `artifacts/phase2c/FJ-P2C-FOUNDATION/essence_extraction_ledger.md`
  - `artifacts/phase2c/FJ-P2C-FOUNDATION/legacy_anchor_map.v1.json`
  - `artifacts/phase2c/FJ-P2C-001/behavior_extraction_ledger.md`
  - `artifacts/phase2c/FJ-P2C-001/legacy_anchor_map.v1.json`

Deferred:

- Formal calibration bounds (conformal/e-values/PAC-Bayes) for production controllers.

## 2. extreme-software-optimization artifacts

Produced:

- Benchmark harness: `crates/fj-dispatch/benches/dispatch_baseline.rs`.
- Updated baseline run artifact: `artifacts/performance/dispatch_baseline_2026-02-13.md`.

Deferred:

- Full hotspot flamegraph and opportunity matrix scoring per feature family.

## 3. RaptorQ-everywhere durability artifacts

Produced:

- Durability module implementation: `crates/fj-conformance/src/durability.rs`.
- Durability CLI: `crates/fj-conformance/src/bin/fj_durability.rs`.
- Sidecar/scrub/decode-proof artifacts:
  - `artifacts/durability/legacy_transform_cases.v1.sidecar.json`
  - `artifacts/durability/legacy_transform_cases.v1.scrub.json`
  - `artifacts/durability/legacy_transform_cases.v1.decode-proof.json`
  - `artifacts/durability/dispatch_baseline_2026-02-13.sidecar.json`
  - `artifacts/durability/dispatch_baseline_2026-02-13.scrub.json`
  - `artifacts/durability/dispatch_baseline_2026-02-13.decode-proof.json`

Deferred:

- Automatic sidecar generation coverage for all future long-lived artifacts.

## 4. Compatibility-security doctrine artifacts

Produced:

- Strict/hardened mode contract in specs.
- Strict-mode fail-closed unknown-feature handling in `crates/fj-cache/src/lib.rs`.
- Transform sequence fail-closed checks for unsupported stacks in `crates/fj-core/src/lib.rs`.
- Foundation threat matrix + compatibility envelope artifact:
  - `artifacts/phase2c/FJ-P2C-FOUNDATION/security_threat_matrix.md`
  - `artifacts/phase2c/global/compatibility_matrix.v1.json`
- Fail-closed audit artifact:
  - `artifacts/phase2c/FJ-P2C-FOUNDATION/fail_closed_audit.md`
- Foundation risk record:
  - `artifacts/phase2c/FJ-P2C-FOUNDATION/risk_note.v1.json`
- Dispatch-level strict/hardened unknown-feature path tests in `crates/fj-dispatch/src/lib.rs`.

Deferred:

- Expanded compatibility matrix + drift-gate automation across full scoped API surface.
- Full adversarial fixture suite for malformed graph/shape signatures.
