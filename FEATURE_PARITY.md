# FEATURE_PARITY

Audit timestamp: **2026-03-03** (`cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test --workspace` passed via `rch`).

## Status Legend

- `not_started`
- `in_progress`
- `parity_green`
- `parity_gap`

## Feature Family Matrix

| Feature Family | Status | Current Evidence | Next Required Artifact |
|---|---|---|---|
| Canonical IR + TTL model | parity_green | Canonical IR/value model in `crates/fj-core/src/lib.rs`; transform evidence flow in `crates/fj-dispatch/src/lib.rs`; E2E traces in `artifacts/e2e/e2e_p2c001_trace_to_ir_roundtrip.e2e.json` and `artifacts/e2e/e2e_p2c001_transform_stack_composition.e2e.json` | Expand structural oracle comparisons to larger program families |
| Primitive semantics (implemented subset) | parity_green | `Primitive` enum (87 ops) in `crates/fj-core/src/lib.rs`; evaluator + extensive tests in `crates/fj-lax/src/lib.rs`; workspace tests green on 2026-03-03 | Continue expanding oracle-backed primitive fixture families |
| Primitive semantics (full V1-scoped surface) | parity_gap | Known gaps tracked in `[V2-PRIM]` (`bd-38qy`) and dependent beads | Complete missing V1-scoped primitives and conformance coverage |
| Interpreter path over canonical IR | in_progress | Interpreter/eval coverage in `crates/fj-interpreters/src/lib.rs` and staging tests in `crates/fj-interpreters/src/staging.rs`; current evaluator still enforces single-output equations | Add broader higher-rank/multi-output oracle parity fixtures and remove remaining execution constraints |
| Dispatch path + transform wrappers (`jit`/`grad`/`vmap`) | in_progress | Dispatch + composition tests in `crates/fj-dispatch/src/lib.rs`; E2E dispatch/ordering artifacts under `artifacts/e2e/e2e_p2c001_*.e2e.json`; grad-order constraints still documented in dispatch path | Extend parity report against broader legacy transform matrix and close current transform constraints |
| Cache-key determinism + strict/hardened split | in_progress | Determinism/mode-split tests in `crates/fj-cache/src/lib.rs`; strict-vs-hardened E2E in `artifacts/e2e/e2e_p2c001_strict_vs_hardened_mode_split.e2e.json` | Component-by-component parity ledger against legacy cache-key behavior |
| Decision/evidence ledger foundation | parity_green | Ledger/test coverage in `crates/fj-ledger/src/lib.rs`; audit-trail E2E in `artifacts/e2e/e2e_p2c004_evidence_ledger_audit_trail.e2e.json` | Add calibration/drift confidence reporting artifacts |
| Conformance harness + transform bundle runner (V1 acceptance scope) | parity_gap | Harness/reporting code in `crates/fj-conformance/src/lib.rs`; transform fixtures in `crates/fj-conformance/fixtures/transforms/legacy_transform_cases.v1.json`; smoke/integration tests in `crates/fj-conformance/tests/` | Full V1 oracle family coverage with acceptance-gate parity artifact |
| Legacy fixture capture automation | in_progress | Capture pipeline script in `crates/fj-conformance/scripts/capture_legacy_fixtures.py` (strict + fallback modes) | Strict-mode run in real `jax`/`jaxlib` environment with reproducible capture log |
| `jit` transform semantics (current fixture families) | parity_green | Transform fixture + API E2E coverage in `crates/fj-conformance/tests/transforms.rs` and `crates/fj-conformance/tests/e2e_p2c002.rs` | Expand to broader `tests/jax_jit_test.py` oracle slices |
| `grad` transform semantics (current fixture families) | parity_gap | AD + transform integration tests in `crates/fj-ad/src/lib.rs` and `crates/fj-conformance/tests/api_transforms_oracle.rs`; remaining grad constraints/gaps tracked in `bd-159q` | Extend oracle families for richer autodiff operator coverage and close known grad limitations |
| `vmap` transform semantics (current fixture families) | in_progress | BatchTrace path in `crates/fj-dispatch/src/batching.rs`; conformance in `crates/fj-conformance/tests/vmap_conformance.rs` and `multirank_conformance.rs`; batching-completeness gaps tracked in `bd-cvtq` | Complete parity for advanced batching/control-flow compositions |
| RNG implementation foundation (ThreeFry + samplers) | in_progress | ThreeFry/key/sampler implementation + tests in `crates/fj-lax/src/threefry.rs` | Integrate RNG path into conformance fixture families |
| RNG determinism vs JAX oracle | parity_gap | No random fixture family currently emitted by `capture_legacy_fixtures.py` | Add differential random fixtures (`tests/random_test.py` anchor set) and parity-gate output |
| RaptorQ sidecar durability pipeline | in_progress | Durability implementation in `crates/fj-conformance/src/durability.rs`; CLI in `crates/fj-conformance/src/bin/fj_durability.rs`; sidecar/scrub/proof artifacts under `artifacts/durability/`; some phase artifacts still tracked as partial | Expand automated sidecar generation to all long-lived evidence bundles and close remaining partials |
| DType system expansion (BF16/F16/U32/U64 and promotion rules) | parity_gap | Baseline + complex dtypes in `crates/fj-core/src/lib.rs`; expansion tasks open (`bd-gsad`, `bd-2983`) | Implement missing dtypes/promotions and add oracle-backed dtype fixtures |
| Vmap batching completeness (`in_axes`/`out_axes`, broader primitive surface) | parity_gap | BatchTrace baseline in `crates/fj-dispatch/src/batching.rs`; completion tasks open (`bd-cvtq` epic and children) | Full per-primitive batching coverage + axis policy parity evidence |
| AD completeness (custom rules, remaining VJP/JVP parity gaps) | parity_gap | Strong core AD coverage in `crates/fj-ad/src/lib.rs`; open AD completion tasks (`bd-159q` epic and children) | Close remaining AD gaps with conformance-backed parity artifacts |
| Control flow completion (`while`/`scan`/`fori` + transform interactions) | parity_gap | Control-flow primitives exist with partial coverage in `crates/fj-lax/src/lib.rs`; control-flow conformance task open (`bd-3q10`) | Add full control-flow fixture families and transform-composition parity reports |
| Tracing from user code + nested transform tracing | in_progress | Closure tracing API in `crates/fj-trace/src/lib.rs` (`make_jaxpr*`) and re-export in `crates/fj-api/src/lib.rs`; nested trace completion task open (`bd-3gxs`) | Complete nested trace composition parity and broaden trace-time validation evidence |
| Special/missing primitive backlog (beyond current subset) | parity_gap | Core special math primitives present (`Erf`, `Erfc`, `Atan2`, etc.); additional scoped primitives tracked in `bd-38qy` | Implement remaining scoped special/missing primitives + add conformance fixtures |

## Required Evidence Per Family

1. Differential conformance report.
2. Invariant checklist entry/update.
3. Benchmark delta report for perf-sensitive changes.
4. Risk note update if compatibility/security surface changed.

## Coverage Objective

Target: 100% coverage for declared V1 scope, with explicit parity exceptions documented by artifact and linked to open bead IDs.
