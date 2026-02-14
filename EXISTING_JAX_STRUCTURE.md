# EXISTING_JAX_STRUCTURE

DOC-PASS-00 baseline matrix: `artifacts/docs/bd-3dl.23.1_gap_matrix.v1.md`
DOC-PASS-01 module/package cartography snapshot: `bd-3dl.23.2` (2026-02-14)

## 1. Legacy Oracle

- Root: `/data/projects/frankenjax/legacy_jax_code/jax`
- Legacy alias path used in older docs/scripts: `/dp/frankenjax/legacy_jax_code/jax`
- Upstream: `jax-ml/jax`

## 2. High-Value File/Function Anchors

### Transform API entry points

| Anchor | Primary symbols | Ownership boundary | Downstream coupling |
|---|---|---|---|
| `jax/_src/api.py` | `jit`, `vmap`, `grad`, `value_and_grad`, `_vjp` | User-visible transform contract boundary | `pjit.py`, `stages.py`, interpreter transforms |

### JIT and staging/lowering path

| Anchor | Primary symbols | Ownership boundary | Downstream coupling |
|---|---|---|---|
| `jax/_src/pjit.py` | `JitWrapped`, `_parse_jit_arguments`, `_run_python_pjit` | JIT argument normalization and staged execution boundary | `dispatch.py`, compilation/cache modules |
| `jax/_src/stages.py` | `Traced` | Trace materialization and lowering metadata boundary | `partial_eval.py`, interpreter/lowering pipeline |

### Trace/Jaxpr construction and partial-eval boundary

| Anchor | Primary symbols | Ownership boundary | Downstream coupling |
|---|---|---|---|
| `jax/_src/core.py` | `Trace`, `Tracer`, `Jaxpr` | Canonical IR typing and tracer lifecycle boundary | AD, batching, dispatch interpreters |
| `jax/_src/interpreters/partial_eval.py` | `trace_to_jaxpr` | Trace-to-IR lowering + residual handling boundary | `dispatch.py`, `ad.py`, `batching.py` |

### AD (reverse/forward transform semantics)

| Anchor | Primary symbols | Ownership boundary | Downstream coupling |
|---|---|---|---|
| `jax/_src/interpreters/ad.py` | `linearize_subtrace`, `linearize_jaxpr`, `backward_pass3` | Reverse/forward transform semantics boundary | API transform composition (`grad`, `value_and_grad`) |

### Batching / vmap semantics

| Anchor | Primary symbols | Ownership boundary | Downstream coupling |
|---|---|---|---|
| `jax/_src/interpreters/batching.py` | `BatchTrace`, `BatchTracer`, `batch_subtrace`, `batch_jaxpr` | Batch-axis semantics boundary | API transform composition (`vmap`) and dispatch path |

### Cache-key and compilation cache semantics

| Anchor | Primary symbols | Ownership boundary | Downstream coupling |
|---|---|---|---|
| `jax/_src/cache_key.py` | `get`, `_hash_serialized_compile_options`, `_hash_xla_flags`, `_hash_accelerator_config` | Cache identity construction boundary | `compiler.py`, `compilation_cache.py`, backend/device metadata |
| `jax/_src/compiler.py` | `compile_or_get_cached`, `_resolve_compilation_strategy`, `_cache_read`, `_cache_write` | Compile/cache lifecycle boundary | `dispatch.py`, backend bridge, persistent cache storage |
| `jax/_src/compilation_cache.py` | `get_cache_key`, `get_executable_and_time`, `put_executable_and_time` | Persistent cache storage boundary | compiler strategy + backend-specific executable handling |

### Dispatch-level memoization and trace caching

| Anchor | Primary symbols | Ownership boundary | Downstream coupling |
|---|---|---|---|
| `jax/_src/dispatch.py` | `xla_primitive_callable`, `_is_supported_cross_host_transfer` | Runtime dispatch + memoization boundary | compiler/cache modules + backend bridge |
| `jax/_src/interpreters/partial_eval.py` | `trace_to_jaxpr` | Trace caching boundary (pre-dispatch IR reuse) | staging + dispatch + transform interpreters |

### Backend bridge and FFI boundary

| Anchor | Primary symbols | Ownership boundary | Downstream coupling |
|---|---|---|---|
| `jax/_src/xla_bridge.py` | backend client resolution | backend selection + runtime protocol boundary | `dispatch.py`, `compiler.py`, device runtime |
| `jax/_src/ffi.py`, `jaxlib/*.cc` | FFI registration/callback surfaces | native bridge lifetime and callback safety boundary | runtime client/device/program ownership |

## 3. Semantic Hotspots (Non-Negotiable)

1. Transform composition semantics for `jit`, `grad`, `vmap`.
2. Deterministic trace/Jaxpr construction and lowering metadata.
3. Cache-key soundness and compatibility-sensitive inputs.
4. Dispatch cache behavior and memoization invariants.
5. Error-path behavior when metadata or shape contracts are incompatible.

## 4. Conformance Fixture Family Anchors

- `jit`: `tests/jax_jit_test.py`
- `grad`: `tests/lax_autodiff_test.py` and `tests/lax_test.py`
- `vmap`: `tests/lax_vmap_test.py` and `tests/lax_vmap_op_test.py`
- `lax primitives`: `tests/lax_test.py`
- `random`: `tests/random_test.py` and `tests/random_lax_test.py`

## 5. Compatibility-Critical Inputs for Cache Keying

The legacy cache key uses deterministic hashing over a canonicalized input bundle that includes:

- canonicalized module bytecode
- backend/version fields
- normalized compile options
- normalized XLA flags
- accelerator topology/device descriptors
- optional runtime custom hook bits

FrankenJAX MUST preserve input-surface semantics for scoped compatibility.

## 6. Security and Reliability Risk Areas

- cache confusion via incompatible/unknown metadata
- transform-order drift leading to semantic mismatch
- malformed shape/graph signatures
- stale cache artifacts or corruption in persistent storage

## 7. Extraction Boundary (Current)

Included now:

- transform API semantics and composition invariants
- deterministic key derivation contract
- core conformance fixture families

Deferred but tracked:

- broad plugin/distributed backend breadth
- long-tail API surfaces outside scoped parity matrix

## 8. FrankenJAX Workspace Topology (Current Rust Slice)

### 8.1 Crate Ownership and Boundary Matrix

| Crate | Owns | Depends on (normal) | Boundary contract |
|---|---|---|---|
| `fj-core` | canonical IR (`Jaxpr`), TTL (`TraceTransformLedger`), transforms/types | none | root semantic model; no runtime/backend side effects |
| `fj-lax` | primitive eval semantics (`eval_primitive`) | `fj-core` | pure primitive semantics over core values |
| `fj-interpreters` | IR interpreter (`eval_jaxpr`) | `fj-core`, `fj-lax` | executes canonical IR without transform orchestration |
| `fj-ad` | autodiff path (`grad_first`, tape/backward logic) | `fj-core`, `fj-lax` | gradient semantics; consumes IR/value model |
| `fj-cache` | deterministic cache key derivation | `fj-core` | strict/hardened key gate for unknown metadata |
| `fj-ledger` | evidence/decision ledger + calibration logic | `fj-core` | policy/evidence state, no dispatch execution |
| `fj-dispatch` | orchestration (`dispatch`) and transform wrapper execution | `fj-ad`, `fj-cache`, `fj-core`, `fj-interpreters`, `fj-ledger` | integration choke point for transform order + cache behavior |
| `fj-runtime` | admission policy model + optional bridges | `fj-core`, `fj-ledger` | operational runtime decision boundary |
| `fj-egraph` | e-graph optimizer + IR round trip | `fj-core` | optimization sandbox, not yet dispatch-integrated |
| `fj-conformance` | fixture/parity harness + durability pipeline | `fj-core`, `fj-dispatch` | oracle comparison and durability evidence generation |
| `fj-test-utils` | common test log schema + fixture IDs | none | cross-crate test contract only (dev dependency) |

### 8.2 Cross-Module Dependency Direction Map (Cycle-Checked)

Normal dependency edges (from `cargo metadata --no-deps --format-version 1`):

- `fj-lax -> fj-core`
- `fj-interpreters -> fj-core`, `fj-interpreters -> fj-lax`
- `fj-ad -> fj-core`, `fj-ad -> fj-lax`
- `fj-cache -> fj-core`
- `fj-ledger -> fj-core`
- `fj-dispatch -> fj-ad`, `fj-dispatch -> fj-cache`, `fj-dispatch -> fj-core`, `fj-dispatch -> fj-interpreters`, `fj-dispatch -> fj-ledger`
- `fj-runtime -> fj-core`, `fj-runtime -> fj-ledger`
- `fj-egraph -> fj-core`
- `fj-conformance -> fj-core`, `fj-conformance -> fj-dispatch`

Cycle check:
- Topological sort (`tsort`) over normal edges succeeds without error.
- Result confirms acyclic layering for current workspace crate graph.

### 8.3 Layering Constraints (Current)

| Layer | Crates | Allowed outbound edges | Forbidden patterns |
|---|---|---|---|
| `L0 model` | `fj-core` | none | any dependency on higher execution/runtime/harness layers |
| `L1 semantics` | `fj-lax`, `fj-cache`, `fj-ledger` | `-> fj-core` | cross-calls into dispatch/harness |
| `L2 execution` | `fj-interpreters`, `fj-ad`, `fj-dispatch` | `-> L0/L1` | reverse dependency from `fj-core` into execution |
| `L3 ops/harness` | `fj-runtime`, `fj-egraph`, `fj-conformance` | `-> L0/L2` (as needed) | production execution paths depending on conformance harness |

## 9. Hidden/Implicit Coupling Register

| Coupling | Why implicit | Risk | Mitigation target |
|---|---|---|---|
| Transform-order law split between `fj-core::verify_transform_composition` and `fj-dispatch` transform execution | validation and execution are in separate crates with shared assumptions | semantic drift between proof and execution | add shared transform contract tests in `fj-conformance` |
| Strict/hardened mode semantics span `fj-core`, `fj-cache`, `fj-dispatch`, `fj-runtime` | mode flag is propagated, not centrally enforced | inconsistent fail-closed behavior | centralize mode contract matrix and cross-crate invariant tests |
| `fj-test-utils` schema contract is global across crates | test log shape is shared via dev dependency only | silent logging drift across crate tests | keep schema tests mandatory in every crate |
| `fj-conformance` directly depends on `fj-dispatch` | harness executes current integration choke point | harness and execution semantics can co-evolve accidentally | add explicit fixture-versioning and drift-class gates |
| `fj-egraph` optimization path is not wired into dispatch | optimization correctness is validated in isolation | future integration may bypass parity gates | require conformance parity run before enabling optimization path |

## 10. Conflict Check: Current vs Target Architecture

| Target architecture component | Current owner/location | Status | Conflict impact |
|---|---|---|---|
| `trace` crate (`fj-trace`) | folded into `fj-core` and `fj-interpreters` | missing dedicated crate | trace/lowering boundary not explicit |
| `canonical IR` crate (`fj-ir`) | represented in `fj-core` | partially satisfied | IR ownership exists but not isolated as planned crate |
| `transform stack` crate (`fj-transforms`) | folded into `fj-dispatch` + `fj-core` proof logic | missing dedicated crate | transform policy spread across crates |
| `lowering` crate (`fj-lowering`) | folded into interpreter/dispatch path | missing | lowering contracts hard to isolate and benchmark separately |
| `backend` crate (`fj-backend-cpu`) | backend bridge emulated via current runtime/dispatch slice | missing | backend compatibility matrix remains implicit |
| top-level `frankenjax` API crate | not present | missing | no single stable API facade crate yet |

Conflict-check verdict:
- Current crate dependency graph is acyclic and layered.
- Architecture conflicts are primarily missing crate boundary extractions, not dependency cycles.
- Highest-priority boundary work remains trace/transform/lowering decomposition and backend facade isolation.

## 11. Symbol/API Census and Surface Classification

Census scope for V1 parity slice:
- legacy anchors in section `2` (transform, trace, cache, dispatch, backend/ffi bands),
- Rust workspace exported symbols in `fj-core`, `fj-lax`, `fj-interpreters`, `fj-ad`, `fj-cache`, `fj-dispatch`, `fj-ledger`, `fj-runtime`, `fj-conformance`, `fj-egraph`.

### 11.1 Surface Classes

| Class | Meaning | Stability expectation | Break policy |
|---|---|---|---|
| `S0 user-contract` | user-observable transform behavior/API contract | strict compatibility target | break only with explicit parity exception |
| `S1 compatibility-critical internal` | internal symbol that changes user-visible outcomes | high stability target | guarded by differential/parity gates |
| `S2 workspace-internal` | implementation helper surface | medium stability target | may change if invariants + tests hold |
| `S3 experimental/internal` | optional or not yet integrated surface | low stability target | must not affect strict-mode user contract |

### 11.2 Legacy Symbol Census (Scoped Families)

| Family | Representative symbols | Class | User visibility | Regression tag |
|---|---|---|---|---|
| Transform API front door | `jit`, `vmap`, `grad`, `value_and_grad`, `_vjp` (`jax/_src/api.py`) | `S0` (`jit`/`vmap`/`grad`), `S1` (`_vjp`) | direct | `P0-transform-order` |
| Trace/Jaxpr construction | `Trace`, `Tracer`, `Jaxpr` (`jax/_src/core.py`) | `S1` | indirect but semantics-critical | `P0-trace-shape` |
| Partial-eval boundary | `trace_to_jaxpr` (`jax/_src/interpreters/partial_eval.py`) | `S1` | indirect | `P0-lowering-determinism` |
| AD transform internals | `linearize_subtrace`, `linearize_jaxpr`, `backward_pass3` (`jax/_src/interpreters/ad.py`) | `S1` | indirect (`grad`) | `P1-ad-correctness` |
| Batching transform internals | `BatchTrace`, `BatchTracer`, `batch_subtrace`, `batch_jaxpr` (`jax/_src/interpreters/batching.py`) | `S1` | indirect (`vmap`) | `P1-batch-axis` |
| Cache key construction | `get`, `_hash_serialized_compile_options`, `_hash_xla_flags`, `_hash_accelerator_config` (`jax/_src/cache_key.py`) | `S1` | indirect | `P0-cache-key` |
| Compile/cache lifecycle | `compile_or_get_cached`, `_resolve_compilation_strategy`, `_cache_read`, `_cache_write` (`jax/_src/compiler.py`) | `S1` | indirect | `P0-cache-lifecycle` |
| Persistent compilation cache | `get_cache_key`, `get_executable_and_time`, `put_executable_and_time` (`jax/_src/compilation_cache.py`) | `S1` | indirect | `P0-cache-persistence` |
| Dispatch callable path | `xla_primitive_callable`, `_is_supported_cross_host_transfer` (`jax/_src/dispatch.py`) | `S1` | indirect | `P0-dispatch-routing` |
| Backend/FFI bridge | backend bridge and FFI surfaces (`jax/_src/xla_bridge.py`, `jax/_src/ffi.py`, `jaxlib/*.cc`) | `S1`/`S3` | indirect | `P0-backend-ffi` |

### 11.3 Rust Workspace Symbol Census (Current Slice)

`rg '^pub (struct|enum|fn|mod|type|trait)'` count snapshot (2026-02-14):
- `fj-core=30`, `fj-conformance=43`, `fj-ledger=15`, `fj-test-utils=10`, `fj-cache=7`, `fj-dispatch=6`, `fj-egraph=5`, `fj-ad=3`, `fj-runtime=3`, `fj-interpreters=2`, `fj-lax=2`.

| Family | Key symbols | Class | Boundary note | Regression tag |
|---|---|---|---|---|
| IR + transform proof | `Jaxpr`, `TraceTransformLedger`, `TransformCompositionProof`, `verify_transform_composition` (`fj-core`) | `S1` | public in crate, but stability governed by parity semantics | `P0-transform-proof` |
| Primitive semantics | `eval_primitive`, `EvalError` (`fj-lax`) | `S1` | primitive semantics feed all execution paths | `P0-primitive-semantics` |
| Interpreter execution | `eval_jaxpr`, `InterpreterError` (`fj-interpreters`) | `S1` | execution core for transform wrappers | `P0-ir-exec` |
| AD execution | `grad_jaxpr`, `grad_first`, `AdError` (`fj-ad`) | `S1` | drives `grad` transform behavior | `P1-ad-numerics` |
| Cache keying | `build_cache_key`, `build_cache_key_ref`, `compatibility_matrix_row`, `CacheKeyError` (`fj-cache`) | `S1` | strict/hardened compatibility gate | `P0-cache-key` |
| Dispatch integration | `dispatch`, `DispatchRequest`, `DispatchResponse`, `DispatchError` (`fj-dispatch`) | `S0`/`S1` | integration choke point for user-observable behavior | `P0-dispatch-order` |
| Decision ledger/runtime policy | `EvidenceLedger`, `recommend_action`, `RuntimeAdmissionModel` (`fj-ledger`, `fj-runtime`) | `S1`/`S2` | policy can alter runtime acceptance path | `P1-policy-drift` |
| Conformance and durability | `run_transform_fixture_bundle`, `capture_transform_fixture_bundle_with_oracle`, `encode_artifact_to_sidecar`, `scrub_sidecar`, `generate_decode_proof` (`fj-conformance`) | `S1` | verification plane; guards parity and durability evidence | `P0-parity-gate` |
| E-graph optimization | `optimize_jaxpr`, `algebraic_rules` (`fj-egraph`) | `S3` | currently isolated from dispatch path | `P2-future-optimizer` |

### 11.4 High-Regression Watchlist (Tagged)

| Symbol/band | Why high regression risk | Required guardrail |
|---|---|---|
| `jit`/`grad`/`vmap` + `verify_transform_composition` + `dispatch` | transform-order drift changes user semantics | transform composition differential fixtures + strict drift gate |
| `cache_key.py:get` + `fj-cache::build_cache_key` | cache confusion/collision risk and compatibility drift | component-level cache-key parity ledger |
| `trace_to_jaxpr` + `fj-interpreters::eval_jaxpr` | non-deterministic lowering/execution drift | deterministic trace fixtures and replay corpus |
| `fj-dispatch::dispatch` error path | fail-open vs fail-closed ambiguity | strict/hardened divergence report with allowlist |
| durability pipeline (`encode`/`scrub`/`proof`) | evidence corruption can mask regressions | decode-proof gate on all long-lived artifacts |

Boundary clarification:
- There is not yet a top-level `frankenjax` facade crate; current `pub` Rust symbols are workspace-public, not final end-user API guarantees.
- For V1 compatibility claims, `S0` and `S1` surfaces are normative; `S2`/`S3` remain implementation-detail unless explicitly promoted.

## 12. Data Model, State, and Invariant Mapping (DOC-PASS-03)

### 12.1 Legacy-to-Rust Data Model Mapping

| Legacy model band | Rust owner | Core fields/state | Invariant obligations |
|---|---|---|---|
| Tracer/Jaxpr graph (`core.py`) | `fj-core` (`Jaxpr`, `Equation`, `VarId`, `TraceTransformLedger`) | invars/outvars, equation edges, transform list | var def-use integrity, deterministic fingerprint/signature |
| Partial-eval state (`partial_eval.py`) | `fj-interpreters` + `fj-core` | trace-to-jaxpr boundary, residual equation ordering | stable lowering order and explicit residual handling |
| AD tape/linearization (`ad.py`) | `fj-ad` | primal values, tangent/cotangent propagation | gradient path consistency and scalar-output constraints |
| Batching state (`batching.py`) | `fj-dispatch` + `fj-interpreters` (current slice) | mapped axis semantics and batched argument slicing | vmap order semantics and output-shape consistency |
| Cache identity and lifecycle (`cache_key.py`, `compiler.py`) | `fj-cache` + `fj-dispatch` | key input bundle, backend/options metadata, lookup/write path | strict fail-closed on unknown incompatibilities |
| Dispatch/runtime envelope (`dispatch.py`, bridge bands) | `fj-dispatch`, `fj-runtime` | request, transform stack, admission decision state | transform order preservation and policy traceability |

### 12.2 Critical State Machines

| Component | State sequence | Safety check | Failure semantics |
|---|---|---|---|
| Trace to executable IR | `build trace -> append equations -> seal outvars -> evaluate` | equation inputs must reference defined vars | strict: error and stop; hardened: audit + bounded handling |
| Transform stack execution | `verify composition -> apply transform wrappers -> evaluate root jaxpr` | transform/evidence cardinality and order | strict: reject on mismatch; hardened: log bounded fallback only |
| Cache/dispatch path | `canonicalize request -> key hash -> cache lookup -> hit/miss branch` | key determinism + compatibility gate | strict: fail closed on unknown incompatible metadata |
| Admission decision | `collect evidence -> posterior -> recommend action -> execute` | posterior/action consistency with loss matrix | inconsistent action path is denied and logged |
| Durability verification | `generate sidecar -> scrub integrity -> decode proof` | sidecar/source hash parity + decode success | decode/scrub mismatch blocks artifact acceptance |

### 12.3 Invariant Violation and Recovery Map

| Invariant family | Violation signal | Strict behavior | Hardened behavior |
|---|---|---|---|
| Transform order and composition | unsupported sequence, evidence mismatch | reject request | bounded fallback only when explicitly allowed + ledger entry |
| Cache identity soundness | unknown incompatible feature or unstable key input | reject key path | continue only with explicit compatibility event |
| Shape/graph integrity | malformed vars/shapes/graph signatures | fail-closed | fail-closed with diagnostic context |
| Dispatch policy traceability | missing evidence or ambiguous decision path | abort and emit deterministic error | abort and emit deterministic error + additional policy metadata |
| Durability integrity | scrub hash mismatch or decode-proof failure | block release | block release |

## 13. Execution-Path Tracing and Control-Flow Narratives (DOC-PASS-04)

### 13.1 Workflow Trace A: Transform API Request -> Dispatch Result

Canonical control flow (`jit`/`grad`/`vmap` request):

1. User-facing transform entry (`jax/_src/api.py` analog) receives function + args.
2. Trace/IR boundary constructs canonical program representation (`fj-core` TTL/Jaxpr path).
3. `fj-core::verify_transform_composition` validates transform order + evidence cardinality.
4. `fj-cache` derives compatibility-gated cache identity.
5. `fj-dispatch::dispatch` executes transform wrappers in declared stack order.
6. `fj-interpreters::eval_jaxpr` evaluates terminal IR path.
7. `fj-ledger` records evidence/decision metadata and response returns.

Branch points and outcomes:

| Branch point | Condition | Strict branch | Hardened branch |
|---|---|---|---|
| composition verification | unsupported transform sequence or evidence mismatch | reject request immediately | bounded fallback only if explicitly allowlisted; always log divergence |
| cache compatibility gate | unknown incompatible metadata in key input | fail-closed before dispatch | compatibility event emitted; continuation only through bounded audited path |
| `grad` wrapper execution | AD path unsupported for current composition | deterministic error | bounded finite-difference fallback if allowed by packet policy |
| interpreter execution | malformed var/shape graph | deterministic fail-closed error | fail-closed with richer diagnostics |

### 13.2 Workflow Trace B: Cache Hit/Miss Runtime Branching

1. Canonical request fingerprint materialized.
2. Cache key computed from backend/options/flags/shape context.
3. Lookup branch:
   - hit -> reuse cached executable/derived path.
   - miss -> compile/evaluate path, then cache write.
4. Dispatch returns output + cache-key metadata.

Error/fallback branches:

| Branch point | Failure mode | Recovery law |
|---|---|---|
| key derivation | non-canonical or incompatible input bundle | strict fail-closed; hardened only via explicit compatibility event |
| cache read | stale/corrupt cache artifact | bypass cache, re-evaluate, and emit cache-integrity event |
| cache write | write with unstable key material | block write path and preserve run result without poisoning cache |

### 13.3 Workflow Trace C: Conformance and Differential Validation

1. Fixture bundle load (`fj-conformance` fixture family).
2. Case execution through dispatch pipeline.
3. Comparator branch (`exact`, `approx`, `shape`, `type`) selects drift class.
4. Parity report emitted with strict/hardened outcome split.
5. Failure branch emits minimized repro + mismatch taxonomy.

Failure branch obligations:
- no silent downgrade of mismatch severity,
- include replay pointers for scenario-level reproduction,
- preserve strict/hardened divergence explicitly.

### 13.4 Workflow Trace D: Durability Sidecar Pipeline

1. `generate`: encode artifact to sidecar symbols.
2. `scrub`: decode + hash-compare integrity.
3. `proof`: decode under symbol-loss simulation.
4. `gate`: artifact accepted only if scrub and proof pass.

Failure branches:

| Branch point | Failure signal | Required action |
|---|---|---|
| scrub | hash mismatch | mark artifact invalid, block release path |
| proof | decode failure under configured symbol loss | block release, require sidecar regeneration/recovery evidence |
| versioning | sidecar schema mismatch | fail-closed until migration path is explicitly approved |

### 13.5 Verification-Bead Links

Execution-path narratives in this section are expected to be validated and cross-walked by:
- `bd-3dl.12.5` (unit/property tests + structured logging),
- `bd-3dl.12.6` (differential/metamorphic/adversarial validation),
- `bd-3dl.12.7` (E2E scenario scripts + replay/forensics),
- `bd-3dl.23.10` (unit/E2E test corpus and logging evidence crosswalk),
- `bd-3dl.23.11` (EXISTING_JAX_STRUCTURE expansion draft integration).
