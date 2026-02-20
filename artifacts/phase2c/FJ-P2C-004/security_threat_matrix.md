# FJ-P2C-004 Security + Compatibility Threat Matrix (bd-3dl.15.3)

## Scope

Packet boundary: Dispatch/effects runtime internals — transform execution engine, AD tape mechanics, vmap slice-stack, evidence ledger population, and effect modeling.

Primary subsystems:
- `fj-dispatch` (transform routing, recursive execution, vmap slice-stack)
- `fj-ad` (reverse-mode AD, forward_with_tape, backward, VJP rules)
- `fj-core` (Transform enum, composition proof, TraceTransformLedger, eval_jaxpr)
- `fj-interpreters` (jaxpr evaluation, primitive dispatch)
- `fj-ledger` (Bayesian decision records, evidence signals, conformal calibration)
- `fj-cache` (cache key construction, strict/hardened policy)

## Threat Matrix

| Threat class | Attack vector | Strict mitigation | Hardened mitigation | Fail-closed boundary | Residual risk | Evidence |
|---|---|---|---|---|---|---|
| Transform stack manipulation | Inject extra transforms that bypass `verify_transform_composition` (e.g. via direct `push_transform` with empty or forged evidence strings) | `verify_transform_composition` checks: evidence count == transform count, no empty evidence strings, at-most-one Grad, at-most-one Vmap. Called before cache key generation or execution | Same as strict; composition proof is never relaxed in hardened mode | TransformCompositionError terminates request before any execution or cache key generation | Low: composition proof is comprehensive for the scoped transform set (Jit/Grad/Vmap). Higher-order transforms (e.g. custom_vjp) not yet in scope | `crates/fj-core/src/lib.rs:947-997`, `crates/fj-dispatch/src/lib.rs:136` |
| Effect ordering violation via evidence ledger | Reorder or duplicate evidence signals to mislead downstream consumers about execution order or transform semantics | Evidence signals are computed internally from verified Jaxpr/transform state (eqn_count, transform_depth, stack_hash). Signals are append-only and derived from actual computation, not user input | Same as strict; no user-controllable signal injection path exists | Ledger entries use cache_key as decision_id; cache_key is SHA-256 of verified internal state | Negligible: no external input path influences signal values or ordering | `crates/fj-dispatch/src/lib.rs:154-179`, `crates/fj-ledger/src/lib.rs` |
| Gradient cache confusion | Two semantically different functions produce identical cache keys, causing cached gradient from function A to be returned for function B | Cache key includes SHA-256 of canonical jaxpr fingerprint + full transform stack + mode + backend + compile_options. Distinct jaxprs produce distinct fingerprints | Same as strict; hardened mode adds unknown_features to hash, increasing entropy | Collision probability < 2^-128 (SHA-256 preimage resistance) | Negligible: cryptographic collision resistance well-established. No jaxpr normalization ambiguity: canonical fingerprint is deterministic | `crates/fj-cache/src/lib.rs`, `crates/fj-dispatch/src/lib.rs:138-146` |
| Dispatch amplification via recursive composition | Pathological transform stacks causing exponential work (e.g. nested finite-diff fallback computing O(2^n) evaluations for n-deep grad stacking) | At-most-one Grad enforced by composition proof. Finite-diff fallback triggered only when grad has tail transforms: each level adds 2 evaluations (plus, minus), bounded by stack depth | Same as strict; no composition relaxation | Maximum evaluations = 2 * (stack_depth - grad_position). With at-most-one Grad and at-most-one Vmap, practical maximum ~2-4 evaluations per dispatch | Low: stack depth is bounded by composition proof to ~3 effective transforms. Rust stack guard prevents overflow for pathological cases | `crates/fj-dispatch/src/lib.rs:188-260`, `crates/fj-core/src/lib.rs:966-986` |
| Evidence ledger flooding | Trigger massive ledger growth via repeated dispatch calls to cause OOM | Each dispatch produces exactly one LedgerEntry with 3 signals (fixed). Ledger is per-request, not global. No accumulation across dispatch calls | Same as strict; no ledger size relaxation | Per-request ledger is O(1) in size; caller controls dispatch frequency | Negligible: bounded output per dispatch. Caller-side rate limiting is an application concern, not a dispatch concern | `crates/fj-dispatch/src/lib.rs:154-179` |
| AD tape memory exhaustion | Large Jaxpr causing forward_with_tape to allocate tape entries proportional to equation count, exhausting memory | Tape size = O(E) where E = equation count in root_jaxpr. Bounded by program construction. No tape sharing across dispatch calls | Same as strict; no tape size relaxation | OOM is process-level termination, not undefined behavior. `#![forbid(unsafe_code)]` prevents buffer overflows | Medium: no explicit tape size cap. Very large programs may legitimately require large tapes. Mitigation: Jaxpr equation count is logged as evidence signal | `crates/fj-ad/src/lib.rs` |
| Vmap slice-stack iteration amplification | Tensor with large leading dimension (e.g. 2^24 elements) causing O(N) jaxpr evaluations via slice-stack | Iteration count = leading dimension of first tensor argument. Each iteration evaluates root_jaxpr once. Total work = N * cost(eval_jaxpr) | Same as strict; no iteration cap relaxation | No explicit iteration cap; system OOM/timeout is the backstop. Leading dimension is extracted from tensor metadata, not user-controlled size field | Medium: no explicit iteration cap. Large legitimate workloads may require large batch sizes. Future: configurable vmap batch size limit | `crates/fj-dispatch/src/lib.rs:262-352` |
| VJP rule correctness attack | Adversarial program targeting a primitive whose VJP rule has a bug, producing incorrect gradients that propagate silently | VJP rules are hardcoded per-primitive in fj-ad backward(). 17 binary/unary + 4 reduction ops covered. Oracle tests validate grad output against analytical derivatives | Same as strict; VJP rules are not mode-dependent | Incorrect gradients are semantic errors, not safety errors. Oracle tests in fj-conformance cover all supported primitives | Medium: VJP rule correctness relies on test coverage. Untested primitives would silently produce wrong gradients. Mitigation: comprehensive oracle + metamorphic test suite | `crates/fj-ad/src/lib.rs`, `crates/fj-conformance/tests/api_transforms_oracle.rs` |
| Conformal calibration manipulation | Supply a ConformalPredictor with adversarial calibration data to skew posterior probability estimates | ConformalPredictor is constructed internally; not exposed in DispatchRequest. calibrated_posterior_abandoned is called with server-controlled conformal predictor | Same as strict | No external input path to ConformalPredictor | Negligible: conformal predictor is an internal component. Heuristic posterior is the fallback when no calibration is available | `crates/fj-dispatch/src/lib.rs:362-375`, `crates/fj-ledger/src/lib.rs` |

## Compatibility Envelope

| JAX dispatch behavior | FrankenJAX status | Strict mode | Hardened mode | Evidence |
|---|---|---|---|---|
| `Primitive.bind()` → trace-based dispatch | DIVERGENT: FrankenJAX uses explicit transform stack iteration via `execute_with_transforms` | Semantically equivalent for supported transforms | Same as strict | `crates/fj-dispatch/src/lib.rs:188-202`, anchor P2C004-A21 |
| `find_top_trace()` trace stack model | NOT IMPLEMENTED: replaced by outside-in recursive execution | Equivalent outcome: transforms applied in correct order | Same as strict | anchor P2C004-A27 |
| Forward-mode AD (JVP) via `JVPTrace` | NOT IMPLEMENTED: reverse-mode only (VJP via forward_with_tape + backward) | Gradients computed via reverse-mode; equivalent for scalar-to-scalar | Same as strict | `crates/fj-ad/src/lib.rs`, anchor P2C004-A09, P2C004-A23 |
| Reverse-mode AD via `backward_pass` | SUPPORTED via fj-ad forward_with_tape + backward | guaranteed | guaranteed | `crates/fj-ad/src/lib.rs`, anchor P2C004-A10 |
| VJP rule registry (`primitive_transposes`) | DIVERGENT: hardcoded match arms instead of registry dict | Functionally equivalent for supported primitives | Same as strict | anchor P2C004-A12, P2C004-A29 |
| `BatchTracer` trace-based vmap | DIVERGENT: slice-stack approach instead of BatchTracer | Semantically equivalent: vmap(f)(batch)[i] == f(batch[i]) | Same as strict | `crates/fj-dispatch/src/lib.rs:262-352`, anchor P2C004-A14 |
| `batch_jaxpr` axis transformation | NOT IMPLEMENTED: slice-stack does not transform jaxpr | Equivalent output for axis-0 batching | Same as strict | anchor P2C004-A15 |
| `WrappedFun` generator protocol | DIVERGENT: ComposedTransform with Vec<Transform> builder | Equivalent composition semantics | Same as strict | anchor P2C004-A18 |
| `linear_util.cache` memoization | DIVERGENT: SHA-256 cache key instead of WeakKeyDictionary | Equivalent caching semantics; different key construction | Same as strict | `crates/fj-cache/src/lib.rs`, anchor P2C004-A19 |
| Effect system (OrderedEffect, token threading) | NOT IMPLEMENTED: effects modeled via EvidenceLedger instead | Out-of-scope for V1; no effect-dependent primitives supported | Same as strict | anchor P2C004-A05 through P2C004-A08 |
| `EffectTypeSet` registries | NOT IMPLEMENTED: no effect registries | Out-of-scope for V1 | Same as strict | anchor P2C004-A28 |
| `RuntimeTokenSet` per-effect tokens | NOT IMPLEMENTED | Out-of-scope for V1 | Same as strict | anchor P2C004-A04 |
| `ad.Zero` / `UndefinedPrimal` sentinels | NOT IMPLEMENTED: no sparse gradient optimization | Semantically equivalent but less efficient for sparse cases | Same as strict | anchor P2C004-A13 |
| `Store` write-once semantics | NOT APPLICABLE: no generator protocol | N/A | N/A | anchor P2C004-A20 |
| Sharded lowering / XLA compilation | NOT IMPLEMENTED: direct eval_jaxpr interpreter | Out-of-scope for V1; CPU-only single-backend | Same as strict | anchor P2C004-A03, P2C004-A22 |
| `DeferredShardArg` multi-device batching | NOT APPLICABLE: single-backend CPU | Out-of-scope | Same as strict | anchor P2C004-A30 |

## Explicit Fail-Closed Rules

1. Transform composition proof failure (`TransformCompositionError`) terminates the request before cache key generation or execution.
2. Evidence count mismatch (transform count != evidence count) is fail-closed.
3. Empty evidence strings are rejected at composition proof time.
4. At-most-one Grad and at-most-one Vmap enforced before any execution.
5. Non-scalar gradient input terminates with `NonScalarGradientInput` error.
6. Non-scalar gradient output terminates with `NonScalarGradientOutput` error.
7. Vmap leading-dimension mismatch terminates at the first inconsistent argument.
8. Empty vmap output (zero-length leading dimension) terminates before iteration.
9. Strict mode rejects unknown incompatible features with fail-closed `CacheKeyError`.
10. All dispatch errors propagate via `Result<T, DispatchError>` — no raw panics escape.
11. `#![forbid(unsafe_code)]` on all crates prevents memory safety violations.
12. Evidence ledger signals are derived from verified internal state; no external injection path.
