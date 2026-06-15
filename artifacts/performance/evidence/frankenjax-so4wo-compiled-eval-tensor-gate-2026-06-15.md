# frankenjax-so4wo compiled eval cache — broaden to tensor/reduction programs

Date: 2026-06-15
Agent: SilverMaple
Lever: drop the `has_scalar_fast_path()` gate in `compile_jaxpr_for_repeated_eval`
so the cached compiled dense plan also serves small **non-scalar** Jaxprs
(reductions, dot, small tensor elementwise). These programs previously had no
scalar fast-path plan, so every repeated default-CPU JIT call fell through to the
full per-call dispatch (rebuilding the `DenseEvalPlan` and re-running the
top-level probes each time).

## Why it is safe

`CompiledJaxpr::eval` runs through `run_dense_plan`, whose generic per-equation
fallback `run_dense_env_into` dispatches the same `eval_primitive` /
`eval_equation_outputs_from_resolved` kernels in the same topological order as
`eval_jaxpr_with_consts` (the interpreter the backend already uses). Caching the
plan changes only the plumbing, not the math. The compile pass still rejects
const-bearing, effectful, sub-jaxpr-bearing, or non-uniquely-bound programs and
still requires a buildable dense slot plan, so ineligible programs keep their
existing dispatch/backend route.

## Same-binary A/B rebench

Command:

```bash
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p fj-api --bench api_overhead -- jit_compiled_eval_cache_tensor --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot
```

Worker: remote `vmi1227854`.

Rows (ReduceSum over an 8-element f64 vector, prepared metadata both arms):

- `jit_compiled_eval_cache_tensor/dispatch_prepared/reduce_sum`: 1.0334 us, 1.0878 us, 1.1256 us
- `jit_compiled_eval_cache_tensor/api_compiled/reduce_sum`: 299.81 ns, 307.89 ns, 313.98 ns

Speedup:

- Midpoint: 1087.8 / 307.89 = **3.53x**
- Conservative interval: 1033.4 / 313.98 = **3.29x**
- Score: 10.0 (Impact 4, Confidence 5, Effort 2)

Final closeout rerun:

- Command: `rch exec -- cargo bench -p fj-api --bench api_overhead -- jit_compiled_eval_cache_tensor`
- Worker: `vmi1149989`
- `jit_compiled_eval_cache_tensor/dispatch_prepared/reduce_sum`: 1.1771 us, 1.2631 us, 1.3611 us
- `jit_compiled_eval_cache_tensor/api_compiled/reduce_sum`: 278.61 ns, 286.25 ns, 294.26 ns
- Midpoint: 1263.1 / 286.25 = **4.41x**
- Conservative interval: 1177.1 / 294.26 = **4.00x**

## Isomorphism proof

- Ordering: compiled eval reuses the existing dense plan and runs the same step
  order as `run_dense_plan` / `eval_jaxpr_with_consts`.
- Tie-breaking: no comparison or sort tie policy changes.
- Floating point: no reassociation, FMA, vector lane reduction, or mixed
  precision changes; primitive closures are unchanged (`eval_primitive`).
- RNG: no RNG/effectful program is eligible (effects rejected at compile).
- Error/fallback surface: only no-const, effect-free, sub-jaxpr-free,
  uniquely-bound dense Jaxprs with bound inputs and a buildable dense plan are
  accepted; everything else falls back to the existing dispatch/backend path.
- Golden output sha256 (ReduceSum result):
  `00973a72bf25a5a56152d373c887ddc555877c6e09545140c723080675c2676a`.

## Validation

- `cargo fmt --check -p fj-api -p fj-interpreters`
- `rch exec -- cargo clippy -p fj-api -p fj-interpreters --all-targets -- -D warnings`
- `rch exec -- cargo test -p fj-interpreters -p fj-api --lib` (92 + 191 pass)
  - new `transforms::tests::jit_repeated_call_compiled_cache_matches_dispatch_tensor_programs`
    asserts compiled-cache == backend dispatch for ReduceSum / Dot / Add tensor
    programs.
  - new `transforms::tests::jit_repeated_call_compiled_tensor_golden_sha256`
    pins the ReduceSum golden output.
- `rch exec -- cargo test -p fj-conformance`

Final closeout reruns:

- `cargo fmt --check -p fj-api -p fj-interpreters`
- `rch exec -- cargo test -j 1 -p fj-interpreters -p fj-api --lib`
  - `fj-api`: 92 passed
  - `fj-interpreters`: 191 passed, 17 ignored
- `rch exec -- cargo check -j 1 -p fj-api -p fj-interpreters --all-targets`
- `rch exec -- cargo clippy -j 1 -p fj-api -p fj-interpreters --all-targets -- -D warnings`
