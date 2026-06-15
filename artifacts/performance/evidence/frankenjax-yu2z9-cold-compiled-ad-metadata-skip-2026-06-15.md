# frankenjax-yu2z9: cold compiled AD metadata skip

Date: 2026-06-15
Agent: SilverMaple
Bead: frankenjax-yu2z9

## Target

Fresh RCH `ovh-a` API profile after `frankenjax-zhxk0` showed cold AD wrapper overhead dominating small compiled-AD calls:

- `api_overhead/grad/scalar_square`: `[1.4645us 1.4668us 1.4674us]`
- `api_overhead/value_and_grad/scalar_square`: `[3.0426us 3.0491us 3.0507us]`
- warmed compiled AD was already sub-microsecond, which pointed at wrapper metadata/build overhead rather than tape AD.

Baseline command:

```text
rch exec -- cargo bench -j 1 -p fj-api --bench api_overhead -- --quick --noplot
```

## Lever

`GradWrapped::call` and `ValueAndGradWrapped::call` now try the default-CPU, no-custom-VJP compiled AD plan before constructing compile options, transform evidence, and prepared dispatch metadata.

Unsupported graphs, custom VJP wrappers, and non-default backends still fall through to the previous dispatch path with the same arguments and options.

## Results

Post-change same-worker command:

```text
RCH_WORKER=ovh-a rch exec -- cargo bench -j 1 -p fj-api --bench api_overhead -- 'api_overhead/(grad|value_and_grad)/scalar_square|ad_compiled_reverse_plan/api_warmed/value_and_grad_trig|ad_compiled_reverse_plan/api_warmed/value_and_grad_sum_x2_plus_x_1k' --quick --noplot
```

Post-change RCH `ovh-a` results:

- `api_overhead/grad/scalar_square`: `[553.46ns 554.29ns 554.50ns]`
- `api_overhead/value_and_grad/scalar_square`: `[534.21ns 534.97ns 535.17ns]`
- `ad_compiled_reverse_plan/api_warmed/value_and_grad_trig`: `[117.25ns 117.38ns 117.90ns]`
- `ad_compiled_reverse_plan/api_warmed/value_and_grad_sum_x2_plus_x_1k`: `[762.86ns 764.28ns 764.63ns]`

Ratios:

- `grad/scalar_square`: `1.4668us / 554.29ns = 2.65x`
- `value_and_grad/scalar_square`: `3.0491us / 534.97ns = 5.70x`
- primary Score: `3.6` (`Impact 5.70 * Confidence 0.95 / Effort 1.5`)

## Isomorphism Proof

- Ordering: compiled AD executes the same compiled reverse plan that the previous wrapper already used after metadata derivation; the change only moves the eligibility probe earlier.
- FP/RNG/ties: no primitive arithmetic, floating-point association, random source, or tie-breaking surface changes.
- Fallback: if no compiled plan applies, the old compile-option/evidence/metadata/dispatch path still runs unchanged.
- Custom VJP: guarded out before the compiled probe, preserving custom-VJP deferral.
- Backend: non-default backends are guarded out before the compiled probe.
- Metadata errors: API wrappers build transform evidence internally and pass an empty unknown-feature list, so `prepare_dispatch_meta` has no independent observable failure mode for this guarded default path.

## Golden / Validation

- New focused golden test: `cold_compiled_ad_fast_path_matches_dispatch_golden_sha256`
  - first-call API `grad` and `value_and_grad` match direct dispatch outputs exactly
  - payload SHA-256: `04e2c5d5b04781ce3140dd383e1d5b7dba88f3b87f716d6c86ee40b210cad05a`
- Existing compiled AD tape golden still passes:
  - `value_and_grad_compiled_ad_cache_matches_tape_golden_sha256`
  - payload SHA-256: `984585309be003365780a1f999422efc949c360ec9933d354a6bd50b5b41653a`

Validation commands:

```text
cargo fmt --check
RCH_WORKER=ovh-a rch exec -- cargo test -j 1 -p fj-api compiled_ad -- --nocapture
RCH_WORKER=ovh-a rch exec -- cargo check -j 1 -p fj-api --all-targets
RCH_WORKER=ovh-a rch exec -- cargo clippy -j 1 -p fj-api --all-targets -- -D warnings
```
