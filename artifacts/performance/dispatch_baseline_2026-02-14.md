# Dispatch Baseline (2026-02-14)

## Command

```bash
rch exec -- cargo bench --bench dispatch_baseline
```

## Benchmark Results (23 benchmarks, 100 samples each)

### dispatch_latency

| Benchmark | p50 | p95 (upper CI) |
|-----------|-----|----------------|
| jit/scalar_add | ~2.5 us | ~2.54 us |
| jit/scalar_square_plus_linear | ~3.5 us | ~3.57 us |
| jit/vector_add_one | ~2.8 us | ~2.85 us |
| grad/scalar_square | ~4.2 us | ~4.30 us |
| vmap/vector_add_one | ~3.1 us | ~3.15 us |
| vmap/rank2_add_one | ~8.5 us | ~8.65 us |
| jit_grad/scalar_square | ~5.0 us | ~5.10 us |
| vmap_grad/vector_square | ~7.5 us | ~7.65 us |

### eval_jaxpr_throughput

| Benchmark | Time |
|-----------|------|
| chain_add/10 | ~650 ns |
| chain_add/100 | ~5.5 us |
| chain_add/1000 | ~55 us |

### transform_composition

| Benchmark | Time |
|-----------|------|
| single/jit | ~222 ns |
| single/grad | ~220 ns |
| single/vmap | ~223 ns |
| depth2/jit_grad | ~249 ns |
| depth3/jit_vmap_grad | ~275 ns |
| empty_stack | ~172 ns |

### cache_key_generation

| Benchmark | Time |
|-----------|------|
| simple/1eq_1t | ~856 ns |
| medium/3eq_2t | ~853 ns |
| large/100eq_1t | ~2.38 us |
| hardened/unknown_features | ~855 ns |

### ledger_append

| Benchmark | Time |
|-----------|------|
| single_append | ~81 ns |
| burst_100_appends | ~23 us |

### jaxpr_fingerprint

| Benchmark | Time |
|-----------|------|
| canonical_fingerprint/1 | ~354 ns |
| canonical_fingerprint/10 | ~1.69 us |
| canonical_fingerprint/100 | ~12.9 us |
| cached_fingerprint/10eq | ~1.3 ns |

### jaxpr_validation

| Benchmark | Time |
|-----------|------|
| validate_well_formed/1 | ~38 ns |
| validate_well_formed/10 | ~172 ns |
| validate_well_formed/100 | ~2.70 us |

## Scope

This benchmark suite covers all 6 metric categories from bd-3dl.8:

1. **Dispatch latency**: jit/grad/vmap x scalar/vector + compositions
2. **eval_jaxpr throughput**: 10/100/1000 equation chain programs
3. **Transform composition overhead**: single/depth-2/depth-3/empty stacks
4. **Cache key generation**: simple/medium/large/hardened inputs
5. **Ledger append throughput**: single and burst-100 appends
6. **Jaxpr fingerprint + validation**: fresh computation vs OnceLock cached

Crates exercised: fj-core, fj-cache, fj-interpreters, fj-lax, fj-dispatch, fj-ledger, fj-ad.

## CI Gate

Performance regression gate: `scripts/check_perf_regression.sh`
- Threshold: 5% p95 regression (configurable in `reliability_budgets.v1.json`)
- Evidence schema: `artifacts/schemas/perf_delta.v1.schema.json`
- Integrated into `scripts/enforce_quality_gates.sh`

## Notes

- Criterion baseline saved as `main` for future comparisons.
- Supersedes dispatch_baseline_2026-02-13.md (single benchmark only).
- Future optimization commits must compare via `./scripts/check_perf_regression.sh`.
