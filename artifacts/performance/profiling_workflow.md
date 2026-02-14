# Profiling Workflow

## Benchmark Suite

All benchmarks live in `crates/fj-dispatch/benches/dispatch_baseline.rs` and cover six metric categories:

| Group | Benchmarks | What it measures |
|-------|-----------|------------------|
| `dispatch_latency` | jit/grad/vmap x scalar/vector + compositions | End-to-end dispatch overhead |
| `eval_jaxpr_throughput` | 10/100/1000 equation chains | Interpreter throughput scaling |
| `transform_composition` | single/depth-2/depth-3/empty | Composition verification cost |
| `cache_key_generation` | simple/medium/large/hardened | Cache key derivation cost |
| `ledger_append` | single/burst-100 | Evidence ledger write throughput |
| `jaxpr_fingerprint` | 1/10/100 eq + cached lookup | Fingerprint computation + OnceLock cache |
| `jaxpr_validation` | 1/10/100 eq | Well-formedness check cost |

## Running Benchmarks

Full suite:

```bash
rch exec -- cargo bench --bench dispatch_baseline
```

Single group:

```bash
rch exec -- cargo bench --bench dispatch_baseline -- dispatch_latency
```

Single benchmark:

```bash
rch exec -- cargo bench --bench dispatch_baseline -- "jit/scalar_add"
```

## Saving a Baseline

```bash
./scripts/check_perf_regression.sh --save-baseline
```

This saves criterion results under the baseline name from `reliability_budgets.v1.json` (default: `main`). Custom name:

```bash
./scripts/check_perf_regression.sh --save-baseline --baseline-name pre-optimization
```

## Checking for Regressions

After making changes, compare against the saved baseline:

```bash
./scripts/check_perf_regression.sh
```

The gate fails if any benchmark's p95 regresses more than 5% (configurable in `reliability_budgets.v1.json`) without a risk-note justification.

## Justifying an Accepted Regression

If a regression is intentional (e.g., correctness fix that costs throughput), create a risk note:

```bash
mkdir -p artifacts/performance/risk_notes
```

Create `artifacts/performance/risk_notes/<group>_<function>.risk_note.json` following `artifacts/schemas/risk_note.v1.schema.json`.

## CI Integration

The perf regression gate is integrated into `scripts/enforce_quality_gates.sh`:

```bash
./scripts/enforce_quality_gates.sh                 # runs all gates including perf
./scripts/enforce_quality_gates.sh --skip-perf      # skip perf gate
```

## Evidence Artifact

Each gate run emits `artifacts/ci/perf_regression_report.v1.json` conforming to `artifacts/schemas/perf_delta.v1.schema.json`. Fields:

- `baseline_id` / `candidate_id`: git refs being compared
- `benchmarks[]`: per-benchmark p95 values and delta percentages
- `regressions[]`: benchmarks exceeding threshold, with justification status
- `overall_status`: `pass` or `fail`

## Example Profiling Session

```bash
# 1. Save baseline on clean main
git checkout main
./scripts/check_perf_regression.sh --save-baseline

# 2. Make optimization changes on branch
git checkout -b optimize-cache-key

# 3. Run comparison
./scripts/check_perf_regression.sh

# 4. Review report
jq . artifacts/ci/perf_regression_report.v1.json
```
