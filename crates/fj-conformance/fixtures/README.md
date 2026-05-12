# Conformance Fixtures

This folder stores normalized oracle-vs-target fixtures for `fj-conformance`.

## Files

- `smoke_case.json`: bootstrap fixture ensuring harness wiring works.

| Fixture | Cases | Oracle metadata | Generator |
|---|---:|---|---|
| `transforms/legacy_transform_cases.v1.json` | 613 | JAX 0.9.2.dev20260316+12a2449, x64=true | `capture_legacy_fixtures.py` |
| `rng/rng_determinism.v1.json` | 25 | JAX 0.9.2.dev20260316+12a2449, x64=true | `capture_legacy_fixtures.py` |
| `linalg_fft_oracle.v1.json` | 46 | JAX 0.9.2, x64=true | `capture_linalg_fft_oracle.py` |
| `composition_oracle.v1.json` | 15 | JAX 0.9.2, x64=true | `capture_composition_oracle.py` |
| `dtype_promotion_oracle.v1.json` | 162 | JAX 0.9.2, x64=true | `capture_dtype_promotion_oracle.py` |

## Regeneration

Use the legacy transform/RNG capture script:

```bash
python3 crates/fj-conformance/scripts/capture_legacy_fixtures.py \
  --legacy-root /data/projects/frankenjax/legacy_jax_code/jax \
  --output /data/projects/frankenjax/crates/fj-conformance/fixtures/transforms/legacy_transform_cases.v1.json \
  --rng-output /data/projects/frankenjax/crates/fj-conformance/fixtures/rng/rng_determinism.v1.json
```

Regenerate the dedicated oracle families with:

```bash
python3 crates/fj-conformance/scripts/capture_linalg_fft_oracle.py \
  --legacy-root /data/projects/frankenjax/legacy_jax_code/jax \
  --output /data/projects/frankenjax/crates/fj-conformance/fixtures/linalg_fft_oracle.v1.json

python3 crates/fj-conformance/scripts/capture_composition_oracle.py \
  --legacy-root /data/projects/frankenjax/legacy_jax_code/jax \
  --output /data/projects/frankenjax/crates/fj-conformance/fixtures/composition_oracle.v1.json

python3 crates/fj-conformance/scripts/capture_dtype_promotion_oracle.py \
  --legacy-root /data/projects/frankenjax/legacy_jax_code/jax \
  --output /data/projects/frankenjax/crates/fj-conformance/fixtures/dtype_promotion_oracle.v1.json
```

If JAX/jaxlib are unavailable in the environment, the script will fail with an explicit setup error.

Run the recapture matrix and drift gate with:

```bash
./scripts/run_oracle_recapture_gate.sh
```

The generated matrix records every required fixture family, case count, legacy
anchor, recapture command, oracle version, x64 mode, fixture hash, and drift-gate
issue. Use `--enforce` when CI should fail on stale or unsupported rows.
