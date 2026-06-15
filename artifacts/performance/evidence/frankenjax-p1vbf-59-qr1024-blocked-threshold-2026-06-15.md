# frankenjax-p1vbf.59 -- QR1024 compact-WY threshold

Status: kept

## Target

- Profile-backed lane: `frankenjax-p1vbf`, fj-lax real QR.
- Production lever: lower `QR_BLOCK_MIN` from `2048` to `1024`.
- Alien/no-gaps primitive: communication-avoiding compact-WY block reflector. The
  blocked path sweeps the trailing matrix twice per panel instead of once per
  scalar reflector.

## Baseline

Same-worker RCH Criterion A/B on `vmi1156319`, before the production gate change:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1156319 RCH_WORKERS=vmi1156319 \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline -- \
  'linalg/qr_1024_(scalar|blocked)' --warm-up-time 1 --measurement-time 3 --sample-size 10

linalg/qr_1024_scalar   [76.798 ms 83.725 ms 87.505 ms]
linalg/qr_1024_blocked  [48.504 ms 50.269 ms 51.916 ms]
```

Forced-path midpoint speedup: `1.666x`; conservative interval speedup:
`76.798 / 51.916 = 1.479x`.

## Rebench

Same worker after lowering the production gate:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1156319 RCH_WORKERS=vmi1156319 \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline -- \
  linalg/qr_1024x1024_f64 --warm-up-time 1 --measurement-time 3 --sample-size 10

linalg/qr_1024x1024_f64 [53.298 ms 54.880 ms 56.067 ms]
```

Public-row midpoint speedup vs scalar baseline: `83.725 / 54.880 = 1.526x`.
Conservative interval speedup: `76.798 / 56.067 = 1.370x`.

## Proof

- API / ABI preserved: yes. `eval_qr` inputs, outputs, dtype, shape, error
  behavior, `full_matrices`, and public value layout are unchanged.
- Ordering preserved: small `n < 1024` still uses the existing scalar path, so
  pinned small-n bit identity remains unchanged. Large QR already uses the
  compact-WY tolerance contract at `n >= 2048`; this change extends that same
  contract to `n = 1024`.
- Tie-breaking unchanged: QR has no pivot/tie surface in this path.
- Floating-point drift: bounded by the existing blocked QR tolerance contract.
  `qr_blocked_reconstructs_and_orthonormal` verifies reconstruction,
  orthonormality, and scalar-path agreement under `1e-9`.
- RNG seeds: N/A.
- Golden output: `qr_real_path_golden_output_digest` passed with pinned digest
  `6119fc5cf4759d8cdcd9c34d89a79de89d205203730814fc06aa52bf57ff262b`,
  proving the small-n public QR golden remains unchanged.

## Validation

```text
cargo fmt --check --package fj-lax
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p fj-lax --lib qr -- --nocapture
RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p fj-lax --all-targets
RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p fj-lax --all-targets -- -D warnings
ubs crates/fj-lax/src/linalg.rs crates/fj-lax/benches/lax_baseline.rs
```

Results:

- fmt: passed.
- focused QR tests: passed, `31 passed; 0 failed; 4 ignored`.
- check: passed.
- clippy: passed.
- UBS: exited `1` on existing broad fj-lax panic/direct-indexing inventory. Its
  embedded formatter, clippy, cargo check, and test-build sections were clean;
  no finding was introduced by the threshold/comment/bench-row hunk.

## Score

`Impact 4 * Confidence 5 / Effort 1 = 20.0`.
