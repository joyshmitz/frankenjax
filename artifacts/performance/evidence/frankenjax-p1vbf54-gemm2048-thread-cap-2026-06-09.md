# frankenjax-p1vbf.54 GEMM2048 Thread-Budget Cap

Date: 2026-06-09
Agent: BoldFalcon
Target: `linalg/matmul_2d_2048x2048x2048_f64`

## Profile-Backed Target

RCH worker: `vmi1227854`
Commit: `9cdbfe5a`

Focused profile before this change:

- `linalg/matmul_2d_2048x2048x2048_f64`: 265.47 ms mean, [251.52, 280.78] ms
- `linalg/lu_1024x1024_f64`: 39.961 ms mean, [38.191, 41.177] ms
- `linalg/qr_1024x1024_f64`: 34.748 ms mean, [34.341, 35.429] ms
- `linalg/cholesky_1024x1024_f64`: 29.546 ms mean, [27.970, 31.954] ms
- `linalg/matmul_2d_1024x1024x1024_f64`: 28.978 ms mean, [27.946, 30.687] ms

The prior `frankenjax-p1vbf.53` superpanel schedule regressed, so this pass
tested a different primitive: right-size row-block fanout to avoid overfeeding the
shared packed-B memory stream on 64-thread workers.

## Candidate Lever

One source lever in `crates/fj-lax/src/tensor_contraction.rs`:

```rust
const MAX_MATMUL_THREADS: usize = 16;
```

`matmul_thread_count` now caps fanout at 16 while preserving the existing
work-based lower thresholds.

## Isomorphism Proof

Thread count only changes row-block partitioning. Each output row remains owned by
exactly one worker, and every output element still accumulates products over `k`
in strictly ascending order. No per-cell reassociation, no FMA introduction, no
mixed precision, no packed-B layout change, no tie-breaking change, and no RNG
surface exists.

Remote proof:

```bash
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,RCH_WORKER CARGO_TARGET_DIR=target/rch-boldfalcon-gemm54-test \
  rch exec -- cargo test -p fj-lax tensor_contraction::tests:: -- --nocapture
```

Result: passed on `ovh-a`.

- 28 tests passed
- 4 perf/stress tests remained ignored
- Covered bit-identical matmul, threaded matmul, blocked matmul, packed-B layout,
  parallel/serial packed-B equality, and packed-NR8 golden digest tests

## Same-Worker Rebench

RCH worker: `vmi1227854`

Candidate:

- `linalg/matmul_2d_1024x1024x1024_f64`: 27.600 ms mean, [25.314, 28.929] ms
- `linalg/matmul_2d_2048x2048x2048_f64`: 227.26 ms mean, [208.22, 248.15] ms

Comparison against the same-worker pre-change profile:

- GEMM2048 mean speedup: 265.47 / 227.26 = 1.168x
- Conservative interval: candidate upper 248.15 ms is below baseline lower 251.52 ms
- GEMM1024 did not regress: 28.978 ms -> 27.600 ms
- Score: 3.1, above the required 2.0 keep threshold

An attempted second confirmation run fell back local after RCH `queue_timeout`.
Those local numbers were discarded and are not used as evidence.

## Gates

- `rustfmt --check crates/fj-lax/src/tensor_contraction.rs`: passed
- `ubs crates/fj-lax/src/tensor_contraction.rs`: no critical findings; warnings
  were existing file-wide panic/indexing inventory outside the changed lines
- `cargo check -p fj-lax --all-targets`: passed remotely on `vmi1227854`
  with one pre-existing warning in `cz0g0_f32accum_evidence.rs`
- `cargo clippy -p fj-lax --all-targets -- -D warnings`: remote-required
  admission was refused by RCH pressure before execution; it was not run locally
  for final evidence
- `cargo fmt -p fj-lax --check`: failed on existing unrelated formatting drift in
  peer-owned fj-lax files; the changed file passed direct rustfmt check

## Decision

Kept. The 16-thread cap is a low-risk communication-avoiding scheduling lever
with a same-worker, non-overlapping GEMM2048 win and unchanged arithmetic order.
