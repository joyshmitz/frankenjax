# frankenjax-p1vbf.55 rejection: GEMM2048 streamed KC B-pack slab

## Target

- Bead: `frankenjax-p1vbf.55`
- Crate: `fj-lax`
- Hotspot: `linalg/matmul_2d_2048x2048x2048_f64`
- Profile-backed baseline: pushed parent commit `f05d16e0` left GEMM2048 as the dominant valid RCH row after the thread-cap keep.
- Worker discipline: same-worker RCH on `vmi1227854`, crate-scoped `cargo bench -j 1 -p fj-lax`.

## Lever tested

Stream one `KC`-sized packed-B slab at a time instead of materializing the full `k*n` packed-B panel buffer for blocked GEMM.

Expected benefit: reduce packed-B footprint at 2048^3 from about 32 MiB to about 4 MiB and improve cache residency.

## Isomorphism proof

- Ordering: candidate kept each output cell's `k` terms in ascending `pc` then ascending inner `l` order.
- Tie-breaking: no ordering-sensitive comparisons or branch tie-breaks in the changed path.
- Floating point: no reassociation within an individual output element; recurrence remained `c += a * b` in ascending `k`.
- RNG: none in this path.
- Golden output digest preserved by the remote proof test:
  - `matmul_2d_packed_nr8_golden_output_digest`
  - `e3762befad86e2a81da53a8413643b658a0be2d6136d69e195770b2beba48b3a`
- Proof command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,RCH_REQUIRE_REMOTE \
  CARGO_TARGET_DIR=target/rch-boldfalcon-p1vbf55-test \
  rch exec -- cargo test -j 1 -p fj-lax tensor_contraction::tests:: -- --nocapture
```

Result: `vmi1227854`, 28 passed, 0 failed, 4 ignored.

## Benchmark

Baseline from pushed parent `f05d16e0`, same worker `vmi1227854`:

- `linalg/matmul_2d_1024x1024x1024_f64`: 27.600 ms [25.314, 28.929]
- `linalg/matmul_2d_2048x2048x2048_f64`: 227.26 ms [208.22, 248.15]

Candidate command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,RCH_REQUIRE_REMOTE \
  CARGO_TARGET_DIR=target/rch-boldfalcon-p1vbf55-bench \
  rch exec -- cargo bench -j 1 -p fj-lax --bench lax_baseline -- \
  'linalg/matmul_2d_(1024x1024x1024|2048x2048x2048)_f64' \
  --sample-size 10 --warm-up-time 1 --measurement-time 2
```

Candidate result on `vmi1227854`:

- `linalg/matmul_2d_1024x1024x1024_f64`: 177.34 ms [140.19, 219.41]
- `linalg/matmul_2d_2048x2048x2048_f64`: 1.9205 s [1.7068, 2.1622]

## Decision

Rejected. Score: `0.0` because the lever is a large same-worker regression.

The regression mechanism is likely the new synchronization and repeated per-`pc` row-worker dispatch, which destroys the existing row-block locality despite reducing the B-pack footprint.

Source was restored to `f05d16e0` behavior before committing this evidence. Next attack should be a different algorithmic primitive: register-blocked microkernel/autotuned panel geometry or recursive/Strassen-family GEMM, not another streamed-pack variant.
