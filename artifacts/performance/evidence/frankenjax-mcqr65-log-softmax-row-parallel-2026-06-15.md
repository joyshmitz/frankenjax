# frankenjax-mcqr.65: log_softmax_2d row-parallel exact path

Date: 2026-06-15
Agent: SilverMaple
Bead: frankenjax-mcqr.65
Crate: fj-lax
Surface: `fj_lax::nn::log_softmax_2d`

## Profile-backed target

Fresh fj-lax profile after pass262 on RCH `ovh-a`:

- `nn/log_softmax_2d_65536x16_fused`: `[4.7763 ms 4.7917 ms 4.8088 ms]`
- `nn/softmax_2d_65536x16_fused`: `[1.3789 ms 1.3919 ms 1.4050 ms]`
- `linalg/qr_1024x1024_f64`: `[22.217 ms 22.311 ms 22.407 ms]`
- `linalg/cholesky_1024x1024_f64`: `[18.229 ms 18.404 ms 18.597 ms]`

`frankenjax-mcqr.30` remains assigned to IcyGlacier, so this pass avoided dense-storage work and took the next narrow non-overlapping hotspot.

## Baseline

Focused pre-edit Criterion run:

Command:

```bash
RCH_WORKER=ovh-a rch exec -- cargo bench -p fj-lax --bench lax_baseline -- 'nn/log_softmax_2d_65536x16_fused'
```

RCH selected `vmi1149989`.

Result:

- `nn/log_softmax_2d_65536x16_fused`: `[4.9910 ms 5.1835 ms 5.3837 ms]`

## Lever

One lever only:

- factor the existing serial row body into `log_softmax_row_into`
- route `log_softmax_2d` through the existing scoped-thread row helper already used by `softmax_2d`

Each row still computes:

1. `max = fold(-inf, f64::max)` in ascending column order
2. if `max.is_infinite()`, use `lse = max`
3. otherwise compute `sum(exp(x - max))` in ascending column order, then `max + ln(sum)`
4. write `x - lse` in ascending column order

Threads own disjoint row chunks. There is no shared mutable row state and no unsafe code.

## Post-change benchmark

Focused post-edit Criterion run:

```bash
rch exec -- cargo bench -p fj-lax --bench lax_baseline -- 'nn/log_softmax_2d_65536x16_fused'
```

RCH selected the same worker, `vmi1149989`.

Result:

- `nn/log_softmax_2d_65536x16_fused`: `[2.0899 ms 2.1884 ms 2.2998 ms]`

Delta:

- midpoint speedup: `5.1835 / 2.1884 = 2.37x`
- conservative interval speedup: `4.9910 / 2.2998 = 2.17x`

Score:

- Impact `2.37`
- Confidence `0.95` (same-worker Criterion plus bit/golden proof)
- Effort `1.0`
- Score `2.25`, keep.

## Isomorphism proof

Behavior surfaces unchanged:

- output row-major order unchanged
- per-row floating-point operation order unchanged
- tie surface unchanged
- signed-zero, infinity, and NaN propagation unchanged
- dtype/shape/error behavior unchanged
- RNG surface absent

Golden proof:

- `cargo test -p fj-lax log_softmax_2d -- --nocapture` passed on RCH `ovh-a`
- new parallel-vs-serial fused proof compares every output bit
- pinned SHA-256: `34e1e76bbcfde76e7bb49161efd7e6b7a8225967e35aa16a6c2ab41d96b8e2d2`

## Validation

Passed:

```bash
cargo fmt --check -p fj-lax
ubs crates/fj-lax/src/nn.rs
rch exec -- cargo test -p fj-lax log_softmax_2d -- --nocapture
rch exec -- cargo check -p fj-lax --all-targets
rch exec -- cargo clippy -p fj-lax --all-targets -- -D warnings
rch exec -- cargo bench -p fj-lax --bench lax_baseline -- 'nn/log_softmax_2d_65536x16_fused'
```

UBS exited 0. It reported existing file-wide warning inventories for asserts/direct indexing in `nn.rs`; no critical findings, unsafe, unwrap/expect, fmt, clippy, check, or test-build failures.

## Next target

Close `frankenjax-mcqr.65`, sync beads, push `main` and `main:master`, then reprofile. If linalg remains dominant, avoid QR row-thread fanout and attack a structurally different QR/Cholesky primitive.
