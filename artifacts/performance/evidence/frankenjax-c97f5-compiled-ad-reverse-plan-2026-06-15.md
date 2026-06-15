# frankenjax-c97f5 compiled scalar AD reverse plan

## Lever

`fj-ad` now exposes a reusable scalar-F64 reverse plan for pure, single-output
Jaxprs built from Add/Sub/Mul/Div/Neg/Sin/Cos/Exp/Log. The plan validates the
static graph once, evaluates equations in original source order, and propagates
cotangents in reverse equation order. Unsupported graphs, custom Jaxpr VJPs,
custom primitive VJPs, non-default API backends, custom-VJP wrappers, non-scalar
runtime args, and non-finite scalar values all fall back to the existing tape
and dispatch paths.

`fj-api` caches this plan inside `grad()` and `value_and_grad()` wrappers after
the existing args-independent dispatch metadata is prepared.

## Profile Seed

RCH selected `ovh-a` for the wrapper baseline:

```bash
RCH_REQUIRE_REMOTE=1 RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
CARGO_TARGET_DIR=/data/tmp/frankenjax-c97f5-baseline \
rch exec -- cargo bench -j 1 -p fj-api --bench api_overhead -- \
  'api_overhead/grad/scalar_square|api_overhead/value_and_grad/scalar_square|value_and_grad_runtime/(shared|separate)/(square_plus_linear|deep_100_nodes)|grad_meta_cache_scalar/(recompute|prepared)/scalar_square|prepared_dispatch_metadata/(recompute|prepared)/deep_100_nodes' \
  --sample-size 20 --warm-up-time 1 --measurement-time 3 --noplot
```

Selected rows:

- `api_overhead/grad/scalar_square`: 2.3381 us .. 2.3508 us .. 2.3619 us
- `api_overhead/value_and_grad/scalar_square`: 2.7163 us .. 2.7245 us .. 2.7310 us
- `value_and_grad_runtime/shared/deep_100_nodes`: 4.0556 us .. 4.0620 us .. 4.0695 us
- `value_and_grad_runtime/separate/deep_100_nodes`: 5.2325 us .. 5.2567 us .. 5.2797 us
- `grad_meta_cache_scalar/recompute/scalar_square`: 1.8189 us .. 1.8775 us .. 1.9967 us
- `grad_meta_cache_scalar/prepared/scalar_square`: 1.0139 us .. 1.0529 us .. 1.1302 us

RCH selected `vmi1227854` for the AD baseline:

```bash
RCH_REQUIRE_REMOTE=1 RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
CARGO_TARGET_DIR=/data/tmp/frankenjax-c97f5-ad-baseline \
rch exec -- cargo bench -j 1 -p fj-ad --bench ad_baseline -- \
  'ad/(grad_sin_cos_mul|grad_exp_log|value_and_grad_poly|grad_poly_x3\+x2\+x|grad_square)' \
  --sample-size 20 --warm-up-time 1 --measurement-time 3 --noplot
```

Selected rows:

- `ad/grad_square`: 133.81 ns .. 136.07 ns .. 138.72 ns
- `ad/grad_poly_x3+x2+x`: 72.618 ns .. 73.784 ns .. 74.755 ns
- `ad/grad_sin_cos_mul`: 96.958 ns .. 98.743 ns .. 100.93 ns
- `ad/grad_exp_log`: 70.066 ns .. 71.091 ns .. 72.056 ns
- `ad/value_and_grad_poly`: 60.790 ns .. 61.885 ns .. 63.199 ns

These rows showed the existing hand-written scalar add/mul/trig/exp-log routes
were already fast, so the target became the generic repeated value-and-grad
route that still used the tape path.

## Rebench

Final RCH rebench on `vmi1227854`, after adding the custom-rule guard:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
CARGO_TARGET_DIR=/data/tmp/frankenjax-c97f5-bench-after-guard \
rch exec -- cargo bench -j 1 -p fj-api --bench api_overhead -- \
  'ad_compiled_reverse_plan/(tape|compiled|api_warmed)/value_and_grad_trig' \
  --sample-size 30 --warm-up-time 1 --measurement-time 3 --noplot
```

- `ad_compiled_reverse_plan/tape/value_and_grad_trig`: 976.70 ns .. 990.43 ns .. 1.0050 us
- `ad_compiled_reverse_plan/compiled/value_and_grad_trig`: 132.35 ns .. 134.87 ns .. 137.50 ns
- `ad_compiled_reverse_plan/api_warmed/value_and_grad_trig`: 250.09 ns .. 260.48 ns .. 269.63 ns

Midpoint speedup:

- Direct old tape path to compiled plan: 7.34x
- Direct old tape path to warmed API wrapper: 3.80x

Conservative interval speedup:

- Direct old tape path to compiled plan: 7.10x (`976.70 ns / 137.50 ns`)
- Direct old tape path to warmed API wrapper: 3.62x (`976.70 ns / 269.63 ns`)

Score: Impact 5.0 x Confidence 1.0 / Effort 1.0 = 5.0.

## Isomorphism

- Ordering preserved: yes. Forward equations execute in original Jaxpr order;
  backward cotangents execute in reverse Jaxpr order.
- Tie-breaking preserved: yes. Adjoints use the same insert-then-add behavior
  as the existing scalar reverse path, and literals do not receive cotangents.
- Floating point preserved: yes for finite accepted values, proved bitwise
  against the generic tape route forced through a wrapper custom-VJP key.
  Non-finite inputs/intermediates fall back to the tape path to preserve NaN
  payload and exceptional-value behavior.
- RNG preserved: N/A. Accepted primitives do not consume RNG state.
- Custom derivatives preserved: yes. Compile-time and per-call guards reject
  active custom Jaxpr VJPs and active custom primitive VJPs, including custom
  rules registered after a wrapper has warmed the compiled plan.
- Error/fallback surfaces preserved: unsupported primitives, params, effects,
  sub-Jaxprs, multiple outputs, repeated output slots, non-default backends, and
  custom-VJP wrappers all use the previous dispatch/tape path.

Golden output digests:

```text
fj-ad finite compiled scalar reverse plan: 029b47fad80a2743a681fbcb769f4f256b12990093f3d3248ac1041ac53fb285
fj-api warmed value_and_grad wrapper:       984585309be003365780a1f999422efc949c360ec9933d354a6bd50b5b41653a
```

## Gates

- `cargo fmt --check -p fj-ad -p fj-api`
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -j 1 -p fj-ad compiled_scalar_f64_reverse_plan_matches_generic_bits -- --nocapture`
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -j 1 -p fj-api value_and_grad_compiled_ad_cache_matches_tape_golden_sha256 -- --nocapture`
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -j 1 -p fj-ad -p fj-api --all-targets`
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -j 1 -p fj-ad -p fj-api --all-targets -- -D warnings`
- `ubs crates/fj-ad/src/lib.rs crates/fj-api/src/transforms.rs crates/fj-api/benches/api_overhead.rs`

UBS exited nonzero from existing file-wide panic/assert/indexing/clone warning
inventory in these large files. The new hot-path warnings are reviewed: zero
cotangent equality matches the old reverse-mode skip, `VarId`-derived slot casts
cannot exceed their original `u32` source domain, and input indexing is covered
by compile-time arity checks plus focused bitwise tests. Cargo fmt/check/clippy
and the two focused golden tests are clean.
