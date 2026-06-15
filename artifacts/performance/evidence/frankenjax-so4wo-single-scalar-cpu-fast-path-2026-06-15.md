# frankenjax-so4wo single-equation scalar CPU fast path

## Lever

`fj-backend-cpu` now checks for a guarded single-equation scalar Jaxpr before
building the dependency scheduler state. The accepted shape has no constvars,
effects, sub-Jaxprs, tensors, hidden intermediates, or multi-output behavior.
It resolves equation inputs in source order and delegates the operation to the
existing `fj_lax::eval_primitive` implementation, then falls back to the
previous scheduler for every nonmatching case.

## Baseline

Command:

```bash
rch exec -- cargo bench -p fj-api --bench api_overhead -- \
  "api_overhead/(jit/scalar_add|jit/scalar_add_repeated_call)|api_vs_dispatch/(api_jit_add|dispatch_jit_add)" \
  --warm-up-time 1 --measurement-time 3 --sample-size 20
```

RCH selected `vmi1149989`:

- `api_overhead/jit/scalar_add`: 2.1752 us .. 2.3456 us .. 2.5480 us
- `api_overhead/jit/scalar_add_repeated_call`: 1.0346 us .. 1.0874 us .. 1.1531 us
- `api_vs_dispatch/api_jit_add`: 2.2266 us .. 2.3227 us .. 2.4554 us
- `api_vs_dispatch/dispatch_jit_add`: 2.0816 us .. 2.2043 us .. 2.3548 us

## Rebench

RCH did not expose a worker pin for `exec`, and the original baseline worker
was reported degraded by `rch status`, so the rebench ran where RCH scheduled
it. RCH selected `ovh-a`:

- `api_overhead/jit/scalar_add`: 2.2819 us .. 2.2945 us .. 2.3045 us
- `api_overhead/jit/scalar_add_repeated_call`: 179.36 ns .. 180.63 ns .. 181.91 ns
- `api_vs_dispatch/api_jit_add`: 2.1338 us .. 2.1486 us .. 2.1616 us
- `api_vs_dispatch/dispatch_jit_add`: 1.8797 us .. 1.8851 us .. 1.8898 us

The construct-and-call control stayed comparable across workers
(`2.3456 us` -> `2.2945 us` midpoint), while the repeated-call target moved
`1.0874 us` -> `180.63 ns`: 6.02x midpoint and 5.69x conservative
(`1.0346 us / 181.91 ns`).

Score: Impact 5.0 x Confidence 0.75 / Effort 1.0 = 3.75. Confidence is capped
because the final RCH run is not strict same-worker, but the unchanged control
row and large target delta keep it above the retain threshold.

## Isomorphism

- Ordering preserved: yes. The path only accepts one equation and resolves its
  input atoms in the original equation order.
- Tie-breaking preserved: N/A. There is no ordering tie or parallel reduction.
- Floating point preserved: yes. The path calls the same `fj_lax::eval_primitive`
  used by the existing interpreter/scheduler leaf evaluation.
- RNG preserved: N/A. The accepted primitives here have no RNG state.
- Error/fallback surfaces preserved: non-scalar args, constvars, effects,
  sub-Jaxprs, nontrivial output wiring, tensors, and unknown vars all fall back
  to the previous scheduler.

Golden output digest:

```text
458d418a96e2c77d5f9fb43b857ecd4de5e0004c136eea3c2a2eb4e91b1f7f4b
```

The golden test compares CPU backend output to `fj_interpreters::eval_jaxpr`
for i64 normal addition, i64 wrapping addition, f64 signed-zero addition, and a
finite f64 row.

## Gates

- `rch exec -- cargo test -p fj-backend-cpu cpu_single_equation_scalar_fast_path_matches_interpreter_and_golden_sha256 --lib -- --nocapture`
- `cargo fmt -p fj-backend-cpu --check`
- `ubs crates/fj-backend-cpu/src/executor.rs` (exit 0; existing warning inventory only)
- `rch exec -- cargo check -p fj-backend-cpu --all-targets`
- `rch exec -- cargo clippy -p fj-backend-cpu --all-targets -- -D warnings`
