# frankenjax-06zxm: scalar Exp -> Log grad-only fast path

Date: 2026-06-13
Agent: BeigeMouse
Bead: frankenjax-06zxm
Scope: `crates/fj-ad/src/lib.rs`

## Profile-backed target

Post-`frankenjax-srk4i` AD reprofile, crate-scoped through RCH:

```text
rch exec -- cargo bench -p fj-ad --bench ad_baseline -- 'ad/' --sample-size 20 --measurement-time 2 --warm-up-time 1 --noplot
Worker: vmi1149989
ad/grad_exp_log time: [149.45 ns 152.24 ns 155.47 ns]
```

`ad/grad_exp_log` was the slowest remaining AD row not owned by the claimed fj-lax/dense/linalg beads.

## Lever

One lever: route `grad_jaxpr` for the exact scalar F64 `Exp -> Log` graph through a gradient-only fast path.

The existing `value_and_grad_jaxpr` fast path already recognized this graph and computed the generic reverse-mode gradient. This change factors that gradient arithmetic into `scalar_f64_exp_log_grad_for_generic_reverse` and calls it from `grad_jaxpr`, skipping only the unused forward output construction.

## Baseline

```text
Parent commit: 2aa610a5
Parent worktree: /data/projects/.scratch/frankenjax-06zxm-baseline-2aa610a5
RCH_WORKERS=vmi1149989 rch exec -- cargo bench -p fj-ad --bench ad_baseline -- 'ad/grad_exp_log' --sample-size 30 --measurement-time 3 --warm-up-time 1 --noplot
Worker: vmi1149989
ad/grad_exp_log time: [169.81 ns 187.93 ns 204.54 ns]
```

## Candidate

```text
Candidate commit: 04d8723c
RCH_WORKERS=vmi1149989 rch exec -- cargo bench -p fj-ad --bench ad_baseline -- 'ad/grad_exp_log' --sample-size 30 --measurement-time 3 --warm-up-time 1 --noplot
Worker: vmi1149989
ad/grad_exp_log time: [69.102 ns 71.703 ns 73.590 ns]
```

Midpoint speedup: `187.93 / 71.703 = 2.62x`
Conservative speedup: `169.81 / 73.590 = 2.31x`

Score: `Impact 8 * Confidence 5 / Effort 2 = 20.0`
Decision: keep.

## Isomorphism proof

The recognizer requires exactly:

- one scalar F64 input
- no constvars
- no top-level effects
- one output
- two equations: `Exp(input) -> tmp`, then `Log(tmp) -> output`
- empty params, sub-jaxprs, and equation effects
- builtin VJP rules only; any primitive or wrapper custom VJP falls back

`grad_jaxpr` returns gradients only, so omitting the forward output is unobservable. The gradient arithmetic is the same reverse-mode sequence used by the existing certified `value_and_grad` path:

```text
forward_exp = x.exp()
log_cotangent = 1.0 / forward_exp
if log_cotangent == 0.0 {
    grad = 0.0
} else {
    exp_vjp = x.exp()
    grad = log_cotangent * exp_vjp
}
```

Ordering/tie-breaking: no collection iteration, sorting, or tie-breaking surface.

Floating point: preserves operation order, signed-zero behavior, infinity behavior, and NaN propagation for the generic reverse path. The equality guard is the pre-existing generic-fast-path guard, now shared by gradient-only and value-and-gradient calls.

RNG: no RNG surface.

Golden-output SHA256 proof:

```text
Test: scalar_f64_exp_log_grad_matches_generic_bits
Inputs: 0.0, -0.0, 1.0, -2.5, MIN_POSITIVE, 709.0, +inf, -inf, qNaN payload 0x7ff8000000000042
Comparison: direct fast path == public grad_jaxpr == forced-generic grad_jaxpr_with_custom_vjp_key("force-generic")
Golden SHA256: bbbf792543a06f6978db4103c4bd5246a6b7d5d853d8d8afeb98c57abf8b4f97
```

## Validation

```text
cargo fmt --package fj-ad -- --check
Result: pass

git diff --check
Result: pass

rch exec -- cargo test -j 1 -p fj-ad scalar_f64_exp_log_grad_matches_generic_bits -- --nocapture
Worker: vmi1149989
Result: pass, 1 passed, 0 failed

rch exec -- cargo check -j 1 -p fj-ad --all-targets
Worker: vmi1149989
Result: pass

rch exec -- cargo clippy -j 1 -p fj-ad --all-targets --no-deps -- -D warnings
Worker: vmi1149989
Result: pass
```

After the final rebase onto `2aa610a5`, the parent baseline and candidate benchmark above were refreshed on `vmi1149989`. The upstream parent change touched fj-interpreters staging and bead/artifact metadata, not `fj-ad`.

Ambient warnings observed during RCH builds, unchanged and outside this bead:

- `crates/fj-lax/src/lib.rs:3662` non-snake-case `eval_reduce_window_iN_sum_sat`
- `crates/fj-trace/src/lib.rs:1808` unused `num_spatial`

UBS:

```text
ubs crates/fj-ad/src/lib.rs
Result: exit 1 due broad pre-existing fj-ad inventory.
Subchecks: formatting clean; no clippy warnings/errors; cargo check clean; tests build clean; no unsafe blocks; no hardcoded secrets; no TODO/FIXME/HACK markers.
New-code note: the new helper shares the existing `log_cotangent == 0.0` guard to preserve generic reverse-mode semantics.
```
