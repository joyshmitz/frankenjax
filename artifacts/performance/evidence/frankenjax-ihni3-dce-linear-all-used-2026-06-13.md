# frankenjax-ihni3: all-used linear-chain DCE shortcut

Date: 2026-06-13
Agent: BeigeMouse
Crate: fj-interpreters

## Target

The post-pass242 RCH profile ranked `dce/all_used/1000eq` as the top remaining `fj-interpreters` row:

```text
RCH vmi1227854 pe_baseline:
dce/all_used/1000eq [6.6078 us 6.8427 us 7.0708 us]
```

Because RCH affinity later pinned timing runs to `vmi1152480`, acceptance used a clean detached baseline worktree at commit `5c0ef99d`:

```text
/data/projects/.scratch/frankenjax-ihni3-baseline-20260613T1854
```

## Lever

One lever shipped: an early DCE shortcut for the exact all-used single-output linear-chain shape.

The shortcut only accepts a Jaxpr when:

- `used_outputs.len() == jaxpr.outvars.len()`
- every output is marked used
- there is exactly one input var
- there are no constvars
- there is exactly one output var
- there is at least one equation
- every equation has exactly one output
- every variable input in each equation is the current chain variable
- each equation consumes the current chain variable at least once
- the last equation output is the Jaxpr output

All other DCE inputs fall through to the existing generic reverse bitset pass.

## Benchmark

Clean baseline worktree at `5c0ef99d`, same RCH worker `vmi1152480`:

```bash
rch exec -- cargo bench -j 1 -p fj-interpreters --bench pe_baseline -- dce/all_used/1000eq
```

Before:

```text
dce/all_used/1000eq [7.9409 us 8.2062 us 8.4887 us]
```

Edited worktree, same RCH worker `vmi1152480`:

```bash
rch exec -- cargo bench -j 1 -p fj-interpreters --bench pe_baseline -- dce/all_used/1000eq
```

After:

```text
dce/all_used/1000eq [3.1346 us 3.2315 us 3.3338 us]
```

Mean ratio: `8.2062 / 3.2315 = 2.54x`.

Score: `Impact 2.54 * Confidence 0.95 / Effort 0.50 = 4.83`.

## Behavior Proof

For accepted inputs, the generic DCE path retains every equation and every output, then returns:

- the same input var list
- the same empty constvar list
- the same output var list
- the same equation order
- `used_inputs == [true]`

The shortcut returns exactly `jaxpr.clone()` and `vec![true]` for that same accepted shape. It does not execute any primitive, so floating-point order, NaN/tie behavior, rounding, and RNG behavior are unchanged.

Focused golden proof:

```bash
rch exec -- cargo test -j 1 -p fj-interpreters --lib test_dce_all_used_large_chain_golden_hash -- --nocapture
```

Result:

```text
test partial_eval::tests::test_dce_all_used_large_chain_golden_hash ... ok
```

Golden SHA-256 asserted by the test:

```text
3729e2d5cc19c0abec46fb5b188cc7576b9853ee7d0cd523f3656b1ac57e8ad8
```

## Validation

Passed:

```bash
rustfmt --edition 2024 --check crates/fj-interpreters/src/partial_eval.rs
git diff --check
rch exec -- cargo check -j 1 -p fj-interpreters --lib
```

`cargo check` repeated pre-existing dependency warnings outside this lever:

- `crates/fj-lax/src/lib.rs:3662`: `eval_reduce_window_iN_sum_sat` non-snake-case
- `crates/fj-trace/src/lib.rs:1808`: unused `num_spatial`

Blocked by pre-existing debt outside this lever:

```bash
rch exec -- cargo clippy -j 1 -p fj-interpreters --lib --no-deps -- -D warnings
```

Failure:

```text
crates/fj-interpreters/src/lib.rs:5466: clippy::question_mark
```

`ubs crates/fj-interpreters/src/partial_eval.rs` exited nonzero from existing file-wide inventory. Its built-in formatting, clippy, cargo check, and test-build sections were clean.

## Decision

Keep and close `frankenjax-ihni3`. The same-worker timing gate is above threshold, and the behavior proof is an exact all-retained DCE specialization.
