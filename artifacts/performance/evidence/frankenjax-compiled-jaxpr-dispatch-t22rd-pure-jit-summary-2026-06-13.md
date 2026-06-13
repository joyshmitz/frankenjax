# frankenjax-compiled-jaxpr-dispatch-t22rd: pure-JIT nested trace summary fast path

Date: 2026-06-13
Agent: BeigeMouse
Worker: vmi1293453
Base commit: 838b9e48

## Lever

`simulate_nested_trace_contexts` now handles stacks where every transform is
`Transform::Jit` by constructing the `NestedTraceSummary` directly. The generic
path is unchanged for all non-pure-JIT stacks.

This removes per-dispatch construction of `ShapedArray` inputs and
`SimpleTraceContext` frames on the cold JIT dispatch path while preserving the
same frame order and field values.

## Baseline

Command, from a clean baseline worktree at `838b9e48` (`origin/main` after
the intervening scalar-arena interpreter commits):

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1293453 rch exec -- cargo bench -p fj-dispatch --bench dispatch_baseline -- dispatch_latency/jit/scalar --warm-up-time 1 --measurement-time 3 --sample-size 10
```

Results:

- `dispatch_latency/jit/scalar_add`: `[2.9692, 3.0758, 3.1717] us`
- `dispatch_latency/jit/scalar_square_plus_linear`: `[3.7037, 3.9469, 4.1123] us`

## Rebench

Command, from the candidate worktree:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1293453 rch exec -- cargo bench -p fj-dispatch --bench dispatch_baseline -- dispatch_latency/jit/scalar --warm-up-time 1 --measurement-time 3 --sample-size 10
```

Results:

- `dispatch_latency/jit/scalar_add`: `[2.5092, 2.6045, 2.6949] us`
- `dispatch_latency/jit/scalar_square_plus_linear`: `[3.3212, 3.4606, 3.5925] us`

Median speedups:

- `scalar_add`: `3.0758 / 2.6045 = 1.181x`
- `scalar_square_plus_linear`: `3.9469 / 3.4606 = 1.141x`

Conservative interval speedups:

- `scalar_add`: `2.9692 / 2.6949 = 1.102x`
- `scalar_square_plus_linear`: `3.7037 / 3.5925 = 1.031x`

Score: `Impact 2.4 * Confidence 0.9 / Effort 1.0 = 2.16`, keep. The
Criterion intervals remain non-overlapping on both rows after rebasing onto
the newer scalar-arena interpreter baseline.

## Behavior Proof

The shortcut applies only when `transforms.iter().all(|t| *t == Transform::Jit)`.
For that case the generic trace simulation previously:

1. Created input avals from `args`.
2. Pushed one empty subtrace per JIT transform, assigning trace ids starting at
   2 and depths starting at 2.
3. Popped all frames, producing zero equations, `args.len()` input vars, and
   zero output vars per frame.
4. Reversed the popped frames back to outer-to-inner transform order.

The new path constructs exactly those values directly:

- order: unchanged, outer-to-inner by `enumerate()`
- trace ids: unchanged, `idx + 2`
- depths: unchanged, `idx + 2`
- equation count: unchanged, `0`
- input count: unchanged, `args.len()`
- output count: unchanged, `0`
- `max_depth`: unchanged, `transforms.len() + 1`

No primitive execution, floating-point arithmetic, RNG, ordering ties,
cache-key material, or transform sequence is changed. Non-pure-JIT stacks still
use the original generic path.

Golden summary contract:

```json
[3,[["jit",2,2,0,2,0],["jit",3,3,0,2,0]]]
```

Golden SHA-256:

```text
7ecd3b83d07c77799f97a478915bbca86e8634e86e85e7aea61f1a4c51f3bd6c
```

The SHA is pinned in
`tests::pure_jit_nested_trace_summary_preserves_frame_contract`.

## Verification

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1293453 rch exec -- cargo test -j 1 -p fj-trace pure_jit_nested_trace_summary_preserves_frame_contract -- --nocapture
```

Passed: `1 passed; 0 failed`.

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1293453 rch exec -- cargo test -j 1 -p fj-dispatch jit_scalar_add -- --nocapture
```

Passed: `jit_scalar_add`.

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1293453 rch exec -- cargo test -j 1 -p fj-dispatch dispatch_jit_add_scalar -- --nocapture
```

Passed: `tests::dispatch_jit_add_scalar`.

```bash
cargo fmt -p fj-trace --check
ubs crates/fj-trace/src/lib.rs
```

Both exited 0. UBS reported no critical findings; its broad warning inventory
is pre-existing for the large trace file.

Known pre-existing compiler warnings observed during rch runs:

- `crates/fj-trace/src/lib.rs:1808`: unused `num_spatial`
- `crates/fj-dispatch/src/batching.rs:4966`: unnecessary `mut`
