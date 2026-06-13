# frankenjax-cbmoi I64 eval fusion rejection

Date: 2026-06-13
Agent: BeigeMouse
Worker: vmi1152480
Bead: frankenjax-cbmoi - perf(fj-interpreters): improve large I64 eval fusion chain

## Profile-backed target

Fresh current-code baseline:

```bash
RCH_WORKER=vmi1152480 RCH_WORKERS=vmi1152480 RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo bench -j 1 -p fj-interpreters --bench eval_fusion_speed
```

Baseline row:

```text
EVAL_FUSION_SPEED_I64 n=1048576 ops=8 unfused=4.759ms fused=2.817ms speedup=1.69x
```

This was the lowest remaining positive eval-fusion row after the BF16 reroute.

## Candidate lever

Rejected lever:

- Add a long-tape I64 scalar-register executor.
- For tapes with six or more steps, keep each element's chain value in a local
  scalar and write the dense output once, instead of making one scratch-buffer
  pass per step.

The candidate preserved the same tape order and exact integer closures:
`wrapping_add`, `wrapping_sub`, `wrapping_mul`, `checked_div(...).unwrap_or(0)`,
`wrapping_neg`, `wrapping_abs`, `max`, and `min`.

## Behavior proof

Command:

```bash
RCH_WORKER=vmi1152480 RCH_WORKERS=vmi1152480 RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo test -j 1 -p fj-interpreters --lib \
    fusion_i64_chain_matches_reference_bit_for_bit -- --nocapture
```

Result:

```text
test tests::fusion_i64_chain_matches_reference_bit_for_bit ... ok
```

Golden SHA-256 remained:

```text
7f7e34d693a2f0e9f63a0d4575b03db01882e16d74746eb5b3978d8a6d25b297
```

Isomorphism:

- Ordering preserved: the candidate interpreted the same tape steps in original
  equation order for each element.
- Tie-breaking unchanged: integer max/min used the same total-order closures.
- Floating-point behavior not applicable: this candidate touched only I64.
- RNG unchanged: the fused I64 path has no RNG state or effects.
- Integer overflow/division behavior preserved by the golden test, including
  `i64::MIN`, `i64::MAX`, division by zero, and checked `MIN / -1` behavior.

## Re-benchmark

Command:

```bash
RCH_WORKER=vmi1152480 RCH_WORKERS=vmi1152480 RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo bench -j 1 -p fj-interpreters --bench eval_fusion_speed
```

Candidate row:

```text
EVAL_FUSION_SPEED_I64 n=1048576 ops=8 unfused=4.689ms fused=17.410ms speedup=0.27x
```

Decision:

- Target fused path regressed: 2.817 ms -> 17.410 ms = 0.16x.
- Score is below the keep threshold; the source hunk was removed.
- No production code from this candidate shipped.

## Next attack

The failed result means dynamic per-element tape interpretation is the wrong
primitive. The next I64 attempt should be a branch-free specialized arithmetic
microkernel for safe subsets of long I64 tapes, e.g. a classifier for Add/Sub/Mul
chains that emits fixed opcode slots or pattern-specific straight-line loops
without per-element dynamic matches, while retaining the existing vectorized
step executor for unclassified tapes.
