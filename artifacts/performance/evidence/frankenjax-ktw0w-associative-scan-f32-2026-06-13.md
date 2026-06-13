# frankenjax-ktw0w: dense f32 associative_scan fast path

## Target

Profile-backed follow-up to `frankenjax-kuuqw`: `eval_associative_scan`
already avoided the generic per-slice `eval_primitive` dispatch path for dense
f64 tensors, but dense f32 tensors still performed one slice, dispatch, clone,
and final stack per leading-axis element. This is pure dispatch/allocation
overhead for 1-D scans.

## Change

Extended the existing dense scan path in `crates/fj-lax/src/lib.rs` to dense
f32 tensors for `add`, `mul`, `max`, and `min`.

Score: Impact 5 * Confidence 5 / Effort 1 = 25.0.

## Same-worker RCH timing

Worker: `vmi1227854`

Command:

```text
RCH_WORKER=vmi1227854 RCH_ENV_ALLOWLIST=AGENT_NAME,RCH_WORKER AGENT_NAME=BeigeMouse rch exec -- cargo test -p fj-lax --lib bench_associative_scan_dense_vs_slice_dispatch -- --ignored --nocapture
```

Output:

```text
BENCH associative_scan f64 add(x[1048576],axis=0): slice-dispatch=272.4362ms dense=58.3501ms speedup=4.67x
BENCH associative_scan f32 add(x[1048576],axis=0): slice-dispatch=287.4788ms dense=42.5861ms speedup=6.75x
test tests::bench_associative_scan_dense_vs_slice_dispatch ... ok
```

The f32 baseline comparator is the replicated pre-fast-path slice-dispatch
algorithm in the same test binary. The dense path matched its raw `F32Bits`
digest before the timing line was printed.

## Behavior proof

Command:

```text
RCH_WORKER=vmi1227854 RCH_ENV_ALLOWLIST=AGENT_NAME,RCH_WORKER AGENT_NAME=BeigeMouse rch exec -- cargo test -p fj-lax --lib associative_scan_dense_f32_bit_identical_to_slice_dispatch -- --nocapture
```

Result:

```text
test tests::associative_scan_dense_f32_bit_identical_to_slice_dispatch ... ok
f32 associative_scan golden digest: 922b9e035eec116a5dc10da15bcc68dc45ebacdddb3b1f4a95908e6621966f80
```

Isomorphism:

- Ordering preserved: yes. Forward uses `op(acc, x)` and reverse uses
  `op(x, acc)`, matching the old slice-dispatch algorithm exactly.
- Tie-breaking unchanged: yes. `max` and `min` call the same JAX scalar
  helpers after widening each f32 lane to f64, then narrow to f32 like the
  existing f32 elementwise path.
- Floating-point unchanged: yes for the covered f32 contract. The test compares
  raw `F32Bits` for add/mul/max/min, forward and reverse, including NaN,
  signed zero, and infinities.
- RNG unchanged: N/A.
- Golden output: digest
  `922b9e035eec116a5dc10da15bcc68dc45ebacdddb3b1f4a95908e6621966f80`.

## Validation

Passed:

```text
rustfmt --edition 2024 --check crates/fj-lax/src/lib.rs
RCH_WORKER=vmi1227854 RCH_ENV_ALLOWLIST=AGENT_NAME,RCH_WORKER AGENT_NAME=BeigeMouse rch exec -- cargo check -p fj-lax --lib
```

Known external gate limitation:

```text
RCH_WORKER=vmi1227854 RCH_ENV_ALLOWLIST=AGENT_NAME,RCH_WORKER AGENT_NAME=BeigeMouse rch exec -- cargo clippy -p fj-lax --lib -- -D warnings
```

failed on pre-existing fj-lax lint debt outside this lever: `linalg.rs`
`doc_lazy_continuation` and `needless_range_loop`, plus existing warnings in
`reduction.rs` and `tensor_ops.rs`. This is tracked separately by
`frankenjax-npsnl` / `frankenjax-p7ri2`; no `ktw0w` lines were implicated.
