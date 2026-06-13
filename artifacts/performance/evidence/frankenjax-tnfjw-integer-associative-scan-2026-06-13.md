# frankenjax-tnfjw: dense integer associative_scan fast path

## Target

Profile-backed follow-up to the dense f64/f32 `associative_scan` work:
integer `body_op`s still used the old per-slice dispatcher path. For a
1-D scan, that meant one `slice_axis0`, `eval_primitive`, clone, and final
`stack_axis0` step per element, even though the operation is a simple typed
prefix fold.

## Change

Extended the dense `eval_associative_scan` typed-buffer path to:

- `I64`: `add`, `mul`, `max`, `min`, `and`, `or`, `xor`
- `I32`: `add`, `mul`, `max`, `min`

I32 bitwise tensor scan remains on the generic path because the existing
bitwise tensor evaluator does not support `DType::I32` same-shape tensors.

Score: Impact 5 * Confidence 5 / Effort 1 = 25.0.

## Same-worker RCH timing

Worker: `vmi1153651`

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1153651 RCH_WORKERS=vmi1153651 RCH_ENV_ALLOWLIST=AGENT_NAME,RCH_REQUIRE_REMOTE,RCH_WORKER,RCH_WORKERS AGENT_NAME=BeigeMouse rch exec -- cargo test -p fj-lax --lib bench_associative_scan_dense_integer_vs_slice_dispatch -- --ignored --nocapture
```

Output:

```text
BENCH associative_scan i64 add(x[1048576],axis=0): slice-dispatch=721.5205ms assoc-eval=125.8165ms speedup=5.73x digest=564d3767c9700000
BENCH associative_scan i32 add(x[1048576],axis=0): slice-dispatch=797.3233ms assoc-eval=126.4726ms speedup=6.30x digest=0000000040e80000
test tests::bench_associative_scan_dense_integer_vs_slice_dispatch ... ok
```

The baseline comparator is the old per-slice dispatcher algorithm reproduced
inside the same test binary, so the before/after numbers are same-worker and
same-invocation.

## Behavior proof

Command:

```text
RCH_WORKER=vmi1227854 RCH_ENV_ALLOWLIST=AGENT_NAME,RCH_WORKER AGENT_NAME=BeigeMouse rch exec -- cargo test -p fj-lax --lib associative_scan_dense_integer_bit_identical_to_slice_dispatch -- --nocapture
```

RCH selected worker `vmi1153651` for this command.

Result:

```text
test tests::associative_scan_dense_integer_bit_identical_to_slice_dispatch ... ok
integer associative_scan golden digest: 83ffbd381da3466a8a439360580be405087f409ceb95af001cc3488889aface3
```

Isomorphism:

- Ordering preserved: yes. Forward uses `op(acc, x)` and reverse uses
  `op(x, acc)`, matching the old slice-dispatch algorithm.
- Tie-breaking unchanged: yes. Integer `max` and `min` use the same signed
  `i64::max` / `i64::min` behavior as the dispatcher.
- Overflow unchanged: yes. I64 `add`/`mul` use wrapping two's-complement
  arithmetic. I32 `add`/`mul` narrow after each prefix step with the same
  mod-2^32 rule as `narrow_i32_tensor_result`.
- Floating-point unchanged: N/A.
- RNG unchanged: N/A.
- Golden output: digest
  `83ffbd381da3466a8a439360580be405087f409ceb95af001cc3488889aface3`.

## Validation

Passed:

```text
git diff --check
rustfmt --edition 2024 --unstable-features --check --file-lines '[{"file":"crates/fj-lax/src/lib.rs","range":[900,1120]},{"file":"crates/fj-lax/src/lib.rs","range":[9560,9975]}]' crates/fj-lax/src/lib.rs
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1153651 RCH_WORKERS=vmi1153651 RCH_ENV_ALLOWLIST=AGENT_NAME,RCH_REQUIRE_REMOTE,RCH_WORKER,RCH_WORKERS AGENT_NAME=BeigeMouse rch exec -- cargo check -p fj-lax --lib
```

UBS:

```text
ubs crates/fj-lax/src/lib.rs artifacts/performance/evidence/frankenjax-tnfjw-integer-associative-scan-2026-06-13.md .beads/issues.jsonl
```

UBS exited nonzero from the pre-existing file-wide `fj-lax/src/lib.rs`
inventory (panic/unwrap/indexing and contextual warnings). Its internal
formatting, clippy, cargo-check, test-build, audit, and deny sections were
clean.

Known external gate limitations:

```text
rustfmt --edition 2024 --check crates/fj-lax/src/lib.rs
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1153651 RCH_WORKERS=vmi1153651 RCH_ENV_ALLOWLIST=AGENT_NAME,RCH_REQUIRE_REMOTE,RCH_WORKER,RCH_WORKERS AGENT_NAME=BeigeMouse rch exec -- cargo clippy -p fj-lax --lib -- -D warnings
```

Full-module rustfmt is blocked by pre-existing formatting drift in imported
`fj-lax` modules outside this lever. Clippy is blocked by pre-existing lint
debt in `linalg.rs`, `reduction.rs`, and `tensor_ops.rs`; no `lib.rs`
finding from this change was reported.
