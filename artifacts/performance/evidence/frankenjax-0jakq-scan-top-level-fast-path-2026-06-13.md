# frankenjax-0jakq: top-level i64 scan add-emit fast path

## Target

- Bead: `frankenjax-0jakq`
- Crate: `fj-interpreters`
- Profile-backed target: `eval/scan_sub_jaxpr_add_emit_128`
- Baseline worker: `vmi1227854`
- Baseline command:

```bash
RCH_WORKERS=vmi1227854 RCH_REQUIRE_REMOTE=1 rch exec -- \
  cargo bench -j 1 -p fj-interpreters --bench pe_baseline -- eval/
```

Current post-pass239 profile showed this as the slowest `fj-interpreters` eval row:

```text
eval/scan_sub_jaxpr_add_emit_128
  [526.77 ns 539.59 ns 552.31 ns]
```

## One Lever

Detect only the top-level single-equation `Scan` Jaxpr whose inputs and outputs are exactly the outer Jaxpr inputs/outputs, then delegate to the existing `try_eval_scan_i64_add_emit` body fast path.

The lever removes repeated generic-environment setup for this benchmark shape. It does not add a new arithmetic implementation.

## Isomorphism Proof

- Equation order: unchanged. The top-level guard requires exactly one `Scan` equation, then executes the existing scan fast path for the same sub-Jaxpr.
- Input/output ordering: unchanged. The guard requires `equation.inputs == jaxpr.invars` and `equation.outputs == jaxpr.outvars`.
- Reverse semantics: unchanged. The guard reads the same `reverse` param and passes it through to the existing scan path.
- Integer behavior: unchanged. The delegated path uses the existing i64 wrapping add/emit implementation.
- Floating point behavior: not in scope for this path; non-i64 or non-matching bodies fall through to the generic interpreter.
- Tie-breaking: not in scope; no comparisons are introduced.
- RNG: unchanged. The accepted Jaxpr has no effects, and the delegated path has no RNG source.
- Fallback surface: all constvar, effectful, arity-mismatched, atom-mismatched, multi-equation, multi-sub-Jaxpr, or unsupported-body cases return `None` and use the prior generic path.

## Golden Output

Focused proof compares top-level fast evaluation against forced generic evaluation for forward and reverse scans using wrapping-edge values:

```text
carry0 = i64::MAX - 3
xs = [4, -7, i64::MIN, 11]
```

Golden digest:

```text
775f4b39aa923c00abea919a50d2de053c9a09f18ce1e9758a10eccc8b4d1e3b
```

Proof command:

```bash
RCH_WORKERS=vmi1227854 RCH_REQUIRE_REMOTE=1 rch exec -- \
  cargo test -j 1 -p fj-interpreters --lib \
  eval_top_level_scan_i64_add_emit_fast_path_matches_generic_and_golden -- --nocapture
```

Result: passed, 1 test.

## Benchmark Gate

Same-worker after command:

```bash
RCH_WORKERS=vmi1227854 RCH_REQUIRE_REMOTE=1 rch exec -- \
  cargo bench -j 1 -p fj-interpreters --bench pe_baseline -- eval/scan_sub_jaxpr_add_emit_128
```

```text
baseline: [526.77 ns 539.59 ns 552.31 ns]
after:    [209.23 ns 218.08 ns 227.07 ns]
median:   539.59 ns -> 218.08 ns
ratio:    2.47x
score:    4.69 = 2.47 impact * 0.95 confidence / 0.50 effort
```

Decision: keep.

## Validation

Passed:

```bash
rustfmt --edition 2024 --check crates/fj-interpreters/src/lib.rs
git diff --check
RCH_WORKERS=vmi1227854 RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -j 1 -p fj-interpreters --lib
```

RCH clippy remains blocked by pre-existing lint debt outside this pass:

```text
crates/fj-interpreters/src/lib.rs:5354 clippy::question_mark
```

`ubs crates/fj-interpreters/src/lib.rs` remains nonzero from pre-existing file-wide inventory, while its built-in formatting, clippy, cargo check, test-build, audit, and deny checks reported clean.
