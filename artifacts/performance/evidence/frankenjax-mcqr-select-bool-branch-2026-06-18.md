# frankenjax-mcqr: dense Bool select branch pass

Date: 2026-06-18
Agent: cod-a / RedBeaver
Status: in_progress, code-first batch-test pending

## Lever

Add a boolean-branch `Select` fast path for mask algebra:

- `select(Bool cond, Bool on_true, Bool on_false)` now avoids the boxed
  per-`Literal` loop when all three tensors expose dense or packed bool backing.
- Dense Bool inputs emit dense bool output.
- Packed `BoolWords` inputs bit-test packed `Vec<u64>` words directly and emit a
  packed `BoolWords` output.
- Tensor condition plus scalar bool branches now emits dense bool output instead
  of boxed literals.

This is a storage/layout lever only. The semantic witness is still the original
per-element rule: `out[i] = if cond[i] { on_true[i] } else { on_false[i] }`.

## Alien-source mapping

- `/alien-graveyard` section 7.2 succinct bitvectors: store boolean masks in
  `Vec<u64>` and query bits directly instead of expanding them to element
  objects.
- `/alien-graveyard` section 8.2 vectorized execution: operate over cache-sized
  dense batches rather than tuple-at-a-time interpretation.
- `/alien-graveyard` section 7.7 Swiss Tables: compact metadata streams should
  stay separate from payload and be scanned in cache-friendly form.
- `/alien-artifact-coding`: the proof artifact is a direct isomorphism check
  against the boxed literal path for dense bool, packed BoolWords, and scalar
  bool branches.
- `/extreme-software-optimization`: one lever, one crate, no speedup claim until
  criterion batch evidence lands.

## Correctness guard

Updated `bool_select_dense_and_boolwords_match_generic` in
`crates/fj-lax/src/arithmetic.rs` to compare:

- Dense bool condition/branches vs fully boxed literal condition/branches.
- Packed BoolWords condition/branches vs the same boxed literal path.
- Packed BoolWords condition plus scalar bool branches vs boxed condition plus
  scalar bool branches.

The guard also asserts dense bool outputs stay `as_bool_slice()`-backed and
packed inputs emit `as_bool_words()` output.

## Benchmark guard

Added criterion rows in `crates/fj-lax/benches/lax_baseline.rs`:

- `eval/select_64k_bool_vec`
- `eval/select_64k_bool_literal_ref`

These quantify boolean mask-algebra selection against the literal reference path
in the next batch run.

## Negative-evidence ledger

- Dense SelectN case-output and index-decode expansions already shipped today.
  Do not retry SelectN storage widening without a new profile that names a
  different remaining SelectN bottleneck.
- FMA/SIMD-exp/GEMM remains maintainer-gated under `frankenjax-cntiy`; no retry
  until the policy decision changes.
- Cumsum axis specialization remains a prior rejected/non-comparable family; no
  retry without fresh same-worker evidence.
- Cod-b owns active `fj-core` dense repeat/slice/to_i64 lanes. Do not touch
  those surfaces from this cod-a loop without coordination.
- No benchmark speedup is claimed in this commit. The criterion rows are the
  code-first batch-test target.

## Local validation

Passed:

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a cargo check -p fj-lax
```
