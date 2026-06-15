# frankenjax-mcqr.62: VmapWrapped repeated-call metadata cache

Date: 2026-06-15
Agent: SilverMaple
Bead: frankenjax-mcqr.62

## Target

`VmapWrapped::call` still used `dispatch_with_options`, so repeated calls rebuilt
the args-independent dispatch metadata: transform evidence and canonical Jaxpr
cache key. This pass applies the same pure memo pattern already used by
`GradWrapped` and `ValueAndGradWrapped`.

## One Lever

Add `DispatchMetaCache` to `VmapWrapped`, reset it when backend, mode, `in_axes`,
or `out_axes` changes, and call `dispatch_with_options_prepared` with the cached
metadata. The batch-trace evaluation and all argument-dependent work still run
per call.

## Baseline And Rebench

Clean HEAD public-row baseline, RCH `ovh-a`:

```text
cargo bench -j 1 -p fj-api --bench api_overhead -- api_overhead/vmap/vector_add_one
api_overhead/vmap/vector_add_one: [2.8869 us 3.1908 us 3.5159 us]
```

Candidate public row, RCH `vmi1152480` (cross-worker; routing evidence only):

```text
api_overhead/vmap/vector_add_one: [6.4516 us 7.1572 us 7.8689 us]
```

Same-binary A/B acceptance row, RCH `vmi1152480`:

```text
vmap_meta_cache/recompute/add_one: [4.8365 us 5.2076 us 5.7230 us]
vmap_meta_cache/prepared/add_one:  [3.2861 us 3.5711 us 3.8877 us]
```

Midpoint speedup: `5.2076 / 3.5711 = 1.46x`.
Conservative interval speedup: `4.8365 / 3.8877 = 1.24x`.

Score: `Impact 3 * Confidence 4 / Effort 2 = 6.0`.

## Isomorphism Proof

- Ordering preserved: yes. `Transform::Vmap`, compile options, and argument order
  are passed to dispatch unchanged.
- Tie-breaking unchanged: yes. No comparison or sorting behavior changes.
- Floating-point: unchanged. The same dispatch path and primitive kernels execute;
  only args-independent proof/key preparation is memoized.
- RNG seeds: N/A. This Vmap AddOne benchmark/test has no RNG surface.
- Fallback behavior: unchanged. Metadata preparation failure stores `None` and the
  prepared dispatch wrapper falls back to the unprepared path.
- Cache invalidation: backend, mode, `in_axes`, and `out_axes` reset the cache
  because they feed the cache key or compile options.

## Golden

Payload:

```text
vmap_meta_cache add_one warmed=[2,3,4,5,6] reference=[2,3,4,5,6]
```

SHA-256:

```text
71cc9ad54fc35b1240b1419bccafd0ed93ade694ca3368fc2aa008ab4184e65d  artifacts/performance/evidence/frankenjax-mcqr62-vmap-meta-cache-2026-06-15.golden
```

## Validation

```text
cargo fmt --check -p fj-api
RCH cargo test -j 1 -p fj-api --lib vmap_repeated_call_meta_cache_matches_dispatch -- --nocapture
RCH cargo check -j 1 -p fj-api --all-targets
RCH cargo clippy -j 1 -p fj-api --all-targets -- -D warnings
ubs crates/fj-api/src/transforms.rs crates/fj-api/benches/api_overhead.rs
sha256sum -c artifacts/performance/evidence/frankenjax-mcqr62-vmap-meta-cache-2026-06-15.golden.sha256
```

Result: all passed. UBS exited 0; it still reports existing file-wide warning
inventory, but its built-in fmt, clippy, check, test-build, audit, and deny
sections were clean.

## Next Route

Reprofile API/interpreter rows after landing. Do not repeat metadata-cache work
unless a fresh profile ranks a wrapper that still calls unprepared dispatch.
