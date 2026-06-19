# FrankenJAX Perf Release Readiness Scorecard

Updated: 2026-06-19

Scope: verify recent code-first `fj-lax` perf backlog against original JAX on
realistic warmed CPU workloads. This scorecard records measured readiness only;
unmeasured `code-first batch-test pending` entries remain outside the score.

## Environment

- Agent: cod-b / WildForge
- Host: AMD Ryzen Threadripper PRO 5975WX, 64 logical CPUs, Linux 6.17.0-35
- Rust: `rustc 1.98.0-nightly (f20a92ec0 2026-06-07)`
- Cargo target dir: `/data/projects/.rch-targets/frankenjax-cod-b`
- Rust bench command family: `cargo bench -p fj-lax --bench lax_baseline`
- JAX oracle: `uv run --with 'jax[cpu]' --with numpy python`
- JAX/JAXLIB: 0.10.2 / 0.10.2, `jax_enable_x64=true`, CPU backend
- JAX timing protocol: warmed `block_until_ready()` execution, 12 batches x 20
  iterations per workload

## Measured Workloads

| Bead | Workload | Rust median | JAX median | Rust/JAX | Outcome |
| --- | --- | ---: | ---: | ---: | --- |
| frankenjax-19wst | `tile_scalar_f32_1024x1024` | 51.435 us | 204.302 us | 0.252 | Rust 3.97x faster |
| frankenjax-19wst | `tile_scalar_complex128_1024x1024` | 412.679 us | 495.802 us | 0.832 | Rust 1.20x faster |
| frankenjax-1z7k9 | `complex_f32_tensor_scalar_1m` | 1.379 ms | 1.071 ms | 1.287 | Rust 1.29x slower |
| frankenjax-1z7k9 | `complex_f64_tensor_scalar_1m` | 0.914 ms | 1.855 ms | 0.493 | Rust 2.03x faster |

## Readiness

- JAX domination score for this measured cluster: 75/100.
- Basis: 3 of 4 measured realistic workloads beat warmed original JAX CPU.
- Release blocker for this cluster: `complex_f32_tensor_scalar_1m` remains a
  JAX loss at Rust/JAX 1.287.
- Reverts: none. No measured workload showed that reverting the committed dense
  path would improve results; the mixed complex constructor cluster keeps the
  F64 win and routes the F32 loss to deeper representation work.
- Next measured gate: Complex64/F32 constructor must target packed Complex64
  output construction or a fused real-to-complex path, not another retry of the
  boxed-literal-elision lever.
