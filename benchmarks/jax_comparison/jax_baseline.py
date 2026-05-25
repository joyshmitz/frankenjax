#!/usr/bin/env python3
"""
JAX Baseline Benchmarks for FrankenJAX Comparison

These benchmarks measure the same operations as FrankenJAX's Criterion benchmarks
for apples-to-apples performance comparison.

Run with: python jax_baseline.py --runs 20
Output: JSON with p50/p95/p99 timing statistics
"""

import argparse
import json
import time
import platform
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Callable, Any, Dict
from functools import partial

# Defer JAX import to handle missing dependency gracefully
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap, value_and_grad
    import jax.lax as lax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@dataclass
class BenchmarkResult:
    name: str
    runs: int
    p50_ns: float
    p95_ns: float
    p99_ns: float
    mean_ns: float
    std_ns: float
    throughput_ops_per_sec: float


def percentile(data: List[float], p: float) -> float:
    """Calculate percentile of a sorted list."""
    k = (len(data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(data) else f
    return data[f] + (k - f) * (data[c] - data[f])


def benchmark_fn(fn: Callable, warmup: int = 5, runs: int = 20) -> BenchmarkResult:
    """Benchmark a function with warmup and multiple runs."""
    # Warmup
    for _ in range(warmup):
        fn()

    # Timed runs
    times_ns = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        fn()
        end = time.perf_counter_ns()
        times_ns.append(end - start)

    times_ns.sort()
    mean = np.mean(times_ns)
    std = np.std(times_ns)

    return BenchmarkResult(
        name="",
        runs=runs,
        p50_ns=percentile(times_ns, 50),
        p95_ns=percentile(times_ns, 95),
        p99_ns=percentile(times_ns, 99),
        mean_ns=mean,
        std_ns=std,
        throughput_ops_per_sec=1e9 / mean if mean > 0 else 0,
    )


# ===========================================================================
# 1. Transform Overhead Benchmarks
# ===========================================================================

def bench_jit_scalar_add(runs: int) -> BenchmarkResult:
    """Equivalent to FrankenJAX: api_overhead/jit/scalar_add"""
    @jit
    def add_two(a, b):
        return a + b

    a, b = jnp.array(3, dtype=jnp.int64), jnp.array(4, dtype=jnp.int64)
    add_two(a, b).block_until_ready()  # JIT compile

    def run():
        add_two(a, b).block_until_ready()

    result = benchmark_fn(run, runs=runs)
    result.name = "jit/scalar_add"
    return result


def bench_grad_scalar_square(runs: int) -> BenchmarkResult:
    """Equivalent to FrankenJAX: api_overhead/grad/scalar_square"""
    def square(x):
        return x * x

    grad_square = jit(grad(square))
    x = jnp.array(3.0, dtype=jnp.float64)
    grad_square(x).block_until_ready()  # JIT compile

    def run():
        grad_square(x).block_until_ready()

    result = benchmark_fn(run, runs=runs)
    result.name = "grad/scalar_square"
    return result


def bench_vmap_vector_add_one(runs: int) -> BenchmarkResult:
    """Equivalent to FrankenJAX: api_overhead/vmap/vector_add_one"""
    def add_one(x):
        return x + 1

    vmap_add_one = jit(vmap(add_one))
    x = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int64)
    vmap_add_one(x).block_until_ready()

    def run():
        vmap_add_one(x).block_until_ready()

    result = benchmark_fn(run, runs=runs)
    result.name = "vmap/vector_add_one"
    return result


def bench_value_and_grad_scalar_square(runs: int) -> BenchmarkResult:
    """Equivalent to FrankenJAX: api_overhead/value_and_grad/scalar_square"""
    def square(x):
        return x * x

    vag_square = jit(value_and_grad(square))
    x = jnp.array(3.0, dtype=jnp.float64)
    vag_square(x)  # JIT compile

    def run():
        val, g = vag_square(x)
        val.block_until_ready()
        g.block_until_ready()

    result = benchmark_fn(run, runs=runs)
    result.name = "value_and_grad/scalar_square"
    return result


def bench_jit_grad_scalar_square(runs: int) -> BenchmarkResult:
    """Equivalent to FrankenJAX: dispatch_latency/jit_grad/scalar_square"""
    def square(x):
        return x * x

    jit_grad_square = jit(grad(square))
    x = jnp.array(3.0, dtype=jnp.float64)
    jit_grad_square(x).block_until_ready()

    def run():
        jit_grad_square(x).block_until_ready()

    result = benchmark_fn(run, runs=runs)
    result.name = "jit_grad/scalar_square"
    return result


def bench_vmap_grad_vector_square(runs: int) -> BenchmarkResult:
    """Equivalent to FrankenJAX: dispatch_latency/vmap_grad/vector_square"""
    def square(x):
        return x * x

    vmap_grad_square = jit(vmap(grad(square)))
    x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
    vmap_grad_square(x).block_until_ready()

    def run():
        vmap_grad_square(x).block_until_ready()

    result = benchmark_fn(run, runs=runs)
    result.name = "vmap_grad/vector_square"
    return result


# ===========================================================================
# 2. AD Benchmarks
# ===========================================================================

def bench_ad_grad_polynomial(runs: int) -> BenchmarkResult:
    """Equivalent to FrankenJAX: ad/grad_poly_x3+x2+x"""
    def poly(x):
        return x**3 + x**2 + x

    grad_poly = jit(grad(poly))
    x = jnp.array(2.0, dtype=jnp.float64)
    grad_poly(x).block_until_ready()

    def run():
        grad_poly(x).block_until_ready()

    result = benchmark_fn(run, runs=runs)
    result.name = "ad/grad_poly_x3+x2+x"
    return result


def bench_ad_grad_trig(runs: int) -> BenchmarkResult:
    """Equivalent to FrankenJAX: ad/grad_sin_cos_mul"""
    def trig(x):
        return jnp.sin(x) * jnp.cos(x)

    grad_trig = jit(grad(trig))
    x = jnp.array(1.0, dtype=jnp.float64)
    grad_trig(x).block_until_ready()

    def run():
        grad_trig(x).block_until_ready()

    result = benchmark_fn(run, runs=runs)
    result.name = "ad/grad_sin_cos_mul"
    return result


def bench_ad_grad_exp_log(runs: int) -> BenchmarkResult:
    """Equivalent to FrankenJAX: ad/grad_exp_log"""
    def exp_log(x):
        return jnp.log(jnp.exp(x))

    grad_exp_log = jit(grad(exp_log))
    x = jnp.array(1.0, dtype=jnp.float64)
    grad_exp_log(x).block_until_ready()

    def run():
        grad_exp_log(x).block_until_ready()

    result = benchmark_fn(run, runs=runs)
    result.name = "ad/grad_exp_log"
    return result


# ===========================================================================
# 3. LAX Primitive Benchmarks
# ===========================================================================

def bench_eval_add_1k_vector(runs: int) -> BenchmarkResult:
    """Equivalent to FrankenJAX: eval/add_1k_i64_vec"""
    @jit
    def add_vecs(a, b):
        return a + b

    a = jnp.arange(1000, dtype=jnp.int64)
    b = jnp.arange(1000, dtype=jnp.int64)
    add_vecs(a, b).block_until_ready()

    def run():
        add_vecs(a, b).block_until_ready()

    result = benchmark_fn(run, runs=runs)
    result.name = "eval/add_1k_i64_vec"
    return result


def bench_eval_mul_1k_vector(runs: int) -> BenchmarkResult:
    """Equivalent to FrankenJAX: eval/mul_1k_f64_vec"""
    @jit
    def mul_vecs(a, b):
        return a * b

    a = jnp.arange(1000, dtype=jnp.float64) * 0.001
    b = jnp.arange(1000, dtype=jnp.float64) * 0.001
    mul_vecs(a, b).block_until_ready()

    def run():
        mul_vecs(a, b).block_until_ready()

    result = benchmark_fn(run, runs=runs)
    result.name = "eval/mul_1k_f64_vec"
    return result


def bench_eval_sin_1k(runs: int) -> BenchmarkResult:
    """Equivalent to FrankenJAX: eval/sin_1k_f64"""
    @jit
    def sin_vec(x):
        return jnp.sin(x)

    x = jnp.arange(1000, dtype=jnp.float64) * 0.001
    sin_vec(x).block_until_ready()

    def run():
        sin_vec(x).block_until_ready()

    result = benchmark_fn(run, runs=runs)
    result.name = "eval/sin_1k_f64"
    return result


def bench_eval_exp_1k(runs: int) -> BenchmarkResult:
    """Equivalent to FrankenJAX: eval/exp_1k_f64"""
    @jit
    def exp_vec(x):
        return jnp.exp(x)

    x = jnp.arange(1000, dtype=jnp.float64) * 0.001
    exp_vec(x).block_until_ready()

    def run():
        exp_vec(x).block_until_ready()

    result = benchmark_fn(run, runs=runs)
    result.name = "eval/exp_1k_f64"
    return result


def bench_eval_reduce_sum_1k(runs: int) -> BenchmarkResult:
    """Equivalent to FrankenJAX: eval/reduce_sum_1k_i64"""
    @jit
    def reduce_sum(x):
        return jnp.sum(x)

    x = jnp.arange(1000, dtype=jnp.int64)
    reduce_sum(x).block_until_ready()

    def run():
        reduce_sum(x).block_until_ready()

    result = benchmark_fn(run, runs=runs)
    result.name = "eval/reduce_sum_1k_i64"
    return result


def bench_eval_dot_100(runs: int) -> BenchmarkResult:
    """Equivalent to FrankenJAX: eval/dot_100_i64"""
    @jit
    def dot_vecs(a, b):
        return jnp.dot(a, b)

    a = jnp.arange(100, dtype=jnp.int64)
    b = jnp.arange(100, dtype=jnp.int64)
    dot_vecs(a, b).block_until_ready()

    def run():
        dot_vecs(a, b).block_until_ready()

    result = benchmark_fn(run, runs=runs)
    result.name = "eval/dot_100_i64"
    return result


# ===========================================================================
# 4. FFT Benchmarks
# ===========================================================================

def bench_eval_fft_256(runs: int) -> BenchmarkResult:
    """Equivalent to FrankenJAX: eval/fft_256_complex128"""
    @jit
    def fft(x):
        return jnp.fft.fft(x)

    x = jnp.array([complex(np.sin(i * 0.125), np.cos(i * 0.25)) for i in range(256)], dtype=jnp.complex128)
    fft(x).block_until_ready()

    def run():
        fft(x).block_until_ready()

    result = benchmark_fn(run, runs=runs)
    result.name = "eval/fft_256_complex128"
    return result


def bench_eval_rfft_256(runs: int) -> BenchmarkResult:
    """Equivalent to FrankenJAX: eval/rfft_256_f64"""
    @jit
    def rfft(x):
        return jnp.fft.rfft(x)

    x = jnp.array([np.sin(i * 0.125) + np.cos(i * 0.03125) for i in range(256)], dtype=jnp.float64)
    rfft(x).block_until_ready()

    def run():
        rfft(x).block_until_ready()

    result = benchmark_fn(run, runs=runs)
    result.name = "eval/rfft_256_f64"
    return result


# ===========================================================================
# 5. Linear Algebra Benchmarks
# ===========================================================================

def bench_vmap_qr_512x3x2(runs: int) -> BenchmarkResult:
    """Equivalent to FrankenJAX: vmap_qr/batched_matrix_3x2"""
    @jit
    def batched_qr(x):
        return vmap(jnp.linalg.qr)(x)

    # Generate 512 3x2 matrices
    matrices = jnp.array([
        [[1.0 + (b % 11) * 0.01, 0.0], [0.0, 1.0 + (b % 11) * 0.01], [1.0 + (b % 11) * 0.01, 1.0 + (b % 11) * 0.01]]
        for b in range(512)
    ], dtype=jnp.float64)

    batched_qr(matrices)  # JIT compile

    def run():
        q, r = batched_qr(matrices)
        q.block_until_ready()
        r.block_until_ready()

    result = benchmark_fn(run, runs=runs)
    result.name = "vmap_qr/batched_matrix_3x2"
    return result


def bench_vmap_svd_512x3x2(runs: int) -> BenchmarkResult:
    """Equivalent to FrankenJAX: vmap_svd/batched_matrix_3x2"""
    @jit
    def batched_svd(x):
        return vmap(jnp.linalg.svd, in_axes=0)(x)

    matrices = jnp.array([
        [[1.0 + (b % 11) * 0.01, 0.1], [0.0, 1.5 + (b % 11) * 0.01], [0.2, 2.0 + (b % 11) * 0.01]]
        for b in range(512)
    ], dtype=jnp.float64)

    batched_svd(matrices)  # JIT compile

    def run():
        u, s, vh = batched_svd(matrices)
        u.block_until_ready()
        s.block_until_ready()
        vh.block_until_ready()

    result = benchmark_fn(run, runs=runs)
    result.name = "vmap_svd/batched_matrix_3x2"
    return result


def bench_vmap_eigh_512x3x3(runs: int) -> BenchmarkResult:
    """Equivalent to FrankenJAX: vmap_eigh/batched_matrix_3x3"""
    @jit
    def batched_eigh(x):
        return vmap(jnp.linalg.eigh)(x)

    # Generate 512 symmetric 3x3 matrices
    matrices = jnp.array([
        [[2.0 + (b % 13) * 0.01, 0.1, 0.0],
         [0.1, 3.0 + (b % 13) * 0.01, 0.2],
         [0.0, 0.2, 4.0 + (b % 13) * 0.01]]
        for b in range(512)
    ], dtype=jnp.float64)

    batched_eigh(matrices)  # JIT compile

    def run():
        w, v = batched_eigh(matrices)
        w.block_until_ready()
        v.block_until_ready()

    result = benchmark_fn(run, runs=runs)
    result.name = "vmap_eigh/batched_matrix_3x3"
    return result


# ===========================================================================
# 6. Control Flow Benchmarks
# ===========================================================================

def bench_vmap_scan_128x64(runs: int) -> BenchmarkResult:
    """Equivalent to FrankenJAX: vmap_scan/shared_init_batched_xs_128x64"""
    def scan_body(carry, x):
        return carry + x, carry + x

    @jit
    def batched_scan(init, xs):
        return vmap(lambda xs_slice: lax.scan(scan_body, init, xs_slice))(xs)

    init = jnp.int64(0)
    xs = jnp.array([[(i * 17) % 17 for i in range(64)] for _ in range(128)], dtype=jnp.int64)

    batched_scan(init, xs)  # JIT compile

    def run():
        final, outputs = batched_scan(init, xs)
        final.block_until_ready()
        outputs.block_until_ready()

    result = benchmark_fn(run, runs=runs)
    result.name = "vmap_scan/shared_init_batched_xs_128x64"
    return result


# ===========================================================================
# Main
# ===========================================================================

def get_environment_fingerprint() -> Dict[str, str]:
    """Capture environment details for reproducibility."""
    fingerprint = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }

    if JAX_AVAILABLE:
        fingerprint["jax_version"] = jax.__version__
        fingerprint["jax_devices"] = str(jax.devices())

    return fingerprint


def main():
    parser = argparse.ArgumentParser(description="JAX Baseline Benchmarks")
    parser.add_argument("--runs", type=int, default=20, help="Number of runs per benchmark")
    parser.add_argument("--output", type=str, default="jax_baseline_results.json", help="Output JSON file")
    args = parser.parse_args()

    if not JAX_AVAILABLE:
        print("JAX not installed. Install with: pip install jax jaxlib")
        return 1

    # Force CPU execution for fair comparison
    jax.config.update('jax_platform_name', 'cpu')
    # Enable 64-bit precision for fair comparison with FrankenJAX
    jax.config.update('jax_enable_x64', True)

    benchmarks = [
        # Transform overhead
        bench_jit_scalar_add,
        bench_grad_scalar_square,
        bench_vmap_vector_add_one,
        bench_value_and_grad_scalar_square,
        bench_jit_grad_scalar_square,
        bench_vmap_grad_vector_square,
        # AD
        bench_ad_grad_polynomial,
        bench_ad_grad_trig,
        bench_ad_grad_exp_log,
        # LAX primitives
        bench_eval_add_1k_vector,
        bench_eval_mul_1k_vector,
        bench_eval_sin_1k,
        bench_eval_exp_1k,
        bench_eval_reduce_sum_1k,
        bench_eval_dot_100,
        # FFT
        bench_eval_fft_256,
        bench_eval_rfft_256,
        # Linear algebra
        bench_vmap_qr_512x3x2,
        bench_vmap_svd_512x3x2,
        bench_vmap_eigh_512x3x3,
        # Control flow
        bench_vmap_scan_128x64,
    ]

    print(f"Running {len(benchmarks)} benchmarks with {args.runs} runs each...")
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print()

    results = []
    for bench_fn in benchmarks:
        print(f"  {bench_fn.__name__}...", end=" ", flush=True)
        result = bench_fn(args.runs)
        results.append(asdict(result))
        print(f"p50={result.p50_ns/1000:.1f}us p99={result.p99_ns/1000:.1f}us")

    output = {
        "environment": get_environment_fingerprint(),
        "runs_per_benchmark": args.runs,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())
