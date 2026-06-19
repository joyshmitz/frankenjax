#!/usr/bin/env python3
"""JAX CPU head-to-head for the einsum permute-copy trailing-block memcpy lever.

Mirrors crates/fj-lax/benches/lax_baseline.rs:
eval/einsum2_general_bqhd_bkhd_bhqk_f64 uses an attention-score pattern with
the head axis in the middle, forcing the general einsum path:

  bqhd,bkhd->bhqk

Run:
  benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/einsum_permute_copy_gauntlet.py \
      --runs 50 --warmup 10 --inner-loops 20 --output /tmp/frankenjax_einsum_permute_jax.json
"""

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

B, Q, H, D, K = 4, 64, 8, 64, 64


def percentile(sorted_values, pct):
    k = (len(sorted_values) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


def bench(name, fn, runs, warmup, inner_loops):
    fn()
    for _ in range(warmup):
        for _ in range(inner_loops):
            fn()

    times = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        for _ in range(inner_loops):
            fn()
        times.append((time.perf_counter_ns() - start) / inner_loops)

    times.sort()
    mean = statistics.fmean(times)
    std = statistics.pstdev(times) if len(times) > 1 else 0.0
    return {
        "name": name,
        "engine": "jax_jit_cpu",
        "p50_ns": percentile(times, 50),
        "p95_ns": percentile(times, 95),
        "p99_ns": percentile(times, 99),
        "mean_ns": mean,
        "cv_pct": (std / mean * 100.0) if mean else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--inner-loops", type=int, default=20)
    ap.add_argument("--output", type=str, default="")
    args = ap.parse_args()

    a = jnp.sin(jnp.arange(B * Q * H * D, dtype=jnp.float64) * 0.0011).reshape(B, Q, H, D)
    b = jnp.cos(jnp.arange(B * K * H * D, dtype=jnp.float64) * 0.0009).reshape(B, K, H, D)

    einsum = jax.jit(lambda lhs, rhs: jnp.einsum("bqhd,bkhd->bhqk", lhs, rhs))
    einsum(a, b).block_until_ready()

    def ready():
        einsum(a, b).block_until_ready()

    def host_copy():
        np.asarray(einsum(a, b).block_until_ready())

    results = [
        bench(
            "einsum_bqhd_bkhd_bhqk_f64_ready",
            ready,
            args.runs,
            args.warmup,
            args.inner_loops,
        ),
        bench(
            "einsum_bqhd_bkhd_bhqk_f64_host_copy",
            host_copy,
            args.runs,
            args.warmup,
            args.inner_loops,
        ),
    ]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine": "jax_jit_cpu",
        "jax_version": jax.__version__,
        "platform": platform.platform(),
        "workload": {
            "expr": "jax.jit(lambda lhs, rhs: jnp.einsum('bqhd,bkhd->bhqk', lhs, rhs))",
            "lhs_shape": [B, Q, H, D],
            "rhs_shape": [B, K, H, D],
            "out_shape": [B, H, Q, K],
            "dtype": "float64",
        },
        "results": results,
    }
    text = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    for result in results:
        print(
            f"{result['name']}: mean={result['mean_ns']:.1f}ns "
            f"p50={result['p50_ns']:.1f}ns p95={result['p95_ns']:.1f}ns "
            f"p99={result['p99_ns']:.1f}ns cv={result['cv_pct']:.2f}%"
        )


if __name__ == "__main__":
    main()
