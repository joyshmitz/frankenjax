#!/usr/bin/env python3
"""JAX CPU head-to-head for the FrankenJAX dense integer_pow lever (bead
frankenjax-hfq7o): x**2 on a 1M f64/f32 vector. Mirrors
crates/fj-lax/benches/integer_pow_gauntlet.rs.

Run:
  benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/integer_pow_gauntlet.py \
      --runs 50 --warmup 10 --inner-loops 100 --output /tmp/frankenjax_integer_pow_jax.json
"""

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

N = 1_048_576


def percentile(sorted_values, pct):
    k = (len(sorted_values) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


def bench(name, fn, args, runs, warmup, inner_loops):
    compiled = jax.jit(fn)
    compiled(*args).block_until_ready()
    for _ in range(warmup):
        for _ in range(inner_loops):
            compiled(*args).block_until_ready()
    times = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        for _ in range(inner_loops):
            compiled(*args).block_until_ready()
        times.append((time.perf_counter_ns() - start) / inner_loops)
    times.sort()
    mean = statistics.fmean(times)
    std = statistics.pstdev(times) if len(times) > 1 else 0.0
    return {
        "name": name,
        "engine": "jax_jit_cpu",
        "p50_ns": percentile(times, 50),
        "mean_ns": mean,
        "cv_pct": (std / mean * 100.0) if mean else 0.0,
    }


def square(x):
    return x * x  # lax.integer_pow[2] lowers to this


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--inner-loops", type=int, default=100)
    ap.add_argument("--output", type=str, default="")
    args = ap.parse_args()

    xf64 = jnp.arange(N, dtype=jnp.float64) * 1e-6 - 0.5
    xf32 = (jnp.arange(N, dtype=jnp.float32) * 1e-6 - 0.5)
    results = [
        bench("integer_pow2_f64_1m", square, (xf64,), args.runs, args.warmup, args.inner_loops),
        bench("integer_pow2_f32_1m", square, (xf32,), args.runs, args.warmup, args.inner_loops),
    ]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine": "jax_jit_cpu",
        "jax_version": jax.__version__,
        "platform": platform.platform(),
        "n": N,
        "results": results,
    }
    text = json.dumps(payload, indent=2)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(text)
    print(text)
    for r in results:
        print(f"{r['name']}: mean={r['mean_ns']:.1f}ns p50={r['p50_ns']:.1f}ns cv={r['cv_pct']:.2f}%")


if __name__ == "__main__":
    main()
