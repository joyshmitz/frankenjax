#!/usr/bin/env python3
"""JAX CPU head-to-head for the FrankenJAX broadcast_replicate block-copy lever
(bead frankenjax-thnjs). Mirrors crates/fj-lax/benches/broadcast_gauntlet.rs: a
bias broadcast [D] -> [ROWS, D] on f32 (materialized).

Run:
  benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/broadcast_gauntlet.py \
      --runs 30 --warmup 5 --inner-loops 50 --output /tmp/frankenjax_broadcast_jax.json
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

ROWS, D = 4096, 768


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


def broadcast_bias(bias):
    # +0.0 forces materialization (a bare broadcast_to is a lazy view in XLA; the
    # Rust eval_primitive always materializes the replicated tensor).
    return jnp.broadcast_to(bias, (ROWS, D)) + jnp.float32(0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--inner-loops", type=int, default=50)
    ap.add_argument("--output", type=str, default="")
    args = ap.parse_args()

    bias = (jnp.arange(D, dtype=jnp.float32) * 1e-3 - 0.25)
    results = [bench("broadcast_bias_D768_to_4096x768_f32", broadcast_bias, (bias,),
                     args.runs, args.warmup, args.inner_loops)]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine": "jax_jit_cpu",
        "jax_version": jax.__version__,
        "platform": platform.platform(),
        "shape": [ROWS, D],
        "elements": ROWS * D,
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
