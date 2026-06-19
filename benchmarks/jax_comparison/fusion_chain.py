#!/usr/bin/env python3
"""JAX CPU head-to-head for the FrankenJAX eval_jaxpr elementwise-FUSION cluster.

Mirrors the workloads in `crates/fj-interpreters/benches/eval_fusion_speed.rs`:
a deep single-use cheap-elementwise chain over a 1M-element vector. The Rust
side measures the INTERPRETER (eval_jaxpr, now fused) vs the per-op reference;
this measures jax.jit (XLA-compiled) on the same logical computation, so the
ledger can record the honest interpreter-vs-compiler absolute gap alongside the
Rust-internal fused-vs-unfused win.

Run:
  benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/fusion_chain.py \
      --runs 50 --warmup 10 --inner-loops 100 --output /tmp/frankenjax_fusion_chain_jax.json
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
    cv_pct = std / mean * 100.0 if mean else 0.0
    return {
        "name": name,
        "engine": "jax_jit_cpu",
        "runs": runs,
        "inner_loops": inner_loops,
        "p50_ns": percentile(times, 50),
        "p95_ns": percentile(times, 95),
        "mean_ns": mean,
        "cv_pct": cv_pct,
    }


# Mirrors eval_fusion_speed.rs F64 chain (8 cheap elementwise ops):
#   v1=x*x; v2=v1+0.5; v3=v2-x; v4=v3*y; v5=v4+1; v6=v5-y; v7=v6*2; out=v7+x
def chain_arith8(x, y):
    v1 = x * x
    v2 = v1 + 0.5
    v3 = v2 - x
    v4 = v3 * y
    v5 = v4 + 1.0
    v6 = v5 - y
    v7 = v6 * 2.0
    return v7 + x


# Square-bearing chain (exercises the new Square/integer_pow[2] fusion lever):
#   out = ((x - 0.5) ** 2) * y + x   -> sub, square, mul, add
def chain_square(x, y):
    c = x - 0.5
    return (c * c) * y + x


# bf16 bias-add + activation tail (exercises the half broadcast fusion lever):
#   out = ((x + bias) * scale - bias) * 2 + x   with bias/scale broadcast [D] over [B,D]
def chain_bf16_broadcast(x, bias, scale):
    v = x + bias
    v = v * scale
    v = v - bias
    v = v * jnp.bfloat16(2.0)
    return v + x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--inner-loops", type=int, default=100)
    ap.add_argument("--output", type=str, default="")
    args = ap.parse_args()

    x = jnp.arange(N, dtype=jnp.float64) * 1e-6 - 0.5
    y = jnp.arange(N, dtype=jnp.float64) * 2e-6 + 0.25

    rows, cols = 4096, 256  # rows*cols == 1,048,576
    xb = (jnp.arange(rows * cols, dtype=jnp.float32) * 1e-3).astype(jnp.bfloat16).reshape(rows, cols)
    bias = (jnp.arange(cols, dtype=jnp.float32) * 1e-2).astype(jnp.bfloat16)
    scale = (jnp.arange(cols, dtype=jnp.float32) * 3e-3 + 1.0).astype(jnp.bfloat16)

    results = [
        bench("fusion_arith8_f64_1m", chain_arith8, (x, y), args.runs, args.warmup, args.inner_loops),
        bench("fusion_square_f64_1m", chain_square, (x, y), args.runs, args.warmup, args.inner_loops),
        bench("fusion_bf16_broadcast_1m", chain_bf16_broadcast, (xb, bias, scale), args.runs, args.warmup, args.inner_loops),
    ]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine": "jax_jit_cpu",
        "jax_version": jax.__version__,
        "platform": platform.platform(),
        "n": N,
        "x64": True,
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
