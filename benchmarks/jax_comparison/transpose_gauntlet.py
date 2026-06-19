#!/usr/bin/env python3
"""JAX CPU head-to-head for the FrankenJAX transpose trailing-block memcpy lever
(bead frankenjax-f62hx). Mirrors crates/fj-lax/benches/transpose_gauntlet.rs:
the attention transpose [B,S,H,D] -> [B,H,S,D] (perm (0,2,1,3)) on f32.

Run:
  benchmarks/jax_comparison/.venv/bin/python benchmarks/jax_comparison/transpose_gauntlet.py \
      --runs 50 --warmup 10 --inner-loops 100 --output /tmp/frankenjax_transpose_jax.json
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

B, S, H, D = 8, 512, 8, 64


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


def transpose_bshd(x):
    # Force materialization (XLA can fuse a bare transpose into a no-op view; the
    # Rust path always materializes, so add a cheap touch to keep it honest).
    return jnp.transpose(x, (0, 2, 1, 3)) + jnp.float32(0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--inner-loops", type=int, default=100)
    ap.add_argument("--output", type=str, default="")
    args = ap.parse_args()

    x = (jnp.arange(B * S * H * D, dtype=jnp.float32) * 1e-5 - 0.5).reshape(B, S, H, D)
    results = [bench("transpose_attn_BSHD_to_BHSD_f32", transpose_bshd, (x,),
                     args.runs, args.warmup, args.inner_loops)]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine": "jax_jit_cpu",
        "jax_version": jax.__version__,
        "platform": platform.platform(),
        "shape": [B, S, H, D],
        "elements": B * S * H * D,
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
