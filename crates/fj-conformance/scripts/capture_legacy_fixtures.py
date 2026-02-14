#!/usr/bin/env python3
"""Capture transform fixtures from the legacy JAX oracle.

Usage:
  python crates/fj-conformance/scripts/capture_legacy_fixtures.py \
      --legacy-root /data/projects/frankenjax/legacy_jax_code/jax \
      --output /data/projects/frankenjax/crates/fj-conformance/fixtures/transforms/legacy_transform_cases.v1.json

Default behavior:
- try to run true legacy JAX capture
- if JAX import fails, fallback to deterministic analytical capture

Use `--strict` to disable fallback and fail hard on JAX import errors.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _import_jax(legacy_root: Path):
    sys.path.insert(0, str(legacy_root))
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore

    return jax, jnp


@dataclass
class Case:
    case_id: str
    family: str
    mode: str
    program: str
    transforms: list[str]
    args: list[dict[str, Any]]
    expected: list[dict[str, Any]]
    atol: float
    rtol: float
    comparator: str = "approx_atol_rtol"
    baseline_mismatch: bool = False
    flaky: bool = False
    simulated_delay_ms: int = 0


def fixture_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"kind": "scalar_i64", "value": int(value)}
    if isinstance(value, int):
        return {"kind": "scalar_i64", "value": value}
    if isinstance(value, float):
        return {"kind": "scalar_f64", "value": value}
    if isinstance(value, (list, tuple)):
        if all(isinstance(item, int) for item in value):
            return {"kind": "vector_i64", "values": [int(item) for item in value]}
        if all(isinstance(item, (int, float)) for item in value):
            return {"kind": "vector_f64", "values": [float(item) for item in value]}
        raise ValueError("list/tuple fixture values must be numeric")

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(value)
        if arr.ndim == 0:
            if np.issubdtype(arr.dtype, np.integer):
                return {"kind": "scalar_i64", "value": int(arr.item())}
            return {"kind": "scalar_f64", "value": float(arr.item())}
        if arr.ndim == 1:
            if np.issubdtype(arr.dtype, np.integer):
                return {"kind": "vector_i64", "values": [int(x) for x in arr.tolist()]}
            return {"kind": "vector_f64", "values": [float(x) for x in arr.tolist()]}
        raise ValueError(f"Unsupported array rank for fixture capture: {arr.ndim}")
    except ModuleNotFoundError as err:
        raise ValueError(
            "fixture_value received an unsupported object and numpy is unavailable"
        ) from err


def _scalar_samples() -> list[float]:
    return [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0]


def _int_add_pairs() -> list[tuple[int, int]]:
    return [(2, 5), (-3, 7), (0, 0), (11, -4), (42, -7), (8, 8), (15, 6), (-12, -9)]


def _dot_vectors() -> list[tuple[list[int], list[int]]]:
    return [
        ([1, 2, 3], [4, 5, 6]),
        ([2, 0, -1], [3, 7, 2]),
        ([9, 9, 9], [1, 0, -1]),
        ([-2, -3, -4], [5, 6, 7]),
    ]


def _reduce_vectors() -> list[list[int]]:
    return [[1, 2, 3], [10, -2, 4], [0, 0, 0], [-5, 9, 2]]


def _vmap_vectors_i64() -> list[list[int]]:
    return [[1, 2, 3], [3, 4, 5], [10, 20, 30]]


def _vmap_vectors_f64() -> list[list[float]]:
    return [[1.0, 2.0, 3.0], [-1.5, 0.0, 1.5], [0.25, 0.5, 0.75]]


def build_cases_with_oracle(jax, jnp) -> list[Case]:
    def add2(x, y):
        return x + y

    def square(x):
        return x * x

    def square_plus_linear(x):
        return x * x + 2.0 * x

    def add_one(x):
        return x + 1

    def sin_x(x):
        return jnp.sin(x)

    def cos_x(x):
        return jnp.cos(x)

    def dot3(x, y):
        return jnp.dot(x, y)

    def reduce_sum_vec(x):
        return jnp.sum(x)

    jit_add2 = jax.jit(add2)
    jit_square = jax.jit(square)
    jit_square_plus_linear = jax.jit(square_plus_linear)
    jit_sin = jax.jit(sin_x)
    jit_cos = jax.jit(cos_x)
    jit_dot3 = jax.jit(dot3)
    jit_reduce_sum = jax.jit(reduce_sum_vec)
    grad_square = jax.grad(square)
    grad_square_plus_linear = jax.grad(square_plus_linear)
    grad_sin = jax.grad(sin_x)
    grad_cos = jax.grad(cos_x)
    vmap_add_one = jax.vmap(add_one)
    vmap_grad_square = jax.vmap(grad_square)
    vmap_sin = jax.vmap(sin_x)
    jit_vmap_add_one = jax.jit(vmap_add_one)

    cases: list[Case] = []

    def add_case(
        case_id: str,
        family: str,
        program: str,
        transforms: list[str],
        args: list[Any],
        expected: list[Any],
        *,
        atol: float,
        rtol: float,
        comparator: str = "approx_atol_rtol",
    ) -> None:
        cases.append(
            Case(
                case_id=case_id,
                family=family,
                mode="strict",
                program=program,
                transforms=transforms,
                args=[fixture_value(arg) for arg in args],
                expected=[fixture_value(item) for item in expected],
                atol=atol,
                rtol=rtol,
                comparator=comparator,
            )
        )

    for idx, (lhs, rhs) in enumerate(_int_add_pairs()):
        add_case(
            f"jit_add2_i64_{idx}",
            "jit",
            "add2",
            ["jit"],
            [lhs, rhs],
            [jit_add2(lhs, rhs)],
            atol=0.0,
            rtol=0.0,
            comparator="exact",
        )

    for idx, x in enumerate(_scalar_samples()):
        add_case(
            f"jit_square_f64_{idx}",
            "jit",
            "square",
            ["jit"],
            [x],
            [jit_square(x)],
            atol=1e-6,
            rtol=1e-6,
        )
        add_case(
            f"jit_square_plus_linear_f64_{idx}",
            "jit",
            "square_plus_linear",
            ["jit"],
            [x],
            [jit_square_plus_linear(x)],
            atol=1e-6,
            rtol=1e-6,
        )
        add_case(
            f"jit_sin_x_f64_{idx}",
            "jit",
            "sin_x",
            ["jit"],
            [x],
            [jit_sin(x)],
            atol=1e-6,
            rtol=1e-6,
        )
        add_case(
            f"jit_cos_x_f64_{idx}",
            "jit",
            "cos_x",
            ["jit"],
            [x],
            [jit_cos(x)],
            atol=1e-6,
            rtol=1e-6,
        )

    for idx, (lhs, rhs) in enumerate(_dot_vectors()):
        lhs_arr = jnp.array(lhs)
        rhs_arr = jnp.array(rhs)
        add_case(
            f"jit_dot3_i64_{idx}",
            "jit",
            "dot3",
            ["jit"],
            [lhs_arr, rhs_arr],
            [jit_dot3(lhs_arr, rhs_arr)],
            atol=0.0,
            rtol=0.0,
            comparator="exact",
        )

    for idx, vec in enumerate(_reduce_vectors()):
        arr = jnp.array(vec)
        add_case(
            f"jit_reduce_sum_vec_i64_{idx}",
            "jit",
            "reduce_sum_vec",
            ["jit"],
            [arr],
            [jit_reduce_sum(arr)],
            atol=0.0,
            rtol=0.0,
            comparator="exact",
        )

    for idx, x in enumerate(_scalar_samples()[:4]):
        add_case(
            f"grad_square_f64_{idx}",
            "grad",
            "square",
            ["grad"],
            [x],
            [grad_square(x)],
            atol=1e-3,
            rtol=1e-3,
        )
        add_case(
            f"grad_square_plus_linear_f64_{idx}",
            "grad",
            "square_plus_linear",
            ["grad"],
            [x],
            [grad_square_plus_linear(x)],
            atol=1e-3,
            rtol=1e-3,
        )
        add_case(
            f"grad_sin_x_f64_{idx}",
            "grad",
            "sin_x",
            ["grad"],
            [x],
            [grad_sin(x)],
            atol=1e-3,
            rtol=1e-3,
        )
        add_case(
            f"grad_cos_x_f64_{idx}",
            "grad",
            "cos_x",
            ["grad"],
            [x],
            [grad_cos(x)],
            atol=1e-3,
            rtol=1e-3,
        )

    for idx, vec in enumerate(_vmap_vectors_i64()):
        arr = jnp.array(vec)
        add_case(
            f"vmap_add_one_i64_{idx}",
            "vmap",
            "add_one",
            ["vmap"],
            [arr],
            [vmap_add_one(arr)],
            atol=0.0,
            rtol=0.0,
            comparator="exact",
        )
        add_case(
            f"jit_vmap_add_one_i64_{idx}",
            "jit",
            "add_one",
            ["jit", "vmap"],
            [arr],
            [jit_vmap_add_one(arr)],
            atol=0.0,
            rtol=0.0,
            comparator="exact",
        )

    for idx, vec in enumerate(_vmap_vectors_f64()):
        arr = jnp.array(vec)
        add_case(
            f"vmap_grad_square_f64_{idx}",
            "vmap",
            "square",
            ["vmap", "grad"],
            [arr],
            [vmap_grad_square(arr)],
            atol=1e-3,
            rtol=1e-3,
        )
        add_case(
            f"vmap_sin_x_f64_{idx}",
            "vmap",
            "sin_x",
            ["vmap"],
            [arr],
            [vmap_sin(arr)],
            atol=1e-6,
            rtol=1e-6,
        )

    return cases


def build_cases_fallback() -> list[Case]:
    import math

    cases: list[Case] = []

    def add_case(
        case_id: str,
        family: str,
        program: str,
        transforms: list[str],
        args: list[Any],
        expected: list[Any],
        *,
        atol: float,
        rtol: float,
        comparator: str = "approx_atol_rtol",
    ) -> None:
        cases.append(
            Case(
                case_id=case_id,
                family=family,
                mode="strict",
                program=program,
                transforms=transforms,
                args=[fixture_value(arg) for arg in args],
                expected=[fixture_value(item) for item in expected],
                atol=atol,
                rtol=rtol,
                comparator=comparator,
            )
        )

    for idx, (lhs, rhs) in enumerate(_int_add_pairs()):
        add_case(
            f"jit_add2_i64_{idx}",
            "jit",
            "add2",
            ["jit"],
            [lhs, rhs],
            [lhs + rhs],
            atol=0.0,
            rtol=0.0,
            comparator="exact",
        )

    for idx, x in enumerate(_scalar_samples()):
        add_case(
            f"jit_square_f64_{idx}",
            "jit",
            "square",
            ["jit"],
            [x],
            [x * x],
            atol=1e-6,
            rtol=1e-6,
        )
        add_case(
            f"jit_square_plus_linear_f64_{idx}",
            "jit",
            "square_plus_linear",
            ["jit"],
            [x],
            [x * x + 2.0 * x],
            atol=1e-6,
            rtol=1e-6,
        )
        add_case(
            f"jit_sin_x_f64_{idx}",
            "jit",
            "sin_x",
            ["jit"],
            [x],
            [math.sin(x)],
            atol=1e-6,
            rtol=1e-6,
        )
        add_case(
            f"jit_cos_x_f64_{idx}",
            "jit",
            "cos_x",
            ["jit"],
            [x],
            [math.cos(x)],
            atol=1e-6,
            rtol=1e-6,
        )

    for idx, (lhs, rhs) in enumerate(_dot_vectors()):
        add_case(
            f"jit_dot3_i64_{idx}",
            "jit",
            "dot3",
            ["jit"],
            [lhs, rhs],
            [sum(a * b for a, b in zip(lhs, rhs))],
            atol=0.0,
            rtol=0.0,
            comparator="exact",
        )

    for idx, vec in enumerate(_reduce_vectors()):
        add_case(
            f"jit_reduce_sum_vec_i64_{idx}",
            "jit",
            "reduce_sum_vec",
            ["jit"],
            [vec],
            [sum(vec)],
            atol=0.0,
            rtol=0.0,
            comparator="exact",
        )

    for idx, x in enumerate(_scalar_samples()[:4]):
        add_case(
            f"grad_square_f64_{idx}",
            "grad",
            "square",
            ["grad"],
            [x],
            [2.0 * x],
            atol=1e-3,
            rtol=1e-3,
        )
        add_case(
            f"grad_square_plus_linear_f64_{idx}",
            "grad",
            "square_plus_linear",
            ["grad"],
            [x],
            [2.0 * x + 2.0],
            atol=1e-3,
            rtol=1e-3,
        )
        add_case(
            f"grad_sin_x_f64_{idx}",
            "grad",
            "sin_x",
            ["grad"],
            [x],
            [math.cos(x)],
            atol=1e-3,
            rtol=1e-3,
        )
        add_case(
            f"grad_cos_x_f64_{idx}",
            "grad",
            "cos_x",
            ["grad"],
            [x],
            [-math.sin(x)],
            atol=1e-3,
            rtol=1e-3,
        )

    for idx, vec in enumerate(_vmap_vectors_i64()):
        add_case(
            f"vmap_add_one_i64_{idx}",
            "vmap",
            "add_one",
            ["vmap"],
            [vec],
            [[value + 1 for value in vec]],
            atol=0.0,
            rtol=0.0,
            comparator="exact",
        )
        add_case(
            f"jit_vmap_add_one_i64_{idx}",
            "jit",
            "add_one",
            ["jit", "vmap"],
            [vec],
            [[value + 1 for value in vec]],
            atol=0.0,
            rtol=0.0,
            comparator="exact",
        )

    for idx, vec in enumerate(_vmap_vectors_f64()):
        add_case(
            f"vmap_grad_square_f64_{idx}",
            "vmap",
            "square",
            ["vmap", "grad"],
            [vec],
            [[2.0 * value for value in vec]],
            atol=1e-3,
            rtol=1e-3,
        )
        add_case(
            f"vmap_sin_x_f64_{idx}",
            "vmap",
            "sin_x",
            ["vmap"],
            [vec],
            [[math.sin(value) for value in vec]],
            atol=1e-6,
            rtol=1e-6,
        )

    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Fail if legacy JAX capture cannot run; do not fallback.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.skip_existing and args.output.exists():
        print(f"skip-existing set and output exists: {args.output}")
        return 0

    legacy_root = args.legacy_root
    if not legacy_root.exists():
        print(f"legacy root does not exist: {legacy_root}", file=sys.stderr)
        return 2

    capture_mode = "legacy_jax"
    try:
        jax, jnp = _import_jax(legacy_root)
        cases = build_cases_with_oracle(jax, jnp)
    except Exception as exc:
        if args.strict:
            print(
                "Failed to import/execute JAX from legacy root under --strict mode. "
                "Ensure jax + jaxlib are installed and compatible.",
                file=sys.stderr,
            )
            print(str(exc), file=sys.stderr)
            return 3

        capture_mode = "fallback_analytical"
        cases = build_cases_fallback()

    bundle = {
        "schema_version": "frankenjax.transform-fixtures.v1",
        "generated_by": "legacy_jax_capture_script",
        "generated_at_unix_ms": int(time.time() * 1000),
        "oracle_root": str(legacy_root),
        "capture_mode": capture_mode,
        "strict_capture": bool(args.strict),
        "cases": [
            {
                "case_id": case.case_id,
                "family": case.family,
                "mode": case.mode,
                "program": case.program,
                "transforms": case.transforms,
                "comparator": case.comparator,
                "baseline_mismatch": case.baseline_mismatch,
                "flaky": case.flaky,
                "simulated_delay_ms": case.simulated_delay_ms,
                "args": case.args,
                "expected": case.expected,
                "atol": case.atol,
                "rtol": case.rtol,
            }
            for case in cases
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases to {args.output} (capture_mode={capture_mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
