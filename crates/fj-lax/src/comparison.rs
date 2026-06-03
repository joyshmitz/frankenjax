#![forbid(unsafe_code)]

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};

use crate::EvalError;
use crate::type_promotion::compare_literals;

fn broadcast_shape(lhs: &Shape, rhs: &Shape) -> Option<Shape> {
    let max_rank = lhs.rank().max(rhs.rank());
    let mut dims = Vec::with_capacity(max_rank);

    for offset in 0..max_rank {
        let lhs_dim = if offset < lhs.rank() {
            lhs.dims[lhs.rank() - 1 - offset]
        } else {
            1
        };
        let rhs_dim = if offset < rhs.rank() {
            rhs.dims[rhs.rank() - 1 - offset]
        } else {
            1
        };

        let out_dim = if lhs_dim == rhs_dim {
            lhs_dim
        } else if lhs_dim == 1 {
            rhs_dim
        } else if rhs_dim == 1 {
            lhs_dim
        } else {
            return None;
        };
        dims.push(out_dim);
    }

    dims.reverse();
    Some(Shape { dims })
}

fn compute_strides(dims: &[u32]) -> Vec<usize> {
    let mut strides = vec![1_usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1] as usize;
    }
    strides
}

fn flat_to_multi(flat: usize, strides: &[usize]) -> Vec<usize> {
    let mut multi = Vec::with_capacity(strides.len());
    let mut remainder = flat;
    for &stride in strides {
        multi.push(remainder / stride);
        remainder %= stride;
    }
    multi
}

fn broadcast_strides(shape: &Shape, out_shape: &Shape) -> Vec<usize> {
    let rank = shape.rank();
    let out_rank = out_shape.rank();
    let real_strides = compute_strides(&shape.dims);

    let mut result = vec![0_usize; out_rank];
    for (i, &stride) in real_strides.iter().enumerate().take(rank) {
        let out_axis = out_rank - rank + i;
        if shape.dims[i] == 1 {
            result[out_axis] = 0;
        } else {
            result[out_axis] = stride;
        }
    }
    result
}

fn broadcast_flat_index(multi: &[usize], strides: &[usize]) -> usize {
    multi.iter().zip(strides.iter()).map(|(&m, &s)| m * s).sum()
}

/// Comparison operators: return Bool scalars/tensors.
#[inline]
pub(crate) fn eval_comparison(
    primitive: Primitive,
    inputs: &[Value],
    int_cmp: impl Fn(i128, i128) -> bool,
    float_cmp: impl Fn(f64, f64) -> bool,
) -> Result<Value, EvalError> {
    if inputs.len() != 2 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2,
            actual: inputs.len(),
        });
    }

    match (&inputs[0], &inputs[1]) {
        (Value::Scalar(lhs), Value::Scalar(rhs)) => {
            let result = compare_literals(*lhs, *rhs, primitive, &int_cmp, &float_cmp)?;
            Ok(Value::scalar_bool(result))
        }
        (Value::Tensor(lhs), Value::Tensor(rhs)) => {
            if lhs.shape == rhs.shape {
                if lhs.dtype == DType::F64
                    && rhs.dtype == DType::F64
                    && let Some(value) = eval_same_shape_f64_compare(lhs, rhs, &float_cmp)?
                {
                    return Ok(value);
                }

                let mut elements = Vec::with_capacity(lhs.elements.len());
                for (lhs, rhs) in lhs
                    .elements
                    .iter()
                    .copied()
                    .zip(rhs.elements.iter().copied())
                {
                    elements.push(Literal::Bool(compare_literals(
                        lhs, rhs, primitive, &int_cmp, &float_cmp,
                    )?));
                }
                return Ok(Value::Tensor(TensorValue::new(
                    DType::Bool,
                    lhs.shape.clone(),
                    elements,
                )?));
            }

            let out_shape =
                broadcast_shape(&lhs.shape, &rhs.shape).ok_or(EvalError::ShapeMismatch {
                    primitive,
                    left: lhs.shape.clone(),
                    right: rhs.shape.clone(),
                })?;

            let out_count = out_shape.element_count().unwrap_or(0) as usize;
            let out_strides = compute_strides(&out_shape.dims);
            let lhs_strides = broadcast_strides(&lhs.shape, &out_shape);
            let rhs_strides = broadcast_strides(&rhs.shape, &out_shape);

            let mut elements = Vec::with_capacity(out_count);
            for flat_idx in 0..out_count {
                let multi = flat_to_multi(flat_idx, &out_strides);
                let lhs_idx = broadcast_flat_index(&multi, &lhs_strides);
                let rhs_idx = broadcast_flat_index(&multi, &rhs_strides);

                let l = lhs.elements[lhs_idx];
                let r = rhs.elements[rhs_idx];
                elements.push(Literal::Bool(compare_literals(
                    l, r, primitive, &int_cmp, &float_cmp,
                )?));
            }

            Ok(Value::Tensor(TensorValue::new(
                DType::Bool,
                out_shape,
                elements,
            )?))
        }
        (Value::Scalar(lhs), Value::Tensor(rhs)) => {
            if let Some(value) = eval_f64_scalar_compare(*lhs, rhs, true, &float_cmp)? {
                return Ok(value);
            }

            let mut elements = Vec::with_capacity(rhs.elements.len());
            for rhs in rhs.elements.iter().copied() {
                elements.push(Literal::Bool(compare_literals(
                    *lhs, rhs, primitive, &int_cmp, &float_cmp,
                )?));
            }
            Ok(Value::Tensor(TensorValue::new(
                DType::Bool,
                rhs.shape.clone(),
                elements,
            )?))
        }
        (Value::Tensor(lhs), Value::Scalar(rhs)) => {
            if let Some(value) = eval_f64_scalar_compare(*rhs, lhs, false, &float_cmp)? {
                return Ok(value);
            }

            let mut elements = Vec::with_capacity(lhs.elements.len());
            for lhs in lhs.elements.iter().copied() {
                elements.push(Literal::Bool(compare_literals(
                    lhs, *rhs, primitive, &int_cmp, &float_cmp,
                )?));
            }
            Ok(Value::Tensor(TensorValue::new(
                DType::Bool,
                lhs.shape.clone(),
                elements,
            )?))
        }
    }
}

/// Same-shape F64⊗F64 comparison fast path producing a `DType::Bool` tensor.
///
/// Bit-for-bit identical to the generic `compare_literals` path: for F64
/// operands that path falls through to `float_cmp(lhs.as_f64(), rhs.as_f64())`,
/// which is exactly what this applies here (same closure), skipping the
/// complex-operand check and the integral `literal_to_i128` probe per element.
/// Returns `Ok(None)` if any element is not `F64Bits` so the caller falls
/// through to the generic path.
#[inline]
fn eval_same_shape_f64_compare(
    lhs: &TensorValue,
    rhs: &TensorValue,
    float_cmp: &impl Fn(f64, f64) -> bool,
) -> Result<Option<Value>, EvalError> {
    let mut elements = Vec::with_capacity(lhs.elements.len());
    for (left, right) in lhs.elements.iter().zip(&rhs.elements) {
        let (Literal::F64Bits(left_bits), Literal::F64Bits(right_bits)) = (*left, *right) else {
            return Ok(None);
        };
        elements.push(Literal::Bool(float_cmp(
            f64::from_bits(left_bits),
            f64::from_bits(right_bits),
        )));
    }

    Ok(Some(Value::Tensor(TensorValue::new(
        DType::Bool,
        lhs.shape.clone(),
        elements,
    )?)))
}

/// F64 scalar/tensor broadcast comparison fast path producing a `DType::Bool`
/// tensor. `scalar_on_left` preserves operand order (`Scalar ⊗ Tensor` vs
/// `Tensor ⊗ Scalar`) for the asymmetric ordered comparisons. Bit-for-bit
/// identical to the generic `compare_literals` path for F64 operands. Returns
/// `Ok(None)` for non-F64 scalar/elements so the caller uses the generic path.
#[inline]
fn eval_f64_scalar_compare(
    scalar: Literal,
    tensor: &TensorValue,
    scalar_on_left: bool,
    float_cmp: &impl Fn(f64, f64) -> bool,
) -> Result<Option<Value>, EvalError> {
    let Literal::F64Bits(scalar_bits) = scalar else {
        return Ok(None);
    };
    if tensor.dtype != DType::F64 {
        return Ok(None);
    }
    let scalar = f64::from_bits(scalar_bits);
    let mut elements = Vec::with_capacity(tensor.elements.len());
    for &elem in &tensor.elements {
        let Literal::F64Bits(bits) = elem else {
            return Ok(None);
        };
        let value = f64::from_bits(bits);
        let out = if scalar_on_left {
            float_cmp(scalar, value)
        } else {
            float_cmp(value, scalar)
        };
        elements.push(Literal::Bool(out));
    }

    Ok(Some(Value::Tensor(TensorValue::new(
        DType::Bool,
        tensor.shape.clone(),
        elements,
    )?)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn f64_compare_fast_paths_bit_identical_to_scalar() {
        // Same-shape and both scalar-broadcast orders, all six comparisons,
        // with NaN / +-inf / signed zero, must match the per-element scalar
        // comparison exactly (Bool output).
        let lhs = [
            1.5,
            -0.0,
            f64::INFINITY,
            f64::NAN,
            7.0,
            -3.25,
            0.0,
            f64::NEG_INFINITY,
        ];
        let rhs = [2.0, 0.0, -4.0, 5.0, 7.0, -3.25, f64::NAN, f64::NEG_INFINITY];
        let scalar = 0.0_f64;
        let params = BTreeMap::new();
        for p in [
            Primitive::Eq,
            Primitive::Ne,
            Primitive::Lt,
            Primitive::Le,
            Primitive::Gt,
            Primitive::Ge,
        ] {
            let fcmp = |a: f64, b: f64| match p {
                Primitive::Eq => a == b,
                Primitive::Ne => a != b,
                Primitive::Lt => a < b,
                Primitive::Le => a <= b,
                Primitive::Gt => a > b,
                Primitive::Ge => a >= b,
                _ => unreachable!(),
            };
            let extract = |v: Value| -> Vec<bool> {
                let Value::Tensor(t) = v else {
                    panic!("expected tensor for {p:?}");
                };
                assert_eq!(t.dtype, DType::Bool);
                t.elements
                    .iter()
                    .map(|e| matches!(e, Literal::Bool(true)))
                    .collect()
            };

            // same-shape
            let got =
                extract(crate::eval_primitive(p, &[v_f64(&lhs), v_f64(&rhs)], &params).unwrap());
            let want: Vec<bool> = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(&a, &b)| fcmp(a, b))
                .collect();
            assert_eq!(got, want, "{p:?} same-shape mismatch");

            // scalar on left: Scalar ⊗ Tensor
            let got =
                extract(crate::eval_primitive(p, &[s_f64(scalar), v_f64(&rhs)], &params).unwrap());
            let want: Vec<bool> = rhs.iter().map(|&b| fcmp(scalar, b)).collect();
            assert_eq!(got, want, "{p:?} scalar-left mismatch");

            // scalar on right: Tensor ⊗ Scalar
            let got =
                extract(crate::eval_primitive(p, &[v_f64(&lhs), s_f64(scalar)], &params).unwrap());
            let want: Vec<bool> = lhs.iter().map(|&a| fcmp(a, scalar)).collect();
            assert_eq!(got, want, "{p:?} scalar-right mismatch");
        }
    }

    fn s_f64(v: f64) -> Value {
        Value::Scalar(Literal::from_f64(v))
    }
    fn s_i64(v: i64) -> Value {
        Value::Scalar(Literal::I64(v))
    }
    fn v_f64(data: &[f64]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::F64,
                fj_core::Shape {
                    dims: vec![data.len() as u32],
                },
                data.iter().map(|&v| Literal::from_f64(v)).collect(),
            )
            .unwrap(),
        )
    }
    fn extract_bool(val: &Value) -> bool {
        match val {
            Value::Scalar(Literal::Bool(b)) => *b,
            _ => std::panic::panic_any(format!("expected bool scalar, got {val:?}")),
        }
    }
    fn extract_bools(val: &Value) -> Vec<bool> {
        val.as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| match l {
                Literal::Bool(b) => *b,
                _ => std::panic::panic_any("expected bool element"),
            })
            .collect()
    }

    #[test]
    fn eq_scalar_true() {
        let result = eval_comparison(
            Primitive::Eq,
            &[s_f64(5.0), s_f64(5.0)],
            |a, b| a == b,
            |a, b| a == b,
        )
        .unwrap();
        assert!(extract_bool(&result));
    }

    #[test]
    fn eq_scalar_false() {
        let result = eval_comparison(
            Primitive::Eq,
            &[s_f64(5.0), s_f64(3.0)],
            |a, b| a == b,
            |a, b| a == b,
        )
        .unwrap();
        assert!(!extract_bool(&result));
    }

    #[test]
    fn ne_scalar() {
        let result = eval_comparison(
            Primitive::Ne,
            &[s_i64(1), s_i64(2)],
            |a, b| a != b,
            |a, b| a != b,
        )
        .unwrap();
        assert!(extract_bool(&result));
    }

    #[test]
    fn lt_scalar() {
        let result = eval_comparison(
            Primitive::Lt,
            &[s_f64(1.0), s_f64(2.0)],
            |a, b| a < b,
            |a, b| a < b,
        )
        .unwrap();
        assert!(extract_bool(&result));
        let result = eval_comparison(
            Primitive::Lt,
            &[s_f64(2.0), s_f64(1.0)],
            |a, b| a < b,
            |a, b| a < b,
        )
        .unwrap();
        assert!(!extract_bool(&result));
    }

    #[test]
    fn ge_scalar() {
        let result = eval_comparison(
            Primitive::Ge,
            &[s_f64(5.0), s_f64(5.0)],
            |a, b| a >= b,
            |a, b| a >= b,
        )
        .unwrap();
        assert!(extract_bool(&result));
        let result = eval_comparison(
            Primitive::Ge,
            &[s_f64(4.0), s_f64(5.0)],
            |a, b| a >= b,
            |a, b| a >= b,
        )
        .unwrap();
        assert!(!extract_bool(&result));
    }

    #[test]
    fn eq_tensor_elementwise() {
        let a = v_f64(&[1.0, 2.0, 3.0]);
        let b = v_f64(&[1.0, 9.0, 3.0]);
        let result = eval_comparison(Primitive::Eq, &[a, b], |a, b| a == b, |a, b| a == b).unwrap();
        assert_eq!(extract_bools(&result), vec![true, false, true]);
    }

    #[test]
    fn lt_tensor_elementwise() {
        let a = v_f64(&[1.0, 5.0, 3.0]);
        let b = v_f64(&[2.0, 4.0, 3.0]);
        let result = eval_comparison(Primitive::Lt, &[a, b], |a, b| a < b, |a, b| a < b).unwrap();
        assert_eq!(extract_bools(&result), vec![true, false, false]);
    }

    #[test]
    fn comparison_shape_mismatch() {
        let a = v_f64(&[1.0, 2.0]);
        let b = v_f64(&[1.0, 2.0, 3.0]);
        let result = eval_comparison(Primitive::Eq, &[a, b], |a, b| a == b, |a, b| a == b);
        assert!(result.is_err());
    }

    #[test]
    fn comparison_scalar_tensor_broadcast() {
        let a = s_f64(2.0);
        let b = v_f64(&[1.0, 2.0, 3.0]);
        let result = eval_comparison(Primitive::Eq, &[a, b], |a, b| a == b, |a, b| a == b).unwrap();
        assert_eq!(extract_bools(&result), vec![false, true, false]);
    }

    #[test]
    fn comparison_arity_error() {
        let result = eval_comparison(Primitive::Eq, &[s_f64(1.0)], |a, b| a == b, |a, b| a == b);
        assert!(result.is_err());
    }
}
