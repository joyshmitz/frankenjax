//! Oracle tests for BesselI0e and BesselI1e primitives.
//!
//! BesselI0e(x) = I0(x) * exp(-|x|) (exponentially scaled modified Bessel I0)
//! BesselI1e(x) = I1(x) * exp(-|x|) (exponentially scaled modified Bessel I1)
//!
//! Key properties:
//! - I0e(0) = 1
//! - I0e is symmetric: I0e(-x) = I0e(x)
//! - I1e(0) = 0
//! - I1e is odd: I1e(-x) = -I1e(x)
//! - For large |x|, both approach 1/sqrt(2*pi*|x|)
//!
//! Tests:
//! - Zero values
//! - Symmetry properties
//! - Known values
//! - Special values: infinity, NaN
//! - Tensor shapes

// Golden reference constants are transcribed at full source precision on purpose.
#![allow(clippy::excessive_precision)]

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;
use std::f64::consts::PI;

fn make_f64_tensor(shape: &[u32], data: Vec<f64>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: shape.to_vec(),
            },
            data.into_iter().map(Literal::from_f64).collect(),
        )
        .unwrap(),
    )
}

fn extract_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
        _ => unreachable!("expected tensor"),
    }
}

fn extract_f64_scalar(v: &Value) -> f64 {
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.shape.dims.len(), 0, "expected scalar");
            t.elements[0].as_f64().unwrap()
        }
        Value::Scalar(l) => l.as_f64().unwrap(),
    }
}

fn extract_shape(v: &Value) -> Vec<u32> {
    match v {
        Value::Tensor(t) => t.shape.dims.clone(),
        _ => unreachable!("expected tensor"),
    }
}

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

// ======================== BesselI0e Cases ========================

#[test]
fn oracle_bessel_i0e_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::BesselI0e, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 1.0, "I0e(0) = 1");
}

#[test]
fn oracle_bessel_i0e_symmetry() {
    for x in [0.5, 1.0, 2.0, 5.0, 10.0] {
        let pos_input = make_f64_tensor(&[], vec![x]);
        let neg_input = make_f64_tensor(&[], vec![-x]);
        let pos_result = eval_primitive(Primitive::BesselI0e, &[pos_input], &no_params()).unwrap();
        let neg_result = eval_primitive(Primitive::BesselI0e, &[neg_input], &no_params()).unwrap();
        let pos_val = extract_f64_scalar(&pos_result);
        let neg_val = extract_f64_scalar(&neg_result);
        assert!(
            (pos_val - neg_val).abs() < 1e-14,
            "I0e({}) = I0e(-{}) = {} vs {}",
            x,
            x,
            pos_val,
            neg_val
        );
    }
}

#[test]
fn oracle_bessel_i0e_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::BesselI0e, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    // True I0e(1) = exp(-1)*I0(1) = 0.46575960759364043 (scipy.special.i0e,
    // matches JAX/XLA Cephes). The previous assertion locked in the old
    // Abramowitz & Stegun value 0.46575959666109185 (~1.1e-8 too small) at a
    // 1e-10 tolerance — a circular oracle that masked the inaccuracy.
    assert!(
        (actual - 0.46575960759364043).abs() < 1e-12,
        "I0e(1) = 0.46575960759364043, got {}",
        actual
    );
}

#[test]
fn oracle_bessel_i0e_large() {
    // For large x, I0e(x) ~ 1/sqrt(2*pi*x)
    let x = 100.0;
    let input = make_f64_tensor(&[], vec![x]);
    let result = eval_primitive(Primitive::BesselI0e, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    let asymptotic = 1.0 / (2.0 * PI * x).sqrt();
    assert!(
        (actual - asymptotic).abs() < 0.01,
        "I0e(100) ~ 1/sqrt(200*pi), got {} vs {}",
        actual,
        asymptotic
    );
}

#[test]
fn oracle_bessel_i0e_inf() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::BesselI0e, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert_eq!(actual, 0.0, "I0e(inf) = 0 (approaches 0 asymptotically)");
}

#[test]
fn oracle_bessel_i0e_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::BesselI0e, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "I0e(NaN) = NaN");
}

// ======================== BesselI1e Cases ========================

#[test]
fn oracle_bessel_i1e_zero() {
    let input = make_f64_tensor(&[], vec![0.0]);
    let result = eval_primitive(Primitive::BesselI1e, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "I1e(0) = 0");
}

#[test]
fn oracle_bessel_i1e_odd() {
    for x in [0.5, 1.0, 2.0, 5.0, 10.0] {
        let pos_input = make_f64_tensor(&[], vec![x]);
        let neg_input = make_f64_tensor(&[], vec![-x]);
        let pos_result = eval_primitive(Primitive::BesselI1e, &[pos_input], &no_params()).unwrap();
        let neg_result = eval_primitive(Primitive::BesselI1e, &[neg_input], &no_params()).unwrap();
        let pos_val = extract_f64_scalar(&pos_result);
        let neg_val = extract_f64_scalar(&neg_result);
        assert!(
            (pos_val + neg_val).abs() < 1e-14,
            "I1e({}) = -I1e(-{}) = {} vs {}",
            x,
            x,
            pos_val,
            neg_val
        );
    }
}

#[test]
fn oracle_bessel_i1e_one() {
    let input = make_f64_tensor(&[], vec![1.0]);
    let result = eval_primitive(Primitive::BesselI1e, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    // True I1e(1) = exp(-1)*I1(1) = 0.2079104153497085 (scipy.special.i1e,
    // matches JAX/XLA Cephes). The previous assertion locked in the old
    // Abramowitz & Stegun value 0.207910412991402 (~2.4e-9 too small) at a
    // 1e-10 tolerance — a circular oracle that masked the inaccuracy.
    assert!(
        (actual - 0.2079104153497085).abs() < 1e-12,
        "I1e(1) = 0.2079104153497085, got {}",
        actual
    );
}

#[test]
fn oracle_bessel_i1e_large() {
    // For large x, I1e(x) ~ 1/sqrt(2*pi*x)
    let x = 100.0;
    let input = make_f64_tensor(&[], vec![x]);
    let result = eval_primitive(Primitive::BesselI1e, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    let asymptotic = 1.0 / (2.0 * PI * x).sqrt();
    assert!(
        (actual - asymptotic).abs() < 0.01,
        "I1e(100) ~ 1/sqrt(200*pi), got {} vs {}",
        actual,
        asymptotic
    );
}

#[test]
fn oracle_bessel_i1e_inf() {
    let input = make_f64_tensor(&[], vec![f64::INFINITY]);
    let result = eval_primitive(Primitive::BesselI1e, &[input], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    assert_eq!(actual, 0.0, "I1e(inf) = 0 (approaches 0 asymptotically)");
}

#[test]
fn oracle_bessel_i1e_nan() {
    let input = make_f64_tensor(&[], vec![f64::NAN]);
    let result = eval_primitive(Primitive::BesselI1e, &[input], &no_params()).unwrap();
    assert!(extract_f64_scalar(&result).is_nan(), "I1e(NaN) = NaN");
}

// ============== Accuracy vs true values (JAX/XLA Cephes) ==============

#[test]
fn oracle_bessel_i0e_i1e_accuracy_table() {
    // Reference values from scipy.special.i0e/i1e (== JAX/XLA Cephes). The 1e-12
    // tolerance is far tighter than the old ~1e-7 Abramowitz & Stegun error, so
    // it is a genuine (non-circular) accuracy guard, not a snapshot of fj's own
    // approximation.
    let table: &[(f64, f64, f64)] = &[
        (0.5, 0.64503527044914999, 0.15642080318487173),
        (2.0, 0.308508322553671, 0.21526928924893771),
        (3.75, 0.21445705123004871, 0.18296842093089091),
        (5.0, 0.18354081260932834, 0.16397226694454234),
        (8.0, 0.14343178185685029, 0.13414249329269812),
        (10.0, 0.1278333371634286, 0.12126268138445551),
        (50.0, 0.056561626647454184, 0.055993123892895395),
        (100.0, 0.03994437929909668, 0.039744153025130249),
    ];
    for &(x, ref_i0e, ref_i1e) in table {
        let i0e = extract_f64_scalar(
            &eval_primitive(
                Primitive::BesselI0e,
                &[make_f64_tensor(&[], vec![x])],
                &no_params(),
            )
            .unwrap(),
        );
        let i1e = extract_f64_scalar(
            &eval_primitive(
                Primitive::BesselI1e,
                &[make_f64_tensor(&[], vec![x])],
                &no_params(),
            )
            .unwrap(),
        );
        assert!(
            (i0e - ref_i0e).abs() < 1e-12,
            "I0e({x}) = {ref_i0e}, got {i0e} (err {:.2e})",
            (i0e - ref_i0e).abs()
        );
        assert!(
            (i1e - ref_i1e).abs() < 1e-12,
            "I1e({x}) = {ref_i1e}, got {i1e} (err {:.2e})",
            (i1e - ref_i1e).abs()
        );
    }
}

// ======================== Tensor Shapes ========================

#[test]
fn oracle_bessel_i0e_vector() {
    let input = make_f64_tensor(&[3], vec![0.0, 1.0, -1.0]);
    let result = eval_primitive(Primitive::BesselI0e, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals[0], 1.0);
    assert!((vals[1] - vals[2]).abs() < 1e-15, "I0e symmetric");
}

#[test]
fn oracle_bessel_i1e_vector() {
    let input = make_f64_tensor(&[3], vec![0.0, 1.0, -1.0]);
    let result = eval_primitive(Primitive::BesselI1e, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals[0], 0.0);
    assert!((vals[1] + vals[2]).abs() < 1e-15, "I1e odd function");
}

#[test]
fn oracle_bessel_i0e_preserves_dtype() {
    let input = make_f64_tensor(&[3], vec![0.0, 1.0, 2.0]);
    let result = eval_primitive(Primitive::BesselI0e, &[input], &no_params()).unwrap();
    match &result {
        Value::Tensor(t) => assert_eq!(t.dtype, DType::F64),
        _ => panic!("expected tensor"),
    }
}

#[test]
fn oracle_bessel_i1e_preserves_dtype() {
    let input = make_f64_tensor(&[3], vec![0.0, 1.0, 2.0]);
    let result = eval_primitive(Primitive::BesselI1e, &[input], &no_params()).unwrap();
    match &result {
        Value::Tensor(t) => assert_eq!(t.dtype, DType::F64),
        _ => panic!("expected tensor"),
    }
}

#[test]
fn oracle_bessel_i0e_2d() {
    let input = make_f64_tensor(&[2, 2], vec![0.0, 1.0, -1.0, 2.0]);
    let result = eval_primitive(Primitive::BesselI0e, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals[0], 1.0); // I0e(0)
    assert!((vals[1] - vals[2]).abs() < 1e-15, "I0e symmetric");
}

#[test]
fn oracle_bessel_i0e_empty_tensor() {
    let input = make_f64_tensor(&[0], vec![]);
    let result = eval_primitive(Primitive::BesselI0e, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![0]);
    assert_eq!(extract_f64_vec(&result), vec![] as Vec<f64>);
}

// ======================== Additional Coverage ========================

#[test]
fn oracle_bessel_i0e_3d() {
    let input = make_f64_tensor(&[2, 2, 2], vec![0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 0.0]);
    let result = eval_primitive(Primitive::BesselI0e, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals[0], 1.0); // I0e(0)
    assert_eq!(vals[7], 1.0); // I0e(0)
}

#[test]
fn oracle_bessel_i1e_3d() {
    let input = make_f64_tensor(&[2, 2, 2], vec![0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 0.0]);
    let result = eval_primitive(Primitive::BesselI1e, &[input], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![2, 2, 2]);
    let vals = extract_f64_vec(&result);
    assert_eq!(vals[0], 0.0); // I1e(0)
    assert_eq!(vals[7], 0.0); // I1e(0)
}

#[test]
fn oracle_bessel_i0e_neg_inf() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::BesselI0e, &[input], &no_params()).unwrap();
    assert_eq!(extract_f64_scalar(&result), 0.0, "I0e(-inf) = 0");
}

#[test]
fn oracle_bessel_i1e_neg_inf() {
    let input = make_f64_tensor(&[], vec![f64::NEG_INFINITY]);
    let result = eval_primitive(Primitive::BesselI1e, &[input], &no_params()).unwrap();
    let val = extract_f64_scalar(&result);
    assert!(val == 0.0 || val.is_nan(), "I1e(-inf) = 0 or NaN");
}

// ======================== PROPERTY: dtype preservation ========================

#[test]
fn property_bessel_i0e_preserves_all_float_dtypes() {
    fn make_vec(dtype: DType, values: &[f64]) -> Value {
        let lits: Vec<Literal> = values
            .iter()
            .map(|&v| match dtype {
                DType::BF16 => Literal::from_bf16_f32(v as f32),
                DType::F16 => Literal::from_f16_f32(v as f32),
                DType::F32 => Literal::from_f32(v as f32),
                DType::F64 => Literal::from_f64(v),
                _ => panic!("not a float dtype"),
            })
            .collect();
        Value::Tensor(TensorValue::new(dtype, Shape { dims: vec![3] }, lits).unwrap())
    }

    let values = [0.0_f64, 1.0, 2.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let result = eval_primitive(Primitive::BesselI0e, &[input], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "bessel_i0e {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}

#[test]
fn property_bessel_i1e_preserves_all_float_dtypes() {
    fn make_vec(dtype: DType, values: &[f64]) -> Value {
        let lits: Vec<Literal> = values
            .iter()
            .map(|&v| match dtype {
                DType::BF16 => Literal::from_bf16_f32(v as f32),
                DType::F16 => Literal::from_f16_f32(v as f32),
                DType::F32 => Literal::from_f32(v as f32),
                DType::F64 => Literal::from_f64(v),
                _ => panic!("not a float dtype"),
            })
            .collect();
        Value::Tensor(TensorValue::new(dtype, Shape { dims: vec![3] }, lits).unwrap())
    }

    let values = [0.0_f64, 1.0, 2.0];
    for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
        let input = make_vec(dtype, &values);
        let result = eval_primitive(Primitive::BesselI1e, &[input], &no_params()).unwrap();
        let t = result.as_tensor().expect("tensor result");
        assert_eq!(t.dtype, dtype, "bessel_i1e {dtype:?}: dtype mismatch");
        t.validate_dtype_consistency()
            .expect("literal/dtype consistency");
    }
}

// ======================== Complex Type Tests ========================

fn make_complex64_tensor(shape: &[u32], data: Vec<(f32, f32)>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex64,
            Shape {
                dims: shape.to_vec(),
            },
            data.into_iter()
                .map(|(re, im)| Literal::from_complex64(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

fn make_complex128_tensor(shape: &[u32], data: Vec<(f64, f64)>) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape {
                dims: shape.to_vec(),
            },
            data.into_iter()
                .map(|(re, im)| Literal::from_complex128(re, im))
                .collect(),
        )
        .unwrap(),
    )
}

#[test]
fn oracle_bessel_i0e_complex64_real_axis() {
    // bessel_i0e on real axis should match real version
    let input = make_complex64_tensor(&[2], vec![(0.0, 0.0), (1.0, 0.0)]);
    let result = eval_primitive(Primitive::BesselI0e, &[input], &no_params())
        .expect("bessel_i0e complex64 should succeed");
    assert_eq!(result.dtype(), DType::Complex64);
}

#[test]
fn oracle_bessel_i1e_complex64_real_axis() {
    let input = make_complex64_tensor(&[2], vec![(0.0, 0.0), (1.0, 0.0)]);
    let result = eval_primitive(Primitive::BesselI1e, &[input], &no_params())
        .expect("bessel_i1e complex64 should succeed");
    assert_eq!(result.dtype(), DType::Complex64);
}

#[test]
fn oracle_bessel_i0e_complex128_preserves_dtype() {
    let input = make_complex128_tensor(&[2], vec![(0.0, 0.0), (1.0, 0.0)]);
    let result = eval_primitive(Primitive::BesselI0e, &[input], &no_params())
        .expect("bessel_i0e complex128 should succeed");
    assert_eq!(result.dtype(), DType::Complex128);
}

#[test]
fn property_bessel_preserves_complex_dtypes() {
    for dtype in [DType::Complex64, DType::Complex128] {
        let input = match dtype {
            DType::Complex64 => make_complex64_tensor(&[2], vec![(0.0, 0.0), (1.0, 0.0)]),
            DType::Complex128 => make_complex128_tensor(&[2], vec![(0.0, 0.0), (1.0, 0.0)]),
            _ => unreachable!(),
        };
        for primitive in [Primitive::BesselI0e, Primitive::BesselI1e] {
            let result = eval_primitive(primitive, std::slice::from_ref(&input), &no_params())
                .expect("bessel should succeed for complex dtype");
            assert_eq!(
                result.dtype(),
                dtype,
                "{primitive:?} {dtype:?}: dtype mismatch"
            );
        }
    }
}
