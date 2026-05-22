//! Oracle tests for Zeta primitive.
//!
//! zeta(x) = Riemann zeta function ζ(x) = Σ_{n=1}^∞ 1/n^x
//!
//! Key properties:
//! - zeta(2) = π²/6 ≈ 1.6449
//! - zeta(4) = π⁴/90 ≈ 1.0823
//! - zeta(3) ≈ 1.2020569 (Apéry's constant)
//!
//! Tests:
//! - Known Riemann zeta values
//! - Tensor shapes

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

// ======================== Riemann Zeta ========================

#[test]
fn oracle_zeta_2() {
    // ζ(2) = π²/6
    let x = make_f64_tensor(&[], vec![2.0]);
    let result = eval_primitive(Primitive::Zeta, &[x], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    let expected = PI * PI / 6.0;
    assert!(
        (actual - expected).abs() < 1e-4,
        "zeta(2) = π²/6 ≈ {}, got {}",
        expected, actual
    );
}

#[test]
fn oracle_zeta_4() {
    // ζ(4) = π⁴/90
    let x = make_f64_tensor(&[], vec![4.0]);
    let result = eval_primitive(Primitive::Zeta, &[x], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    let expected = PI.powi(4) / 90.0;
    assert!(
        (actual - expected).abs() < 1e-4,
        "zeta(4) = π⁴/90 ≈ {}, got {}",
        expected, actual
    );
}

#[test]
fn oracle_zeta_3() {
    // ζ(3) ≈ 1.2020569... (Apéry's constant)
    let x = make_f64_tensor(&[], vec![3.0]);
    let result = eval_primitive(Primitive::Zeta, &[x], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    let expected = 1.2020569031595942;
    assert!(
        (actual - expected).abs() < 1e-4,
        "zeta(3) ≈ {}, got {}",
        expected, actual
    );
}

// ======================== Large s (rapid convergence) ========================

#[test]
fn oracle_zeta_large_s() {
    // For large s, ζ(s) ≈ 1 + 2^(-s) + ...
    let x = make_f64_tensor(&[], vec![10.0]);
    let result = eval_primitive(Primitive::Zeta, &[x], &no_params()).unwrap();
    let actual = extract_f64_scalar(&result);
    // ζ(10) = 1.000994575...
    assert!(
        (actual - 1.0).abs() < 0.01,
        "zeta(10) ≈ 1, got {}",
        actual
    );
}

// ======================== Tensor Shapes ========================

#[test]
fn oracle_zeta_vector() {
    let x = make_f64_tensor(&[3], vec![2.0, 3.0, 4.0]);
    let result = eval_primitive(Primitive::Zeta, &[x], &no_params()).unwrap();
    assert_eq!(extract_shape(&result), vec![3]);
    let vals = extract_f64_vec(&result);
    assert!((vals[0] - PI * PI / 6.0).abs() < 1e-4);
    assert!((vals[1] - 1.2020569031595942).abs() < 1e-4);
    assert!((vals[2] - PI.powi(4) / 90.0).abs() < 1e-4);
}
