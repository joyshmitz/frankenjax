//! Oracle tests for the LU primitive.
//!
//! Upstream `jax.lax.linalg.lu` returns `(lu, pivots, permutation)`, where
//! `lu` stores L below the diagonal and U on and above it.

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive_multi;
use std::collections::BTreeMap;

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

fn f64_matrix(rows: u32, cols: u32, values: &[f64]) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape {
                dims: vec![rows, cols],
            },
            values.iter().copied().map(Literal::from_f64).collect(),
        )
        .expect("valid f64 matrix"),
    )
}

fn shape(value: &Value) -> Option<Shape> {
    value.as_tensor().map(|tensor| tensor.shape.clone())
}

fn f64_values(value: &Value) -> Option<Vec<f64>> {
    value
        .as_tensor()?
        .elements
        .iter()
        .map(|literal| literal.as_f64())
        .collect()
}

fn i64_values(value: &Value) -> Option<Vec<i64>> {
    value
        .as_tensor()?
        .elements
        .iter()
        .map(|literal| literal.as_i64())
        .collect()
}

fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (actual_value, expected_value) in actual.iter().zip(expected.iter()) {
        assert!(
            (actual_value - expected_value).abs() <= tolerance,
            "got {actual_value}, expected {expected_value} within {tolerance}",
        );
    }
}

#[test]
fn lu_identity_returns_identity_factors() {
    let input = f64_matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]);
    let result = eval_primitive_multi(Primitive::Lu, &[input], &no_params())
        .expect("identity LU should succeed");

    assert_eq!(result.len(), 3);
    assert_eq!(shape(&result[0]), Some(Shape { dims: vec![2, 2] }));
    assert_eq!(shape(&result[1]), Some(Shape { dims: vec![2] }));
    assert_eq!(shape(&result[2]), Some(Shape { dims: vec![2] }));

    let lu = f64_values(&result[0]).expect("expected f64 LU matrix");
    assert_close(&lu, &[1.0, 0.0, 0.0, 1.0], 1e-12);
    assert_eq!(i64_values(&result[1]), Some(vec![0, 1]));
    assert_eq!(i64_values(&result[2]), Some(vec![0, 1]));
}

#[test]
fn lu_pivots_rows_to_make_nonzero_leading_diagonal() {
    let input = f64_matrix(2, 2, &[0.0, 2.0, 1.0, 3.0]);
    let result = eval_primitive_multi(Primitive::Lu, &[input], &no_params())
        .expect("pivoted LU should succeed");

    assert_eq!(result.len(), 3);
    assert_eq!(shape(&result[0]), Some(Shape { dims: vec![2, 2] }));

    let lu = f64_values(&result[0]).expect("expected f64 LU matrix");
    assert_close(&lu, &[1.0, 3.0, 0.0, 2.0], 1e-12);
    assert_eq!(i64_values(&result[1]), Some(vec![1, 1]));
    assert_eq!(i64_values(&result[2]), Some(vec![1, 0]));
}

#[test]
fn lu_reconstructs_permuted_matrix_from_compact_factor() {
    let input = f64_matrix(2, 2, &[2.0, 1.0, 4.0, 3.0]);
    let result = eval_primitive_multi(Primitive::Lu, &[input], &no_params())
        .expect("pivoted LU should succeed");

    let lu = f64_values(&result[0]).expect("expected f64 LU matrix");
    assert_close(&lu, &[4.0, 3.0, 0.5, -0.5], 1e-12);
    assert_eq!(i64_values(&result[1]), Some(vec![1, 1]));
    assert_eq!(i64_values(&result[2]), Some(vec![1, 0]));

    let l = [1.0, 0.0, lu[2], 1.0];
    let u = [lu[0], lu[1], 0.0, lu[3]];
    let reconstructed = [
        l[0] * u[0] + l[1] * u[2],
        l[0] * u[1] + l[1] * u[3],
        l[2] * u[0] + l[3] * u[2],
        l[2] * u[1] + l[3] * u[3],
    ];

    assert_close(&reconstructed, &[4.0, 3.0, 2.0, 1.0], 1e-12);
}

#[test]
fn lu_rectangular_matrix_preserves_input_shape_and_min_pivots() {
    let input = f64_matrix(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result = eval_primitive_multi(Primitive::Lu, &[input], &no_params())
        .expect("rectangular LU should succeed");

    assert_eq!(result.len(), 3);
    assert_eq!(shape(&result[0]), Some(Shape { dims: vec![2, 3] }));
    assert_eq!(shape(&result[1]), Some(Shape { dims: vec![2] }));
    assert_eq!(shape(&result[2]), Some(Shape { dims: vec![2] }));
    assert_eq!(i64_values(&result[1]), Some(vec![1, 1]));
    assert_eq!(i64_values(&result[2]), Some(vec![1, 0]));
}

#[test]
fn lu_rejects_non_matrix_input() {
    let err = eval_primitive_multi(Primitive::Lu, &[Value::scalar_f64(1.0)], &no_params())
        .expect_err("scalar LU input should fail");

    assert!(
        err.to_string().contains("rank-2")
            || err.to_string().contains("matrix")
            || err.to_string().contains("tensor"),
        "unexpected LU scalar-input error: {err}"
    );
}
