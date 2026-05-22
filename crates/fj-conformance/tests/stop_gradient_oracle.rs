//! StopGradient conformance tests.
//!
//! JAX treats `stop_gradient(x)` as an identity in primal evaluation and as a
//! gradient barrier for both reverse- and forward-mode AD.

use fj_core::{Atom, DType, Equation, Jaxpr, Literal, Primitive, Shape, TensorValue, Value, VarId};
use fj_lax::eval_primitive;
use smallvec::smallvec;
use std::collections::BTreeMap;

fn make_f64_vector(values: &[f64]) -> Value {
    Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape::vector(values.len() as u32),
            values.iter().copied().map(Literal::from_f64).collect(),
        )
        .unwrap(),
    )
}

fn extract_f64_vec(value: &Value) -> Vec<f64> {
    value
        .as_tensor()
        .expect("expected tensor")
        .elements
        .iter()
        .map(|literal| literal.as_f64().expect("expected f64 literal"))
        .collect()
}

fn stop_gradient_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(2)],
        vec![Equation {
            primitive: Primitive::StopGradient,
            inputs: smallvec![Atom::Var(VarId(1))],
            outputs: smallvec![VarId(2)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    )
}

#[test]
fn stop_gradient_preserves_scalar_value() {
    let input = Value::scalar_f64(-3.25);
    let output = eval_primitive(
        Primitive::StopGradient,
        std::slice::from_ref(&input),
        &BTreeMap::new(),
    )
    .expect("stop_gradient scalar eval should succeed");

    assert_eq!(output.as_f64_scalar(), Some(-3.25));
    assert_eq!(output.dtype(), DType::F64);
}

#[test]
fn stop_gradient_preserves_tensor_value_shape_and_dtype() {
    let input = make_f64_vector(&[-2.0, 0.5, 7.0]);
    let output = eval_primitive(
        Primitive::StopGradient,
        std::slice::from_ref(&input),
        &BTreeMap::new(),
    )
    .expect("stop_gradient tensor eval should succeed");

    assert_eq!(extract_f64_vec(&output), vec![-2.0, 0.5, 7.0]);
    let tensor = output.as_tensor().expect("expected tensor output");
    assert_eq!(tensor.dtype, DType::F64);
    assert_eq!(tensor.shape, Shape::vector(3));
}

#[test]
fn stop_gradient_vjp_returns_zero_cotangent_for_scalar() {
    let input = Value::scalar_f64(4.0);
    let output = Value::scalar_f64(4.0);
    let incoming_cotangent = Value::scalar_f64(9.0);

    let cotangents = fj_ad::vjp(
        Primitive::StopGradient,
        std::slice::from_ref(&input),
        std::slice::from_ref(&incoming_cotangent),
        std::slice::from_ref(&output),
        &BTreeMap::new(),
    )
    .expect("stop_gradient VJP should succeed");

    assert_eq!(cotangents.len(), 1);
    assert_eq!(cotangents[0].as_f64_scalar(), Some(0.0));
}

#[test]
fn stop_gradient_vjp_returns_zero_cotangent_for_tensor() {
    let input = make_f64_vector(&[1.0, -2.0, 3.5]);
    let output = input.clone();
    let incoming_cotangent = make_f64_vector(&[5.0, 6.0, 7.0]);

    let cotangents = fj_ad::vjp(
        Primitive::StopGradient,
        std::slice::from_ref(&input),
        std::slice::from_ref(&incoming_cotangent),
        std::slice::from_ref(&output),
        &BTreeMap::new(),
    )
    .expect("stop_gradient tensor VJP should succeed");

    assert_eq!(cotangents.len(), 1);
    assert_eq!(extract_f64_vec(&cotangents[0]), vec![0.0, 0.0, 0.0]);
    let tensor = cotangents[0]
        .as_tensor()
        .expect("expected tensor cotangent");
    assert_eq!(tensor.dtype, DType::F64);
    assert_eq!(tensor.shape, Shape::vector(3));
}

#[test]
fn stop_gradient_jvp_preserves_primal_and_blocks_tangent() {
    let jaxpr = stop_gradient_jaxpr();
    let primal = make_f64_vector(&[2.0, -1.0, 0.25]);
    let tangent = make_f64_vector(&[10.0, 20.0, 30.0]);

    let result = fj_ad::jvp(&jaxpr, &[primal], &[tangent]).expect("stop_gradient JVP should pass");

    assert_eq!(result.primals.len(), 1);
    assert_eq!(result.tangents.len(), 1);
    assert_eq!(extract_f64_vec(&result.primals[0]), vec![2.0, -1.0, 0.25]);
    assert_eq!(extract_f64_vec(&result.tangents[0]), vec![0.0, 0.0, 0.0]);
}
