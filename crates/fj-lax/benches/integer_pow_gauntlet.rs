//! Gauntlet bench for the dense integer_pow lever (bead frankenjax-hfq7o): x**2
//! (the ubiquitous square in variance/MSE/poly). The dense path reads the typed
//! slice, maps v.powi(2)==v*v into dense storage; the boxed path materializes a
//! Vec<Literal> per element then densifies. Tests whether de-box on a SIMPLE
//! compute op (one mul) approaches JAX, unlike the heavier clamp cluster.
//!
//! Arm A: dense input -> dense integer_pow path (committed).
//! Arm B: boxed input -> per-Literal integer_pow_literal path (pre-hfq7o).
//! JAX head-to-head: benchmarks/jax_comparison/integer_pow_gauntlet.py.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use fj_core::{DType, Literal, LiteralBuffer, Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Duration;

const N: usize = 1_048_576;

fn vshape(len: usize) -> Shape {
    Shape::vector(u32::try_from(len).unwrap())
}

fn f64_dense(v: &[f64]) -> Value {
    Value::Tensor(TensorValue::new_f64_values(vshape(v.len()), v.to_vec()).unwrap())
}
fn f64_boxed(v: &[f64]) -> Value {
    Value::Tensor(
        TensorValue::new_with_literal_buffer(
            DType::F64,
            vshape(v.len()),
            LiteralBuffer::new(v.iter().map(|&x| Literal::from_f64(x)).collect()),
        )
        .unwrap(),
    )
}
fn f32_dense(v: &[f32]) -> Value {
    Value::Tensor(TensorValue::new_f32_values(vshape(v.len()), v.to_vec()).unwrap())
}
fn f32_boxed(v: &[f32]) -> Value {
    Value::Tensor(
        TensorValue::new_with_literal_buffer(
            DType::F32,
            vshape(v.len()),
            LiteralBuffer::new(v.iter().map(|&x| Literal::from_f32(x)).collect()),
        )
        .unwrap(),
    )
}

fn bench_one(label: &str, dense: Value, boxed: Value, c: &mut Criterion) {
    let mut params = BTreeMap::new();
    params.insert("exponent".to_owned(), "2".to_owned());

    // Sanity: dense path == boxed path (bit-identical values).
    let d = eval_primitive(Primitive::IntegerPow, std::slice::from_ref(&dense), &params).unwrap();
    let b = eval_primitive(Primitive::IntegerPow, std::slice::from_ref(&boxed), &params).unwrap();
    if let (Value::Tensor(dt), Value::Tensor(bt)) = (&d, &b) {
        for i in [0usize, N / 2, N - 1] {
            assert_eq!(dt.elements[i], bt.elements[i], "dense != boxed integer_pow");
        }
    }

    let mut group = c.benchmark_group(label);
    group.throughput(Throughput::Elements(N as u64));
    group.bench_function("dense", |bn| {
        bn.iter(|| {
            black_box(
                eval_primitive(
                    Primitive::IntegerPow,
                    std::slice::from_ref(black_box(&dense)),
                    black_box(&params),
                )
                .unwrap(),
            )
        });
    });
    group.bench_function("boxed", |bn| {
        bn.iter(|| {
            black_box(
                eval_primitive(
                    Primitive::IntegerPow,
                    std::slice::from_ref(black_box(&boxed)),
                    black_box(&params),
                )
                .unwrap(),
            )
        });
    });
    group.finish();
}

fn bench_integer_pow(c: &mut Criterion) {
    let xf64: Vec<f64> = (0..N).map(|i| (i as f64) * 1e-6 - 0.5).collect();
    let xf32: Vec<f32> = (0..N).map(|i| (i as f32) * 1e-6 - 0.5).collect();
    bench_one("integer_pow2_f64_1m", f64_dense(&xf64), f64_boxed(&xf64), c);
    bench_one("integer_pow2_f32_1m", f32_dense(&xf32), f32_boxed(&xf32), c);
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    targets = bench_integer_pow
}
criterion_main!(benches);
