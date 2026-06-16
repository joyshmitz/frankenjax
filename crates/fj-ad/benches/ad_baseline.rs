use criterion::{Criterion, criterion_group, criterion_main};
use fj_ad::{grad_jaxpr, jacobian_jaxpr, jvp, value_and_grad_jaxpr};
use fj_core::{Atom, Equation, Jaxpr, Primitive, Shape, TensorValue, Value, VarId};
use fj_lax::eval_primitive;
use smallvec::smallvec;
use std::collections::BTreeMap;
use std::hint::black_box;

const JACFWD_DIAG_N: usize = 384;

fn build_square_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(0)],
        vec![],
        vec![VarId(1)],
        vec![Equation {
            primitive: Primitive::Mul,
            inputs: smallvec![Atom::Var(VarId(0)), Atom::Var(VarId(0))],
            outputs: smallvec![VarId(1)],
            params: BTreeMap::new(),
            effects: Vec::new(),
            sub_jaxprs: Vec::new(),
        }],
    )
}

fn build_poly_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(0)],
        vec![],
        vec![VarId(4)],
        vec![
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec![Atom::Var(VarId(0)), Atom::Var(VarId(0))],
                outputs: smallvec![VarId(1)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(0))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            },
            Equation {
                primitive: Primitive::Add,
                inputs: smallvec![Atom::Var(VarId(2)), Atom::Var(VarId(1))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            },
            Equation {
                primitive: Primitive::Add,
                inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(0))],
                outputs: smallvec![VarId(4)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            },
        ],
    )
}

fn build_trig_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(0)],
        vec![],
        vec![VarId(3)],
        vec![
            Equation {
                primitive: Primitive::Sin,
                inputs: smallvec![Atom::Var(VarId(0))],
                outputs: smallvec![VarId(1)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            },
            Equation {
                primitive: Primitive::Cos,
                inputs: smallvec![Atom::Var(VarId(0))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            },
        ],
    )
}

fn build_exp_log_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(0)],
        vec![],
        vec![VarId(2)],
        vec![
            Equation {
                primitive: Primitive::Exp,
                inputs: smallvec![Atom::Var(VarId(0))],
                outputs: smallvec![VarId(1)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            },
            Equation {
                primitive: Primitive::Log,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            },
        ],
    )
}

fn build_vector_reduce_jaxpr() -> Jaxpr {
    // f(x) = sum(x*x + x): the intermediates v1 (x*x) and v2 (x*x + x) are
    // full-length tensors, so the forward pass stores tensor-valued
    // output_values on the tape (exercises the per-equation clone).
    Jaxpr::new(
        vec![VarId(0)],
        vec![],
        vec![VarId(3)],
        vec![
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec![Atom::Var(VarId(0)), Atom::Var(VarId(0))],
                outputs: smallvec![VarId(1)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            },
            Equation {
                primitive: Primitive::Add,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(0))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            },
            Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec![Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            },
        ],
    )
}

fn build_chain_cumsum_jaxpr(chain: &[Primitive]) -> Jaxpr {
    let mut equations = Vec::new();
    let mut cur = VarId(0);
    let mut next = 1u32;
    for &primitive in chain {
        let out = VarId(next);
        next += 1;
        equations.push(Equation {
            primitive,
            inputs: smallvec![Atom::Var(cur)],
            outputs: smallvec![out],
            params: BTreeMap::new(),
            effects: Vec::new(),
            sub_jaxprs: Vec::new(),
        });
        cur = out;
    }
    let out = VarId(next);
    equations.push(Equation {
        primitive: Primitive::Cumsum,
        inputs: smallvec![Atom::Var(cur)],
        outputs: smallvec![out],
        params: axis_params(0),
        effects: Vec::new(),
        sub_jaxprs: Vec::new(),
    });
    Jaxpr::new(vec![VarId(0)], vec![], vec![out], equations)
}

fn no_params() -> BTreeMap<String, String> {
    BTreeMap::new()
}

fn axis_params(axis: usize) -> BTreeMap<String, String> {
    let mut params = BTreeMap::new();
    params.insert("axis".to_owned(), axis.to_string());
    params
}

fn dense_f64_tensor(dims: Vec<u32>, values: Vec<f64>) -> Value {
    Value::Tensor(TensorValue::new_f64_values(Shape { dims }, values).unwrap())
}

fn jacfwd_diag_input(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i as f64) * 0.007).sin() * 0.5).collect()
}

fn jacfwd_diag_coeff(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.75 + ((i as f64) * 0.013).cos() * 0.125)
        .collect()
}

fn jacfwd_diag_column_tangents(n: usize) -> Vec<Value> {
    (0..n)
        .map(|basis| {
            let mut values = vec![0.0_f64; n];
            values[basis] = 1.0;
            dense_f64_tensor(vec![n as u32], values)
        })
        .collect()
}

fn jacfwd_diag_batched_tangents(n: usize) -> Value {
    let mut values = vec![0.0_f64; n * n];
    for row in 0..n {
        values[row * n + row] = 1.0;
    }
    dense_f64_tensor(vec![n as u32, n as u32], values)
}

fn bench_grad_square(c: &mut Criterion) {
    let jaxpr = build_square_jaxpr();
    let args = vec![Value::scalar_f64(3.0)];
    c.bench_function("ad/grad_square", |b| b.iter(|| grad_jaxpr(&jaxpr, &args)));
}

fn bench_grad_vector_1k(c: &mut Criterion) {
    let jaxpr = build_vector_reduce_jaxpr();
    let data: Vec<f64> = (0..1024).map(|i| i as f64 * 0.001).collect();
    let args = vec![Value::vector_f64(&data).unwrap()];
    c.bench_function("ad/grad_sum_x2_plus_x_1k", |b| {
        b.iter(|| grad_jaxpr(&jaxpr, &args))
    });
}

fn bench_grad_polynomial(c: &mut Criterion) {
    let jaxpr = build_poly_jaxpr();
    let args = vec![Value::scalar_f64(2.0)];
    c.bench_function("ad/grad_poly_x3+x2+x", |b| {
        b.iter(|| grad_jaxpr(&jaxpr, &args))
    });
}

fn bench_grad_trig(c: &mut Criterion) {
    let jaxpr = build_trig_jaxpr();
    let args = vec![Value::scalar_f64(1.0)];
    c.bench_function("ad/grad_sin_cos_mul", |b| {
        b.iter(|| grad_jaxpr(&jaxpr, &args))
    });
}

fn bench_grad_exp_log(c: &mut Criterion) {
    let jaxpr = build_exp_log_jaxpr();
    let args = vec![Value::scalar_f64(1.0)];
    c.bench_function("ad/grad_exp_log", |b| b.iter(|| grad_jaxpr(&jaxpr, &args)));
}

fn bench_value_and_grad_poly(c: &mut Criterion) {
    let jaxpr = build_poly_jaxpr();
    let args = vec![Value::scalar_f64(2.0)];
    c.bench_function("ad/value_and_grad_poly", |b| {
        b.iter(|| value_and_grad_jaxpr(&jaxpr, &args))
    });
}

fn bench_jvp_square(c: &mut Criterion) {
    let jaxpr = build_square_jaxpr();
    let primals = vec![Value::scalar_f64(3.0)];
    let tangents = vec![Value::scalar_f64(1.0)];
    c.bench_function("ad/jvp_square", |b| {
        b.iter(|| jvp(&jaxpr, &primals, &tangents))
    });
}

fn bench_jvp_polynomial(c: &mut Criterion) {
    let jaxpr = build_poly_jaxpr();
    let primals = vec![Value::scalar_f64(2.0)];
    let tangents = vec![Value::scalar_f64(1.0)];
    c.bench_function("ad/jvp_poly_x3+x2+x", |b| {
        b.iter(|| jvp(&jaxpr, &primals, &tangents))
    });
}

fn bench_jvp_trig(c: &mut Criterion) {
    let jaxpr = build_trig_jaxpr();
    let primals = vec![Value::scalar_f64(1.0)];
    let tangents = vec![Value::scalar_f64(1.0)];
    c.bench_function("ad/jvp_sin_cos_mul", |b| {
        b.iter(|| jvp(&jaxpr, &primals, &tangents))
    });
}

fn bench_jacfwd_isolated_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("ad_jacfwd_isolated_ops");
    let n = JACFWD_DIAG_N;
    let params = no_params();
    let axis0 = axis_params(0);
    let axis1 = axis_params(1);
    let one = Value::scalar_f64(1.0);

    let input_1d = dense_f64_tensor(vec![n as u32], jacfwd_diag_input(n));
    let coeff_1d = dense_f64_tensor(vec![n as u32], jacfwd_diag_coeff(n));
    let column_tangents = jacfwd_diag_column_tangents(n);
    let batched_tangents = jacfwd_diag_batched_tangents(n);

    group.bench_function("value_mul/per_column_384x384", |b| {
        b.iter(|| {
            for tangent in &column_tangents {
                black_box(
                    eval_primitive(
                        Primitive::Mul,
                        &[coeff_1d.clone(), tangent.clone()],
                        &params,
                    )
                    .expect("per-column mul"),
                );
            }
        });
    });

    group.bench_function("value_mul/batched_384x384", |b| {
        b.iter(|| {
            black_box(
                eval_primitive(
                    Primitive::Mul,
                    &[coeff_1d.clone(), batched_tangents.clone()],
                    &params,
                )
                .expect("batched mul"),
            );
        });
    });

    group.bench_function("log1p_jvp/per_column_384x384", |b| {
        b.iter(|| {
            for tangent in &column_tangents {
                let denom =
                    eval_primitive(Primitive::Add, &[one.clone(), input_1d.clone()], &params)
                        .expect("per-column log1p denominator");
                black_box(
                    eval_primitive(Primitive::Div, &[tangent.clone(), denom], &params)
                        .expect("per-column log1p jvp"),
                );
            }
        });
    });

    group.bench_function("log1p_jvp/batched_384x384", |b| {
        b.iter(|| {
            let denom = eval_primitive(Primitive::Add, &[one.clone(), input_1d.clone()], &params)
                .expect("batched log1p denominator");
            black_box(
                eval_primitive(Primitive::Div, &[batched_tangents.clone(), denom], &params)
                    .expect("batched log1p jvp"),
            );
        });
    });

    group.bench_function("cumsum/per_column_384x384", |b| {
        b.iter(|| {
            for tangent in &column_tangents {
                black_box(
                    eval_primitive(Primitive::Cumsum, std::slice::from_ref(tangent), &axis0)
                        .expect("per-column cumsum"),
                );
            }
        });
    });

    group.bench_function("cumsum/batched_axis1_384x384", |b| {
        b.iter(|| {
            black_box(
                eval_primitive(
                    Primitive::Cumsum,
                    std::slice::from_ref(&batched_tangents),
                    &axis1,
                )
                .expect("batched cumsum"),
            );
        });
    });

    group.finish();
}

fn bench_jacfwd_chain_cumsum(c: &mut Criterion) {
    let chain: &[Primitive] = &[
        Primitive::Sin,
        Primitive::Cos,
        Primitive::Tanh,
        Primitive::Exp,
        Primitive::Sinh,
        Primitive::Erf,
        Primitive::Logistic,
        Primitive::Cosh,
    ];
    let jaxpr = build_chain_cumsum_jaxpr(chain);
    let input = dense_f64_tensor(vec![JACFWD_DIAG_N as u32], jacfwd_diag_input(JACFWD_DIAG_N));
    let args = vec![input];

    let mut group = c.benchmark_group("ad_jacfwd_chain_cumsum");
    group.bench_function("jacobian_384", |b| {
        b.iter(|| black_box(jacobian_jaxpr(&jaxpr, &args).expect("jacobian")))
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_grad_square,
    bench_grad_vector_1k,
    bench_grad_polynomial,
    bench_grad_trig,
    bench_grad_exp_log,
    bench_value_and_grad_poly,
    bench_jvp_square,
    bench_jvp_polynomial,
    bench_jvp_trig,
    bench_jacfwd_isolated_ops,
    bench_jacfwd_chain_cumsum,
);
criterion_main!(benches);
