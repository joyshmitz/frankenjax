//! Compiled-dispatch (CompiledJaxpr) vs eager `eval_jaxpr` on interpreter-bound workloads.
//!
//! Scalar `Add` chains are DISPATCH-bound — the per-equation kernel is trivial, so the
//! cost is the interpreter tax (slot env setup, per-equation `eval_primitive` dispatch,
//! and `BTreeMap<String,String>` param handling). This isolates exactly what the dense
//! compiled plan targets, so it quantifies the existing `compile_jaxpr_for_repeated_eval`
//! win over per-call `eval_jaxpr`, and BASELINES the tensor-param-prescan lever
//! (frankenjax-6dfew): re-run this bench before/after that change to measure it.
//!
//! Bit-exactness of compiled-vs-eager is guarded by the unit test
//! `compiled_jaxpr_eval_matches_eager_eval_jaxpr`; this file only measures speed.
use criterion::{Criterion, criterion_group, criterion_main};
use fj_core::{Atom, Equation, Jaxpr, Literal, Primitive, Value, VarId};
use fj_interpreters::{compile_jaxpr_for_repeated_eval, eval_jaxpr};
use std::collections::BTreeMap;
use std::hint::black_box;

/// `x -> x+1 -> x+2 -> ... ` : an n-equation scalar Add chain (dispatch-bound).
fn build_chain_jaxpr(n: usize) -> Jaxpr {
    let mut equations = Vec::with_capacity(n);
    for i in 0..n {
        equations.push(Equation {
            primitive: Primitive::Add,
            inputs: smallvec::smallvec![
                Atom::Var(VarId((i + 1) as u32)),
                Atom::Lit(Literal::I64(1))
            ],
            outputs: smallvec::smallvec![VarId((i + 2) as u32)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        });
    }
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId((n + 1) as u32)],
        equations,
    )
}

fn bench_compiled_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("compiled_dispatch");
    let args = [Value::scalar_i64(0)];
    for &n in &[8usize, 32, 128] {
        let jaxpr = build_chain_jaxpr(n);
        let compiled =
            compile_jaxpr_for_repeated_eval(&jaxpr).expect("scalar add chain should compile");
        group.bench_function(format!("eager/n={n}"), |b| {
            b.iter(|| eval_jaxpr(black_box(&jaxpr), black_box(&args)).unwrap())
        });
        group.bench_function(format!("compiled/n={n}"), |b| {
            b.iter(|| compiled.eval(black_box(&args)).unwrap())
        });
    }
    group.finish();
}

criterion_group!(benches, bench_compiled_dispatch);
criterion_main!(benches);
