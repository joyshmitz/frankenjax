//! Gauntlet bench for dense contiguous gather (embedding lookup): gather rows of
//! an [vocab, dim] f32 table at random token indices. The dense path memcpy's each
//! contiguous [dim] row via extend_from_slice; the naive reference copies per
//! element. Random-index row access (cache-miss-bound) — a different regime than
//! the contiguous block-copies (transpose/broadcast/slice).
//!
//! Arm A: eval_primitive(Gather) (committed dense contiguous-row path).
//! Arm B: naive_gather_f32 — per-element gather reference.
//! JAX head-to-head: benchmarks/jax_comparison/gather_gauntlet.py.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use fj_core::{Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Duration;

const VOCAB: usize = 16384;
const DIM: usize = 768;
const NIDX: usize = 4096;

fn idx_at(i: usize) -> i64 {
    // deterministic pseudo-random token index in [0, VOCAB)
    ((i.wrapping_mul(2654435761) ^ 0x9e3779b9) % VOCAB) as i64
}

fn table_f32() -> Value {
    let data: Vec<f32> = (0..VOCAB * DIM).map(|i| (i as f32) * 1e-7 - 0.5).collect();
    Value::Tensor(
        TensorValue::new_f32_values(
            Shape {
                dims: vec![VOCAB as u32, DIM as u32],
            },
            data,
        )
        .unwrap(),
    )
}

fn indices() -> Value {
    let data: Vec<i64> = (0..NIDX).map(idx_at).collect();
    Value::Tensor(
        TensorValue::new_i64_values(
            Shape {
                dims: vec![NIDX as u32],
            },
            data,
        )
        .unwrap(),
    )
}

fn naive_gather_f32(table: &[f32], idx: &[i64]) -> Vec<f32> {
    let mut out = Vec::with_capacity(NIDX * DIM);
    for &row in idx {
        let base = (row as usize) * DIM;
        for c in 0..DIM {
            out.push(table[base + c]);
        }
    }
    out
}

fn bench_gather(c: &mut Criterion) {
    let table = table_f32();
    let idx = indices();
    let mut params = BTreeMap::new();
    params.insert("slice_sizes".to_owned(), format!("1,{DIM}"));

    let inputs = [table.clone(), idx.clone()];
    let opt = eval_primitive(Primitive::Gather, &inputs, &params).unwrap();
    let raw: Vec<f32> = match &table {
        Value::Tensor(t) => t.elements.as_f32_slice().unwrap().to_vec(),
        _ => unreachable!(),
    };
    let idxv: Vec<i64> = match &idx {
        Value::Tensor(t) => t.elements.as_i64_slice().unwrap().to_vec(),
        _ => unreachable!(),
    };
    let naive = naive_gather_f32(&raw, &idxv);
    if let Value::Tensor(ot) = &opt {
        let os = ot.elements.as_f32_slice().unwrap();
        assert_eq!(os.len(), naive.len());
        for i in [0usize, DIM - 1, naive.len() / 2, naive.len() - 1] {
            assert_eq!(os[i].to_bits(), naive[i].to_bits(), "dense gather != naive");
        }
    }

    let mut group = c.benchmark_group("gather_embed_16384x768_take4096_f32");
    group.throughput(Throughput::Elements((NIDX * DIM) as u64));
    group.bench_function("dense_eval_primitive", |bencher| {
        bencher.iter(|| {
            black_box(
                eval_primitive(Primitive::Gather, black_box(&inputs), black_box(&params)).unwrap(),
            )
        });
    });
    group.bench_function("naive_per_element", |bencher| {
        bencher.iter(|| black_box(naive_gather_f32(black_box(&raw), black_box(&idxv))));
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(3));
    targets = bench_gather
}
criterion_main!(benches);
