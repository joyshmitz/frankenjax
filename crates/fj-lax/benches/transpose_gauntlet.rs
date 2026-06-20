//! Gauntlet bench for the transpose_general trailing-block memcpy lever (bead
//! frankenjax-f62hx): the attention transpose [B,S,H,D] -> [B,H,S,D] (perm
//! [0,2,1,3]) keeps the [D] feature vector contiguous, so the optimized path
//! memcpy-replicates a `D`-element block instead of decoding every output element.
//!
//! Arm A: `eval_primitive(Transpose)` (the committed block-copy path).
//! Arm B: `naive_transpose_f32` — the pre-f62hx per-element incremental odometer
//!        (push one src element per output position). Same values, no block memcpy.
//! The A/B ratio isolates the block-copy win; the JAX head-to-head lives in
//! benchmarks/jax_comparison/transpose_gauntlet.py.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use fj_core::{Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Duration;

// Realistic attention activation: batch=8, seq=512, heads=8, head_dim=64.
const B: usize = 8;
const S: usize = 512;
const H: usize = 8;
const D: usize = 64;

fn total() -> usize {
    B * S * H * D
}

fn input_f32() -> Value {
    let n = total();
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 1e-5 - 0.5).collect();
    Value::Tensor(
        TensorValue::new_f32_values(
            Shape {
                dims: vec![B as u32, S as u32, H as u32, D as u32],
            },
            data,
        )
        .unwrap(),
    )
}

// Pre-f62hx reference: incremental-odometer per-element transpose (no block memcpy).
fn naive_transpose_f32(src: &[f32], old_dims: &[usize], perm: &[usize]) -> Vec<f32> {
    let rank = old_dims.len();
    let new_dims: Vec<usize> = perm.iter().map(|&p| old_dims[p]).collect();
    let mut old_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        old_strides[i] = old_strides[i + 1] * old_dims[i + 1];
    }
    let step: Vec<usize> = perm.iter().map(|&p| old_strides[p]).collect();
    let total: usize = old_dims.iter().product();
    let mut out = Vec::with_capacity(total);
    let mut coord = vec![0usize; rank];
    let mut old_flat = 0usize;
    for _ in 0..total {
        out.push(src[old_flat]);
        let mut axis = rank;
        while axis > 0 {
            axis -= 1;
            coord[axis] += 1;
            old_flat += step[axis];
            if coord[axis] < new_dims[axis] {
                break;
            }
            coord[axis] = 0;
            old_flat -= step[axis] * new_dims[axis];
        }
    }
    out
}

fn bench_transpose(c: &mut Criterion) {
    let input = input_f32();
    let mut params = BTreeMap::new();
    params.insert("permutation".to_owned(), "0,2,1,3".to_owned());

    // Sanity: block-copy path == naive reference (bit-identical).
    let opt = eval_primitive(Primitive::Transpose, std::slice::from_ref(&input), &params).unwrap();
    let raw: Vec<f32> = match &input {
        Value::Tensor(t) => t.elements.as_f32_slice().unwrap().to_vec(),
        _ => unreachable!(),
    };
    let naive = naive_transpose_f32(&raw, &[B, S, H, D], &[0, 2, 1, 3]);
    if let Value::Tensor(ot) = &opt {
        let os = ot.elements.as_f32_slice().unwrap();
        assert_eq!(os.len(), naive.len());
        for i in [0usize, D - 1, naive.len() / 2, naive.len() - 1] {
            assert_eq!(os[i].to_bits(), naive[i].to_bits(), "block-copy != naive");
        }
    }

    let mut group = c.benchmark_group("transpose_attn_BSHD_to_BHSD_f32");
    group.throughput(Throughput::Elements(total() as u64));

    group.bench_function("blockcopy_eval_primitive", |bencher| {
        bencher.iter(|| {
            black_box(
                eval_primitive(
                    Primitive::Transpose,
                    std::slice::from_ref(black_box(&input)),
                    black_box(&params),
                )
                .unwrap(),
            )
        });
    });

    group.bench_function("naive_per_element_odometer", |bencher| {
        bencher.iter(|| {
            black_box(naive_transpose_f32(
                black_box(&raw),
                &[B, S, H, D],
                &[0, 2, 1, 3],
            ))
        });
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(3));
    targets = bench_transpose
}
criterion_main!(benches);
