//! Gauntlet bench for the broadcast_replicate trailing-block memcpy lever (bead
//! frankenjax-thnjs): bias/feature broadcast [D] -> [rows, D] (broadcast_dimensions
//! [1]) replicates the contiguous [D] source block `rows` times via
//! extend_from_slice instead of decoding every output element.
//!
//! Arm A: eval_primitive(BroadcastInDim) (the committed block-copy path).
//! Arm B: naive_broadcast_f32 — the pre-thnjs per-element coordinate-decode replicate.
//! JAX head-to-head: benchmarks/jax_comparison/broadcast_gauntlet.py.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use fj_core::{Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Duration;

// Bias broadcast: D=768 model dim replicated over ROWS=4096 (batch*seq) tokens.
const ROWS: usize = 4096;
const D: usize = 768;

fn bias_input() -> Value {
    let data: Vec<f32> = (0..D).map(|i| (i as f32) * 1e-3 - 0.25).collect();
    Value::Tensor(
        TensorValue::new_f32_values(
            Shape {
                dims: vec![D as u32],
            },
            data,
        )
        .unwrap(),
    )
}

// Pre-thnjs reference: per-element row-major coordinate-decode replicate.
fn naive_broadcast_f32(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let total = rows * cols;
    let mut out = Vec::with_capacity(total);
    for i in 0..total {
        // out[i] = src[i % cols] (row broadcast of a [cols] vector over [rows, cols]).
        out.push(src[i % cols]);
    }
    out
}

fn bench_broadcast(c: &mut Criterion) {
    let input = bias_input();
    let mut params = BTreeMap::new();
    params.insert("shape".to_owned(), format!("{ROWS},{D}"));
    params.insert("broadcast_dimensions".to_owned(), "1".to_owned());

    let opt = eval_primitive(
        Primitive::BroadcastInDim,
        std::slice::from_ref(&input),
        &params,
    )
    .unwrap();
    let raw: Vec<f32> = match &input {
        Value::Tensor(t) => t.elements.as_f32_slice().unwrap().to_vec(),
        _ => unreachable!(),
    };
    let naive = naive_broadcast_f32(&raw, ROWS, D);
    if let Value::Tensor(ot) = &opt {
        let os = ot.elements.as_f32_slice().unwrap();
        assert_eq!(os.len(), naive.len());
        for i in [0usize, D - 1, naive.len() / 2, naive.len() - 1] {
            assert_eq!(os[i].to_bits(), naive[i].to_bits(), "block-copy != naive");
        }
    }

    let mut group = c.benchmark_group("broadcast_bias_D768_to_4096x768_f32");
    group.throughput(Throughput::Elements((ROWS * D) as u64));

    group.bench_function("blockcopy_eval_primitive", |bencher| {
        bencher.iter(|| {
            black_box(
                eval_primitive(
                    Primitive::BroadcastInDim,
                    std::slice::from_ref(black_box(&input)),
                    black_box(&params),
                )
                .unwrap(),
            )
        });
    });

    group.bench_function("naive_per_element_decode", |bencher| {
        bencher.iter(|| black_box(naive_broadcast_f32(black_box(&raw), ROWS, D)));
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(3));
    targets = bench_broadcast
}
criterion_main!(benches);
