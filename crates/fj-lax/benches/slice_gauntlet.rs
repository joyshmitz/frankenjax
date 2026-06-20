//! Gauntlet bench for the slice_strided_gather trailing-block memcpy lever (bead
//! frankenjax-idunl): a 2D crop [1024,1024] -> [512,512] copies each contiguous
//! 512-col row-block via extend_from_slice instead of per-element decode.
//!
//! Arm A: eval_primitive(Slice) (committed block-copy path).
//! Arm B: naive_slice_f32 — per-element strided gather reference (pre-idunl).
//! JAX head-to-head: benchmarks/jax_comparison/slice_gauntlet.py.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use fj_core::{Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Duration;

const ROWS: usize = 1024;
const COLS: usize = 1024;
const LO: usize = 256;
const HI: usize = 768; // crop [256:768, 256:768] -> [512,512]
const OUT: usize = HI - LO;

fn input_f32() -> Value {
    let data: Vec<f32> = (0..ROWS * COLS).map(|i| (i as f32) * 1e-6 - 0.5).collect();
    Value::Tensor(
        TensorValue::new_f32_values(
            Shape {
                dims: vec![ROWS as u32, COLS as u32],
            },
            data,
        )
        .unwrap(),
    )
}

fn naive_slice_f32(src: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(OUT * OUT);
    for r in LO..HI {
        for c in LO..HI {
            out.push(src[r * COLS + c]);
        }
    }
    out
}

fn bench_slice(c: &mut Criterion) {
    let input = input_f32();
    let mut params = BTreeMap::new();
    params.insert("start_indices".to_owned(), format!("{LO},{LO}"));
    params.insert("limit_indices".to_owned(), format!("{HI},{HI}"));

    let opt = eval_primitive(Primitive::Slice, std::slice::from_ref(&input), &params).unwrap();
    let raw: Vec<f32> = match &input {
        Value::Tensor(t) => t.elements.as_f32_slice().unwrap().to_vec(),
        _ => unreachable!(),
    };
    let naive = naive_slice_f32(&raw);
    if let Value::Tensor(ot) = &opt {
        let os = ot.elements.as_f32_slice().unwrap();
        assert_eq!(os.len(), naive.len());
        for i in [0usize, OUT - 1, naive.len() / 2, naive.len() - 1] {
            assert_eq!(os[i].to_bits(), naive[i].to_bits(), "block-copy != naive");
        }
    }

    let mut group = c.benchmark_group("slice_crop_1024x1024_to_512x512_f32");
    group.throughput(Throughput::Elements((OUT * OUT) as u64));
    group.bench_function("blockcopy_eval_primitive", |bencher| {
        bencher.iter(|| {
            black_box(
                eval_primitive(
                    Primitive::Slice,
                    std::slice::from_ref(black_box(&input)),
                    black_box(&params),
                )
                .unwrap(),
            )
        });
    });
    group.bench_function("naive_per_element", |bencher| {
        bencher.iter(|| black_box(naive_slice_f32(black_box(&raw))));
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(3));
    targets = bench_slice
}
criterion_main!(benches);
