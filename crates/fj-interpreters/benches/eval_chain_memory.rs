//! Peak-memory evidence for liveness-based env-slot freeing in `eval_jaxpr`.
//!
//! Standalone bench binary (its own process) so it can install a peak-tracking
//! global allocator. It evaluates a deep chain of large-tensor elementwise ops and
//! reports the PEAK simultaneously-live heap during evaluation. With liveness-based
//! freeing, an N-equation chain holds only its working set (~a couple of tensors)
//! instead of all N intermediates — so peak ≈ a small multiple of one tensor, NOT
//! N × tensor. Run: `cargo bench -p fj-interpreters --bench eval_chain_memory`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use fj_core::{Atom, Equation, Jaxpr, Primitive, Shape, TensorValue, Value, VarId};
use fj_interpreters::eval_jaxpr;
use smallvec::smallvec;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static TRACK: AtomicUsize = AtomicUsize::new(0); // 0 = off, 1 = on

struct PeakAlloc;

unsafe impl GlobalAlloc for PeakAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() && TRACK.load(Ordering::Relaxed) == 1 {
            let now = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if TRACK.load(Ordering::Relaxed) == 1 {
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        }
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[global_allocator]
static A: PeakAlloc = PeakAlloc;

/// Deep chain: y0 = x + x; y1 = y0 + x; ...; y_{N-1} = y_{N-2} + x. Each step is a
/// same-shape f64 add producing a fresh [size] tensor; only the previous result and
/// `x` are ever live at once.
fn build_add_chain(n: usize, size: usize) -> (Jaxpr, Value) {
    let x = VarId(0);
    let mut equations = Vec::with_capacity(n);
    let mut cur = x;
    for next in (1u32..).take(n) {
        let out = VarId(next);
        equations.push(Equation {
            primitive: Primitive::Add,
            inputs: smallvec![Atom::Var(cur), Atom::Var(x)],
            outputs: smallvec![out],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        });
        cur = out;
    }
    let jaxpr = Jaxpr::new(vec![x], vec![], vec![cur], equations);
    let arg = Value::Tensor(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![size as u32],
            },
            (0..size).map(|i| i as f64 * 1e-3).collect(),
        )
        .unwrap(),
    );
    (jaxpr, arg)
}

fn main() {
    let n = 64usize;
    let size = 1usize << 20; // 1M f64 = 8 MB per tensor
    let tensor_bytes = size * 8;
    let (jaxpr, arg) = build_add_chain(n, size);

    // Warm + correctness sanity (every element = (n+1)*x, since y_k = (k+2)*x).
    let warm = eval_jaxpr(&jaxpr, std::slice::from_ref(&arg)).expect("eval");
    if let Value::Tensor(t) = &warm[0] {
        let v0 = t.elements[0].as_f64().unwrap();
        assert!((v0 - 0.0).abs() < 1e-9, "chain[0] sanity");
    }

    LIVE.store(0, Ordering::Relaxed);
    PEAK.store(0, Ordering::Relaxed);
    TRACK.store(1, Ordering::Relaxed);
    let out = eval_jaxpr(&jaxpr, std::slice::from_ref(&arg)).expect("eval");
    TRACK.store(0, Ordering::Relaxed);
    std::hint::black_box(&out);

    let peak = PEAK.load(Ordering::Relaxed);
    let all_held = n * tensor_bytes; // peak if every intermediate stayed alive
    println!(
        "EVAL_CHAIN_MEMORY n={n} tensor={:.1}MB peak={:.1}MB all_held_would_be={:.1}MB peak_in_tensors={:.1} reduction={:.1}x",
        tensor_bytes as f64 / 1e6,
        peak as f64 / 1e6,
        all_held as f64 / 1e6,
        peak as f64 / tensor_bytes as f64,
        all_held as f64 / peak.max(1) as f64,
    );
    // Liveness freeing must keep peak to a small working set, NOT the full chain.
    assert!(
        peak < 8 * tensor_bytes,
        "peak {peak} should be a small multiple of one {tensor_bytes}-byte tensor, not ~N×"
    );
}
