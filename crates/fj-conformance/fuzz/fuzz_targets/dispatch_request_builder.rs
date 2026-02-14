#![no_main]

mod common;

use common::{
    ByteCursor, sample_backend, sample_compile_options, sample_mode, sample_program,
    sample_transform, sample_unknown_features, sample_values,
};
use fj_core::{TraceTransformLedger, build_program};
use fj_dispatch::{DispatchRequest, dispatch};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut cursor = ByteCursor::new(data);

    let mut ledger = TraceTransformLedger::new(build_program(sample_program(&mut cursor)));
    let transform_count = cursor.take_usize(5);
    for idx in 0..transform_count {
        ledger.push_transform(sample_transform(&mut cursor), format!("fuzz-ev-{}", idx));
    }

    if cursor.take_bool() && !ledger.transform_evidence.is_empty() {
        ledger.transform_evidence[0] = String::new();
    }

    let request = DispatchRequest {
        mode: sample_mode(&mut cursor),
        ledger,
        args: sample_values(&mut cursor, 4),
        backend: sample_backend(&mut cursor),
        compile_options: sample_compile_options(&mut cursor, 6),
        custom_hook: if cursor.take_bool() {
            Some(cursor.take_string(16))
        } else {
            None
        },
        unknown_incompatible_features: sample_unknown_features(&mut cursor, 3),
    };

    let _ = dispatch(request);
});
