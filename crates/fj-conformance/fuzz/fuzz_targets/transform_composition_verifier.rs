#![no_main]

mod common;

use common::{ByteCursor, sample_evidence_id, sample_program, sample_transform};
use fj_core::{TraceTransformLedger, build_program, verify_transform_composition};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut cursor = ByteCursor::new(data);
    let root = build_program(sample_program(&mut cursor));
    let mut ledger = TraceTransformLedger::new(root);

    let transforms = cursor.take_usize(6);
    for idx in 0..transforms {
        let transform = sample_transform(&mut cursor);
        let evidence = sample_evidence_id(&mut cursor, idx);
        ledger.push_transform(transform, evidence);
    }

    if cursor.take_bool() && !ledger.transform_evidence.is_empty() {
        let _ = ledger.transform_evidence.pop();
    }

    let _ = ledger.composition_signature();
    let _ = verify_transform_composition(&ledger);
});
