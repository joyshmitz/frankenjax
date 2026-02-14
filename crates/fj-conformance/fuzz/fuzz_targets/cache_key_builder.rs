#![no_main]

mod common;

use common::{
    ByteCursor, sample_backend, sample_compile_options, sample_mode, sample_program,
    sample_transform, sample_unknown_features,
};
use fj_cache::{CacheKeyInput, build_cache_key, compatibility_matrix_row};
use fj_core::build_program;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut cursor = ByteCursor::new(data);

    let mut transform_stack = Vec::new();
    let transform_count = cursor.take_usize(5);
    for _ in 0..transform_count {
        transform_stack.push(sample_transform(&mut cursor));
    }

    let input = CacheKeyInput {
        mode: sample_mode(&mut cursor),
        backend: sample_backend(&mut cursor),
        jaxpr: build_program(sample_program(&mut cursor)),
        transform_stack,
        compile_options: sample_compile_options(&mut cursor, 5),
        custom_hook: if cursor.take_bool() {
            Some(cursor.take_string(16))
        } else {
            None
        },
        unknown_incompatible_features: sample_unknown_features(&mut cursor, 4),
    };

    let _ = compatibility_matrix_row(&input);
    let _ = build_cache_key(&input);
});
