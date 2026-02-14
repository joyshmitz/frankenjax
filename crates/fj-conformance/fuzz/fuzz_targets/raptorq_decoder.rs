#![no_main]

mod common;

use common::ByteCursor;
use fj_conformance::durability::{
    SidecarConfig, encode_artifact_to_sidecar, generate_decode_proof, read_sidecar_manifest,
    scrub_sidecar,
};
use libfuzzer_sys::fuzz_target;
use std::fs;

fuzz_target!(|data: &[u8]| {
    let mut cursor = ByteCursor::new(data);
    let Ok(tmp_dir) = tempfile::tempdir() else {
        return;
    };

    let artifact_path = tmp_dir.path().join("artifact.bin");
    let sidecar_path = tmp_dir.path().join("artifact.sidecar.json");
    let scrub_path = tmp_dir.path().join("artifact.scrub.json");
    let proof_path = tmp_dir.path().join("artifact.proof.json");

    let max_len = data.len().min(2_048);
    let artifact_len = cursor.take_usize(max_len);
    let artifact_len = artifact_len.min(data.len());
    if fs::write(&artifact_path, &data[..artifact_len]).is_err() {
        return;
    }

    let config = SidecarConfig {
        symbol_size: 64_u16 + (cursor.take_u16() % 448_u16),
        max_block_size: 512 + cursor.take_usize(2_048),
        repair_overhead: 1.0 + f64::from(cursor.take_u8() % 20) / 10.0,
    };

    if encode_artifact_to_sidecar(&artifact_path, &sidecar_path, &config).is_err() {
        return;
    }

    if cursor.take_bool() {
        let Ok(mut sidecar_raw) = fs::read(&sidecar_path) else {
            return;
        };
        if !sidecar_raw.is_empty() {
            let flip_index = cursor.take_usize(sidecar_raw.len() - 1);
            sidecar_raw[flip_index] ^= cursor.take_u8();
            let _ = fs::write(&sidecar_path, sidecar_raw);
        }
    }

    let _ = read_sidecar_manifest(&sidecar_path);
    let _ = scrub_sidecar(&sidecar_path, &artifact_path, &scrub_path);
    let _ = generate_decode_proof(
        &sidecar_path,
        &artifact_path,
        &proof_path,
        cursor.take_usize(5),
    );
});
