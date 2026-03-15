//! Durability coverage tests.
//!
//! Verifies that all long-lived conformance fixtures pass the full
//! RaptorQ durability pipeline: generate → scrub → decode proof.
//! This ensures that fixture bundles underpinning correctness guarantees
//! are protected against bit rot and storage corruption.

use fj_conformance::durability::{
    SidecarConfig, encode_artifact_to_sidecar, generate_decode_proof, scrub_sidecar,
};
use std::path::{Path, PathBuf};
use tempfile::tempdir;

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures")
}

/// Run the full durability pipeline on a single artifact file.
/// Returns Ok(()) if generate + scrub + decode-proof all succeed.
fn run_durability_pipeline(artifact_path: &Path) -> Result<(), String> {
    let tmp = tempdir().map_err(|e| format!("tempdir: {e}"))?;
    let stem = artifact_path
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .to_string();

    let sidecar = tmp.path().join(format!("{stem}.sidecar.json"));
    let scrub_report = tmp.path().join(format!("{stem}.scrub.json"));
    let proof = tmp.path().join(format!("{stem}.proof.json"));

    // Generate sidecar
    let manifest = encode_artifact_to_sidecar(artifact_path, &sidecar, &SidecarConfig::default())
        .map_err(|e| format!("generate failed for {}: {e}", artifact_path.display()))?;

    assert!(
        manifest.total_symbols > 0,
        "sidecar should have symbols for {}",
        artifact_path.display()
    );
    assert!(
        manifest.source_symbols > 0,
        "sidecar should have source symbols for {}",
        artifact_path.display()
    );
    assert!(
        manifest.repair_symbols > 0,
        "sidecar should have repair symbols for {}",
        artifact_path.display()
    );

    // Scrub: verify sidecar decodes to match artifact
    let scrub = scrub_sidecar(&sidecar, artifact_path, &scrub_report)
        .map_err(|e| format!("scrub failed for {}: {e}", artifact_path.display()))?;

    assert!(
        scrub.decoded_matches_expected,
        "scrub integrity check failed for {}",
        artifact_path.display()
    );

    // Decode proof: drop 2 source symbols, verify recovery
    let decode = generate_decode_proof(&sidecar, artifact_path, &proof, 2)
        .map_err(|e| format!("proof failed for {}: {e}", artifact_path.display()))?;

    assert!(
        decode.recovered,
        "decode proof recovery failed for {} (dropped {} symbols)",
        artifact_path.display(),
        decode.dropped_symbols.len()
    );

    Ok(())
}

fn collect_json_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(collect_json_files(&path));
            } else if path.extension().is_some_and(|ext| ext == "json") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

#[test]
fn all_conformance_fixtures_pass_durability_pipeline() {
    let root = fixture_root();
    let fixtures = collect_json_files(&root);

    assert!(
        !fixtures.is_empty(),
        "expected at least one fixture in {}",
        root.display()
    );

    let mut failures = Vec::new();
    for fixture in &fixtures {
        if let Err(e) = run_durability_pipeline(fixture) {
            failures.push(e);
        }
    }

    assert!(
        failures.is_empty(),
        "durability pipeline failures:\n{}",
        failures.join("\n")
    );
}

#[test]
fn smoke_fixture_durability_round_trip() {
    let smoke_path = fixture_root().join("smoke_case.json");
    if !smoke_path.exists() {
        eprintln!("SKIP: smoke_case.json not found");
        return;
    }
    run_durability_pipeline(&smoke_path).expect("smoke fixture durability pipeline should pass");
}

#[test]
fn rng_fixture_durability_round_trip() {
    let rng_path = fixture_root().join("rng/rng_determinism.v1.json");
    if !rng_path.exists() {
        eprintln!("SKIP: rng_determinism.v1.json not found");
        return;
    }
    run_durability_pipeline(&rng_path).expect("RNG fixture durability pipeline should pass");
}

#[test]
fn transform_fixture_durability_round_trip() {
    let transform_path = fixture_root().join("transforms/legacy_transform_cases.v1.json");
    if !transform_path.exists() {
        eprintln!("SKIP: legacy_transform_cases.v1.json not found");
        return;
    }
    run_durability_pipeline(&transform_path)
        .expect("transform fixture durability pipeline should pass");
}

/// Verify that the CI reliability budgets artifact passes durability.
#[test]
fn ci_reliability_budgets_durability() {
    let budgets_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../artifacts/ci/reliability_budgets.v1.json");
    if !budgets_path.exists() {
        eprintln!("SKIP: reliability_budgets.v1.json not found");
        return;
    }
    run_durability_pipeline(&budgets_path)
        .expect("reliability budgets durability pipeline should pass");
}

/// Verify that the conformance parity report artifact passes durability.
#[test]
fn conformance_parity_report_durability() {
    let parity_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../artifacts/conformance/v1_parity_report.json");
    if !parity_path.exists() {
        eprintln!("SKIP: v1_parity_report.json not found");
        return;
    }
    run_durability_pipeline(&parity_path).expect("parity report durability pipeline should pass");
}
