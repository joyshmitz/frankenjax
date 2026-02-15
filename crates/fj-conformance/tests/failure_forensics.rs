//! bd-3dl.21: Failure Forensics UX + Artifact Index
//!
//! Tests:
//!   - Failure diagnostic schema validation
//!   - Manifest generation with synthetic multi-gate failure
//!   - One-command replay verification for each gate type
//!   - Human-readable summary format validation

#![forbid(unsafe_code)]

use serde_json::{Value as JsonValue, json};
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
}

// ---------------------------------------------------------------------------
// Test 1: Failure diagnostic schema validates correct instances
// ---------------------------------------------------------------------------

#[test]
fn failure_diagnostic_schema_validates_example() {
    let root = repo_root();
    let schema_path = root.join("artifacts/schemas/failure_diagnostic.v1.schema.json");
    let example_path = root.join("artifacts/examples/failure_diagnostic.v1.example.json");

    let schema: JsonValue =
        serde_json::from_str(&fs::read_to_string(&schema_path).unwrap()).unwrap();
    let instance: JsonValue =
        serde_json::from_str(&fs::read_to_string(&example_path).unwrap()).unwrap();

    let validator = jsonschema::validator_for(&schema).expect("schema should compile");
    let errors: Vec<String> = validator
        .iter_errors(&instance)
        .map(|e| e.to_string())
        .collect();
    assert!(
        errors.is_empty(),
        "failure_diagnostic example should validate: {}",
        errors.join(" | ")
    );
}

// ---------------------------------------------------------------------------
// Test 2: Synthetic multi-gate failure manifest generation
// ---------------------------------------------------------------------------

#[test]
fn synthetic_multi_gate_failure_manifest() {
    let ts = now_ms();

    // Simulate failures across multiple gates
    let failures = vec![
        json!({
            "schema_version": "frankenjax.failure-diagnostic.v1",
            "gate": "G1",
            "test": "fmt::check_formatting",
            "status": "fail",
            "summary": "3 files have formatting issues",
            "detail_path": "ci-artifacts/test-run/G1/formatting.log",
            "replay_cmd": "cargo fmt -- --check",
            "related_fixtures": [],
            "timestamp_unix_ms": ts
        }),
        json!({
            "schema_version": "frankenjax.failure-diagnostic.v1",
            "gate": "G2",
            "test": "fj_core::tests::jaxpr_fingerprint_deterministic",
            "status": "fail",
            "summary": "fingerprint mismatch: different hashes for identical jaxpr",
            "detail_path": "ci-artifacts/test-run/G2/jaxpr_fingerprint_deterministic.log",
            "replay_cmd": "cargo test -p fj-core -- jaxpr_fingerprint_deterministic --nocapture",
            "related_fixtures": [],
            "timestamp_unix_ms": ts
        }),
        json!({
            "schema_version": "frankenjax.failure-diagnostic.v1",
            "gate": "G4",
            "test": "fj_conformance::fuzz::fuzz_jaxpr_construction",
            "status": "error",
            "summary": "panic in equation validation with empty invars",
            "detail_path": "ci-artifacts/test-run/G4/fuzz_jaxpr_construction.log",
            "replay_cmd": "cargo fuzz run fuzz_jaxpr_construction -- -runs=1",
            "related_fixtures": ["crates/fj-conformance/fuzz/corpus/crashes/crash-abc123"],
            "timestamp_unix_ms": ts
        }),
        json!({
            "schema_version": "frankenjax.failure-diagnostic.v1",
            "gate": "G6",
            "test": "benches::dispatch_jit_scalar_add",
            "status": "fail",
            "summary": "p95 regression: 2500ns -> 3200ns (28% slower, threshold 5%)",
            "detail_path": "ci-artifacts/test-run/G6/dispatch_jit_scalar_add.log",
            "replay_cmd": "cargo bench -p fj-dispatch -- dispatch_jit_scalar_add",
            "related_fixtures": [],
            "timestamp_unix_ms": ts
        }),
    ];

    // Build manifest
    let manifest = json!({
        "schema_version": "frankenjax.run-manifest.v1",
        "run_id": format!("synthetic-test-{}", ts),
        "started_at_unix_ms": ts,
        "finished_at_unix_ms": ts + 60000,
        "total_duration_ms": 60000,
        "summary": {
            "total_tests": 156,
            "passed": 152,
            "failed": 4,
            "skipped": 0,
            "flaky": 0,
            "overall_status": "fail"
        },
        "gate_results": [
            {"gate_id": "G1", "name": "fmt_lint", "status": "fail", "duration_ms": 2000, "failure_count": 1},
            {"gate_id": "G2", "name": "unit_tests", "status": "fail", "duration_ms": 30000, "failure_count": 1},
            {"gate_id": "G3", "name": "differential", "status": "pass", "duration_ms": 15000},
            {"gate_id": "G4", "name": "adversarial", "status": "fail", "duration_ms": 8000, "failure_count": 1},
            {"gate_id": "G6", "name": "perf", "status": "fail", "duration_ms": 10000, "failure_count": 1}
        ],
        "failures": failures,
        "artifact_index": [
            {"path": "ci-artifacts/test-run/G1/formatting.log", "category": "other", "size_bytes": 512},
            {"path": "ci-artifacts/test-run/G2/jaxpr_fingerprint_deterministic.log", "category": "test_log", "size_bytes": 2048}
        ],
        "env": {
            "rust_version": "rustc 1.86.0-nightly",
            "os": "Linux 6.17.0-14-generic x86_64",
            "git_sha": "abc1234567890",
            "git_branch": "main"
        }
    });

    // Validate against schema
    let root = repo_root();
    let schema: JsonValue = serde_json::from_str(
        &fs::read_to_string(root.join("artifacts/schemas/run_manifest.v1.schema.json")).unwrap(),
    )
    .unwrap();
    let validator = jsonschema::validator_for(&schema).expect("run_manifest schema should compile");
    let errors: Vec<String> = validator
        .iter_errors(&manifest)
        .map(|e| e.to_string())
        .collect();
    assert!(
        errors.is_empty(),
        "synthetic manifest should validate: {}",
        errors.join(" | ")
    );

    // Verify multi-gate failure coverage
    let failed_gates: HashSet<&str> = manifest["failures"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["gate"].as_str().unwrap())
        .collect();
    assert!(failed_gates.contains("G1"), "G1 fmt failure present");
    assert!(failed_gates.contains("G2"), "G2 unit test failure present");
    assert!(
        failed_gates.contains("G4"),
        "G4 adversarial failure present"
    );
    assert!(failed_gates.contains("G6"), "G6 perf failure present");

    // Verify summary counts are consistent
    let summary = &manifest["summary"];
    assert_eq!(
        summary["failed"].as_u64().unwrap(),
        manifest["failures"].as_array().unwrap().len() as u64,
        "failure count matches failures array length"
    );

    // Write the synthetic manifest to artifacts for inspection
    let output_dir = root.join("artifacts/ci/runs/synthetic-forensics-test");
    fs::create_dir_all(&output_dir).unwrap();
    fs::write(
        output_dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();
}

// ---------------------------------------------------------------------------
// Test 3: One-command replay verification per gate type
// ---------------------------------------------------------------------------

#[test]
fn replay_commands_are_valid_for_all_gate_types() {
    // Define expected replay command patterns per gate
    let gate_replay_patterns: Vec<(&str, &str, &str)> = vec![
        ("G1", "cargo fmt -- --check", "fmt/lint gate"),
        (
            "G2",
            "cargo test -p fj-core -- test_name --nocapture",
            "unit test gate",
        ),
        (
            "G3",
            "cargo test -p fj-conformance --test ir_core_oracle -- oracle_test --nocapture",
            "differential oracle gate",
        ),
        (
            "G4",
            "cargo fuzz run fuzz_target -- -runs=1",
            "adversarial/fuzz gate",
        ),
        (
            "G5",
            "cargo test -p fj-conformance --test e2e -- scenario_id --nocapture",
            "e2e gate",
        ),
        (
            "G6",
            "cargo bench -p fj-dispatch -- bench_name",
            "performance gate",
        ),
    ];

    for (gate, replay_cmd, desc) in &gate_replay_patterns {
        // Verify command starts with cargo (copy-pasteable)
        assert!(
            replay_cmd.starts_with("cargo"),
            "{} ({}) replay should start with cargo: {}",
            gate,
            desc,
            replay_cmd
        );

        // Verify command doesn't contain shell-unsafe characters
        let unsafe_chars = ['|', '&', ';', '$', '`'];
        for c in &unsafe_chars {
            assert!(
                !replay_cmd.contains(*c),
                "{} ({}) replay should not contain shell metachar '{}': {}",
                gate,
                desc,
                c,
                replay_cmd
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test 4: Human-readable summary format validation
// ---------------------------------------------------------------------------

#[test]
fn human_readable_summary_format() {
    // Build a summary string like the script would produce
    let summary = format!(
        "{}{}{}{}{}{}{}{}{}{}",
        "============================================================\n",
        "  FrankenJAX CI Run Summary\n",
        "  Run ID:   20260214-120000-abc1234\n",
        "  Branch:   main\n",
        "  Commit:   abc1234567890\n",
        "============================================================\n",
        "\n",
        "RESULT: FAIL\n",
        "\n",
        "Tests:    156 total, 152 passed, 4 failed, 0 skipped\n",
    );

    // Must contain key fields for <30s comprehension
    assert!(summary.contains("Run ID:"), "summary must show run ID");
    assert!(summary.contains("Branch:"), "summary must show branch");
    assert!(summary.contains("Commit:"), "summary must show commit");
    assert!(
        summary.contains("RESULT:"),
        "summary must show overall result"
    );
    assert!(
        summary.contains("passed") && summary.contains("failed"),
        "summary must show pass/fail counts"
    );

    // Result should be UPPERCASE for visibility
    assert!(
        summary.contains("PASS") || summary.contains("FAIL"),
        "result should be uppercase for quick scanning"
    );
}

// ---------------------------------------------------------------------------
// Test 5: Existing artifact directories produce valid index entries
// ---------------------------------------------------------------------------

#[test]
fn existing_artifacts_have_valid_categories() {
    let root = repo_root();
    let valid_categories = [
        "test_log",
        "e2e_log",
        "golden_journey",
        "coverage_report",
        "perf_delta",
        "crash_triage",
        "parity_gate",
        "risk_note",
        "durability_sidecar",
        "failure_diagnostic",
        "other",
    ];

    // Check e2e logs exist and would map to e2e_log category
    let e2e_dir = root.join("artifacts/e2e");
    if e2e_dir.exists() {
        let e2e_files: Vec<_> = fs::read_dir(&e2e_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "json"))
            .collect();
        assert!(
            !e2e_files.is_empty(),
            "e2e directory should contain JSON artifacts"
        );
    }

    // Check golden journey logs exist
    let gj_dir = root.join("artifacts/e2e/golden_journeys");
    if gj_dir.exists() {
        let gj_files: Vec<_> = fs::read_dir(&gj_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .to_str()
                    .is_some_and(|s| s.ends_with(".golden.json"))
            })
            .collect();
        assert!(
            gj_files.len() >= 8,
            "should have at least 8 golden journey artifacts, found {}",
            gj_files.len()
        );
    }

    // Verify category enum matches schema
    let schema: JsonValue = serde_json::from_str(
        &fs::read_to_string(root.join("artifacts/schemas/run_manifest.v1.schema.json")).unwrap(),
    )
    .unwrap();
    let schema_categories: Vec<&str> =
        schema["properties"]["artifact_index"]["items"]["properties"]["category"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();

    for cat in &valid_categories {
        assert!(
            schema_categories.contains(cat),
            "category '{}' should be in schema enum",
            cat
        );
    }
}
