#![forbid(unsafe_code)]

use fj_conformance::{HarnessConfig, read_transform_fixture_bundle, run_transform_fixture_bundle};
use fj_core::{
    CompatibilityMode, DType, ProgramSpec, Shape, TensorValue, TraceTransformLedger, Transform,
    Value, build_program,
};
use fj_dispatch::{DispatchError, DispatchRequest, TransformExecutionError, dispatch};
use serde::Serialize;
use serde_json::{Value as JsonValue, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Serialize)]
struct E2EForensicLogV1 {
    schema_version: &'static str,
    scenario_id: String,
    test_id: String,
    packet_id: String,
    fixture_id: String,
    seed: Option<u64>,
    mode: String,
    ts_utc_unix_ms: u128,
    env: JsonValue,
    artifact_refs: Vec<String>,
    replay_command: String,
    input_capture: JsonValue,
    intermediate_states: Vec<JsonValue>,
    output_capture: Option<JsonValue>,
    result: String,
    duration_ms: u128,
    details: Option<String>,
}

#[derive(Debug)]
struct ScenarioOutcome {
    output_capture: JsonValue,
    artifact_refs: Vec<String>,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn artifact_dir() -> PathBuf {
    if let Ok(path) = std::env::var("FJ_E2E_ARTIFACT_DIR") {
        return PathBuf::from(path);
    }
    repo_root().join("artifacts/e2e")
}

fn replay_command(scenario_id: &str) -> String {
    format!(
        "cargo test -p fj-conformance --test e2e -- {} --exact --nocapture",
        scenario_id
    )
}

fn fixture_id(input_capture: &JsonValue) -> String {
    fj_test_utils::fixture_id_from_json(input_capture)
        .unwrap_or_else(|_| "fixture-id-error".to_owned())
}

fn env_fingerprint() -> JsonValue {
    json!({
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "rust_toolchain": std::env::var("RUSTUP_TOOLCHAIN").unwrap_or_else(|_| "unknown".to_owned()),
    })
}

fn write_log(log: &E2EForensicLogV1) {
    let dir = artifact_dir();
    fs::create_dir_all(&dir).expect("e2e artifact dir should be creatable");
    let path = dir.join(format!("{}.e2e.json", log.scenario_id));
    let raw = serde_json::to_string_pretty(log).expect("log serialization should succeed");
    fs::write(&path, raw).expect("e2e forensic log write should succeed");
}

fn run_e2e_scenario(
    scenario_id: &str,
    packet_id: &str,
    mode: CompatibilityMode,
    input_capture: JsonValue,
    run: impl FnOnce(&mut Vec<JsonValue>) -> Result<ScenarioOutcome, String>,
) {
    let started = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after epoch")
        .as_millis();
    let start = Instant::now();
    let mut states = Vec::new();

    let result = run(&mut states);
    let duration_ms = start.elapsed().as_millis();

    let mut log = E2EForensicLogV1 {
        schema_version: "frankenjax.e2e.log.v1",
        scenario_id: scenario_id.to_owned(),
        test_id: format!("{}::{}", module_path!(), scenario_id),
        packet_id: packet_id.to_owned(),
        fixture_id: fixture_id(&input_capture),
        seed: None,
        mode: match mode {
            CompatibilityMode::Strict => "strict".to_owned(),
            CompatibilityMode::Hardened => "hardened".to_owned(),
        },
        ts_utc_unix_ms: started,
        env: env_fingerprint(),
        artifact_refs: Vec::new(),
        replay_command: replay_command(scenario_id),
        input_capture,
        intermediate_states: states,
        output_capture: None,
        result: "fail".to_owned(),
        duration_ms,
        details: None,
    };

    match result {
        Ok(outcome) => {
            log.result = "pass".to_owned();
            log.output_capture = Some(outcome.output_capture);
            log.artifact_refs = outcome.artifact_refs;
            write_log(&log);
        }
        Err(detail) => {
            log.details = Some(detail.clone());
            write_log(&log);
            panic!("{} failed: {}", scenario_id, detail);
        }
    }
}

fn ledger(program: ProgramSpec, transforms: &[Transform]) -> TraceTransformLedger {
    let mut ttl = TraceTransformLedger::new(build_program(program));
    for (idx, transform) in transforms.iter().enumerate() {
        ttl.push_transform(*transform, format!("e2e-{}-{}", idx, transform.as_str()));
    }
    ttl
}

#[test]
fn e2e_p2c001_full_dispatch_pipeline() {
    let cfg = HarnessConfig::default_paths();
    let fixture_path = cfg
        .fixture_root
        .join("transforms")
        .join("legacy_transform_cases.v1.json");

    run_e2e_scenario(
        "e2e_p2c001_full_dispatch_pipeline",
        "P2C-001",
        CompatibilityMode::Strict,
        json!({
            "fixture_path": fixture_path.display().to_string(),
            "strict_mode": true,
        }),
        |states| {
            states.push(json!({
                "step": "fixture_exists",
                "exists": fixture_path.exists(),
            }));

            let bundle = read_transform_fixture_bundle(&fixture_path)
                .map_err(|err| format!("failed to load fixture bundle: {err}"))?;
            states.push(json!({
                "step": "bundle_loaded",
                "case_count": bundle.cases.len(),
                "schema_version": bundle.schema_version,
            }));

            let report = run_transform_fixture_bundle(&cfg, &bundle);
            states.push(json!({
                "step": "parity_run_complete",
                "matched_cases": report.matched_cases,
                "mismatched_cases": report.mismatched_cases,
            }));

            if report.total_cases != bundle.cases.len() {
                return Err(format!(
                    "report case count mismatch: report={} bundle={}",
                    report.total_cases,
                    bundle.cases.len()
                ));
            }
            if report.mismatched_cases != 0 {
                return Err(format!(
                    "expected zero mismatches, got {}",
                    report.mismatched_cases
                ));
            }

            Ok(ScenarioOutcome {
                output_capture: json!({
                    "total_cases": report.total_cases,
                    "matched_cases": report.matched_cases,
                    "mismatched_cases": report.mismatched_cases,
                }),
                artifact_refs: vec![fixture_path.display().to_string()],
            })
        },
    );
}

#[test]
fn e2e_p2c001_transform_order_enforcement() {
    run_e2e_scenario(
        "e2e_p2c001_transform_order_enforcement",
        "P2C-001",
        CompatibilityMode::Strict,
        json!({
            "good_stack": ["vmap", "grad"],
            "bad_stack": ["grad", "vmap"],
            "input_vector": [1.0, 2.0, 3.0],
        }),
        |states| {
            let args = vec![Value::vector_f64(&[1.0, 2.0, 3.0]).map_err(|err| err.to_string())?];
            let good = dispatch(DispatchRequest {
                mode: CompatibilityMode::Strict,
                ledger: ledger(ProgramSpec::Square, &[Transform::Vmap, Transform::Grad]),
                args: args.clone(),
                backend: "cpu".to_owned(),
                compile_options: BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: Vec::new(),
            })
            .map_err(|err| format!("good transform order failed: {err}"))?;

            let good_values = good.outputs[0]
                .as_tensor()
                .and_then(TensorValue::to_f64_vec)
                .ok_or_else(|| "good path did not produce numeric tensor".to_owned())?;
            states.push(json!({
                "step": "good_stack_executed",
                "outputs": good_values,
            }));

            let bad = dispatch(DispatchRequest {
                mode: CompatibilityMode::Strict,
                ledger: ledger(ProgramSpec::Square, &[Transform::Grad, Transform::Vmap]),
                args,
                backend: "cpu".to_owned(),
                compile_options: BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: Vec::new(),
            });

            match bad {
                Err(DispatchError::TransformExecution(
                    TransformExecutionError::NonScalarGradientInput,
                )) => {
                    states.push(json!({
                        "step": "bad_stack_rejected",
                        "error": "NonScalarGradientInput",
                    }));
                }
                Err(other) => {
                    return Err(format!("unexpected bad-stack rejection class: {other}"));
                }
                Ok(response) => {
                    return Err(format!(
                        "bad transform order unexpectedly succeeded with outputs={:?}",
                        response.outputs
                    ));
                }
            }

            Ok(ScenarioOutcome {
                output_capture: json!({
                    "good_outputs": good_values,
                    "bad_stack_rejection": "NonScalarGradientInput",
                }),
                artifact_refs: vec![],
            })
        },
    );
}

#[test]
fn e2e_p2c001_vmap_rank2_tensor_support() {
    run_e2e_scenario(
        "e2e_p2c001_vmap_rank2_tensor_support",
        "P2C-001",
        CompatibilityMode::Strict,
        json!({
            "input_shape": [2, 2],
            "input_values": [1, 2, 3, 4],
            "transform_stack": ["vmap"],
        }),
        |states| {
            let matrix = Value::Tensor(
                TensorValue::new(
                    DType::I64,
                    Shape { dims: vec![2, 2] },
                    vec![
                        fj_core::Literal::I64(1),
                        fj_core::Literal::I64(2),
                        fj_core::Literal::I64(3),
                        fj_core::Literal::I64(4),
                    ],
                )
                .map_err(|err| err.to_string())?,
            );

            let response = dispatch(DispatchRequest {
                mode: CompatibilityMode::Strict,
                ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
                args: vec![matrix],
                backend: "cpu".to_owned(),
                compile_options: BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: Vec::new(),
            })
            .map_err(|err| format!("vmap rank2 dispatch failed: {err}"))?;

            let out_tensor = response.outputs[0]
                .as_tensor()
                .ok_or_else(|| "expected tensor output".to_owned())?;
            let out_values = out_tensor
                .elements
                .iter()
                .map(|lit| lit.as_i64().ok_or_else(|| "non-i64 element".to_owned()))
                .collect::<Result<Vec<_>, _>>()?;

            states.push(json!({
                "step": "rank2_vmap_output",
                "shape": out_tensor.shape.dims,
                "values": out_values,
            }));

            if out_tensor.shape.dims != vec![2, 2] {
                return Err(format!(
                    "unexpected output shape {:?}",
                    out_tensor.shape.dims
                ));
            }
            if out_values != vec![2, 3, 4, 5] {
                return Err(format!("unexpected output values {:?}", out_values));
            }

            Ok(ScenarioOutcome {
                output_capture: json!({
                    "shape": out_tensor.shape.dims,
                    "values": out_values,
                }),
                artifact_refs: vec![],
            })
        },
    );
}
