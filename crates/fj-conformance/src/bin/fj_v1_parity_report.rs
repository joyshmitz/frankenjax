#![forbid(unsafe_code)]

use fj_conformance::{
    FamilyReport, HarnessConfig, ParityReportSummary, ParityReportV1,
    read_transform_fixture_bundle, run_transform_fixture_bundle,
};
use serde::Serialize;
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug)]
struct Args {
    fixtures: PathBuf,
    output_json: PathBuf,
    output_markdown: PathBuf,
    ci_json: PathBuf,
    e2e_json: PathBuf,
    mode: String,
    fj_version: String,
    oracle_version: String,
}

#[derive(Debug, Clone, Serialize)]
struct PrimitiveBreakdown {
    total: usize,
    matched: usize,
    mismatched: usize,
    cases: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ParityException {
    case_id: String,
    family: String,
    primitive: String,
    drift: String,
    expected: String,
    actual: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct CoverageException {
    family: String,
    reason: String,
    justification: String,
}

#[derive(Debug, Clone, Serialize)]
struct ComprehensiveParityReport {
    version: String,
    timestamp: String,
    fj_version: String,
    jax_version: String,
    mode: String,
    summary: ParityReportSummary,
    families: BTreeMap<String, FamilyReport>,
    per_primitive: BTreeMap<String, PrimitiveBreakdown>,
    parity_exceptions: Vec<ParityException>,
    coverage_exceptions: Vec<CoverageException>,
    gate_status: String,
}

#[derive(Debug, Clone, Serialize)]
struct FamilyRollup {
    total: usize,
    matched: usize,
    mismatched: usize,
}

#[derive(Debug, Clone, Serialize)]
struct E2EFinalParityLog {
    total_cases: usize,
    matched: usize,
    mismatched: usize,
    pass_rate: f64,
    per_family: BTreeMap<String, FamilyRollup>,
    gate_status: String,
    pass: bool,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("..")
        })
}

fn default_paths(root: &Path) -> Args {
    Args {
        fixtures: root
            .join("crates/fj-conformance/fixtures/transforms/legacy_transform_cases.v1.json"),
        output_json: root.join("artifacts/conformance/v1_parity_report.json"),
        output_markdown: root.join("artifacts/conformance/v1_parity_report.md"),
        ci_json: root.join("artifacts/ci/runs/v1-parity-report/parity_report.v1.json"),
        e2e_json: root.join("artifacts/e2e/e2e_parity_report_final.e2e.json"),
        mode: "strict".to_owned(),
        fj_version: env!("CARGO_PKG_VERSION").to_owned(),
        oracle_version: "jax-0.9.0.1".to_owned(),
    }
}

fn usage() -> &'static str {
    "Usage:
  cargo run -p fj-conformance --bin fj_v1_parity_report -- [options]

Options:
  --fixtures <path>        Input transform fixture bundle JSON
  --output-json <path>     Comprehensive parity JSON output
  --output-md <path>       Comprehensive parity markdown output
  --ci-json <path>         Spec Section 8.3 parity-report JSON output
  --e2e-json <path>        E2E forensic parity log output
  --mode <strict|hardened> Report mode label (default: strict)
  --fj-version <string>    FrankenJAX version in report metadata
  --oracle-version <str>   Oracle/JAX version in report metadata
  --help                   Show this help"
}

fn next_value<I>(args: &mut I, flag: &str) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| format!("missing value for {flag}"))
}

fn parse_args(root: &Path) -> Result<Args, String> {
    let mut parsed = default_paths(root);
    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--fixtures" => parsed.fixtures = PathBuf::from(next_value(&mut it, "--fixtures")?),
            "--output-json" => {
                parsed.output_json = PathBuf::from(next_value(&mut it, "--output-json")?);
            }
            "--output-md" => {
                parsed.output_markdown = PathBuf::from(next_value(&mut it, "--output-md")?);
            }
            "--ci-json" => parsed.ci_json = PathBuf::from(next_value(&mut it, "--ci-json")?),
            "--e2e-json" => parsed.e2e_json = PathBuf::from(next_value(&mut it, "--e2e-json")?),
            "--mode" => parsed.mode = next_value(&mut it, "--mode")?,
            "--fj-version" => parsed.fj_version = next_value(&mut it, "--fj-version")?,
            "--oracle-version" => parsed.oracle_version = next_value(&mut it, "--oracle-version")?,
            "--help" | "-h" => return Err(usage().to_owned()),
            _ => return Err(format!("unknown flag: {arg}\n\n{}", usage())),
        }
    }
    Ok(parsed)
}

fn family_name(name: &str) -> String {
    name.to_owned()
}

fn ensure_parent(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create directory {}: {err}", parent.display()))?;
    }
    Ok(())
}

fn write_text(path: &Path, content: &str) -> Result<(), String> {
    ensure_parent(path)?;
    fs::write(path, content).map_err(|err| format!("failed to write {}: {err}", path.display()))
}

fn to_markdown(
    v1: &ParityReportV1,
    per_primitive: &BTreeMap<String, PrimitiveBreakdown>,
    parity_exceptions: &[ParityException],
    coverage_exceptions: &[CoverageException],
) -> String {
    let matched_total: usize = v1.families.values().map(|family| family.matched).sum();
    let mismatched_total: usize = v1.families.values().map(|family| family.mismatched).sum();

    let mut out = String::new();
    out.push_str("# V1 Parity Report\n\n");
    out.push_str(&format!("Mode: `{}`\n\n", v1.mode));
    out.push_str(&format!(
        "FrankenJAX: `{}` | Oracle: `{}`\n\n",
        v1.fj_version, v1.oracle_version
    ));
    out.push_str("## Summary\n\n");
    out.push_str("| Metric | Value |\n");
    out.push_str("|---|---|\n");
    out.push_str(&format!("| Total Cases | {} |\n", v1.summary.total));
    out.push_str(&format!("| Matched | {} |\n", matched_total));
    out.push_str(&format!("| Mismatched | {} |\n", mismatched_total));
    out.push_str(&format!(
        "| Pass Rate | {:.2}% |\n",
        v1.summary.pass_rate * 100.0
    ));
    out.push_str(&format!("| Gate | **{}** |\n\n", v1.summary.gate));

    out.push_str("## Per-Family Breakdown\n\n");
    out.push_str("| Family | Total | Matched | Mismatched |\n");
    out.push_str("|---|---|---|---|\n");
    for family in ["jit", "grad", "vmap", "lax", "random"] {
        let stats = v1.families.get(family).cloned().unwrap_or(FamilyReport {
            total: 0,
            matched: 0,
            mismatched: 0,
            cases: Vec::new(),
        });
        out.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            family, stats.total, stats.matched, stats.mismatched
        ));
    }
    out.push('\n');

    out.push_str("## Per-Primitive Breakdown\n\n");
    out.push_str("| Primitive | Total | Matched | Mismatched |\n");
    out.push_str("|---|---|---|---|\n");
    for (primitive, stats) in per_primitive {
        out.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            primitive, stats.total, stats.matched, stats.mismatched
        ));
    }
    out.push('\n');

    out.push_str("## Coverage Exceptions\n\n");
    if coverage_exceptions.is_empty() {
        out.push_str("None.\n\n");
    } else {
        for exception in coverage_exceptions {
            out.push_str(&format!(
                "- `{}`: {} ({})\n",
                exception.family, exception.reason, exception.justification
            ));
        }
        out.push('\n');
    }

    out.push_str("## Parity Exceptions\n\n");
    if parity_exceptions.is_empty() {
        out.push_str("None.\n");
    } else {
        for exception in parity_exceptions {
            out.push_str(&format!(
                "- `{}` (`{}` / `{}`): {}\n",
                exception.case_id,
                exception.family,
                exception.primitive,
                exception
                    .error
                    .clone()
                    .unwrap_or_else(|| "mismatch without explicit error detail".to_owned())
            ));
        }
    }
    out
}

fn main() -> Result<(), String> {
    let root = repo_root();
    let args = parse_args(&root)?;

    let bundle = read_transform_fixture_bundle(&args.fixtures).map_err(|err| {
        format!(
            "failed reading fixture bundle {}: {err}",
            args.fixtures.display()
        )
    })?;

    let case_program: BTreeMap<String, String> = bundle
        .cases
        .iter()
        .map(|case| (case.case_id.clone(), format!("{:?}", case.program)))
        .collect();

    let config = HarnessConfig::default();
    let transform_report = run_transform_fixture_bundle(&config, &bundle);

    let mut v1 = ParityReportV1::from_transform_report(
        &transform_report,
        &args.mode,
        &args.fj_version,
        &args.oracle_version,
    );

    for family in ["jit", "grad", "vmap", "lax", "random"] {
        v1.families
            .entry(family_name(family))
            .or_insert_with(|| FamilyReport {
                total: 0,
                matched: 0,
                mismatched: 0,
                cases: Vec::new(),
            });
    }

    let mut per_primitive: BTreeMap<String, PrimitiveBreakdown> = BTreeMap::new();
    let mut parity_exceptions = Vec::new();
    let mut coverage_exceptions = Vec::new();

    for case in &transform_report.reports {
        let primitive = case_program
            .get(&case.case_id)
            .cloned()
            .unwrap_or_else(|| "unknown".to_owned());
        let stats = per_primitive
            .entry(primitive.clone())
            .or_insert_with(|| PrimitiveBreakdown {
                total: 0,
                matched: 0,
                mismatched: 0,
                cases: Vec::new(),
            });
        stats.total += 1;
        if case.matched {
            stats.matched += 1;
        } else {
            stats.mismatched += 1;
            parity_exceptions.push(ParityException {
                case_id: case.case_id.clone(),
                family: format!("{:?}", case.family).to_lowercase(),
                primitive,
                drift: format!("{:?}", case.drift_classification),
                expected: case.expected_json.clone(),
                actual: case.actual_json.clone(),
                error: case.error.clone(),
            });
        }
        stats.cases.push(case.case_id.clone());
    }

    for family in ["jit", "grad", "vmap", "lax", "random"] {
        if let Some(stats) = v1.families.get(family)
            && stats.total == 0
        {
            coverage_exceptions.push(CoverageException {
                family: family.to_owned(),
                reason: "no fixture cases captured for this family".to_owned(),
                justification: "tracked as known conformance gap pending fixture expansion"
                    .to_owned(),
            });
        }
    }

    let comprehensive = ComprehensiveParityReport {
        version: "frankenjax.v1-comprehensive-parity.v1".to_owned(),
        timestamp: v1.timestamp.clone(),
        fj_version: v1.fj_version.clone(),
        jax_version: v1.oracle_version.clone(),
        mode: v1.mode.clone(),
        summary: v1.summary.clone(),
        families: v1.families.clone(),
        per_primitive: per_primitive.clone(),
        parity_exceptions: parity_exceptions.clone(),
        coverage_exceptions: coverage_exceptions.clone(),
        gate_status: v1.summary.gate.clone(),
    };

    let markdown = to_markdown(
        &v1,
        &per_primitive,
        &parity_exceptions,
        &coverage_exceptions,
    );

    let per_family = ["jit", "grad", "vmap", "lax", "random"]
        .into_iter()
        .map(|name| {
            let family = v1.families.get(name).cloned().unwrap_or(FamilyReport {
                total: 0,
                matched: 0,
                mismatched: 0,
                cases: Vec::new(),
            });
            (
                name.to_owned(),
                FamilyRollup {
                    total: family.total,
                    matched: family.matched,
                    mismatched: family.mismatched,
                },
            )
        })
        .collect::<BTreeMap<_, _>>();

    let e2e_log = E2EFinalParityLog {
        total_cases: v1.summary.total,
        matched: transform_report.matched_cases,
        mismatched: transform_report.mismatched_cases,
        pass_rate: v1.summary.pass_rate,
        per_family,
        gate_status: v1.summary.gate.clone(),
        pass: v1.gate_passes(),
    };

    let comprehensive_json = serde_json::to_string_pretty(&comprehensive)
        .map_err(|err| format!("failed to serialize comprehensive report: {err}"))?;
    let ci_json = v1
        .to_json()
        .map_err(|err| format!("failed to serialize V1 report: {err}"))?;
    let e2e_json = serde_json::to_string_pretty(&e2e_log)
        .map_err(|err| format!("failed to serialize e2e parity log: {err}"))?;

    write_text(&args.output_json, &comprehensive_json)?;
    write_text(&args.output_markdown, &markdown)?;
    write_text(&args.ci_json, &ci_json)?;
    write_text(&args.e2e_json, &e2e_json)?;

    println!("generated comprehensive parity artifacts:");
    println!("  {}", args.output_json.display());
    println!("  {}", args.output_markdown.display());
    println!("  {}", args.ci_json.display());
    println!("  {}", args.e2e_json.display());
    println!(
        "gate={} total={} matched={} mismatched={}",
        v1.summary.gate,
        transform_report.total_cases,
        transform_report.matched_cases,
        transform_report.mismatched_cases
    );
    Ok(())
}
