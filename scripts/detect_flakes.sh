#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS=10
COMMAND="cargo test --workspace --quiet"
OUT_JSON="$ROOT_DIR/artifacts/ci/flake_report.v1.json"
LOG_DIR="$ROOT_DIR/artifacts/ci/flake-runs"

usage() {
  cat <<'USAGE'
Usage: ./scripts/detect_flakes.sh [options]

Options:
  --runs <n>         Number of repeated runs (default: 10).
  --command <cmd>    Command to execute repeatedly.
  --out <path>       JSON report output path.
  --log-dir <path>   Directory for per-run stdout/stderr logs.
  -h, --help         Show help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs)
      RUNS="$2"
      shift 2
      ;;
    --command)
      COMMAND="$2"
      shift 2
      ;;
    --out)
      OUT_JSON="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument '$1'" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "$RUNS" =~ ^[0-9]+$ ]] || [[ "$RUNS" -lt 1 ]]; then
  echo "error: --runs must be a positive integer" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUT_JSON")" "$LOG_DIR"

run_entries_tmp="$(mktemp)"
trap 'rm -f "$run_entries_tmp"' EXIT

pass_runs=0
fail_runs=0

printf 'Running flake detector for %s runs...\n' "$RUNS"
for ((i = 1; i <= RUNS; i++)); do
  run_log="$LOG_DIR/run_${i}.log"
  start_ms="$(date +%s%3N)"

  set +e
  bash -lc "cd '$ROOT_DIR' && $COMMAND" >"$run_log" 2>&1
  rc=$?
  set -e

  end_ms="$(date +%s%3N)"
  duration_ms=$((end_ms - start_ms))

  status="pass"
  if [[ $rc -ne 0 ]]; then
    status="fail"
    ((fail_runs += 1))
  else
    ((pass_runs += 1))
  fi

  jq -nc \
    --argjson run_index "$i" \
    --arg status "$status" \
    --argjson exit_code "$rc" \
    --argjson duration_ms "$duration_ms" \
    --arg log_path "$run_log" \
    '{run_index, status, exit_code, duration_ms, log_path}' >>"$run_entries_tmp"

  printf '  [%s/%s] %s (%sms)\n' "$i" "$RUNS" "$status" "$duration_ms"
done

suite_flake_rate="$(awk -v fails="$fail_runs" -v runs="$RUNS" 'BEGIN { printf "%.6f", fails / runs }')"
flaky_detected="false"
if [[ "$pass_runs" -gt 0 && "$fail_runs" -gt 0 ]]; then
  flaky_detected="true"
fi

generated_at_ms="$(date +%s%3N)"

jq -s \
  --arg schema_version "frankenjax.flake-report.v1" \
  --arg command "$COMMAND" \
  --argjson runs_requested "$RUNS" \
  --argjson pass_runs "$pass_runs" \
  --argjson fail_runs "$fail_runs" \
  --argjson generated_at_unix_ms "$generated_at_ms" \
  --argjson suite_flake_rate "$suite_flake_rate" \
  --argjson flaky_detected "$flaky_detected" \
  --arg log_dir "$LOG_DIR" \
  '{
    schema_version: $schema_version,
    generated_at_unix_ms: $generated_at_unix_ms,
    command: $command,
    runs_requested: $runs_requested,
    pass_runs: $pass_runs,
    fail_runs: $fail_runs,
    suite_flake_rate: $suite_flake_rate,
    flaky_detected: $flaky_detected,
    log_dir: $log_dir,
    runs: .
  }' "$run_entries_tmp" >"$OUT_JSON"

printf 'Flake report written: %s\n' "$OUT_JSON"

if [[ "$fail_runs" -gt 0 ]]; then
  echo "flake detector found failing runs" >&2
  exit 1
fi
