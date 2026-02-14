#!/usr/bin/env bash
# generate_run_manifest.sh — Produce a CI run manifest with failure diagnostics
# and artifact index for the current run.
#
# Usage:
#   ./scripts/generate_run_manifest.sh [--run-id <id>] [--output <path>]
#
# If --run-id is omitted, generates one from timestamp + git sha.
# Scans artifacts/ for test logs, e2e logs, golden journeys, coverage, etc.
# Produces manifest.json + summary.txt for human review.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID=""
OUTPUT_DIR=""
STARTED_AT=""

usage() {
  echo "Usage: $0 [--run-id <id>] [--output <dir>]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --output) OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

# Generate run ID if not provided
if [[ -z "$RUN_ID" ]]; then
  GIT_SHA=$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo "unknown")
  TIMESTAMP=$(date +%Y%m%d-%H%M%S)
  RUN_ID="${TIMESTAMP}-${GIT_SHA}"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$ROOT_DIR/artifacts/ci/runs/$RUN_ID"
fi

mkdir -p "$OUTPUT_DIR"

STARTED_AT=$(date +%s%3N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1000))')

# ---- Collect environment ----
RUST_VERSION=$(rustc --version 2>/dev/null || echo "unknown")
OS_INFO=$(uname -srm 2>/dev/null || echo "unknown")
GIT_SHA_FULL=$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_BRANCH=$(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

# ---- Scan for artifacts ----
declare -a ARTIFACT_INDEX=()
ARTIFACT_COUNT=0

scan_artifacts() {
  local dir="$1"
  local category="$2"
  local pattern="$3"

  if [[ ! -d "$dir" ]]; then
    return
  fi

  while IFS= read -r -d '' file; do
    local size
    size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo 0)
    local rel_path="${file#$ROOT_DIR/}"
    local sha=""
    sha=$(sha256sum "$file" 2>/dev/null | cut -d' ' -f1 || echo "")

    ARTIFACT_INDEX+=("{\"path\":\"${rel_path}\",\"category\":\"${category}\",\"size_bytes\":${size},\"sha256\":\"${sha}\"}")
    ARTIFACT_COUNT=$((ARTIFACT_COUNT + 1))
  done < <(find "$dir" -name "$pattern" -type f -print0 2>/dev/null)
}

scan_artifacts "$ROOT_DIR/artifacts/testing/logs" "test_log" "*.json"
scan_artifacts "$ROOT_DIR/artifacts/e2e" "e2e_log" "*.e2e.json"
scan_artifacts "$ROOT_DIR/artifacts/e2e/golden_journeys" "golden_journey" "*.golden.json"
scan_artifacts "$ROOT_DIR/artifacts/testing" "coverage_report" "*.coverage.*.json"
scan_artifacts "$ROOT_DIR/artifacts/ci" "other" "*.v1.json"
scan_artifacts "$ROOT_DIR/artifacts/performance/evidence" "perf_delta" "*.json"

# ---- Scan for failures from gate reports ----
GATE_REPORT="$ROOT_DIR/artifacts/ci/reliability_gate_report.v1.json"
declare -a FAILURES=()
declare -a GATE_RESULTS=()
TOTAL_TESTS=0
PASSED=0
FAILED=0
SKIPPED=0
OVERALL="pass"

if [[ -f "$GATE_REPORT" ]]; then
  # Extract gate-level results
  GATES=$(python3 -c "
import json, sys
with open('$GATE_REPORT') as f:
    data = json.load(f)
gates = []
for key in ['coverage', 'flake', 'runtime', 'crash', 'perf']:
    section = data.get(key, {})
    status = 'pass' if section.get('passed', section.get('pass', True)) else 'fail'
    gates.append({'gate_id': key, 'name': key, 'status': status, 'duration_ms': 0})
overall = 'pass' if data.get('overall_pass', True) else 'fail'
print(json.dumps({'gates': gates, 'overall': overall}))
" 2>/dev/null || echo '{"gates":[],"overall":"pass"}')

  OVERALL=$(echo "$GATES" | python3 -c "import json,sys; print(json.load(sys.stdin)['overall'])" 2>/dev/null || echo "pass")

  while IFS= read -r gate_json; do
    GATE_RESULTS+=("$gate_json")
  done < <(echo "$GATES" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for g in data['gates']:
    print(json.dumps(g))
" 2>/dev/null)
fi

# ---- Count test results from test log files ----
if [[ -d "$ROOT_DIR/artifacts/testing/logs" ]]; then
  while IFS= read -r -d '' logfile; do
    result=$(python3 -c "import json; print(json.load(open('$logfile'))['result'])" 2>/dev/null || echo "unknown")
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    case "$result" in
      pass) PASSED=$((PASSED + 1)) ;;
      fail)
        FAILED=$((FAILED + 1))
        OVERALL="fail"
        # Extract failure diagnostic
        diag=$(python3 -c "
import json, os, time
with open('$logfile') as f:
    data = json.load(f)
rel = os.path.relpath('$logfile', '$ROOT_DIR')
print(json.dumps({
    'schema_version': 'frankenjax.failure-diagnostic.v1',
    'gate': 'G2',
    'test': data.get('test_id', 'unknown'),
    'status': 'fail',
    'summary': (data.get('details') or 'test failed')[:200],
    'detail_path': rel,
    'replay_cmd': 'cargo test -- ' + data.get('test_id', '').split('::')[-1] + ' --nocapture',
    'related_fixtures': [],
    'timestamp_unix_ms': data.get('env', {}).get('timestamp_unix_ms', int(time.time()*1000))
}))
" 2>/dev/null || echo "")
        if [[ -n "$diag" ]]; then
          FAILURES+=("$diag")
        fi
        ;;
      skip) SKIPPED=$((SKIPPED + 1)) ;;
    esac
  done < <(find "$ROOT_DIR/artifacts/testing/logs" -name "*.json" -type f -print0 2>/dev/null)
fi

# ---- Also check e2e logs for failures ----
for e2e_log in "$ROOT_DIR"/artifacts/e2e/*.e2e.json; do
  [[ -f "$e2e_log" ]] || continue
  result=$(python3 -c "import json; print(json.load(open('$e2e_log')).get('result','pass'))" 2>/dev/null || echo "pass")
  TOTAL_TESTS=$((TOTAL_TESTS + 1))
  if [[ "$result" == "pass" ]]; then
    PASSED=$((PASSED + 1))
  else
    FAILED=$((FAILED + 1))
    OVERALL="fail"
  fi
done

# ---- Also check golden journey logs ----
for gj_log in "$ROOT_DIR"/artifacts/e2e/golden_journeys/*.golden.json; do
  [[ -f "$gj_log" ]] || continue
  result=$(python3 -c "import json; print(json.load(open('$gj_log')).get('result','pass'))" 2>/dev/null || echo "pass")
  TOTAL_TESTS=$((TOTAL_TESTS + 1))
  if [[ "$result" == "pass" ]]; then
    PASSED=$((PASSED + 1))
  else
    FAILED=$((FAILED + 1))
    OVERALL="fail"
  fi
done

FINISHED_AT=$(date +%s%3N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1000))')
DURATION=$((FINISHED_AT - STARTED_AT))

# ---- Build manifest JSON ----
GATE_RESULTS_JSON="[]"
if [[ ${#GATE_RESULTS[@]} -gt 0 ]]; then
  GATE_RESULTS_JSON=$(printf '%s\n' "${GATE_RESULTS[@]}" | python3 -c "import json,sys; print(json.dumps([json.loads(l) for l in sys.stdin]))")
fi

FAILURES_JSON="[]"
if [[ ${#FAILURES[@]} -gt 0 ]]; then
  FAILURES_JSON=$(printf '%s\n' "${FAILURES[@]}" | python3 -c "import json,sys; print(json.dumps([json.loads(l) for l in sys.stdin]))")
fi

ARTIFACT_INDEX_JSON="[]"
if [[ ${#ARTIFACT_INDEX[@]} -gt 0 ]]; then
  ARTIFACT_INDEX_JSON=$(printf '%s\n' "${ARTIFACT_INDEX[@]}" | python3 -c "import json,sys; print(json.dumps([json.loads(l) for l in sys.stdin]))")
fi

python3 -c "
import json

manifest = {
    'schema_version': 'frankenjax.run-manifest.v1',
    'run_id': '$RUN_ID',
    'started_at_unix_ms': $STARTED_AT,
    'finished_at_unix_ms': $FINISHED_AT,
    'total_duration_ms': $DURATION,
    'summary': {
        'total_tests': $TOTAL_TESTS,
        'passed': $PASSED,
        'failed': $FAILED,
        'skipped': $SKIPPED,
        'flaky': 0,
        'overall_status': '$OVERALL'
    },
    'gate_results': json.loads('''$GATE_RESULTS_JSON'''),
    'failures': json.loads('''$FAILURES_JSON'''),
    'artifact_index': json.loads('''$ARTIFACT_INDEX_JSON'''),
    'env': {
        'rust_version': '''$RUST_VERSION''',
        'os': '''$OS_INFO''',
        'git_sha': '$GIT_SHA_FULL',
        'git_branch': '$GIT_BRANCH'
    }
}

print(json.dumps(manifest, indent=2))
" > "$OUTPUT_DIR/manifest.json"

echo "[manifest] wrote $OUTPUT_DIR/manifest.json ($ARTIFACT_COUNT artifacts indexed)"

# ---- Generate human-readable summary ----
cat > "$OUTPUT_DIR/summary.txt" <<SUMMARY
============================================================
  FrankenJAX CI Run Summary
  Run ID:   $RUN_ID
  Branch:   $GIT_BRANCH
  Commit:   $GIT_SHA_FULL
  Date:     $(date -u '+%Y-%m-%d %H:%M:%S UTC')
============================================================

RESULT: $(echo "$OVERALL" | tr '[:lower:]' '[:upper:]')

Tests:    $TOTAL_TESTS total, $PASSED passed, $FAILED failed, $SKIPPED skipped
Artifacts: $ARTIFACT_COUNT indexed

SUMMARY

if [[ ${#GATE_RESULTS[@]} -gt 0 ]]; then
  echo "Gate Results:" >> "$OUTPUT_DIR/summary.txt"
  for gate_json in "${GATE_RESULTS[@]}"; do
    gid=$(echo "$gate_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['gate_id'])" 2>/dev/null)
    gstatus=$(echo "$gate_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['status'])" 2>/dev/null)
    printf "  %-12s %s\n" "$gid" "$gstatus" >> "$OUTPUT_DIR/summary.txt"
  done
  echo "" >> "$OUTPUT_DIR/summary.txt"
fi

if [[ $FAILED -gt 0 ]]; then
  echo "FAILURES:" >> "$OUTPUT_DIR/summary.txt"
  echo "------------------------------------------------------------" >> "$OUTPUT_DIR/summary.txt"
  for fail_json in "${FAILURES[@]}"; do
    test_name=$(echo "$fail_json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['test'])" 2>/dev/null)
    summary_text=$(echo "$fail_json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['summary'])" 2>/dev/null)
    replay=$(echo "$fail_json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['replay_cmd'])" 2>/dev/null)
    echo "  TEST:   $test_name" >> "$OUTPUT_DIR/summary.txt"
    echo "  REASON: $summary_text" >> "$OUTPUT_DIR/summary.txt"
    echo "  REPLAY: $replay" >> "$OUTPUT_DIR/summary.txt"
    echo "" >> "$OUTPUT_DIR/summary.txt"
  done
else
  echo "No failures detected." >> "$OUTPUT_DIR/summary.txt"
fi

echo "------------------------------------------------------------" >> "$OUTPUT_DIR/summary.txt"
echo "Manifest: $OUTPUT_DIR/manifest.json" >> "$OUTPUT_DIR/summary.txt"
echo "Artifacts dir: $ROOT_DIR/artifacts/" >> "$OUTPUT_DIR/summary.txt"

echo "[manifest] wrote $OUTPUT_DIR/summary.txt"
cat "$OUTPUT_DIR/summary.txt"
