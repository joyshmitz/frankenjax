#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FUZZ_DIR="$ROOT_DIR/crates/fj-conformance/fuzz"

TARGET=""
INPUT_PATH=""
SEVERITY_OVERRIDE=""
STATUS="open"
CLASSIFICATION_REASON_OVERRIDE=""
BACKTRACE_PATH=""
REPRO_LOG_PATH=""
INDEX_PATH="$ROOT_DIR/crates/fj-conformance/fuzz/corpus/crashes/index.v1.jsonl"
CRASH_ROOT="$ROOT_DIR/crates/fj-conformance/fuzz/corpus/crashes"
BEAD_ID=""
OPEN_BEAD=0
MINIMIZE=1
DROP_SOURCE_COUNT=1

usage() {
  cat <<'USAGE'
Usage: ./scripts/triage_crash.sh --target <name> --input <path> [options]

Required:
  --target <name>            Fuzz target name (e.g., ir_deserializer).
  --input <path>             Crash artifact input path.

Options:
  --severity <P0|P1|P2>      Override severity classification.
  --status <open|closed|resolved>
                             Crash record status (default: open).
  --reason <text>            Override classification reason.
  --backtrace <path>         Optional backtrace log path.
  --repro-log <path>         Optional reproduce-command output log path.
  --index <path>             Crash index JSONL path.
  --crash-root <path>        Root directory for triaged crash artifacts.
  --bead <id>                Related bead ID (e.g., bd-3dl.7).
  --open-bead                Open a new crash bead via `bd create`.
  --no-minimize              Skip cargo-fuzz minimization and copy input directly.
  --drop-source <n>          Decode-proof source symbol drop count (default: 1).
  -h, --help                 Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET="$2"
      shift 2
      ;;
    --input)
      INPUT_PATH="$2"
      shift 2
      ;;
    --severity)
      SEVERITY_OVERRIDE="$2"
      shift 2
      ;;
    --status)
      STATUS="$2"
      shift 2
      ;;
    --reason)
      CLASSIFICATION_REASON_OVERRIDE="$2"
      shift 2
      ;;
    --backtrace)
      BACKTRACE_PATH="$2"
      shift 2
      ;;
    --repro-log)
      REPRO_LOG_PATH="$2"
      shift 2
      ;;
    --index)
      INDEX_PATH="$2"
      shift 2
      ;;
    --crash-root)
      CRASH_ROOT="$2"
      shift 2
      ;;
    --bead)
      BEAD_ID="$2"
      shift 2
      ;;
    --open-bead)
      OPEN_BEAD=1
      shift
      ;;
    --no-minimize)
      MINIMIZE=0
      shift
      ;;
    --drop-source)
      DROP_SOURCE_COUNT="$2"
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

if [[ -z "$TARGET" || -z "$INPUT_PATH" ]]; then
  echo "error: --target and --input are required" >&2
  usage >&2
  exit 2
fi

if [[ ! -f "$INPUT_PATH" ]]; then
  echo "error: input file not found: $INPUT_PATH" >&2
  exit 1
fi

if ! [[ "$DROP_SOURCE_COUNT" =~ ^[0-9]+$ ]]; then
  echo "error: --drop-source must be a non-negative integer" >&2
  exit 2
fi

case "$STATUS" in
  open|closed|resolved) ;;
  *)
    echo "error: --status must be one of open|closed|resolved" >&2
    exit 2
    ;;
esac

if [[ -n "$SEVERITY_OVERRIDE" ]]; then
  case "$SEVERITY_OVERRIDE" in
    P0|P1|P2) ;;
    *)
      echo "error: --severity must be one of P0|P1|P2" >&2
      exit 2
      ;;
  esac
fi

if [[ -n "$BACKTRACE_PATH" && ! -f "$BACKTRACE_PATH" ]]; then
  echo "error: backtrace log not found: $BACKTRACE_PATH" >&2
  exit 1
fi

if [[ -n "$REPRO_LOG_PATH" && ! -f "$REPRO_LOG_PATH" ]]; then
  echo "error: reproduce log not found: $REPRO_LOG_PATH" >&2
  exit 1
fi

mkdir -p "$CRASH_ROOT" "$(dirname "$INDEX_PATH")"

crash_sha="$(sha256sum "$INPUT_PATH" | awk '{print $1}')"
crash_dir="$CRASH_ROOT/$TARGET/$crash_sha"
mkdir -p "$crash_dir"

raw_path="$crash_dir/input.raw"
minimized_path="$crash_dir/input.minimized.raw"
backtrace_copy="$crash_dir/backtrace.log"
repro_copy="$crash_dir/repro.log"
triage_log="$crash_dir/triage.log"

durability_sidecar="$crash_dir/${crash_sha}.sidecar.json"
durability_scrub="$crash_dir/${crash_sha}.scrub.json"
durability_proof="$crash_dir/${crash_sha}.decode-proof.json"
durability_log="$crash_dir/durability.log"

cp -f "$INPUT_PATH" "$raw_path"

if [[ -n "$BACKTRACE_PATH" ]]; then
  cp -f "$BACKTRACE_PATH" "$backtrace_copy"
else
  : >"$backtrace_copy"
fi

if [[ -n "$REPRO_LOG_PATH" ]]; then
  cp -f "$REPRO_LOG_PATH" "$repro_copy"
else
  : >"$repro_copy"
fi

run_cmd_in_repo() {
  local cmd="$1"
  local allow_remote="${2:-1}"
  if [[ "$allow_remote" == "1" ]] && command -v rch >/dev/null 2>&1; then
    rch exec -- bash -lc "cd '$ROOT_DIR' && $cmd"
  else
    bash -lc "cd '$ROOT_DIR' && $cmd"
  fi
}

path_under_root() {
  local resolved
  resolved="$(realpath -m "$1")"
  [[ "$resolved" == "$ROOT_DIR" || "$resolved" == "$ROOT_DIR/"* ]]
}

remote_eligible=1
for required_path in "$raw_path" "$minimized_path" "$durability_sidecar" "$durability_scrub" "$durability_proof"; do
  if ! path_under_root "$required_path"; then
    remote_eligible=0
    break
  fi
done

minimization_status="copied_without_minimize"
if [[ $MINIMIZE -eq 1 ]]; then
  if command -v cargo >/dev/null 2>&1; then
    set +e
    run_cmd_in_repo "cd '$FUZZ_DIR' && cargo fuzz tmin '$TARGET' '$raw_path' -runs=4096 -exact_artifact_path='$minimized_path'" \
      "$remote_eligible" \
      >"$crash_dir/tmin.log" 2>&1
    tmin_rc=$?
    set -e

    if [[ $tmin_rc -eq 0 && -s "$minimized_path" ]]; then
      minimization_status="cargo_fuzz_tmin"
    else
      cp -f "$raw_path" "$minimized_path"
      minimization_status="fallback_copy_after_tmin_failure"
    fi
  else
    cp -f "$raw_path" "$minimized_path"
    minimization_status="fallback_copy_no_cargo"
  fi
else
  cp -f "$raw_path" "$minimized_path"
fi

classification_text=""
if [[ -s "$backtrace_copy" ]]; then
  classification_text+="$(cat "$backtrace_copy")\n"
fi
if [[ -s "$repro_copy" ]]; then
  classification_text+="$(cat "$repro_copy")\n"
fi
classification_text+="target:$TARGET\n"

severity=""
classification_reason=""

if [[ -n "$SEVERITY_OVERRIDE" ]]; then
  severity="$SEVERITY_OVERRIDE"
  classification_reason="manual severity override"
else
  if echo "$classification_text" | grep -Eiq "invariant|integrity|segmentation fault|stack overflow|cache confusion|decode failed|assertion failed|out of bounds"; then
    severity="P0"
    classification_reason="matched safety-invariant/signature pattern"
  elif echo "$classification_text" | grep -Eiq "panic|panicked at|thread '.*' panicked"; then
    severity="P1"
    classification_reason="panic signature detected"
  else
    severity="P2"
    classification_reason="no panic/invariant signature detected"
  fi
fi

if [[ -n "$CLASSIFICATION_REASON_OVERRIDE" ]]; then
  classification_reason="$CLASSIFICATION_REASON_OVERRIDE"
fi

set +e
run_cmd_in_repo "cargo run -p fj-conformance --bin fj_durability -- pipeline --artifact '$minimized_path' --sidecar '$durability_sidecar' --report '$durability_scrub' --proof '$durability_proof' --drop-source '$DROP_SOURCE_COUNT'" \
  "$remote_eligible" \
  >"$durability_log" 2>&1
pipeline_rc=$?
set -e

if [[ $pipeline_rc -ne 0 ]]; then
  echo "error: durability pipeline failed; see $durability_log" >&2
  exit 1
fi

timestamp_ms="$(date +%s%3N)"
timestamp_iso="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
minimized_sha="$(sha256sum "$minimized_path" | awk '{print $1}')"

record_json="$(jq -nc \
  --arg schema_version "frankenjax.crash-triage.v1" \
  --arg crash_hash_sha256 "$crash_sha" \
  --arg minimized_hash_sha256 "$minimized_sha" \
  --arg target "$TARGET" \
  --arg severity "$severity" \
  --arg status "$STATUS" \
  --arg classification_reason "$classification_reason" \
  --arg timestamp_iso "$timestamp_iso" \
  --arg source_input_path "$INPUT_PATH" \
  --arg stored_input_path "$raw_path" \
  --arg minimized_input_path "$minimized_path" \
  --arg minimization_status "$minimization_status" \
  --arg backtrace_path "$backtrace_copy" \
  --arg repro_log_path "$repro_copy" \
  --arg sidecar_path "$durability_sidecar" \
  --arg scrub_report_path "$durability_scrub" \
  --arg decode_proof_path "$durability_proof" \
  --arg durability_log_path "$durability_log" \
  --arg bead_id "$BEAD_ID" \
  --argjson updated_at_unix_ms "$timestamp_ms" \
  '{
    schema_version: $schema_version,
    crash_hash_sha256: $crash_hash_sha256,
    minimized_hash_sha256: $minimized_hash_sha256,
    target: $target,
    severity: $severity,
    status: $status,
    classification_reason: $classification_reason,
    triaged_at_iso8601: $timestamp_iso,
    updated_at_unix_ms: $updated_at_unix_ms,
    source_input_path: $source_input_path,
    stored_input_path: $stored_input_path,
    minimized_input_path: $minimized_input_path,
    minimization_status: $minimization_status,
    backtrace_path: $backtrace_path,
    repro_log_path: $repro_log_path,
    sidecar_path: $sidecar_path,
    scrub_report_path: $scrub_report_path,
    decode_proof_path: $decode_proof_path,
    durability_log_path: $durability_log_path,
    bead_id: (if $bead_id == "" then null else $bead_id end)
  }')"

index_tmp="$(mktemp)"
trap 'rm -f "$index_tmp"' EXIT

updated_existing=0
if [[ -f "$INDEX_PATH" ]]; then
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    if echo "$line" | jq -e --arg hash "$crash_sha" --arg target "$TARGET" '.crash_hash_sha256 == $hash and .target == $target' >/dev/null 2>&1; then
      echo "$record_json" >>"$index_tmp"
      updated_existing=1
    else
      echo "$line" >>"$index_tmp"
    fi
  done <"$INDEX_PATH"
fi

if [[ $updated_existing -eq 0 ]]; then
  echo "$record_json" >>"$index_tmp"
fi

mv "$index_tmp" "$INDEX_PATH"

opened_bead_id=""
if [[ $OPEN_BEAD -eq 1 ]] && command -v bd >/dev/null 2>&1; then
  case "$severity" in
    P0) bead_priority="0" ;;
    P1) bead_priority="1" ;;
    P2) bead_priority="2" ;;
  esac

  title="[CRASH][$severity][$TARGET] ${crash_sha:0:12}"
  description="Crash hash: $crash_sha\nTarget: $TARGET\nSeverity: $severity\nStatus: $STATUS\nCrash dir: $crash_dir\nIndex: $INDEX_PATH"

  set +e
  opened_bead_id="$(bd create --title "$title" --type bug --priority "$bead_priority" --description "$description" --labels "fuzz,crash,$severity" --silent 2>/dev/null)"
  bead_create_rc=$?
  set -e

  if [[ $bead_create_rc -ne 0 ]]; then
    echo "warning: unable to create bead for crash record" >&2
    opened_bead_id=""
  fi
fi

{
  echo "target=$TARGET"
  echo "crash_sha=$crash_sha"
  echo "severity=$severity"
  echo "status=$STATUS"
  echo "classification_reason=$classification_reason"
  echo "minimization_status=$minimization_status"
  echo "crash_dir=$crash_dir"
  echo "index_path=$INDEX_PATH"
  if [[ -n "$opened_bead_id" ]]; then
    echo "opened_bead_id=$opened_bead_id"
  fi
} | tee "$triage_log"

echo "Triage complete for $TARGET ($severity): $crash_dir"
