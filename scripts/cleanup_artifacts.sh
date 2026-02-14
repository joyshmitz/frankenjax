#!/usr/bin/env bash
# cleanup_artifacts.sh — Artifact retention policy enforcement
#
# Retention rules:
#   - Keep last 10 CI run manifests (by creation date)
#   - Keep failed runs for 30 days
#   - Remove passing runs older than 7 days
#   - Generate RaptorQ sidecars for failure artifacts older than 7 days (if fj_durability exists)
#
# Usage:
#   ./scripts/cleanup_artifacts.sh [--dry-run] [--keep-runs <N>] [--fail-retention-days <N>]

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=0
KEEP_RUNS=10
FAIL_RETENTION_DAYS=30
PASS_RETENTION_DAYS=7

usage() {
  echo "Usage: $0 [--dry-run] [--keep-runs N] [--fail-retention-days N] [--pass-retention-days N]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --keep-runs) KEEP_RUNS="$2"; shift 2 ;;
    --fail-retention-days) FAIL_RETENTION_DAYS="$2"; shift 2 ;;
    --pass-retention-days) PASS_RETENTION_DAYS="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

RUNS_DIR="$ROOT_DIR/artifacts/ci/runs"
REMOVED=0
KEPT=0
DURABILITY_QUEUED=0

log() {
  echo "[cleanup] $*"
}

if [[ ! -d "$RUNS_DIR" ]]; then
  log "No runs directory found at $RUNS_DIR — nothing to clean"
  exit 0
fi

# Collect all run directories sorted by modification time (newest first)
mapfile -t ALL_RUNS < <(find "$RUNS_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -rn | cut -d' ' -f2-)

TOTAL_RUNS=${#ALL_RUNS[@]}
log "Found $TOTAL_RUNS run(s) in $RUNS_DIR"

NOW_EPOCH=$(date +%s)

for idx in "${!ALL_RUNS[@]}"; do
  run_dir="${ALL_RUNS[$idx]}"
  run_name=$(basename "$run_dir")
  manifest="$run_dir/manifest.json"

  # Always keep the most recent N runs
  if [[ $idx -lt $KEEP_RUNS ]]; then
    log "KEEP (recent): $run_name"
    KEPT=$((KEPT + 1))
    continue
  fi

  # Check overall status from manifest
  status="unknown"
  if [[ -f "$manifest" ]]; then
    status=$(python3 -c "import json; print(json.load(open('$manifest')).get('summary',{}).get('overall_status','unknown'))" 2>/dev/null || echo "unknown")
  fi

  # Get directory age in days
  dir_mtime=$(stat -c%Y "$run_dir" 2>/dev/null || stat -f%m "$run_dir" 2>/dev/null || echo "$NOW_EPOCH")
  age_days=$(( (NOW_EPOCH - dir_mtime) / 86400 ))

  if [[ "$status" == "fail" ]]; then
    if [[ $age_days -gt $FAIL_RETENTION_DAYS ]]; then
      log "REMOVE (failed, ${age_days}d old > ${FAIL_RETENTION_DAYS}d retention): $run_name"
      if [[ $DRY_RUN -eq 0 ]]; then
        rm -rf "$run_dir"
      fi
      REMOVED=$((REMOVED + 1))
    else
      # Queue durability protection for aging failure artifacts
      if [[ $age_days -gt 7 ]] && command -v cargo &>/dev/null; then
        DURABILITY_BIN="$ROOT_DIR/target/debug/fj_durability"
        if [[ -f "$DURABILITY_BIN" ]]; then
          for artifact in "$run_dir"/*.json; do
            sidecar="${artifact}.sidecar"
            if [[ ! -f "$sidecar" ]]; then
              log "DURABILITY: queueing sidecar for $artifact"
              if [[ $DRY_RUN -eq 0 ]]; then
                "$DURABILITY_BIN" generate --artifact "$artifact" --output "$sidecar" 2>/dev/null || true
              fi
              DURABILITY_QUEUED=$((DURABILITY_QUEUED + 1))
            fi
          done
        fi
      fi
      log "KEEP (failed, ${age_days}d old <= ${FAIL_RETENTION_DAYS}d retention): $run_name"
      KEPT=$((KEPT + 1))
    fi
  else
    if [[ $age_days -gt $PASS_RETENTION_DAYS ]]; then
      log "REMOVE (passed, ${age_days}d old > ${PASS_RETENTION_DAYS}d retention): $run_name"
      if [[ $DRY_RUN -eq 0 ]]; then
        rm -rf "$run_dir"
      fi
      REMOVED=$((REMOVED + 1))
    else
      log "KEEP (passed, ${age_days}d old <= ${PASS_RETENTION_DAYS}d retention): $run_name"
      KEPT=$((KEPT + 1))
    fi
  fi
done

log "------------------------------------------------------------"
log "Summary: $KEPT kept, $REMOVED removed, $DURABILITY_QUEUED durability sidecars queued"
if [[ $DRY_RUN -eq 1 ]]; then
  log "(dry run — no files were actually removed)"
fi
