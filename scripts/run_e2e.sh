#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="${FJ_E2E_ARTIFACT_DIR:-$ROOT_DIR/artifacts/e2e}"
PACKET_FILTER=""
SCENARIO_FILTER=""

usage() {
  cat <<'USAGE'
Usage: ./scripts/run_e2e.sh [--packet P2C-001] [--scenario e2e_p2c001_*]

Options:
  --packet <P2C-###>     Run only scenarios associated with one packet.
  --scenario <id>        Run only one exact scenario id.
  -h, --help             Show this help.

Examples:
  ./scripts/run_e2e.sh
  ./scripts/run_e2e.sh --packet P2C-001
  ./scripts/run_e2e.sh --scenario e2e_p2c001_full_dispatch_pipeline
USAGE
}

normalize_packet() {
  local raw="$1"
  local upper
  upper="$(printf '%s' "$raw" | tr '[:lower:]' '[:upper:]')"
  upper="${upper//_/\-}"
  if [[ "$upper" =~ ^P2C-?([0-9]{3})$ ]]; then
    printf 'P2C-%s' "${BASH_REMATCH[1]}"
    return 0
  fi
  printf '%s' "$upper"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --packet)
      if [[ $# -lt 2 ]]; then
        echo "error: --packet requires a value" >&2
        exit 2
      fi
      PACKET_FILTER="$(normalize_packet "$2")"
      shift 2
      ;;
    --scenario)
      if [[ $# -lt 2 ]]; then
        echo "error: --scenario requires a value" >&2
        exit 2
      fi
      SCENARIO_FILTER="$2"
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

mkdir -p "$ARTIFACT_DIR"

mapfile -t E2E_FILES < <(find "$ROOT_DIR/crates" -type f -path '*/tests/e2e*.rs' | sort)
if [[ ${#E2E_FILES[@]} -eq 0 ]]; then
  echo "No e2e test binaries discovered (expected files matching crates/*/tests/e2e*.rs)." >&2
  exit 1
fi

SCENARIOS=()
for test_file in "${E2E_FILES[@]}"; do
  crate_dir="$(dirname "$(dirname "$test_file")")"
  pkg="$(basename "$crate_dir")"
  test_bin="$(basename "$test_file" .rs)"

  mapfile -t discovered < <(
    cargo test -p "$pkg" --test "$test_bin" -- --list 2>/dev/null \
      | awk -F': test' '/^e2e_[A-Za-z0-9_]+: test$/ {print $1}'
  )

  for scenario in "${discovered[@]}"; do
    packet="UNSPECIFIED"
    if [[ "$scenario" =~ p2c([0-9]{3}) ]]; then
      packet="P2C-${BASH_REMATCH[1]}"
    fi
    SCENARIOS+=("$pkg|$test_bin|$scenario|$packet")
  done
done

if [[ ${#SCENARIOS[@]} -eq 0 ]]; then
  echo "No e2e scenarios discovered from e2e test binaries." >&2
  exit 1
fi

SELECTED=()
for entry in "${SCENARIOS[@]}"; do
  IFS='|' read -r pkg test_bin scenario packet <<<"$entry"

  if [[ -n "$PACKET_FILTER" && "$packet" != "$PACKET_FILTER" ]]; then
    continue
  fi
  if [[ -n "$SCENARIO_FILTER" && "$scenario" != "$SCENARIO_FILTER" ]]; then
    continue
  fi

  SELECTED+=("$entry")
done

if [[ ${#SELECTED[@]} -eq 0 ]]; then
  echo "No scenarios matched filters (packet='${PACKET_FILTER:-*}', scenario='${SCENARIO_FILTER:-*}')." >&2
  exit 1
fi

PASS_COUNT=0
FAIL_COUNT=0
SUMMARY_ROWS=()

printf 'Running %d E2E scenario(s)...\n' "${#SELECTED[@]}"

for entry in "${SELECTED[@]}"; do
  IFS='|' read -r pkg test_bin scenario packet <<<"$entry"

  stdout_log="$ARTIFACT_DIR/${scenario}.stdout.log"
  forensic_log="$ARTIFACT_DIR/${scenario}.e2e.json"
  replay_cmd="cargo test -p $pkg --test $test_bin -- $scenario --exact --nocapture"

  start_ms="$(date +%s%3N)"
  set +e
  FJ_E2E_ARTIFACT_DIR="$ARTIFACT_DIR" \
    cargo test -p "$pkg" --test "$test_bin" -- "$scenario" --exact --nocapture \
    >"$stdout_log" 2>&1
  rc=$?
  set -e
  end_ms="$(date +%s%3N)"
  duration_ms=$((end_ms - start_ms))

  if [[ ! -f "$forensic_log" ]]; then
    cat >"$forensic_log" <<JSON
{
  "schema_version": "frankenjax.e2e.log.v1",
  "scenario_id": "$scenario",
  "packet_id": "$packet",
  "result": "$( [[ $rc -eq 0 ]] && echo pass || echo fail )",
  "duration_ms": $duration_ms,
  "details": "scenario did not emit forensic log; see stdout log",
  "replay_command": "$replay_cmd",
  "artifact_refs": ["$stdout_log"]
}
JSON
  fi

  if [[ $rc -eq 0 ]]; then
    ((PASS_COUNT += 1))
    SUMMARY_ROWS+=("PASS | $packet | $scenario | ${duration_ms}ms")
    printf '[PASS] %s (%s)\n' "$scenario" "$packet"
  else
    ((FAIL_COUNT += 1))
    SUMMARY_ROWS+=("FAIL | $packet | $scenario | ${duration_ms}ms | replay: $replay_cmd")
    printf '[FAIL] %s (%s)\n' "$scenario" "$packet"
    printf '  replay: %s\n' "$replay_cmd"
    printf '  stdout: %s\n' "$stdout_log"
  fi
done

TOTAL_COUNT=$((PASS_COUNT + FAIL_COUNT))
printf '\nE2E Summary\n'
printf '  total: %d\n' "$TOTAL_COUNT"
printf '  pass:  %d\n' "$PASS_COUNT"
printf '  fail:  %d\n' "$FAIL_COUNT"
printf '  logs:  %s\n' "$ARTIFACT_DIR"

for row in "${SUMMARY_ROWS[@]}"; do
  printf '  %s\n' "$row"
done

if [[ $FAIL_COUNT -gt 0 ]]; then
  exit 1
fi
