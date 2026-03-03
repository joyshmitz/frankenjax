#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
threat_doc="${repo_root}/artifacts/phase2c/FJ-P2C-FOUNDATION/security_threat_matrix.md"

ci_out="${repo_root}/artifacts/ci/runs/threat-model-review/result.json"
e2e_out="${repo_root}/artifacts/e2e/e2e_threat_model_validation.e2e.json"

mkdir -p "$(dirname "${ci_out}")" "$(dirname "${e2e_out}")"

python3 - "${threat_doc}" "${ci_out}" "${e2e_out}" <<'PY'
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path


def parse_table(lines: list[str], header_row: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    headers: list[str] | None = None
    in_table = False
    for line in lines:
        if line.strip() == header_row.strip():
            in_table = True
            headers = [c.strip() for c in line.strip().split("|")[1:-1]]
            continue
        if not in_table:
            continue
        stripped = line.strip()
        if not stripped.startswith("|"):
            break
        if stripped.startswith("|---"):
            continue
        cells = [c.strip() for c in stripped.split("|")[1:-1]]
        if headers is None or len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells, strict=True)))
    return rows


def threat_check(
    checks: list[dict[str, object]],
    area: str,
    pattern: str,
    threat_rows: list[dict[str, str]],
) -> None:
    row = next(
        (
            r
            for r in threat_rows
            if re.search(pattern, r.get("Threat class", ""), flags=re.IGNORECASE)
        ),
        None,
    )
    documented = row is not None
    mitigated = bool(
        documented
        and row.get("Strict mitigation", "").strip()
        and row.get("Hardened mitigation", "").strip()
    )
    risk_level = "unknown"
    threat = "missing"
    if row is not None:
        threat = row.get("Threat class", "unknown")
        residual = row.get("Residual risk", "")
        match = re.match(r"([A-Za-z-]+)", residual)
        risk_level = match.group(1) if match else (residual or "unknown")
    checks.append(
        {
            "area": area,
            "threat": threat,
            "documented": documented,
            "mitigated": mitigated,
            "risk_level": risk_level,
            "pass": documented and mitigated,
        }
    )


doc_path = Path(sys.argv[1])
ci_out = Path(sys.argv[2])
e2e_out = Path(sys.argv[3])
doc_text = doc_path.read_text(encoding="utf-8")
lines = doc_text.splitlines()

threat_rows = parse_table(
    lines,
    "| Threat class | Packet family | Subsystem | Strict mitigation | Hardened mitigation | Residual risk | Evidence |",
)
adv_rows = parse_table(
    lines,
    "| Class ID | Malicious input family | Target subsystem(s) | Current coverage | Required next test hook |",
)

checks: list[dict[str, object]] = []

threat_check(checks, "dtypes", r"complex numeric edge-case abuse", threat_rows)
threat_check(checks, "rng", r"rng key lifecycle misuse", threat_rows)
threat_check(checks, "control_flow", r"control-flow safety hazards", threat_rows)
threat_check(checks, "effects", r"effect-token misuse", threat_rows)
threat_check(checks, "custom_derivatives", r"custom-derivative rule abuse", threat_rows)

required_adv = {"ADV-007", "ADV-008", "ADV-009", "ADV-010", "ADV-011"}
present_adv = {row.get("Class ID", "").strip().strip("`") for row in adv_rows}
missing_adv = sorted(required_adv - present_adv)
checks.append(
    {
        "area": "adversarial_fixtures",
        "threat": ",".join(sorted(required_adv)),
        "documented": not missing_adv,
        "mitigated": not missing_adv,
        "risk_level": "n/a",
        "pass": not missing_adv,
        "missing": missing_adv,
    }
)

stale_patterns = [
    r"\bv1-only\b",
    r"\bv1 only\b",
    r"obsolete v1",
]
stale_hits = []
for pat in stale_patterns:
    if re.search(pat, doc_text, flags=re.IGNORECASE):
        stale_hits.append(pat)
checks.append(
    {
        "area": "stale_entries",
        "threat": "v1-stale-threat-vectors",
        "documented": not stale_hits,
        "mitigated": not stale_hits,
        "risk_level": "n/a",
        "pass": not stale_hits,
        "matches": stale_hits,
    }
)

all_pass = all(bool(c["pass"]) for c in checks)
now_ms = int(time.time() * 1000)

ci_payload = {
    "test_name": "threat_model_review",
    "generated_at_unix_ms": now_ms,
    "pass": all_pass,
    "checks": checks,
}
ci_out.write_text(json.dumps(ci_payload, indent=2) + "\n", encoding="utf-8")

e2e_checks = []
for c in checks:
    if c["area"] == "stale_entries":
        e2e_checks.append(
            {
                "area": c["area"],
                "threats_documented": 1 if c["documented"] else 0,
                "mitigations_present": bool(c["mitigated"]),
                "pass": bool(c["pass"]),
            }
        )
        continue
    e2e_checks.append(
        {
            "area": c["area"],
            "threats_documented": 1 if c["documented"] else 0,
            "mitigations_present": bool(c["mitigated"]),
            "pass": bool(c["pass"]),
        }
    )

e2e_payload = {
    "scenario": "e2e_threat_model_validation",
    "generated_at_unix_ms": now_ms,
    "checks": e2e_checks,
}
e2e_out.write_text(json.dumps(e2e_payload, indent=2) + "\n", encoding="utf-8")

print(f"threat-model validation: {'PASS' if all_pass else 'FAIL'}")
print(f"  checks: {len(checks)}")
print(f"  ci: {ci_out}")
print(f"  e2e: {e2e_out}")

if not all_pass:
    sys.exit(1)
PY
