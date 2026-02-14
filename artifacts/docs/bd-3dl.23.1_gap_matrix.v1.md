# bd-3dl.23.1 Baseline Gap Matrix + Quantitative Expansion Targets (v1)

Date: 2026-02-14
Bead: `bd-3dl.23.1`
Scope:
- `EXHAUSTIVE_LEGACY_ANALYSIS.md`
- `EXISTING_JAX_STRUCTURE.md`

## Method (Auditable)

Section metrics were captured from Markdown heading boundaries (`##` and `###`):
- `current_words`: tokenized word count inside each section body.
- `anchor_count`: count of inline code anchors (`` `...` ``) inside each section body.

Target formula:
- `target_words = current_words * expansion_multiplier`

Interpretation:
- Higher multipliers are assigned to sections with high semantic-risk concentration (transform order, cache keying, strict/hardened drift, runtime boundaries, recovery semantics).
- Container sections with `current_words = 0` are tracked explicitly and budgeted through child sections.

## Baseline Summary

| Document | Current words | Section count (`##`/`###`) | Inline anchors | Target words floor | Expansion factor |
|---|---:|---:|---:|---:|---:|
| `EXHAUSTIVE_LEGACY_ANALYSIS.md` | 1422 | 20 | 130 | 13776 | 9.69x |
| `EXISTING_JAX_STRUCTURE.md` | 327 | 12 | 58 | 4067 | 12.44x |
| **Combined** | **1749** | **32** | **188** | **17843** | **10.20x** |

## Matrix: `EXHAUSTIVE_LEGACY_ANALYSIS.md`

| Section | Level | Current words | Anchors | Risk | Expansion multiplier | Target words floor | Primary next pass |
|---|---:|---:|---:|---|---:|---:|---|
| `0. Mission and Completion Criteria` | 2 | 41 | 0 | high | 8x | 328 | `bd-3dl.23.4` |
| `1. Source-of-Truth Crosswalk` | 2 | 95 | 9 | high | 7x | 665 | `bd-3dl.23.2` |
| `2. Quantitative Legacy Inventory (Measured)` | 2 | 44 | 12 | medium | 8x | 352 | `bd-3dl.23.2` |
| `3. Subsystem Extraction Matrix (Legacy -> Rust)` | 2 | 213 | 35 | critical | 7x | 1491 | `bd-3dl.23.2` |
| `4. Alien-Artifact Invariant Ledger (Formal Obligations)` | 2 | 88 | 5 | critical | 12x | 1056 | `bd-3dl.23.4` |
| `5. Native/XLA/FFI Boundary Register` | 2 | 71 | 8 | critical | 12x | 852 | `bd-3dl.23.2` |
| `6. Compatibility and Security Doctrine (Mode-Split)` | 2 | 129 | 5 | critical | 11x | 1419 | `bd-3dl.23.4` |
| `7. Conformance Program (Exhaustive First Wave)` | 2 | 0 | 0 | high | container | child-driven | `bd-3dl.23.10` |
| `7.1 Fixture families` | 3 | 34 | 3 | high | 12x | 408 | `bd-3dl.23.10` |
| `7.2 Differential harness outputs` | 3 | 26 | 0 | high | 12x | 312 | `bd-3dl.23.10` |
| `8. Extreme Optimization Program` | 2 | 83 | 0 | high | 12x | 996 | `bd-3dl.23.6` |
| `9. RaptorQ-Everywhere Artifact Contract` | 2 | 29 | 0 | high | 14x | 406 | `bd-3dl.23.8` |
| `10. Phase-2 Execution Backlog (Concrete)` | 2 | 137 | 2 | high | 7x | 959 | `bd-3dl.23.14` |
| `11. Residual Gaps and Risks` | 2 | 42 | 2 | high | 8x | 336 | `bd-3dl.23.9` |
| `12. Deep-Pass Hotspot Inventory (Measured)` | 2 | 108 | 18 | medium | 7x | 756 | `bd-3dl.23.6` |
| `13. Phase-2C Extraction Payload Contract (Per Ticket)` | 2 | 112 | 9 | high | 10x | 1120 | `bd-3dl.23.14` |
| `14. Strict/Hardened Compatibility Drift Budgets` | 2 | 48 | 8 | critical | 12x | 576 | `bd-3dl.23.4` |
| `15. Extreme-Software-Optimization Execution Law` | 2 | 59 | 6 | high | 11x | 649 | `bd-3dl.23.6` |
| `16. RaptorQ Evidence Topology and Recovery Drills` | 2 | 45 | 3 | high | 11x | 495 | `bd-3dl.23.8` |
| `17. Phase-2C Exit Checklist (Operational)` | 2 | 60 | 1 | high | 10x | 600 | `bd-3dl.23.14` |

## Matrix: `EXISTING_JAX_STRUCTURE.md`

| Section | Level | Current words | Anchors | Risk | Expansion multiplier | Target words floor | Primary next pass |
|---|---:|---:|---:|---|---:|---:|---|
| `1. Legacy Oracle` | 2 | 11 | 2 | medium | 10x | 110 | `bd-3dl.23.2` |
| `2. High-Value File/Function Anchors` | 2 | 0 | 0 | critical | container | child-driven | `bd-3dl.23.2` |
| `Transform API entry points` | 3 | 11 | 6 | critical | 15x | 165 | `bd-3dl.23.3` |
| `JIT and staging/lowering path` | 3 | 16 | 6 | critical | 15x | 240 | `bd-3dl.23.5` |
| `AD (reverse/forward transform semantics)` | 3 | 11 | 4 | critical | 15x | 165 | `bd-3dl.23.5` |
| `Batching / vmap semantics` | 3 | 11 | 5 | critical | 15x | 165 | `bd-3dl.23.5` |
| `Cache-key and compilation cache semantics` | 3 | 47 | 14 | critical | 14x | 658 | `bd-3dl.23.3` |
| `Dispatch-level memoization and trace caching` | 3 | 21 | 5 | critical | 14x | 294 | `bd-3dl.23.5` |
| `3. Semantic Hotspots (Non-Negotiable)` | 2 | 42 | 3 | high | 10x | 420 | `bd-3dl.23.4` |
| `4. Conformance Fixture Family Anchors` | 2 | 47 | 13 | high | 12x | 564 | `bd-3dl.23.10` |
| `5. Compatibility-Critical Inputs for Cache Keying` | 2 | 44 | 0 | critical | 14x | 616 | `bd-3dl.23.4` |
| `6. Security and Reliability Risk Areas` | 2 | 25 | 0 | critical | 14x | 350 | `bd-3dl.23.9` |
| `7. Extraction Boundary (Current)` | 2 | 32 | 0 | high | 10x | 320 | `bd-3dl.23.14` |

## Priority Queue: High-Risk Omissions to Attack First

1. `EXISTING_JAX_STRUCTURE.md` section `2` subtree and section `5`: currently outline-level only, but these define compatibility and cache-key correctness boundaries.
2. `EXHAUSTIVE_LEGACY_ANALYSIS.md` sections `4`, `6`, and `14`: invariants and strict/hardened drift budgets are underspecified for release-grade proof obligations.
3. `EXHAUSTIVE_LEGACY_ANALYSIS.md` section `5`: backend/FFI boundary register lacks lifecycle state machines and failure choreography.
4. `EXHAUSTIVE_LEGACY_ANALYSIS.md` sections `7.1` and `7.2`: conformance corpus and differential output taxonomy are not yet dense enough for forensic regression work.
5. `EXHAUSTIVE_LEGACY_ANALYSIS.md` sections `9` and `16`: RaptorQ durability policy lacks adversarial recovery scenarios and decode-proof edge handling detail.

## Section-Level Audit Gates

For each expanded section, completion is accepted only when all numeric gates pass:

1. Word-floor gate: section `word_count >= target_words_floor`.
2. Anchor-density gate:
   - critical/high risk sections: `>= 1` concrete source anchor per `80` words.
   - medium risk sections: `>= 1` concrete source anchor per `120` words.
3. Evidence-crosslink gate:
   - critical sections: `>= 3` links to conformance tests/fixtures/artifacts.
   - high sections: `>= 2` links.
   - medium sections: `>= 1` link.
4. Mode-split gate (where applicable): explicit strict vs hardened behavior table is present.
5. Drift gate (where applicable): section states measurable parity or risk budget thresholds.

## Acceptance Mapping to `bd-3dl.23.1`

- Gap matrix coverage: all `##` and `###` sections in both docs are included above.
- Quantitative targets: each non-container section has explicit `current_words`, `multiplier`, and `target_words_floor`.
- High-risk prioritization: ordered queue provided for early passes (`23.2`, `23.3`, `23.4`, then `23.5+`).
