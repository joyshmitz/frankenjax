# FJ-P2C-FOUNDATION Fail-Closed Audit (bd-3dl.2)

## Objective

Verify unknown incompatible feature handling is fail-closed in strict mode and only deviates in explicitly allowlisted hardened mode.

## Audit Surface

| Path | Entry point | Strict behavior | Hardened behavior | Evidence |
|---|---|---|---|---|
| Cache key creation (owned) | `fj_cache::build_cache_key` | returns `CacheKeyError::UnknownIncompatibleFeatures` when unknown list is non-empty | accepts and hashes unknown feature list | `crates/fj-cache/src/lib.rs` tests: `strict_mode_rejects_unknown_features`, `hardened_mode_accepts_unknown_features` |
| Cache key creation (borrowed) | `fj_cache::build_cache_key_ref` | returns `CacheKeyError::UnknownIncompatibleFeatures` under strict mode | accepts and hashes unknown feature list | `crates/fj-dispatch/src/lib.rs` tests: `strict_mode_rejects_unknown_features_fail_closed`, `hardened_mode_allowlists_unknown_features_for_auditable_progress` |
| Dispatch path | `fj_dispatch::dispatch` | rejects request before execution when strict+unknown features | allows request with allowlisted hardened behavior and records evidence | `crates/fj-dispatch/src/lib.rs` (`dispatch`, tests above) |

## Call-Site Enumeration

`rg -n "build_cache_key_ref\(|build_cache_key\(" crates -g'*.rs'` confirms the only runtime path into borrowed cache-key construction is `fj-dispatch::dispatch`.

## Findings

- Strict mode unknown-feature handling is fail-closed on every active runtime entry point.
- Hardened mode deviation is explicit and limited to allowlisted hashing behavior (`HD-1`).
- No ad hoc hardened-mode bypasses were found in active call paths.

## Residual Risk

- Additional entry points may appear as new crates are added; this audit must be rerun when cache-key APIs gain new call sites.
- Hardened behavior beyond `HD-1` remains deferred and must not be introduced without allowlist updates.
