#![forbid(unsafe_code)]

//! Cache key stability test harness.
//!
//! Detects accidental cache key drift by comparing generated keys against
//! golden reference values. If the canonical payload layout or hash function
//! changes, these tests fail immediately — preventing silent cache invalidation
//! across FrankenJAX versions (threat matrix: "Stale artifact serving").

use crate::{CacheKeyInput, build_cache_key};
use fj_core::{CompatibilityMode, Jaxpr, Transform};
use std::collections::BTreeMap;

/// A golden cache key reference for stability testing.
#[derive(Debug, Clone)]
pub struct GoldenKeyRef {
    /// Human-readable description of this test vector.
    pub description: &'static str,
    /// Expected hex digest (without 'fjx-' prefix).
    pub expected_digest_hex: &'static str,
    /// The input that should produce this key.
    pub input: CacheKeyInput,
}

/// Generate the standard set of golden key references.
///
/// These represent canonical test vectors that must remain stable across
/// releases. If any golden key changes, it indicates a cache key format
/// change that requires a version bump in the namespace prefix.
#[must_use]
pub fn golden_key_refs() -> Vec<GoldenKeyRef> {
    vec![
        GoldenKeyRef {
            description: "empty program, strict mode, no transforms",
            expected_digest_hex: "b46e4618da22175a5f6b387b7a466be89c100263173cb4203e43b31756066cf9",
            input: CacheKeyInput {
                mode: CompatibilityMode::Strict,
                backend: "cpu".to_owned(),
                jaxpr: Jaxpr::new(vec![], vec![], vec![], vec![]),
                transform_stack: vec![],
                compile_options: BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: vec![],
            },
        },
        GoldenKeyRef {
            description: "empty program, hardened mode, jit transform",
            expected_digest_hex: "2631f1bb02ed6a0e66969d3826d43fbe4fd1a80e67fbc2f64a8873dba118c3e9",
            input: CacheKeyInput {
                mode: CompatibilityMode::Hardened,
                backend: "cpu".to_owned(),
                jaxpr: Jaxpr::new(vec![], vec![], vec![], vec![]),
                transform_stack: vec![Transform::Jit],
                compile_options: BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: vec![],
            },
        },
        GoldenKeyRef {
            description: "empty program, strict mode, custom hook",
            expected_digest_hex: "5a5f9f3067b01c0cc4b4adc12c891db9d3e647c7ce7f03618c303b8241ade71c",
            input: CacheKeyInput {
                mode: CompatibilityMode::Strict,
                backend: "cpu".to_owned(),
                jaxpr: Jaxpr::new(vec![], vec![], vec![], vec![]),
                transform_stack: vec![],
                compile_options: BTreeMap::new(),
                custom_hook: Some("my-hook".to_owned()),
                unknown_incompatible_features: vec![],
            },
        },
    ]
}

/// Compute current keys for all golden references and return them as
/// `(description, current_digest_hex)` pairs.
///
/// Use this to capture initial golden values or detect drift.
#[must_use]
pub fn capture_golden_keys() -> Vec<(String, String)> {
    golden_key_refs()
        .into_iter()
        .map(|g| {
            let key = build_cache_key(&g.input).expect("golden ref input must be valid");
            (g.description.to_owned(), key.digest_hex)
        })
        .collect()
}

/// Verify that current key generation matches a set of previously captured
/// golden digests. Returns a list of mismatches (empty = all stable).
#[must_use]
pub fn verify_golden_keys(golden: &[(String, String)]) -> Vec<String> {
    let current = capture_golden_keys();
    let mut mismatches = Vec::new();

    for ((desc, expected), (_desc2, actual)) in golden.iter().zip(current.iter()) {
        if expected != actual {
            mismatches.push(format!(
                "DRIFT: {desc}: expected={expected}, actual={actual}"
            ));
        }
    }

    mismatches
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_keys_are_internally_consistent() {
        // Capture, then immediately verify — should always pass.
        let golden = capture_golden_keys();
        let mismatches = verify_golden_keys(&golden);
        assert!(
            mismatches.is_empty(),
            "golden key drift detected: {mismatches:?}"
        );
    }

    #[test]
    fn golden_refs_produce_valid_cache_keys() {
        for g in golden_key_refs() {
            let key = build_cache_key(&g.input).expect("golden ref should produce valid key");
            assert!(
                key.as_string().starts_with("fjx-"),
                "key should have fjx- prefix: {}",
                g.description
            );
            assert_eq!(
                key.digest_hex.len(),
                64,
                "SHA-256 hex digest should be 64 chars: {}",
                g.description
            );
        }
    }

    #[test]
    fn golden_refs_match_hardcoded_digests() {
        // Regression gate: if the cache key format changes, these will fail,
        // signaling that a namespace version bump is needed.
        for g in golden_key_refs() {
            assert!(
                !g.expected_digest_hex.is_empty(),
                "golden ref '{}' has empty expected_digest_hex — must be filled",
                g.description
            );
            let key = build_cache_key(&g.input).expect("golden ref should produce valid key");
            assert_eq!(
                key.digest_hex, g.expected_digest_hex,
                "cache key drift detected for '{}': expected={}, actual={}",
                g.description, g.expected_digest_hex, key.digest_hex
            );
        }
    }

    #[test]
    fn distinct_golden_refs_produce_distinct_keys() {
        let keys = capture_golden_keys();
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                assert_ne!(
                    keys[i].1, keys[j].1,
                    "golden refs should produce distinct keys: '{}' vs '{}'",
                    keys[i].0, keys[j].0
                );
            }
        }
    }
}
