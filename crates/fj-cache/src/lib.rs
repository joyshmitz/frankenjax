#![forbid(unsafe_code)]

use fj_core::{CompatibilityMode, Jaxpr, Transform};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheKeyInput {
    pub mode: CompatibilityMode,
    pub backend: String,
    pub jaxpr: Jaxpr,
    pub transform_stack: Vec<Transform>,
    pub compile_options: BTreeMap<String, String>,
    pub custom_hook: Option<String>,
    pub unknown_incompatible_features: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheKey {
    pub namespace: &'static str,
    pub digest_hex: String,
}

impl CacheKey {
    #[must_use]
    pub fn as_string(&self) -> String {
        format!("{}-{}", self.namespace, self.digest_hex)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheKeyError {
    UnknownIncompatibleFeatures { features: Vec<String> },
}

impl std::fmt::Display for CacheKeyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownIncompatibleFeatures { features } => {
                write!(
                    f,
                    "strict mode rejected unknown incompatible features: {}",
                    features.join(",")
                )
            }
        }
    }
}

impl std::error::Error for CacheKeyError {}

pub fn build_cache_key(input: &CacheKeyInput) -> Result<CacheKey, CacheKeyError> {
    if input.mode == CompatibilityMode::Strict && !input.unknown_incompatible_features.is_empty() {
        return Err(CacheKeyError::UnknownIncompatibleFeatures {
            features: input.unknown_incompatible_features.clone(),
        });
    }

    let payload = canonical_payload(input);
    let mut hasher = Sha256::new();
    hasher.update(payload.as_bytes());
    let digest = hasher.finalize();

    Ok(CacheKey {
        namespace: "fjx",
        digest_hex: bytes_to_hex(&digest),
    })
}

#[must_use]
pub fn compatibility_matrix_row(input: &CacheKeyInput) -> String {
    format!(
        "mode={:?},backend={},unknown_features={}",
        input.mode,
        input.backend,
        input.unknown_incompatible_features.join(";")
    )
}

pub struct CacheKeyInputRef<'a> {
    pub mode: CompatibilityMode,
    pub backend: &'a str,
    pub jaxpr: &'a Jaxpr,
    pub transform_stack: &'a [Transform],
    pub compile_options: &'a BTreeMap<String, String>,
    pub custom_hook: Option<&'a str>,
    pub unknown_incompatible_features: &'a [String],
}

pub fn build_cache_key_ref(input: &CacheKeyInputRef<'_>) -> Result<CacheKey, CacheKeyError> {
    if input.mode == CompatibilityMode::Strict && !input.unknown_incompatible_features.is_empty() {
        return Err(CacheKeyError::UnknownIncompatibleFeatures {
            features: input.unknown_incompatible_features.to_vec(),
        });
    }

    // Stream canonical payload directly into the hasher to avoid allocating
    // an intermediate String. Each field is written with a separator prefix
    // to match the original canonical_payload_ref layout for compatibility.
    let mut hasher = Sha256::new();
    hash_canonical_payload_ref(&mut hasher, input);
    let digest = hasher.finalize();

    Ok(CacheKey {
        namespace: "fjx",
        digest_hex: bytes_to_hex(&digest),
    })
}

/// Write the canonical payload directly into a Digest, avoiding intermediate
/// String allocation. Layout matches `canonical_payload_ref` exactly.
fn hash_canonical_payload_ref(hasher: &mut Sha256, input: &CacheKeyInputRef<'_>) {
    use std::fmt::Write;

    // mode=<mode>|backend=<backend>|transforms=<t1,t2,...>|compile=<k1=v1;k2=v2>|hook=<hook>|unknown=<u1,u2>|jaxpr=<fp>
    let mut buf = String::new();
    let _ = write!(
        &mut buf,
        "mode={:?}|backend={}|transforms=",
        input.mode, input.backend
    );
    hasher.update(buf.as_bytes());

    for (i, t) in input.transform_stack.iter().enumerate() {
        if i > 0 {
            hasher.update(b",");
        }
        hasher.update(t.as_str().as_bytes());
    }

    hasher.update(b"|compile=");
    for (i, (key, value)) in input.compile_options.iter().enumerate() {
        if i > 0 {
            hasher.update(b";");
        }
        hasher.update(key.as_bytes());
        hasher.update(b"=");
        hasher.update(value.as_bytes());
    }

    hasher.update(b"|hook=");
    hasher.update(input.custom_hook.unwrap_or("none").as_bytes());

    hasher.update(b"|unknown=");
    for (i, feature) in input.unknown_incompatible_features.iter().enumerate() {
        if i > 0 {
            hasher.update(b",");
        }
        hasher.update(feature.as_bytes());
    }

    hasher.update(b"|jaxpr=");
    hasher.update(input.jaxpr.canonical_fingerprint().as_bytes());
}

fn canonical_payload(input: &CacheKeyInput) -> String {
    let transforms = input
        .transform_stack
        .iter()
        .map(|transform| transform.as_str())
        .collect::<Vec<_>>()
        .join(",");

    let compile_options = input
        .compile_options
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(";");

    let unknown = input.unknown_incompatible_features.join(",");

    format!(
        "mode={:?}|backend={}|transforms={}|compile={}|hook={}|unknown={}|jaxpr={}",
        input.mode,
        input.backend,
        transforms,
        compile_options,
        input.custom_hook.as_deref().unwrap_or("none"),
        unknown,
        input.jaxpr.canonical_fingerprint(),
    )
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    const HEX_LUT: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX_LUT[(byte >> 4) as usize] as char);
        out.push(HEX_LUT[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{CacheKeyError, CacheKeyInput, build_cache_key};
    use fj_core::{CompatibilityMode, Jaxpr, Transform};
    use proptest::prelude::*;
    use std::collections::BTreeMap;

    fn empty_jaxpr() -> Jaxpr {
        Jaxpr::new(vec![], vec![], vec![], vec![])
    }

    #[test]
    fn strict_mode_rejects_unknown_features() {
        let input = CacheKeyInput {
            mode: CompatibilityMode::Strict,
            backend: "cpu".to_owned(),
            jaxpr: empty_jaxpr(),
            transform_stack: vec![],
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec!["mystery_field".to_owned()],
        };

        let err = build_cache_key(&input).expect_err("should reject unknown features");
        assert_eq!(
            err,
            CacheKeyError::UnknownIncompatibleFeatures {
                features: vec!["mystery_field".to_owned()],
            }
        );
    }

    #[test]
    fn hardened_mode_accepts_unknown_features() {
        let input = CacheKeyInput {
            mode: CompatibilityMode::Hardened,
            backend: "cpu".to_owned(),
            jaxpr: empty_jaxpr(),
            transform_stack: vec![],
            compile_options: BTreeMap::new(),
            custom_hook: None,
            unknown_incompatible_features: vec!["mystery_field".to_owned()],
        };

        let key = build_cache_key(&input).expect("hardened mode should hash unknown features");
        assert!(key.as_string().starts_with("fjx-"));
    }

    #[test]
    fn cache_key_is_stable_for_identical_inputs() {
        let input = CacheKeyInput {
            mode: CompatibilityMode::Strict,
            backend: "cpu".to_owned(),
            jaxpr: empty_jaxpr(),
            transform_stack: vec![],
            compile_options: BTreeMap::new(),
            custom_hook: Some("hook-a".to_owned()),
            unknown_incompatible_features: vec![],
        };

        let key_a = build_cache_key(&input).expect("key generation should succeed");
        let key_b = build_cache_key(&input).expect("key generation should succeed");
        assert_eq!(key_a, key_b);
    }

    #[test]
    fn test_cache_test_log_schema_contract() {
        let fixture_id =
            fj_test_utils::fixture_id_from_json(&("cpu", "strict")).expect("fixture digest");
        let log = fj_test_utils::TestLogV1::unit(
            fj_test_utils::test_id(module_path!(), "test_cache_test_log_schema_contract"),
            fixture_id,
            fj_test_utils::TestMode::Strict,
            fj_test_utils::TestResult::Pass,
        );
        assert_eq!(log.schema_version, fj_test_utils::TEST_LOG_SCHEMA_VERSION);
    }

    fn arb_transform() -> impl Strategy<Value = Transform> {
        prop_oneof![
            Just(Transform::Jit),
            Just(Transform::Grad),
            Just(Transform::Vmap),
        ]
    }

    #[test]
    fn streaming_hash_matches_owned_hash() {
        use super::{CacheKeyInputRef, build_cache_key_ref};
        use fj_core::ProgramSpec;

        // Test with a real program and transforms
        let jaxpr = fj_core::build_program(ProgramSpec::SquarePlusLinear);
        let transforms = vec![Transform::Jit, Transform::Grad];
        let compile_options = BTreeMap::new();
        let unknown: Vec<String> = vec![];

        let owned_input = CacheKeyInput {
            mode: CompatibilityMode::Strict,
            backend: "cpu".to_owned(),
            jaxpr: jaxpr.clone(),
            transform_stack: transforms.clone(),
            compile_options: compile_options.clone(),
            custom_hook: Some("test-hook".to_owned()),
            unknown_incompatible_features: unknown.clone(),
        };

        let ref_input = CacheKeyInputRef {
            mode: CompatibilityMode::Strict,
            backend: "cpu",
            jaxpr: &jaxpr,
            transform_stack: &transforms,
            compile_options: &compile_options,
            custom_hook: Some("test-hook"),
            unknown_incompatible_features: &unknown,
        };

        let owned_key = build_cache_key(&owned_input).expect("owned key should succeed");
        let ref_key = build_cache_key_ref(&ref_input).expect("ref key should succeed");
        assert_eq!(
            owned_key, ref_key,
            "streaming (ref) and owned cache key must produce identical hashes"
        );
    }

    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(
            fj_test_utils::property_test_case_count()
        ))]
        #[test]
        fn prop_cache_key_stability(
            backend in "[a-z]{3,6}",
            transforms in proptest::collection::vec(arb_transform(), 0..3),
        ) {
            let _seed = fj_test_utils::capture_proptest_seed();
            let input = CacheKeyInput {
                mode: CompatibilityMode::Hardened,
                backend,
                jaxpr: empty_jaxpr(),
                transform_stack: transforms,
                compile_options: BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: vec![],
            };
            let key_a = build_cache_key(&input).expect("key gen");
            let key_b = build_cache_key(&input).expect("key gen");
            prop_assert_eq!(key_a, key_b);
        }
    }
}
