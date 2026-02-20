#![forbid(unsafe_code)]

//! Serialization/deserialization of cached compilation artifacts.
//!
//! V1 scope: FrankenJAX does not persist compiled XLA artifacts (no XLA
//! backend). This module defines the wire format for future cache storage,
//! including integrity verification via SHA-256 digest embedding.
//!
//! Wire format (v1):
//!   [4 bytes: magic "FJC\x01"]
//!   [4 bytes: payload length (big-endian u32)]
//!   [N bytes: payload data]
//!   [32 bytes: SHA-256 digest of payload data]

use crate::backend::CachedArtifact;

/// Magic bytes identifying a FrankenJAX cache artifact file.
const MAGIC: &[u8; 4] = b"FJC\x01";

/// Minimum valid artifact size: magic(4) + length(4) + sha256(32) = 40 bytes.
const MIN_SIZE: usize = 4 + 4 + 32;

/// Errors during artifact serialization/deserialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PersistenceError {
    /// File is too small to contain a valid artifact.
    TooSmall { actual: usize },
    /// Magic bytes do not match expected header.
    BadMagic { actual: [u8; 4] },
    /// Payload length in header does not match actual data length.
    LengthMismatch { declared: u32, actual: u32 },
    /// SHA-256 integrity check failed.
    IntegrityMismatch {
        expected_hex: String,
        actual_hex: String,
    },
}

impl std::fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooSmall { actual } => {
                write!(f, "artifact too small: {actual} bytes (min {MIN_SIZE})")
            }
            Self::BadMagic { actual } => {
                write!(f, "bad magic bytes: {actual:?}")
            }
            Self::LengthMismatch { declared, actual } => {
                write!(
                    f,
                    "payload length mismatch: declared={declared}, actual={actual}"
                )
            }
            Self::IntegrityMismatch {
                expected_hex,
                actual_hex,
            } => {
                write!(
                    f,
                    "SHA-256 integrity mismatch: expected={expected_hex}, actual={actual_hex}"
                )
            }
        }
    }
}

impl std::error::Error for PersistenceError {}

/// Serialize a `CachedArtifact` into the wire format.
#[must_use]
pub fn serialize(artifact: &CachedArtifact) -> Vec<u8> {
    let payload_len = artifact.data.len() as u32;
    let digest = crate::sha256_bytes(&artifact.data);

    let mut buf = Vec::with_capacity(MIN_SIZE + artifact.data.len());
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&payload_len.to_be_bytes());
    buf.extend_from_slice(&artifact.data);
    buf.extend_from_slice(&digest);
    buf
}

/// Deserialize bytes into a `CachedArtifact`, verifying integrity.
pub fn deserialize(bytes: &[u8]) -> Result<CachedArtifact, PersistenceError> {
    if bytes.len() < MIN_SIZE {
        return Err(PersistenceError::TooSmall {
            actual: bytes.len(),
        });
    }

    // Check magic.
    let mut magic = [0u8; 4];
    magic.copy_from_slice(&bytes[0..4]);
    if &magic != MAGIC {
        return Err(PersistenceError::BadMagic { actual: magic });
    }

    // Read declared payload length.
    let mut len_bytes = [0u8; 4];
    len_bytes.copy_from_slice(&bytes[4..8]);
    let declared_len = u32::from_be_bytes(len_bytes);

    // Verify total size consistency.
    let expected_total = 4 + 4 + declared_len as usize + 32;
    if bytes.len() != expected_total {
        return Err(PersistenceError::LengthMismatch {
            declared: declared_len,
            actual: (bytes.len() - MIN_SIZE) as u32,
        });
    }

    // Extract payload and stored digest.
    let payload = &bytes[8..8 + declared_len as usize];
    let stored_digest = &bytes[8 + declared_len as usize..];

    // Verify integrity.
    let computed_digest = crate::sha256_bytes(payload);
    if stored_digest != computed_digest.as_slice() {
        return Err(PersistenceError::IntegrityMismatch {
            expected_hex: crate::bytes_to_hex(stored_digest),
            actual_hex: crate::bytes_to_hex(&computed_digest),
        });
    }

    Ok(CachedArtifact {
        data: payload.to_vec(),
        integrity_sha256_hex: crate::bytes_to_hex(&computed_digest),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_artifact(data: &[u8]) -> CachedArtifact {
        CachedArtifact {
            data: data.to_vec(),
            integrity_sha256_hex: crate::sha256_hex(data),
        }
    }

    #[test]
    fn round_trip_serialization() {
        let artifact = test_artifact(b"hello world");
        let bytes = serialize(&artifact);
        let restored = deserialize(&bytes).expect("should deserialize");
        assert_eq!(restored.data, artifact.data);
        assert_eq!(restored.integrity_sha256_hex, artifact.integrity_sha256_hex);
    }

    #[test]
    fn rejects_too_small() {
        let err = deserialize(b"tiny").unwrap_err();
        assert!(matches!(err, PersistenceError::TooSmall { .. }));
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = serialize(&test_artifact(b"data"));
        bytes[0] = b'X'; // corrupt magic
        let err = deserialize(&bytes).unwrap_err();
        assert!(matches!(err, PersistenceError::BadMagic { .. }));
    }

    #[test]
    fn rejects_corrupted_payload() {
        let mut bytes = serialize(&test_artifact(b"data"));
        bytes[8] ^= 0xFF; // flip a payload byte
        let err = deserialize(&bytes).unwrap_err();
        assert!(matches!(err, PersistenceError::IntegrityMismatch { .. }));
    }

    #[test]
    fn rejects_truncated_file() {
        let bytes = serialize(&test_artifact(b"data"));
        let err = deserialize(&bytes[..bytes.len() - 1]).unwrap_err();
        assert!(matches!(err, PersistenceError::LengthMismatch { .. }));
    }
}
