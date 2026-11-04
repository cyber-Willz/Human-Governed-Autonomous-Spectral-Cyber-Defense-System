//! Integrity checks applied before any downloaded signature file is ever
//! written into the daemon's live signature directory.
//!
//! Two independent layers:
//! - **Per-file SHA-256**, checked against the manifest, so a corrupted or
//!   substituted individual file download is caught even if the manifest
//!   fetch itself was fine.
//! - **Optional Ed25519 signature over the manifest bytes themselves**, so
//!   that -- given an operator-configured public key -- a compromised or
//!   spoofed *distribution point* (not just a flaky download) can't
//!   silently swap in a manifest pointing at malicious files. TLS on the
//!   manifest fetch already defends against on-path tampering; this
//!   defends against a compromised origin server or CDN cache, which TLS
//!   alone does not.

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use sha2::{Digest, Sha256};

#[derive(Debug, thiserror::Error)]
pub enum VerifyError {
    #[error("sha256 mismatch: expected {expected}, got {actual}")]
    HashMismatch { expected: String, actual: String },
    #[error("invalid public key")]
    InvalidPublicKey,
    #[error("invalid signature encoding")]
    InvalidSignatureEncoding,
    #[error("manifest signature verification failed")]
    SignatureInvalid,
    #[error("hex decode error: {0}")]
    Hex(#[from] hex::FromHexError),
}

pub fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

/// Verifies `data` against an expected lowercase-hex SHA-256 digest.
/// Comparison is on the decoded bytes (not string comparison) so case
/// differences in the manifest's hex don't cause spurious rejections.
pub fn verify_sha256(data: &[u8], expected_hex: &str) -> Result<(), VerifyError> {
    let actual = sha256_hex(data);
    let expected_bytes = hex::decode(expected_hex)?;
    let actual_bytes = hex::decode(&actual)?;
    if expected_bytes == actual_bytes {
        Ok(())
    } else {
        Err(VerifyError::HashMismatch {
            expected: expected_hex.to_string(),
            actual,
        })
    }
}

/// Verifies a detached Ed25519 signature over `manifest_bytes`.
/// `public_key_hex` and `signature_hex` are both lowercase hex: 32 bytes
/// (64 hex chars) and 64 bytes (128 hex chars) respectively.
pub fn verify_manifest_signature(
    manifest_bytes: &[u8],
    public_key_hex: &str,
    signature_hex: &str,
) -> Result<(), VerifyError> {
    let key_bytes = hex::decode(public_key_hex)?;
    let key_arr: [u8; 32] = key_bytes
        .try_into()
        .map_err(|_| VerifyError::InvalidPublicKey)?;
    let verifying_key =
        VerifyingKey::from_bytes(&key_arr).map_err(|_| VerifyError::InvalidPublicKey)?;

    let sig_bytes = hex::decode(signature_hex)?;
    let sig_arr: [u8; 64] = sig_bytes
        .try_into()
        .map_err(|_| VerifyError::InvalidSignatureEncoding)?;
    let signature = Signature::from_bytes(&sig_arr);

    verifying_key
        .verify(manifest_bytes, &signature)
        .map_err(|_| VerifyError::SignatureInvalid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    #[test]
    fn sha256_matches_known_vector() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let digest = sha256_hex(b"");
        assert_eq!(
            digest,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn verify_sha256_accepts_correct_hash() {
        let data = b"hello world";
        let good = sha256_hex(data);
        assert!(verify_sha256(data, &good).is_ok());
    }

    #[test]
    fn verify_sha256_rejects_tampered_data() {
        let data = b"hello world";
        let good = sha256_hex(data);
        assert!(verify_sha256(b"goodbye world", &good).is_err());
    }

    #[test]
    fn verify_sha256_is_case_insensitive_on_hex() {
        let data = b"hello world";
        let good = sha256_hex(data).to_uppercase();
        assert!(verify_sha256(data, &good).is_ok());
    }

    #[test]
    fn manifest_signature_roundtrip() {
        let signing_key = SigningKey::from_bytes(&[7u8; 32]);
        let verifying_key = signing_key.verifying_key();
        let manifest_bytes = b"{\"version\":\"v1\",\"files\":[]}";
        let signature = signing_key.sign(manifest_bytes);

        let pubkey_hex = hex::encode(verifying_key.to_bytes());
        let sig_hex = hex::encode(signature.to_bytes());

        assert!(verify_manifest_signature(manifest_bytes, &pubkey_hex, &sig_hex).is_ok());
    }

    #[test]
    fn manifest_signature_rejects_tampered_manifest() {
        let signing_key = SigningKey::from_bytes(&[7u8; 32]);
        let verifying_key = signing_key.verifying_key();
        let manifest_bytes = b"{\"version\":\"v1\",\"files\":[]}";
        let signature = signing_key.sign(manifest_bytes);

        let pubkey_hex = hex::encode(verifying_key.to_bytes());
        let sig_hex = hex::encode(signature.to_bytes());

        let tampered = b"{\"version\":\"v2-evil\",\"files\":[]}";
        assert!(verify_manifest_signature(tampered, &pubkey_hex, &sig_hex).is_err());
    }

    #[test]
    fn manifest_signature_rejects_wrong_key() {
        let signing_key = SigningKey::from_bytes(&[7u8; 32]);
        let manifest_bytes = b"{\"version\":\"v1\",\"files\":[]}";
        let signature = signing_key.sign(manifest_bytes);

        let wrong_key = SigningKey::from_bytes(&[9u8; 32]).verifying_key();
        let pubkey_hex = hex::encode(wrong_key.to_bytes());
        let sig_hex = hex::encode(signature.to_bytes());

        assert!(verify_manifest_signature(manifest_bytes, &pubkey_hex, &sig_hex).is_err());
    }
}
