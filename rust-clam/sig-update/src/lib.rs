//! Library half of the signature-update tool: manifest parsing, integrity
//! verification, and the atomic staging-directory swap. See
//! `src/bin/rclam_sigupdate.rs` for the CLI that wires these together with
//! an actual HTTP fetch, and this crate's own doc comments / README for the
//! threat model each piece addresses.

pub mod apply;
pub mod manifest;
pub mod verify;

pub use apply::{apply_update, rollback, ApplyError};
pub use manifest::{parse_manifest, Manifest, ManifestError, ManifestFile};
pub use verify::{sha256_hex, verify_manifest_signature, verify_sha256, VerifyError};
