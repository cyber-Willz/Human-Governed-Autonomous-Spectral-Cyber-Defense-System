//! The update manifest format: a small JSON document listing every
//! signature file that should exist after an update, alongside its SHA-256
//! so corrupted or tampered downloads are caught before ever being loaded
//! into a running daemon.
//!
//! ```json
//! {
//!   "version": "2026.07.06-01",
//!   "files": [
//!     { "name": "main.ndb", "sha256": "…", "url": "https://example.com/sigs/main.ndb" },
//!     { "name": "daily.hdb", "sha256": "…", "url": "https://example.com/sigs/daily.hdb" }
//!   ]
//! }
//! ```

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct ManifestFile {
    pub name: String,
    pub sha256: String,
    pub url: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct Manifest {
    pub version: String,
    pub files: Vec<ManifestFile>,
}

#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error("invalid manifest JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("manifest lists no files")]
    Empty,
    #[error("duplicate file name in manifest: {0}")]
    DuplicateName(String),
}

pub fn parse_manifest(json: &str) -> Result<Manifest, ManifestError> {
    let manifest: Manifest = serde_json::from_str(json)?;
    if manifest.files.is_empty() {
        return Err(ManifestError::Empty);
    }
    let mut seen = std::collections::HashSet::new();
    for f in &manifest.files {
        if !seen.insert(&f.name) {
            return Err(ManifestError::DuplicateName(f.name.clone()));
        }
    }
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_well_formed_manifest() {
        let json = r#"{
            "version": "2026.07.06-01",
            "files": [
                {"name": "main.ndb", "sha256": "abc123", "url": "https://example.com/main.ndb"}
            ]
        }"#;
        let m = parse_manifest(json).unwrap();
        assert_eq!(m.version, "2026.07.06-01");
        assert_eq!(m.files.len(), 1);
    }

    #[test]
    fn rejects_empty_file_list() {
        let json = r#"{"version": "v1", "files": []}"#;
        assert!(matches!(parse_manifest(json), Err(ManifestError::Empty)));
    }

    #[test]
    fn rejects_duplicate_names() {
        let json = r#"{
            "version": "v1",
            "files": [
                {"name": "a.ndb", "sha256": "x", "url": "https://e/a"},
                {"name": "a.ndb", "sha256": "y", "url": "https://e/b"}
            ]
        }"#;
        assert!(matches!(
            parse_manifest(json),
            Err(ManifestError::DuplicateName(_))
        ));
    }

    #[test]
    fn rejects_malformed_json() {
        assert!(parse_manifest("not json at all").is_err());
    }
}
