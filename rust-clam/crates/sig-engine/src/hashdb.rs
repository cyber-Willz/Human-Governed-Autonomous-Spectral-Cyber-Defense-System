//! Exact-hash signatures (ClamAV's .hdb/.hsb equivalent), keyed for O(1)
//! lookup instead of ClamAV's sorted-array binary search.

use crate::error::{Result, SigError};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct HashHit {
    pub name: String,
    pub expected_size: Option<u64>,
}

#[derive(Debug, Default)]
pub struct HashDb {
    md5: HashMap<[u8; 16], HashHit>,
    sha1: HashMap<[u8; 20], HashHit>,
    sha256: HashMap<[u8; 32], HashHit>,
}

impl HashDb {
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse a `.hdb`-style line: `hexhash:size:name` (size may be `*`).
    pub fn add_line(&mut self, line_no: usize, line: &str) -> Result<()> {
        let mut parts = line.splitn(3, ':');
        let hash_hex = parts.next().ok_or_else(|| SigError::BadHashSig {
            line: line_no,
            reason: "missing hash field".into(),
        })?;
        let size_field = parts.next().ok_or_else(|| SigError::BadHashSig {
            line: line_no,
            reason: "missing size field".into(),
        })?;
        let name = parts.next().ok_or_else(|| SigError::BadHashSig {
            line: line_no,
            reason: "missing name field".into(),
        })?;

        let expected_size = if size_field == "*" {
            None
        } else {
            Some(
                size_field
                    .parse::<u64>()
                    .map_err(|_| SigError::BadHashSig {
                        line: line_no,
                        reason: format!("invalid size '{size_field}'"),
                    })?,
            )
        };
        let hit = HashHit {
            name: name.to_string(),
            expected_size,
        };

        let raw = hex_decode(hash_hex).ok_or_else(|| SigError::BadHashSig {
            line: line_no,
            reason: "invalid hex hash".into(),
        })?;
        match raw.len() {
            16 => {
                let mut key = [0u8; 16];
                key.copy_from_slice(&raw);
                self.md5.insert(key, hit);
            }
            20 => {
                let mut key = [0u8; 20];
                key.copy_from_slice(&raw);
                self.sha1.insert(key, hit);
            }
            32 => {
                let mut key = [0u8; 32];
                key.copy_from_slice(&raw);
                self.sha256.insert(key, hit);
            }
            n => {
                return Err(SigError::BadHashSig {
                    line: line_no,
                    reason: format!("unsupported hash length {n} bytes"),
                })
            }
        }
        Ok(())
    }

    pub fn lookup_md5(&self, digest: &[u8; 16], size: u64) -> Option<&HashHit> {
        self.md5.get(digest).filter(|h| size_matches(h, size))
    }
    pub fn lookup_sha1(&self, digest: &[u8; 20], size: u64) -> Option<&HashHit> {
        self.sha1.get(digest).filter(|h| size_matches(h, size))
    }
    pub fn lookup_sha256(&self, digest: &[u8; 32], size: u64) -> Option<&HashHit> {
        self.sha256.get(digest).filter(|h| size_matches(h, size))
    }

    pub fn len(&self) -> usize {
        self.md5.len() + self.sha1.len() + self.sha256.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn size_matches(hit: &HashHit, size: u64) -> bool {
    hit.expected_size.map(|s| s == size).unwrap_or(true)
}

fn hex_decode(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let byte = u8::from_str_radix(&s[i..i + 2], 16).ok()?;
        out.push(byte);
        i += 2;
    }
    Some(out)
}
