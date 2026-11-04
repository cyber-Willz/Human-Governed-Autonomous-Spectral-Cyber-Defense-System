use crate::error::Result;
use crate::hashdb::HashDb;
use crate::hexsig::HexSignature;
use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use md5::{Digest, Md5};
use sha1::Sha1;
use sha2::Sha256;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatchKindResult {
    HexSignature,
    HashMd5,
    HashSha1,
    HashSha256,
}

#[derive(Debug, Clone)]
pub struct Detection {
    pub name: String,
    pub kind: MatchKindResult,
    pub offset: Option<usize>,
}

pub struct SignatureEngine {
    automaton: AhoCorasick,
    /// Parallel to automaton pattern indices.
    hex_sigs: Vec<HexSignature>,
    hash_db: HashDb,
}

pub struct SignatureEngineBuilder {
    hex_sigs: Vec<HexSignature>,
    hash_db: HashDb,
}

impl Default for SignatureEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SignatureEngineBuilder {
    pub fn new() -> Self {
        Self {
            hex_sigs: Vec::new(),
            hash_db: HashDb::new(),
        }
    }

    /// Load a `.ndb`-style database: lines of `name:hexpattern`.
    pub fn load_ndb(mut self, text: &str) -> Result<Self> {
        for (i, raw_line) in text.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (name, pattern) =
                line.split_once(':')
                    .ok_or_else(|| crate::error::SigError::BadHexSig {
                        line: i + 1,
                        reason: "expected 'name:pattern'".into(),
                    })?;
            let sig = HexSignature::parse(i + 1, name, pattern)?;
            self.hex_sigs.push(sig);
        }
        Ok(self)
    }

    /// Load a `.hdb`-style database: lines of `hexhash:size:name`.
    pub fn load_hdb(mut self, text: &str) -> Result<Self> {
        for (i, raw_line) in text.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            self.hash_db.add_line(i + 1, line)?;
        }
        Ok(self)
    }

    pub fn build(self) -> SignatureEngine {
        let literals: Vec<&[u8]> = self
            .hex_sigs
            .iter()
            .map(|s| s.anchor_literal.as_slice())
            .collect();
        // `Standard` match semantics are required for overlapping search;
        // we want every anchor occurrence (not just the leftmost-first
        // non-overlapping set) because a signature's anchor literal could
        // legitimately recur, and each occurrence needs independent
        // wildcard/gap verification.
        let automaton = AhoCorasickBuilder::new()
            .match_kind(MatchKind::Standard)
            .build(literals)
            .expect("automaton build should not fail on validated literals");

        SignatureEngine {
            automaton,
            hex_sigs: self.hex_sigs,
            hash_db: self.hash_db,
        }
    }
}

impl SignatureEngine {
    pub fn builder() -> SignatureEngineBuilder {
        SignatureEngineBuilder::new()
    }

    pub fn hex_sig_count(&self) -> usize {
        self.hex_sigs.len()
    }
    pub fn hash_sig_count(&self) -> usize {
        self.hash_db.len()
    }

    /// Scan an in-memory buffer against both the wildcard-pattern automaton
    /// and the whole-file hash database. Returns every confirmed detection
    /// (a file may legitimately trip more than one signature).
    pub fn scan_buffer(&self, buf: &[u8]) -> Vec<Detection> {
        let mut hits = Vec::new();

        // --- Hash-based detection (whole-buffer digests) ---
        if !self.hash_db.is_empty() {
            let size = buf.len() as u64;

            let md5_digest: [u8; 16] = Md5::digest(buf).into();
            if let Some(h) = self.hash_db.lookup_md5(&md5_digest, size) {
                hits.push(Detection {
                    name: h.name.clone(),
                    kind: MatchKindResult::HashMd5,
                    offset: None,
                });
            }
            let sha1_digest: [u8; 20] = Sha1::digest(buf).into();
            if let Some(h) = self.hash_db.lookup_sha1(&sha1_digest, size) {
                hits.push(Detection {
                    name: h.name.clone(),
                    kind: MatchKindResult::HashSha1,
                    offset: None,
                });
            }
            let sha256_digest: [u8; 32] = Sha256::digest(buf).into();
            if let Some(h) = self.hash_db.lookup_sha256(&sha256_digest, size) {
                hits.push(Detection {
                    name: h.name.clone(),
                    kind: MatchKindResult::HashSha256,
                    offset: None,
                });
            }
        }

        // --- Wildcard pattern detection via shared AC automaton ---
        // Every anchor-literal hit is a *candidate*; verify() confirms the
        // full pattern (including wildcards/gaps) before we call it a match.
        if !self.hex_sigs.is_empty() {
            for m in self.automaton.find_overlapping_iter(buf) {
                let sig_idx = m.pattern().as_usize();
                let sig = &self.hex_sigs[sig_idx];
                if let Some(start) = sig.verify(buf, m.start()) {
                    hits.push(Detection {
                        name: sig.name.clone(),
                        kind: MatchKindResult::HexSignature,
                        offset: Some(start),
                    });
                }
            }
        }

        hits
    }
}
