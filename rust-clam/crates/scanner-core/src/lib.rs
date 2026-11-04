pub use archive_guard::GuardLimits;
use archive_guard::{extract_gzip, extract_tar, extract_zip, looks_like_tar, GuardBudget};
use memmap2::Mmap;
use pe_analyze::Heuristics;
use rayon::prelude::*;
use sig_engine::{Detection, SignatureEngine};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;
use walkdir::WalkDir;

#[derive(Debug, Error)]
pub enum ScanError {
    #[error("io error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("archive guard tripped: {0}")]
    Guard(#[from] archive_guard::GuardError),
}

#[derive(Debug, Clone)]
pub enum Verdict {
    Clean,
    Infected(Vec<Detection>),
    /// The file was not scanned at all -- e.g. it isn't a regular file
    /// (FIFO, socket, device node), or it exceeds the configured
    /// `max_file_size`. Distinct from `LimitExceeded`, which means a scan
    /// was *in progress* and got cut off by an archive-bomb guard.
    Skipped {
        reason: String,
    },
    /// Scan was aborted partway through because an archive-bomb style
    /// limit was hit. Any detections found before the abort are included.
    LimitExceeded {
        detections: Vec<Detection>,
        reason: String,
    },
}

#[derive(Debug, Clone)]
pub struct FileReport {
    pub path: PathBuf,
    /// Nested path within an archive, e.g. "outer.zip -> inner/payload.exe"
    pub logical_path: String,
    pub verdict: Verdict,
    pub pe_heuristics: Option<Heuristics>,
}

pub struct Scanner {
    engine: Arc<SignatureEngine>,
    limits: GuardLimits,
}

impl Scanner {
    pub fn new(engine: SignatureEngine) -> Self {
        Self {
            engine: Arc::new(engine),
            limits: GuardLimits::default(),
        }
    }

    pub fn with_limits(mut self, limits: GuardLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Scan a single filesystem path, recursing into archives as needed.
    pub fn scan_path(&self, path: &Path) -> Result<Vec<FileReport>, ScanError> {
        let logical_path = path.display().to_string();
        let skipped = |reason: String| {
            vec![FileReport {
                path: path.to_path_buf(),
                logical_path: logical_path.clone(),
                verdict: Verdict::Skipped { reason },
                pe_heuristics: None,
            }]
        };

        // Check the file type via metadata *before* opening. This matters
        // for more than tidiness: opening a FIFO for reading blocks until a
        // writer connects (that's POSIX FIFO semantics, not a bug in
        // `File::open`), so if we opened first and checked the type after,
        // a FIFO with no writer would hang the scan right here regardless
        // of any check we do afterwards. Stat-ing first means a FIFO,
        // socket, or device node is rejected without ever touching it.
        let metadata = std::fs::metadata(path).map_err(|e| ScanError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        if !metadata.is_file() {
            return Ok(skipped("not a regular file".to_string()));
        }

        let len = metadata.len();
        if len > self.limits.max_file_size {
            return Ok(skipped(format!(
                "file size {len} bytes exceeds max_file_size limit ({} bytes)",
                self.limits.max_file_size
            )));
        }

        let file = File::open(path).map_err(|e| ScanError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;

        // `Mmap::map` rejects zero-length mappings outright (there is
        // nothing to map), so an empty file has to be handled as a plain
        // empty buffer rather than going through mmap at all. An empty file
        // is legitimately just "clean" -- there's nothing in it to match
        // any signature, and it can't be a zip/gzip/PE (all of those magic
        // checks already require a nonzero minimum length).
        let mut budget = GuardBudget::new(self.limits);
        let mut reports = Vec::new();
        if len == 0 {
            self.scan_bytes(&[], path, logical_path, &mut budget, &mut reports);
            return Ok(reports);
        }

        // SAFETY-relevant note: mmap avoids a full-file heap copy for large
        // files. The mapping is read-only and the file is not modified
        // concurrently by this process; std file semantics apply for
        // external modification, same caveat as any mmap-based reader.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| ScanError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;

        self.scan_bytes(&mmap, path, logical_path, &mut budget, &mut reports);
        Ok(reports)
    }

    /// Scan a directory tree in parallel (one rayon task per file).
    pub fn scan_directory(&self, root: &Path) -> Vec<FileReport> {
        // `follow_links(false)` is the default, but it's called out
        // explicitly here because it's security-relevant: a directory tree
        // containing a symlink loop, or a symlink pointing outside the
        // intended scan root, must not cause unbounded traversal or scan
        // files the caller didn't intend to include.
        let files: Vec<PathBuf> = WalkDir::new(root)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| match e {
                Ok(entry) => Some(entry),
                Err(err) => {
                    // Permission-denied and similar per-entry errors
                    // shouldn't sink the whole directory scan, but they
                    // also shouldn't vanish silently -- surface them so an
                    // operator can tell "clean" apart from "couldn't read
                    // some files at all".
                    tracing::warn!(%err, "error while walking directory tree");
                    None
                }
            })
            .filter(|e| e.file_type().is_file())
            .map(|e| e.into_path())
            .collect();

        files
            .par_iter()
            .flat_map(|p| match self.scan_path(p) {
                Ok(reports) => reports,
                Err(e) => vec![FileReport {
                    path: p.clone(),
                    logical_path: p.display().to_string(),
                    verdict: Verdict::LimitExceeded {
                        detections: vec![],
                        reason: e.to_string(),
                    },
                    pe_heuristics: None,
                }],
            })
            .collect()
    }

    /// Core recursive scan over an in-memory buffer. Handles: signature
    /// matching, PE heuristic scoring, and zip-archive recursion (bounded
    /// by `budget`).
    fn scan_bytes(
        &self,
        buf: &[u8],
        real_path: &Path,
        logical_path: String,
        budget: &mut GuardBudget,
        out: &mut Vec<FileReport>,
    ) {
        let detections = self.engine.scan_buffer(buf);

        let pe_heuristics = if buf.len() >= 64 && &buf[0..2] == b"MZ" {
            pe_analyze::parse(buf)
                .ok()
                .map(|info| pe_analyze::heuristics(&info))
        } else {
            None
        };

        let verdict = if detections.is_empty() {
            Verdict::Clean
        } else {
            Verdict::Infected(detections)
        };

        out.push(FileReport {
            path: real_path.to_path_buf(),
            logical_path: logical_path.clone(),
            verdict,
            pe_heuristics,
        });

        // Zip magic: "PK\x03\x04" (also covers empty-archive and spanned
        // variants closely enough for this scanner's purposes).
        if buf.len() >= 4 && &buf[0..4] == b"PK\x03\x04" {
            if let Err(e) = budget.enter_archive() {
                out.push(FileReport {
                    path: real_path.to_path_buf(),
                    logical_path,
                    verdict: Verdict::LimitExceeded {
                        detections: vec![],
                        reason: e.to_string(),
                    },
                    pe_heuristics: None,
                });
                return;
            }

            let cursor = std::io::Cursor::new(buf);
            match extract_zip(cursor, budget) {
                Ok((entries, limit_hit)) => {
                    for entry in &entries {
                        let nested_logical = format!("{} -> {}", logical_path, entry.name);
                        self.scan_bytes(&entry.data, real_path, nested_logical, budget, out);
                    }
                    if let Some(err) = limit_hit {
                        out.push(FileReport {
                            path: real_path.to_path_buf(),
                            logical_path,
                            verdict: Verdict::LimitExceeded {
                                detections: vec![],
                                reason: err.to_string(),
                            },
                            pe_heuristics: None,
                        });
                    }
                }
                Err(e) => {
                    out.push(FileReport {
                        path: real_path.to_path_buf(),
                        logical_path,
                        verdict: Verdict::LimitExceeded {
                            detections: vec![],
                            reason: e.to_string(),
                        },
                        pe_heuristics: None,
                    });
                }
            }

            // Depth is released only now, after every descendant of this
            // archive (including nested archives, transitively) has been
            // fully scanned -- that's what makes max_depth an actual bound
            // on recursion depth rather than a per-call no-op.
            budget.leave_archive();
        } else if buf.len() >= 2 && buf[0] == 0x1F && buf[1] == 0x8B {
            // Gzip magic. Treated the same as a zip archive: depth-guarded
            // around the *entire* recursive descent into the decompressed
            // content (so a "gzip quine" -- a gzip stream whose payload is
            // itself gzip, nested arbitrarily deep -- is bounded by
            // max_depth exactly like nested zips are), with the decompressed
            // bytes fed back into scan_bytes as one logical child.
            if let Err(e) = budget.enter_archive() {
                out.push(FileReport {
                    path: real_path.to_path_buf(),
                    logical_path,
                    verdict: Verdict::LimitExceeded {
                        detections: vec![],
                        reason: e.to_string(),
                    },
                    pe_heuristics: None,
                });
                return;
            }

            let cursor = std::io::Cursor::new(buf);
            match extract_gzip(cursor, budget) {
                Ok((data, limit_hit)) => {
                    let nested_logical = format!("{} -> (gunzip)", logical_path);
                    self.scan_bytes(&data, real_path, nested_logical, budget, out);
                    if let Some(err) = limit_hit {
                        out.push(FileReport {
                            path: real_path.to_path_buf(),
                            logical_path,
                            verdict: Verdict::LimitExceeded {
                                detections: vec![],
                                reason: err.to_string(),
                            },
                            pe_heuristics: None,
                        });
                    }
                }
                Err(e) => {
                    out.push(FileReport {
                        path: real_path.to_path_buf(),
                        logical_path,
                        verdict: Verdict::LimitExceeded {
                            detections: vec![],
                            reason: e.to_string(),
                        },
                        pe_heuristics: None,
                    });
                }
            }

            budget.leave_archive();
        } else if looks_like_tar(buf) {
            // Tar magic ("ustar" at header offset 257). Handled the same
            // shape as zip/gzip: depth-guarded around the whole recursive
            // descent, entries fed back into scan_bytes individually so a
            // tar member that is itself an archive still recurses and is
            // still bounded by the same budget.
            if let Err(e) = budget.enter_archive() {
                out.push(FileReport {
                    path: real_path.to_path_buf(),
                    logical_path,
                    verdict: Verdict::LimitExceeded {
                        detections: vec![],
                        reason: e.to_string(),
                    },
                    pe_heuristics: None,
                });
                return;
            }

            let cursor = std::io::Cursor::new(buf);
            match extract_tar(cursor, budget) {
                Ok((entries, limit_hit)) => {
                    for entry in &entries {
                        let nested_logical = format!("{} -> {}", logical_path, entry.name);
                        self.scan_bytes(&entry.data, real_path, nested_logical, budget, out);
                    }
                    if let Some(err) = limit_hit {
                        out.push(FileReport {
                            path: real_path.to_path_buf(),
                            logical_path,
                            verdict: Verdict::LimitExceeded {
                                detections: vec![],
                                reason: err.to_string(),
                            },
                            pe_heuristics: None,
                        });
                    }
                }
                Err(e) => {
                    out.push(FileReport {
                        path: real_path.to_path_buf(),
                        logical_path,
                        verdict: Verdict::LimitExceeded {
                            detections: vec![],
                            reason: e.to_string(),
                        },
                        pe_heuristics: None,
                    });
                }
            }

            budget.leave_archive();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use archive_guard::GuardLimits;
    use sig_engine::SignatureEngine;
    use std::io::Write;
    use tempfile::tempdir;

    fn engine_with(ndb: &str, hdb: &str) -> SignatureEngine {
        let mut b = SignatureEngine::builder();
        if !ndb.is_empty() {
            b = b.load_ndb(ndb).unwrap();
        }
        if !hdb.is_empty() {
            b = b.load_hdb(hdb).unwrap();
        }
        b.build()
    }

    fn zip_bytes(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut buf);
            let mut w = zip::ZipWriter::new(cursor);
            let opts = zip::write::FileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            for (name, data) in entries {
                w.start_file(*name, opts).unwrap();
                w.write_all(data).unwrap();
            }
            w.finish().unwrap();
        }
        buf
    }

    fn tar_bytes(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut builder = tar::Builder::new(Vec::new());
        for (name, data) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_size(data.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder.append_data(&mut header, name, *data).unwrap();
        }
        builder.into_inner().unwrap()
    }

    #[test]
    fn tar_archive_is_scanned_and_finds_signature() {
        let dir = tempdir().unwrap();
        let tar = tar_bytes(&[
            ("clean.txt", b"nothing here"),
            ("payload.txt", b"contains test marker"),
        ]);
        let path = dir.path().join("bundle.tar");
        std::fs::write(&path, tar).unwrap();

        let engine = engine_with("Sig.Test:74657374", "");
        let scanner = Scanner::new(engine);
        let reports = scanner.scan_path(&path).unwrap();

        let infected_logical_paths: Vec<_> = reports
            .iter()
            .filter(|r| matches!(r.verdict, Verdict::Infected(_)))
            .map(|r| r.logical_path.clone())
            .collect();
        assert!(
            infected_logical_paths.iter().any(|p| p.contains("payload.txt")),
            "expected a detection inside bundle.tar -> payload.txt, got: {infected_logical_paths:?}"
        );
    }

    #[test]
    fn tar_gz_is_transparently_unwrapped_and_scanned() {
        use flate2::write::GzEncoder;
        use flate2::Compression;

        let dir = tempdir().unwrap();
        let tar = tar_bytes(&[("payload.txt", b"contains test marker")]);
        let mut enc = GzEncoder::new(Vec::new(), Compression::default());
        enc.write_all(&tar).unwrap();
        let gz = enc.finish().unwrap();
        let path = dir.path().join("bundle.tar.gz");
        std::fs::write(&path, gz).unwrap();

        let engine = engine_with("Sig.Test:74657374", "");
        let scanner = Scanner::new(engine);
        let reports = scanner.scan_path(&path).unwrap();

        let infected = reports.iter().any(
            |r| matches!(&r.verdict, Verdict::Infected(dets) if dets.iter().any(|d| d.name == "Sig.Test")),
        );
        assert!(
            infected,
            "expected a detection inside gunzipped tar payload, got: {reports:?}"
        );
    }

    #[test]
    fn clean_file_scans_clean() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("clean.txt");
        std::fs::write(&path, b"nothing to see here").unwrap();

        let engine = engine_with("Sig.Test:74657374", ""); // "test"
        let scanner = Scanner::new(engine);
        let reports = scanner.scan_path(&path).unwrap();
        assert_eq!(reports.len(), 1);
        assert!(matches!(reports[0].verdict, Verdict::Clean));
    }

    #[test]
    fn wildcard_signature_flags_infected_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("evil.bin");
        std::fs::write(&path, b"XXX test YYY").unwrap();

        let engine = engine_with("Sig.Test:74657374", "");
        let scanner = Scanner::new(engine);
        let reports = scanner.scan_path(&path).unwrap();
        assert_eq!(reports.len(), 1);
        match &reports[0].verdict {
            Verdict::Infected(dets) => {
                assert_eq!(dets.len(), 1);
                assert_eq!(dets[0].name, "Sig.Test");
            }
            other => panic!("expected Infected, got {other:?}"),
        }
    }

    #[test]
    fn hash_signature_flags_by_exact_md5() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("known_bad.bin");
        let content = b"exact bytes for hash match";
        std::fs::write(&path, content).unwrap();

        let digest_bytes: [u8; 16] = {
            use md5::{Digest, Md5};
            Md5::digest(content).into()
        };
        let digest = digest_bytes.iter().fold(String::new(), |mut acc, b| {
            use std::fmt::Write;
            let _ = write!(acc, "{b:02x}");
            acc
        });
        let hdb = format!("{}:{}:Sig.HashHit\n", digest, content.len());

        let engine = engine_with("", &hdb);
        let scanner = Scanner::new(engine);
        let reports = scanner.scan_path(&path).unwrap();
        match &reports[0].verdict {
            Verdict::Infected(dets) => assert_eq!(dets[0].name, "Sig.HashHit"),
            other => panic!("expected Infected, got {other:?}"),
        }
    }

    #[test]
    fn recurses_into_nested_zip_and_finds_signature() {
        let dir = tempdir().unwrap();
        let inner = zip_bytes(&[("payload.txt", b"contains test marker")]);
        let outer = zip_bytes(&[("inner.zip", &inner)]);
        let path = dir.path().join("outer.zip");
        std::fs::write(&path, outer).unwrap();

        let engine = engine_with("Sig.Test:74657374", "");
        let scanner = Scanner::new(engine);
        let reports = scanner.scan_path(&path).unwrap();

        let infected_logical_paths: Vec<_> = reports
            .iter()
            .filter(|r| matches!(r.verdict, Verdict::Infected(_)))
            .map(|r| r.logical_path.clone())
            .collect();
        assert!(
            infected_logical_paths.iter().any(|p| p.contains("payload.txt")),
            "expected a detection nested inside outer.zip -> inner.zip -> payload.txt, got: {infected_logical_paths:?}"
        );
    }

    #[test]
    fn depth_guard_actually_stops_deeply_nested_archives() {
        // Regression test for the bug where enter_archive/leave_archive
        // were scoped inside extract_zip (around the flat one-level
        // extraction) instead of around the caller's recursive descent,
        // which made max_depth a no-op against zip-in-zip nesting.
        let dir = tempdir().unwrap();
        let mut data = b"deepest payload with test marker".to_vec();
        let depth = 20usize;
        for i in 0..depth {
            let name = if i == 0 { "leaf.bin" } else { "inner.zip" };
            data = zip_bytes(&[(name, &data)]);
        }
        let path = dir.path().join("deep.zip");
        std::fs::write(&path, &data).unwrap();

        let engine = engine_with("Sig.Test:74657374", "");
        let scanner = Scanner::new(engine).with_limits(GuardLimits {
            max_depth: 5,
            ..GuardLimits::default()
        });
        let reports = scanner.scan_path(&path).unwrap();

        let hit_limit = reports.iter().any(|r| {
            matches!(&r.verdict, Verdict::LimitExceeded { reason, .. } if reason.contains("depth"))
        });
        assert!(
            hit_limit,
            "expected a depth-exceeded LimitExceeded report among: {reports:?}"
        );

        // And it must have stopped well short of unpacking all 20 levels --
        // otherwise the guard did nothing.
        assert!(
            reports.len() < depth,
            "scanned {} reports for a {}-level archive with max_depth=5 -- depth guard did not stop recursion",
            reports.len(),
            depth
        );
    }

    #[test]
    fn zip_bomb_ratio_guard_stops_extraction() {
        let dir = tempdir().unwrap();
        let bomb = zip_bytes(&[("bomb.bin", &vec![0u8; 20 * 1024 * 1024])]);
        let path = dir.path().join("bomb.zip");
        std::fs::write(&path, bomb).unwrap();

        let engine = engine_with("", "");
        let scanner = Scanner::new(engine).with_limits(GuardLimits {
            max_ratio: 1000,
            ..GuardLimits::default()
        });
        let reports = scanner.scan_path(&path).unwrap();
        assert!(reports.iter().any(|r| matches!(
            &r.verdict,
            Verdict::LimitExceeded { reason, .. } if reason.contains("ratio")
        )));
    }

    #[test]
    fn directory_scan_covers_every_file() {
        let dir = tempdir().unwrap();
        for i in 0..5 {
            std::fs::write(dir.path().join(format!("f{i}.txt")), b"plain content").unwrap();
        }
        let engine = engine_with("", "");
        let scanner = Scanner::new(engine);
        let reports = scanner.scan_directory(dir.path());
        assert_eq!(reports.len(), 5);
        assert!(reports.iter().all(|r| matches!(r.verdict, Verdict::Clean)));
    }

    #[test]
    fn empty_file_scans_clean_instead_of_erroring() {
        // Mmap::map rejects zero-length mappings, so an empty file must be
        // special-cased rather than bubbling up as ScanError::Io.
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.bin");
        std::fs::write(&path, b"").unwrap();

        let engine = engine_with("Sig.Test:74657374", "");
        let scanner = Scanner::new(engine);
        let reports = scanner.scan_path(&path).unwrap();
        assert_eq!(reports.len(), 1);
        assert!(matches!(reports[0].verdict, Verdict::Clean));
    }

    #[test]
    fn oversized_file_is_skipped_not_scanned() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("big.bin");
        std::fs::write(&path, vec![0u8; 4096]).unwrap();

        let engine = engine_with("", "");
        let scanner = Scanner::new(engine).with_limits(GuardLimits {
            max_file_size: 1024,
            ..GuardLimits::default()
        });
        let reports = scanner.scan_path(&path).unwrap();
        assert_eq!(reports.len(), 1);
        assert!(
            matches!(&reports[0].verdict, Verdict::Skipped { reason } if reason.contains("max_file_size"))
        );
    }

    #[cfg(unix)]
    #[test]
    fn fifo_is_skipped_not_scanned() {
        // A FIFO can legitimately be opened, but must never be mmap'd or
        // read synchronously by the scanner -- doing so could block a scan
        // worker indefinitely on an empty pipe with no writer.
        let dir = tempdir().unwrap();
        let path = dir.path().join("a.fifo");
        let c_path = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
        let rc = unsafe { libc_mkfifo(c_path.as_ptr(), 0o600) };
        assert_eq!(rc, 0, "mkfifo failed");

        let engine = engine_with("", "");
        let scanner = Scanner::new(engine);
        let reports = scanner.scan_path(&path).unwrap();
        assert_eq!(reports.len(), 1);
        assert!(
            matches!(&reports[0].verdict, Verdict::Skipped { reason } if reason.contains("regular file"))
        );
    }

    // Minimal libc binding for the FIFO test above, avoiding a dependency on
    // the `libc` crate just for one syscall.
    #[cfg(unix)]
    extern "C" {
        #[link_name = "mkfifo"]
        fn libc_mkfifo(path: *const std::os::raw::c_char, mode: u32) -> i32;
    }

    #[test]
    fn gzip_stream_is_decompressed_and_scanned() {
        use flate2::write::GzEncoder;
        use flate2::Compression;

        let dir = tempdir().unwrap();
        let path = dir.path().join("payload.bin.gz");
        let mut enc = GzEncoder::new(Vec::new(), Compression::default());
        enc.write_all(b"this file contains a test marker inside")
            .unwrap();
        let gz_bytes = enc.finish().unwrap();
        std::fs::write(&path, gz_bytes).unwrap();

        let engine = engine_with("Sig.Test:74657374", "");
        let scanner = Scanner::new(engine);
        let reports = scanner.scan_path(&path).unwrap();

        let infected = reports
            .iter()
            .any(|r| matches!(&r.verdict, Verdict::Infected(dets) if dets.iter().any(|d| d.name == "Sig.Test")));
        assert!(
            infected,
            "expected a detection inside the decompressed gzip payload, got: {reports:?}"
        );
        // And the decompressed logical child should be distinguishable from
        // the outer .gz file itself in the report path.
        assert!(reports.iter().any(|r| r.logical_path.contains("(gunzip)")));
    }

    #[test]
    fn gzip_bomb_is_stopped_by_total_size_guard() {
        use flate2::write::GzEncoder;
        use flate2::Compression;

        let dir = tempdir().unwrap();
        let path = dir.path().join("bomb.gz");
        let mut enc = GzEncoder::new(Vec::new(), Compression::default());
        enc.write_all(&vec![0u8; 5 * 1024 * 1024]).unwrap();
        let gz_bytes = enc.finish().unwrap();
        std::fs::write(&path, gz_bytes).unwrap();

        let engine = engine_with("", "");
        let scanner = Scanner::new(engine).with_limits(GuardLimits {
            max_total_uncompressed: 1024,
            ..GuardLimits::default()
        });
        let reports = scanner.scan_path(&path).unwrap();
        assert!(reports.iter().any(|r| matches!(
            &r.verdict,
            Verdict::LimitExceeded { reason, .. } if reason.contains("uncompressed")
        )));
    }
}
