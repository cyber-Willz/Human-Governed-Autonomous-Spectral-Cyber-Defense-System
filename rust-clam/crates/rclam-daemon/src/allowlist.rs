//! Enforces *what* a connected client is allowed to ask `rclamd` to scan.
//!
//! Socket permissions (`--socket-mode`) control *who* can open a connection
//! to the daemon at all, but say nothing about what paths a connected
//! client may then ask it to `SCAN`. Without this module, any client that
//! can open the socket could ask the daemon -- which typically runs as a
//! service account with broad read access -- to scan `/etc/shadow`, another
//! tenant's uploads, or anything else the daemon's own user can read, and
//! infer something about the target from timing or the response (`OK` vs
//! `SKIPPED` vs `ERROR`) even without ever seeing the file's contents
//! directly. This is exactly the shared-daemon-multi-tenant threat model:
//! socket auth answers "can you talk to rclamd at all", this module answers
//! "can you ask rclamd about *this* path".
//!
//! The allowlist is a set of root directories configured by the operator at
//! startup (`--allow-root`). A requested path is permitted only if it
//! canonicalizes to somewhere underneath one of those roots. Deliberately
//! fails closed: if no roots are configured, every `SCAN`/`CONTSCAN` request
//! is rejected rather than silently defaulting to "allow everything", which
//! is what would happen if this check were merely additive/opt-in.

use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct PathAllowlist {
    /// Canonicalized allowed roots. Empty means "nothing is allowed" (fail
    /// closed), not "everything is allowed".
    roots: Vec<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Denial {
    /// No `--allow-root` was configured at all; the daemon is running with
    /// scanning disabled by default until an operator opts in.
    NoRootsConfigured,
    /// The path either doesn't exist, couldn't be resolved (e.g. a dangling
    /// symlink, a permissions error partway down the tree), or resolved to
    /// somewhere outside every configured root.
    ///
    /// These cases are deliberately *not* distinguished in the response
    /// text sent back to the client: telling an unprivileged client
    /// "doesn't exist" vs "exists but isn't allowed" vs "exists but we
    /// can't read a parent directory" would hand it a path-existence oracle
    /// against parts of the filesystem it has no business probing (e.g.
    /// distinguishing `/etc/shadow` from `/etc/shadowdoesnotexist`).
    NotPermitted,
}

impl PathAllowlist {
    /// Builds an allowlist from operator-supplied root directories.
    /// Each root is canonicalized eagerly at startup (not lazily per
    /// request) so that a misconfigured root -- one that doesn't exist, or
    /// isn't a directory -- is caught immediately at daemon startup rather
    /// than surfacing later as every request from a legitimate client
    /// mysteriously failing.
    pub fn new(configured_roots: &[PathBuf]) -> std::io::Result<Self> {
        let mut roots = Vec::with_capacity(configured_roots.len());
        for r in configured_roots {
            let canon = std::fs::canonicalize(r).map_err(|e| {
                std::io::Error::new(e.kind(), format!("--allow-root {}: {e}", r.display()))
            })?;
            if !canon.is_dir() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("--allow-root {}: not a directory", r.display()),
                ));
            }
            roots.push(canon);
        }
        Ok(Self { roots })
    }

    pub fn is_empty(&self) -> bool {
        self.roots.is_empty()
    }

    /// Checks whether `requested` may be scanned. Canonicalizes the
    /// requested path (resolving `..`, symlinks, etc.) before comparing
    /// against the allowed roots -- comparing un-resolved paths would be
    /// trivially bypassable via a symlink inside an allowed root that
    /// points back out (e.g. an allowed upload directory containing a
    /// symlink to `/etc`).
    pub fn check(&self, requested: &Path) -> Result<PathBuf, Denial> {
        if self.roots.is_empty() {
            return Err(Denial::NoRootsConfigured);
        }
        let canon = std::fs::canonicalize(requested).map_err(|_| Denial::NotPermitted)?;
        if self.roots.iter().any(|root| canon.starts_with(root)) {
            Ok(canon)
        } else {
            Err(Denial::NotPermitted)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_allowlist_denies_everything() {
        let allow = PathAllowlist::new(&[]).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("f.txt");
        std::fs::write(&f, b"hi").unwrap();
        assert_eq!(allow.check(&f), Err(Denial::NoRootsConfigured));
    }

    #[test]
    fn path_inside_allowed_root_is_permitted() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("f.txt");
        std::fs::write(&f, b"hi").unwrap();
        let allow = PathAllowlist::new(&[dir.path().to_path_buf()]).unwrap();
        assert!(allow.check(&f).is_ok());
    }

    #[test]
    fn path_outside_allowed_root_is_denied() {
        let allowed_dir = tempfile::tempdir().unwrap();
        let other_dir = tempfile::tempdir().unwrap();
        let f = other_dir.path().join("f.txt");
        std::fs::write(&f, b"hi").unwrap();
        let allow = PathAllowlist::new(&[allowed_dir.path().to_path_buf()]).unwrap();
        assert_eq!(allow.check(&f), Err(Denial::NotPermitted));
    }

    #[test]
    fn nonexistent_path_is_denied_not_panicked() {
        let allowed_dir = tempfile::tempdir().unwrap();
        let allow = PathAllowlist::new(&[allowed_dir.path().to_path_buf()]).unwrap();
        let missing = allowed_dir.path().join("does-not-exist.txt");
        assert_eq!(allow.check(&missing), Err(Denial::NotPermitted));
    }

    #[cfg(unix)]
    #[test]
    fn symlink_escaping_allowed_root_is_denied() {
        use std::os::unix::fs::symlink;

        let allowed_dir = tempfile::tempdir().unwrap();
        let secret_dir = tempfile::tempdir().unwrap();
        let secret = secret_dir.path().join("secret.txt");
        std::fs::write(&secret, b"top secret").unwrap();

        // A symlink living *inside* the allowed root but pointing *outside*
        // it must not grant access via naive prefix-matching on the
        // unresolved path.
        let link = allowed_dir.path().join("escape");
        symlink(&secret, &link).unwrap();

        let allow = PathAllowlist::new(&[allowed_dir.path().to_path_buf()]).unwrap();
        assert_eq!(allow.check(&link), Err(Denial::NotPermitted));
    }

    #[test]
    fn misconfigured_root_fails_at_construction() {
        let err = PathAllowlist::new(&[PathBuf::from("/this/path/should/not/exist/anywhere")]);
        assert!(err.is_err());
    }
}
