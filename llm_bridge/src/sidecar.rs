//! [`LanguageBackend`] that talks to the `llm_sidecar` process (burn-llm,
//! built in its own workspace) over stdin/stdout. Process isolation means a
//! crash, hang-induced kill, or memory blow-up in model code cannot take
//! down the sensor stack; any I/O failure surfaces as `BridgeError` and the
//! bridge falls back to its deterministic rules.

use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, Command, Stdio};

use crate::{BridgeError, LanguageBackend};

const MAX_REPLY_BYTES: usize = 64 * 1024;

pub struct SidecarBackend {
    child: Option<Child>,
    to: Box<dyn Write + Send>,
    from: Box<dyn BufRead + Send>,
}

fn be<E: std::fmt::Display>(e: E) -> BridgeError {
    BridgeError::Backend(e.to_string())
}

impl SidecarBackend {
    pub fn spawn(bin: &str, model_dir: &str) -> Result<Self, BridgeError> {
        let mut child = Command::new(bin)
            .arg(model_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| BridgeError::Backend(format!("spawn {bin}: {e}")))?;
        let to = child.stdin.take().ok_or_else(|| be("no stdin"))?;
        let from = child.stdout.take().ok_or_else(|| be("no stdout"))?;
        Ok(Self { child: Some(child), to: Box::new(to), from: Box::new(BufReader::new(from)) })
    }

    /// Any reader/writer pair speaking the protocol (tests, sockets).
    pub fn from_streams(to: impl Write + Send + 'static, from: impl BufRead + Send + 'static) -> Self {
        Self { child: None, to: Box::new(to), from: Box::new(from) }
    }
}

impl Drop for SidecarBackend {
    fn drop(&mut self) {
        if let Some(c) = self.child.as_mut() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

impl LanguageBackend for SidecarBackend {
    fn name(&self) -> &str {
        "burn-llm-sidecar"
    }

    fn complete(&mut self, prompt: &str, max_new_tokens: usize) -> Result<String, BridgeError> {
        write!(self.to, "REQ {} {}\n", max_new_tokens, prompt.len()).map_err(be)?;
        self.to.write_all(prompt.as_bytes()).map_err(be)?;
        self.to.flush().map_err(be)?;
        let mut line = String::new();
        if self.from.read_line(&mut line).map_err(be)? == 0 {
            return Err(be("sidecar closed the stream"));
        }
        let mut it = line.split_whitespace();
        match it.next() {
            Some("RES") => {
                let n: usize = it.next().and_then(|x| x.parse().ok()).ok_or_else(|| be("bad RES length"))?;
                if n > MAX_REPLY_BYTES {
                    return Err(be("reply too large"));
                }
                let mut buf = vec![0u8; n];
                self.from.read_exact(&mut buf).map_err(be)?;
                Ok(String::from_utf8_lossy(&buf).into_owned())
            }
            Some("ERR") => Err(BridgeError::Backend(line.trim_start_matches("ERR").trim().to_string())),
            _ => Err(be("malformed sidecar reply")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Default)]
    struct Sink(Arc<Mutex<Vec<u8>>>);
    impl Write for Sink {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn frames_request_and_parses_reply() {
        let sink = Sink::default();
        let mut b = SidecarBackend::from_streams(sink.clone(), Cursor::new(b"RES 14\nHINT: scan\nxyz".to_vec()));
        assert_eq!(b.complete("héllo", 7).unwrap(), "HINT: scan\nxyz");
        assert_eq!(&*sink.0.lock().unwrap(), "REQ 7 6\nhéllo".as_bytes());
    }

    #[test]
    fn errors_and_garbage_become_bridge_errors() {
        let mut b = SidecarBackend::from_streams(Sink::default(), Cursor::new(b"ERR prompt too large\n".to_vec()));
        assert!(b.complete("x", 1).unwrap_err().to_string().contains("prompt too large"));
        let mut b = SidecarBackend::from_streams(Sink::default(), Cursor::new(b"WAT\n".to_vec()));
        assert!(b.complete("x", 1).is_err());
        let mut b = SidecarBackend::from_streams(Sink::default(), Cursor::new(b"RES 99999999\n".to_vec()));
        assert!(b.complete("x", 1).is_err());
        let mut b = SidecarBackend::from_streams(Sink::default(), Cursor::new(Vec::new()));
        assert!(b.complete("x", 1).is_err());
    }
}
