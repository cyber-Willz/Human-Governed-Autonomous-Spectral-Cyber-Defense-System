//! Fuzzes the `.hdb` hash-signature loader.
#![no_main]
use libfuzzer_sys::fuzz_target;
use sig_engine::SignatureEngine;

fuzz_target!(|data: &[u8]| {
    if let Ok(text) = std::str::from_utf8(data) {
        let _ = SignatureEngine::builder().load_hdb(text);
    }
});
