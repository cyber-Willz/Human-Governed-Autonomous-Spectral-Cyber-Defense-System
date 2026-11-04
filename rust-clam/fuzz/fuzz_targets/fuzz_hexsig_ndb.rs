//! Fuzzes the `.ndb` hex-signature loader (`SignatureEngineBuilder::load_ndb`).
//! Signature files are normally operator-supplied, not attacker-controlled,
//! but the parser for the wildcard-gap syntax (`*`, `*{min-max}`, `??`) is
//! non-trivial hand-written parsing over untrusted-shaped text, and a
//! panic here would take the whole daemon down at startup -- worth the
//! same scrutiny as any other parser boundary.
#![no_main]
use libfuzzer_sys::fuzz_target;
use sig_engine::SignatureEngine;

fuzz_target!(|data: &[u8]| {
    if let Ok(text) = std::str::from_utf8(data) {
        let _ = SignatureEngine::builder().load_ndb(text);
    }
});
