//! Fuzzes `pe_analyze::parse`, the one parser in this codebase that has to
//! make sense of a fully attacker-controlled binary header format (DOS
//! stub, PE signature, COFF header, optional header, section table) with a
//! lot of internal offsets that a hostile file can set to anything.
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(info) = pe_analyze::parse(data) {
        // heuristics() consumes whatever parse() considered well-formed
        // enough to return -- exercise it too rather than just the parser,
        // since it re-derives things (entropy, section flag combinations)
        // from attacker-influenced fields.
        let _ = pe_analyze::heuristics(&info);
    }
});
