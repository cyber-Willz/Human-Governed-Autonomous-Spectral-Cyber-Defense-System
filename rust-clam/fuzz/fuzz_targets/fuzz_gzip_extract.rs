//! Fuzzes `extract_gzip` against fully attacker-controlled bytes. Gzip in
//! particular has no reliable declared uncompressed size (see the doc
//! comment on `extract_gzip` itself), so this is the archive path that
//! relies most heavily on the streaming budget check rather than an
//! upfront size check -- a good target for confirming that invariant holds
//! under arbitrary input, not just the crafted bomb in the unit tests.
#![no_main]
use archive_guard::{extract_gzip, GuardBudget, GuardLimits};
use libfuzzer_sys::fuzz_target;
use std::io::Cursor;

fuzz_target!(|data: &[u8]| {
    let mut budget = GuardBudget::new(GuardLimits::default());
    let _ = extract_gzip(Cursor::new(data), &mut budget);
});
