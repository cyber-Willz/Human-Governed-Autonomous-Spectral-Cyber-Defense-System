//! Fuzzes `extract_tar` -- the newest and least battle-tested of the three
//! archive extractors, added alongside this fuzz suite.
#![no_main]
use archive_guard::{extract_tar, GuardBudget, GuardLimits};
use libfuzzer_sys::fuzz_target;
use std::io::Cursor;

fuzz_target!(|data: &[u8]| {
    let mut budget = GuardBudget::new(GuardLimits::default());
    let _ = extract_tar(Cursor::new(data), &mut budget);
});
