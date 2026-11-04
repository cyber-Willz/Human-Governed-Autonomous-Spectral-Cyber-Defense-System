//! Fuzzes `extract_zip` against fully attacker-controlled bytes, with the
//! same `GuardLimits` a real deployment would use. The invariant under test
//! isn't "never errors" (malformed zips are expected to error) -- it's
//! "never panics, and never allocates or reads unboundedly regardless of
//! what the zip header claims", which is exactly what the ratio/entry/size
//! guards exist to enforce.
#![no_main]
use archive_guard::{extract_zip, GuardBudget, GuardLimits};
use libfuzzer_sys::fuzz_target;
use std::io::Cursor;

fuzz_target!(|data: &[u8]| {
    let mut budget = GuardBudget::new(GuardLimits::default());
    let _ = extract_zip(Cursor::new(data), &mut budget);
});
