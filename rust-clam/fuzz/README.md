# Fuzz targets

`cargo-fuzz` scaffold covering the six parsers in this codebase that consume
fully attacker-controlled bytes: PE headers, `.ndb`/`.hdb` signature
definitions, and the zip/gzip/tar archive extractors.

## Status: type-checked, not yet fuzz-run

Every target here has been verified to **compile and type-check** against
the current crate APIs (`cargo check`, stable toolchain). It has **not**
been run under an actual fuzzer (libFuzzer via `cargo fuzz run`) as part of
producing this change, because doing so requires a nightly Rust toolchain
and `cargo-fuzz`'s sanitizer/coverage instrumentation, which the environment
this work was done in could not install (no network path to
`static.rust-lang.org`/`sh.rustup.rs`). Treat this as a scaffold that is
ready to run, not as evidence of fuzzing having actually found (or ruled
out) anything yet -- that's real follow-up work, ideally wired into CI as a
short scheduled job (`cargo fuzz run <target> -- -max_total_time=300`) per
target rather than a one-off local run.

## Running locally

```sh
rustup toolchain install nightly
cargo install cargo-fuzz
cd fuzz
cargo +nightly fuzz run fuzz_zip_extract
cargo +nightly fuzz run fuzz_gzip_extract
cargo +nightly fuzz run fuzz_tar_extract
cargo +nightly fuzz run fuzz_pe_parse
cargo +nightly fuzz run fuzz_hexsig_ndb
cargo +nightly fuzz run fuzz_hashdb
```

## Why these six

- `fuzz_pe_parse` -- the PE header parser has the most internal
  offsets/lengths taken directly from attacker bytes (DOS stub, PE
  signature offset, section table) of anything in this codebase.
- `fuzz_zip_extract` / `fuzz_gzip_extract` / `fuzz_tar_extract` -- exercise
  the archive-bomb guards (`GuardBudget`) against malformed/adversarial
  input directly, rather than only the hand-crafted bombs in the unit
  tests. The invariant under test is "never panics, never reads/allocates
  past what the budget allows" -- errors are an expected, fine outcome.
- `fuzz_hexsig_ndb` / `fuzz_hashdb` -- signature files are normally
  operator-supplied rather than attacker-controlled, but a panic while
  loading signatures at startup would take the whole daemon down, so the
  parser is still worth the scrutiny.

This is a detached `cargo-fuzz`-style workspace (see the empty `[workspace]`
table in `Cargo.toml`) so it isn't swept into the parent workspace's
dependency resolution -- several parent crates pin exact dependency
versions for reasons unrelated to what `libfuzzer-sys` needs.
