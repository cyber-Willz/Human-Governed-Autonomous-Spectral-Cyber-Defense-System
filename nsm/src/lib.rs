//! Library surface for `nsm`.
//!
//! `nsm` was originally a bin-only crate (`main.rs` with private `mod`
//! declarations, same as `rustwall` and `honeypot_engine` started out).
//! This file exposes the same modules publicly so other workspace crates
//! -- specifically `correlation_engine`'s `NsmSender` (see
//! `correlation_engine::NsmSender`) and `unified_runner`'s live-run driver
//! -- can depend on `nsm::alert::{Alert, Severity}`, `nsm::detect`, and
//! `nsm::flow` directly, without touching `main.rs`'s own module tree.
//! Same approach `rustwall/src/lib.rs` already used for the same reason.

pub mod alert;
pub mod capture;
pub mod detect;
pub mod flow;
pub mod packet;
pub mod sim;
#[cfg(all(target_os = "linux", feature = "xdp"))]
pub mod xdp;
