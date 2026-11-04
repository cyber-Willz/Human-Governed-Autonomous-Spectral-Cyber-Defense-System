//! Library surface for `rustwall`, added for this integration so
//! `ips_control_plane` can depend on `Quarantine` directly instead of
//! shelling out to the `rustwall` binary or re-implementing ban-table
//! semantics a second time.
//!
//! `rustwall` was previously bin-only (`src/main.rs` with private `mod`
//! declarations); `main.rs`'s own `mod` tree is untouched, so the binary
//! still builds and runs exactly as before -- this file gives the same
//! source files a second, `pub`, module tree for library consumers. Cargo
//! compiles the `--bin rustwall` and `--lib rustwall` targets as separate
//! units, so there's no conflict between the two trees despite sharing
//! files.
//!
//! Only [`quarantine`] is `pub` for a reason today (it's what
//! `ips_control_plane::RustwallContainmentAction` calls `ban()` on); the
//! rest are `pub` for completeness/future use (e.g. a future integration
//! wanting `engine`'s policy evaluation directly) without needing another
//! round of "oh, that module wasn't exposed" edits.

pub mod chan;
pub mod config;
pub mod conntrack;
pub mod engine;
pub mod log_gate;
pub mod logging;
pub mod metrics;
pub mod nfqueue;
pub mod os_firewall;
pub mod packet;
pub mod quarantine;
pub mod rate_limit;
pub mod reject;
pub mod sync_worker;
pub mod threshold;
