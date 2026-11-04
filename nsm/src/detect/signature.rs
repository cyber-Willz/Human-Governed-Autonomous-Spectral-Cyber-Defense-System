//! Content-match engine: Suricata/Snort-style substring rules over the first
//! `PAYLOAD_CAP` bytes of the L4 payload.
//!
//! The original matched raw bytes only, which the red-team harness shows is
//! trivially evaded. Matching now runs on a *normalised* view of the payload:
//!
//! * ASCII case-folded, NUL bytes dropped (UTF-16-ish padding),
//! * `%xx` and `+` decoded (single pass, plus a second pass for double
//!   encoding),
//! * SQL inline comments `/* ... */` replaced by a space and whitespace runs
//!   collapsed (`UNION/**/SELECT`, `union%20%20select`),
//! * Log4j lookup obfuscation resolved: `${lower:j}`, `${upper:J}`,
//!   `${::-j}`, `${env:X:-j}` ... so `${${lower:j}ndi:` becomes `${jndi:`.
//!
//! Matches are also carried **across TCP segment boundaries**: the tail of
//! the previous payload on the same 4-tuple is prepended, so a marker split
//! over two segments is still seen (bounded table, stale entries expire).
//!
//! Round-2 additions (adaptive adversary):
//!
//! * **Whitespace / comment differentials.** CR, LF, VT, FF and NBSP are SQL
//!   whitespace but were not folded to a space (`union%0aselect`); MySQL
//!   versioned comments (`/*!50000UNION*/`) were deleted instead of executed;
//!   `union all(select`, `union(select`, `union distinct select` and
//!   `union#<newline>select` were not substring matches. The two SQLi rules are
//!   now one structural matcher over an SQL-specific view (line comments
//!   stripped *only* there, so `#` cannot be used to hide other rules' text),
//!   with word boundaries (`reunion selection` no longer matches).
//! * **Slow drip.** Partial matches expired after 30 s; TCP allows any pause.
//!   Now 300 s.
//! * **Flow-table flush.** The partial-match table (20k entries, 8-byte "weak"
//!   rule) could be flushed by opening junk flows between the two halves of a
//!   marker. Now only tails that end in a *proper prefix of some rule* (or an
//!   open `${`, unterminated comment, partial `%xx`, partial `union ...`) are
//!   stored at all; ordinary traffic costs no state. Capacity is 100k, eviction
//!   drops the least specific prefixes first, and an eviction under pressure
//!   raises a Medium alert so the attempt is not silent.
//!
//! Residual limits (documented, not fixed here): out-of-order segments (no TCP
//! sequence number in `PacketMeta`), a flood of >100k *equally specific* crafted
//! flows (alerted, but can still evict), payload beyond the capture
//! window, TLS-encrypted payloads, and full TCP reassembly / IP-fragment
//! reassembly remain out of reach of a stateless header-and-prefix engine.

use super::bounded::make_room;
use crate::alert::{Alert, Severity};
use crate::packet::{L4Proto, PacketMeta};
use dashmap::DashMap;
use std::net::IpAddr;
use std::time::Duration;

pub struct Rule {
    pub name: &'static str,
    /// Lower-case, already-normalised pattern.
    pub pattern: &'static [u8],
    pub severity: Severity,
    /// Only match at the very start of the (normalised) payload. Used for
    /// protocol commands such as FTP `USER`, which must not fire on the
    /// English word "user" in the middle of a case-folded HTTP body.
    pub anchored: bool,
    /// Only evaluate for packets to this destination port.
    pub dst_port: Option<u16>,
    /// Structural `UNION [ALL|DISTINCT] SELECT` matcher on the SQL view.
    pub sql_union: bool,
}

const TAIL_KEEP: usize = 160;
const FLOW_TTL: Duration = Duration::from_secs(300);
const FLOW_CAP: usize = 100_000;
/// Tails whose matched prefix is this short or shorter are cheap for an
/// attacker to mass-produce, so they are evicted first.
const WEAK_SCORE: u16 = 3;

struct Tail {
    bytes: Vec<u8>,
    last: Duration,
    score: u16,
}

pub struct SignatureEngine {
    rules: Vec<Rule>,
    tails: DashMap<(IpAddr, u16, IpAddr, u16), Tail>,
    last_pressure_s: std::sync::atomic::AtomicU64,
}

impl SignatureEngine {
    pub fn with_default_ruleset() -> Self {
        Self {
            rules: vec![
                Rule { name: "cleartext-http-basic-auth", pattern: b"authorization: basic", severity: Severity::Medium, anchored: false, dst_port: None, sql_union: false },
                Rule { name: "ftp-cleartext-user", pattern: b"user ", severity: Severity::Low, anchored: true, dst_port: Some(21), sql_union: false },
                Rule { name: "suspicious-user-agent-curl", pattern: b"user-agent: curl", severity: Severity::Info, anchored: false, dst_port: None, sql_union: false },
                Rule { name: "possible-webshell-cmd", pattern: b"cmd=", severity: Severity::Medium, anchored: false, dst_port: None, sql_union: false },
                Rule { name: "possible-sqli-union-select", pattern: b"union select", severity: Severity::High, anchored: false, dst_port: None, sql_union: true },
                Rule { name: "possible-log4shell", pattern: b"${jndi:", severity: Severity::Critical, anchored: false, dst_port: None, sql_union: false },
                Rule { name: "eicar-test-string", pattern: b"eicar-standard-antivirus-test-file", severity: Severity::High, anchored: false, dst_port: None, sql_union: false },
            ],
            tails: DashMap::new(),
            last_pressure_s: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub fn scan(&self, pkt: &PacketMeta) -> Vec<Alert> {
        if pkt.payload_head.is_empty() {
            return Vec::new();
        }
        let now = pkt.ts_dur();
        let key = (pkt.src_ip, pkt.src_port, pkt.dst_ip, pkt.dst_port);
        let is_tcp = pkt.proto == L4Proto::Tcp;

        // previous tail on this flow (TCP only)
        let prev: Vec<u8> = if is_tcp {
            match self.tails.get(&key) {
                Some(t) if now.saturating_sub(t.last) <= FLOW_TTL => t.bytes.clone(),
                _ => Vec::new(),
            }
        } else {
            Vec::new()
        };

        let tail_norm = normalise(&prev);
        let mut combined_raw = prev.clone();
        combined_raw.extend_from_slice(&pkt.payload_head);
        let (combined, obfuscated) = normalise_flagged(&combined_raw);

        let mut hits = Vec::new();
        // The TCP source address is only trustworthy when the segment belongs
        // to an established flow (ACK set); UDP/ICMP and bare SYNs can be
        // forged by anyone, so a ban keyed on them would let an attacker get
        // arbitrary third parties (gateway, resolver) blackholed.
        let spoofable = !(is_tcp && pkt.tcp_flags & 0x10 != 0);
        for rule in &self.rules {
            if rule.dst_port.map_or(false, |p| p != pkt.dst_port) {
                continue;
            }
            let (found, seen_before) = if rule.sql_union {
                (union_select(&sql_normalise(&combined_raw)), union_select(&sql_normalise(&prev)))
            } else if rule.anchored {
                // anchored rules look only at the start of this packet's own payload
                let own = normalise(&pkt.payload_head);
                (own.starts_with(rule.pattern), false)
            } else {
                (contains(&combined, rule.pattern), contains(&tail_norm, rule.pattern))
            };
            if found && !seen_before {
                hits.push(Alert::new(
                    rule.severity,
                    "signature",
                    format!("payload matched rule '{}' ({} -> {})", rule.name, pkt.src_ip, pkt.dst_ip),
                    Some(pkt.src_ip),
                    Some(pkt.dst_ip),
                    serde_json::json!({
                        "rule": rule.name, "obfuscated": obfuscated, "src_spoofable": spoofable
                    }),
                ));
            }
        }

        if is_tcp {
            let start = combined_raw.len().saturating_sub(TAIL_KEEP);
            let tail_raw = &combined_raw[start..];
            let score = self.interest(tail_raw);
            if score == 0 {
                // Nothing here can be the first half of a match: keep no state.
                self.tails.remove(&key);
            } else {
                if !self.tails.contains_key(&key) && self.tails.len() >= FLOW_CAP {
                    self.tails.retain(|_, t| now.saturating_sub(t.last) <= FLOW_TTL);
                    if self.tails.len() >= FLOW_CAP {
                        // Real pressure: live partial matches are being displaced.
                        let last = self.last_pressure_s.load(std::sync::atomic::Ordering::Relaxed);
                        if now.as_secs().saturating_sub(last) >= 60 {
                            self.last_pressure_s.store(now.as_secs().max(1), std::sync::atomic::Ordering::Relaxed);
                            hits.push(Alert::new(
                                Severity::Medium,
                                "signature",
                                format!("partial-match table full ({FLOW_CAP} live entries): possible signature-evasion flood (latest source {})", pkt.src_ip),
                                Some(pkt.src_ip),
                                Some(pkt.dst_ip),
                                serde_json::json!({ "rule": "tail-table-pressure", "src_spoofable": true }),
                            ));
                        }
                        make_room(&self.tails, FLOW_CAP, |_| false, |t| t.score <= WEAK_SCORE);
                    }
                }
                self.tails.insert(key, Tail { bytes: tail_raw.to_vec(), last: now, score });
            }
        }
        hits
    }

    /// 0 if `raw_tail` cannot be the beginning of any match; otherwise the
    /// specificity (length of the matched rule prefix) of the strongest partial.
    fn interest(&self, raw_tail: &[u8]) -> u16 {
        let n = normalise(raw_tail);
        let mut best = 0usize;
        if raw_tail.last() == Some(&b'%') || (raw_tail.len() >= 2 && raw_tail[raw_tail.len() - 2] == b'%') {
            best = 1; // split inside a %xx escape
        }
        for rule in &self.rules {
            if rule.anchored || rule.sql_union {
                continue;
            }
            // `$` heads the lookup patterns and is rare at end-of-segment, so even
            // one byte is worth keeping; other prefixes need >= 3 bytes.
            let min_k = if rule.pattern.first() == Some(&b'$') { 1 } else { 3 };
            for k in (min_k..rule.pattern.len()).rev() {
                if n.ends_with(&rule.pattern[..k]) {
                    best = best.max(k);
                    break;
                }
            }
        }
        if let Some(p) = rfind(&n, b"${") {
            if !n[p..].contains(&b'}') {
                best = best.max((n.len() - p).min(7)); // open lookup; longer = more specific
            }
        }
        if rfind(raw_tail, b"/*").map_or(false, |p| find(&raw_tail[p + 2..], b"*/").is_none()) {
            best = best.max(2);
        }
        if sql_partial(&sql_normalise(raw_tail)) {
            best = best.max(5);
        }
        best as u16
    }
}

fn rfind(h: &[u8], n: &[u8]) -> Option<usize> {
    if n.is_empty() || n.len() > h.len() {
        return None;
    }
    h.windows(n.len()).rposition(|w| w == n)
}

fn is_word(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn skip_sep(h: &[u8], mut i: usize) -> usize {
    while i < h.len() && (h[i] == b' ' || h[i] == b'(') {
        i += 1;
    }
    i
}

/// `UNION [ALL|DISTINCT] SELECT` on an SQL-normalised view (whitespace folded,
/// comments handled), with word boundaries and any mix of spaces / `(` between
/// the tokens.
fn union_select(h: &[u8]) -> bool {
    let mut from = 0;
    while from < h.len() {
        let Some(rel) = find(&h[from..], b"union") else { return false };
        let p = from + rel;
        from = p + 1;
        if p > 0 && is_word(h[p - 1]) {
            continue;
        }
        let mut i = skip_sep(h, p + 5);
        if i == p + 5 {
            continue; // "unionselect" is not a UNION query
        }
        for w in [&b"all"[..], &b"distinct"[..]] {
            if h[i..].starts_with(w) && h.get(i + w.len()).map_or(false, |c| !is_word(*c)) {
                i = skip_sep(h, i + w.len());
                break;
            }
        }
        if h[i..].starts_with(b"select") && h.get(i + 6).map_or(true, |c| !is_word(*c)) {
            return true;
        }
    }
    false
}

/// Does the SQL view *end* partway through `UNION [ALL|DISTINCT] SELECT`?
fn sql_partial(h: &[u8]) -> bool {
    let Some(p) = rfind(h, b"union") else { return false };
    if p > 0 && is_word(h[p - 1]) {
        return false;
    }
    let mut rest = &h[p + 5..];
    let trim = |r: &[u8]| -> usize { r.iter().take_while(|c| **c == b' ' || **c == b'(').count() };
    rest = &rest[trim(rest)..];
    for w in [&b"all"[..], &b"distinct"[..]] {
        if rest.starts_with(w) {
            rest = &rest[w.len()..];
            rest = &rest[trim(rest)..];
            break;
        }
    }
    b"select".starts_with(rest) && rest.len() < 6 || [&b"all"[..], &b"distinct"[..]].iter().any(|w| w.starts_with(rest) && rest.len() < w.len())
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || needle.len() > haystack.len() {
        return false;
    }
    haystack.windows(needle.len()).any(|w| w == needle)
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn percent_decode(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len());
    let mut i = 0;
    while i < input.len() {
        match input[i] {
            b'%' if i + 2 < input.len() + 0 && hex_val(input[i + 1]).is_some() && hex_val(input[i + 2]).is_some() => {
                out.push(hex_val(input[i + 1]).unwrap() * 16 + hex_val(input[i + 2]).unwrap());
                i += 3;
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    out
}

/// Resolve Log4j-style nested lookups that only exist to hide `jndi`.
fn resolve_lookups(input: &[u8]) -> (Vec<u8>, bool) {
    let mut cur = input.to_vec();
    let mut changed_any = false;
    for _ in 0..12 {
        // find an innermost `${ ... }` (no `${` inside)
        let mut open = None;
        let mut replaced = false;
        let mut out = Vec::with_capacity(cur.len());
        let mut i = 0;
        while i < cur.len() {
            if cur[i] == b'$' && i + 1 < cur.len() && cur[i + 1] == b'{' {
                open = Some(i);
            }
            if cur[i] == b'}' {
                if let Some(o) = open {
                    let inner = &cur[o + 2..i];
                    if !inner.contains(&b'$') {
                        let lower = inner.to_ascii_lowercase();
                        let resolved: Option<Vec<u8>> = if lower.starts_with(b"lower:") || lower.starts_with(b"upper:") {
                            Some(lower[6..].to_vec())
                        } else if let Some(p) = find(&lower, b":-") {
                            Some(lower[p + 2..].to_vec())
                        } else {
                            None
                        };
                        if let Some(r) = resolved {
                            // splice: everything before `${`, the resolution, then continue
                            out.truncate(out.len() - (i - o));
                            out.extend_from_slice(&r);
                            replaced = true;
                            open = None;
                            i += 1;
                            continue;
                        }
                    }
                    open = None;
                }
            }
            out.push(cur[i]);
            i += 1;
        }
        if !replaced {
            break;
        }
        changed_any = true;
        cur = out;
    }
    (cur, changed_any)
}

fn find(h: &[u8], n: &[u8]) -> Option<usize> {
    h.windows(n.len()).position(|w| w == n)
}

fn normalise(raw: &[u8]) -> Vec<u8> {
    normalise_flagged(raw).0
}

fn normalise_flagged(raw: &[u8]) -> (Vec<u8>, bool) {
    // 1. NULs out, then up to two passes of percent-decoding.
    let no_nul: Vec<u8> = raw.iter().copied().filter(|&b| b != 0).collect();
    let mut d = percent_decode(&no_nul);
    if d.contains(&b'%') {
        d = percent_decode(&d);
    }
    // 2. Log4j lookup obfuscation.
    let (resolved, obf) = resolve_lookups(&d);
    // 3. case-fold, handle SQL comments, fold all whitespace.
    (strip_comments(&resolved.to_ascii_lowercase(), false), obf)
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\r' | b'\n' | 0x0b | 0x0c | 0xa0)
}

/// Comment + whitespace folding. `/* ... */` becomes a space; MySQL
/// `/*!12345 body */` is *executed*, so its body is kept. With
/// `line_comments`, `#...EOL` and `-- ...EOL` are dropped too (SQL view only).
fn strip_comments(lowered: &[u8], line_comments: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(lowered.len());
    let mut i = 0;
    let mut last_space = false;
    let mut push_space = |out: &mut Vec<u8>, last: &mut bool| {
        if !*last {
            out.push(b' ');
            *last = true;
        }
    };
    while i < lowered.len() {
        let b = lowered[i];
        if b == b'/' && lowered.get(i + 1) == Some(&b'*') {
            if let Some(end) = find(&lowered[i + 2..], b"*/") {
                let inner = &lowered[i + 2..i + 2 + end];
                push_space(&mut out, &mut last_space);
                if inner.first() == Some(&b'!') {
                    let body = &inner[1..];
                    let digits = body.iter().take_while(|c| c.is_ascii_digit()).count();
                    for &c in &body[digits..] {
                        if is_ws(c) {
                            push_space(&mut out, &mut last_space);
                        } else {
                            out.push(c);
                            last_space = false;
                        }
                    }
                    push_space(&mut out, &mut last_space);
                }
                i += 2 + end + 2;
                continue;
            }
        }
        if line_comments && (b == b'#' || (b == b'-' && lowered.get(i + 1) == Some(&b'-') && lowered.get(i + 2).map_or(true, |c| is_ws(*c)))) {
            while i < lowered.len() && lowered[i] != b'\n' && lowered[i] != b'\r' {
                i += 1;
            }
            push_space(&mut out, &mut last_space);
            continue;
        }
        if is_ws(b) {
            push_space(&mut out, &mut last_space);
        } else {
            out.push(b);
            last_space = false;
        }
        i += 1;
    }
    out
}

/// SQL-specific view: like `normalise` but also drops `#` / `-- ` line comments.
fn sql_normalise(raw: &[u8]) -> Vec<u8> {
    let no_nul: Vec<u8> = raw.iter().copied().filter(|&b| b != 0).collect();
    let mut d = percent_decode(&no_nul);
    if d.contains(&b'%') {
        d = percent_decode(&d);
    }
    strip_comments(&d.to_ascii_lowercase(), true)
}
