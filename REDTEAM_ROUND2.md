# Red-team round 2 — adaptive adversary

Round 1 (`REDTEAM_REPORT.md`) fixed what a naive attacker could do and ended at 42/42.
Round 2 assumes the attacker has read that report. All scenarios are in-process synthetic
metadata against the real detectors / correlation engine / control plane / quarantine
(`attack_sim/src/round2.rs`, layer C). Reproduce: `cargo run --release -p attack_sim`
(`--no-round2` reproduces the round-1 baseline).

| | before (unchanged tree) | after |
|---|---|---|
| Round-1 layers A/B | 42/42 | 42/42 |
| New layer C (20 scored) | **4/20** (only the benign controls) | **20/20** |
| Total scored | 46/62 | **62/62** |

Snapshots: `redteam/round2_before.json`, `redteam/round2_after.json`.

## What was broken, and what changed

| ID | Evasion | Root cause | Fix |
|---|---|---|---|
| C01 | 240 bots x 3 ports | every scan counter is per source | per-target distributed view (>=64 ports, >=16 sources / 15 min); victim-only Medium alert |
| C02 | IPv6, new address per probe in one /64 | per-address keys | `canon_src`: inbound sources keyed per /64, `::ffff:a.b.c.d` = `a.b.c.d` |
| C03 | SYN flood, random dst port | counter keyed (ip, port) | per-IP spread view (4x threshold over >=64 ports) |
| C04 | tunnel via `x.y.abc.io` | zone guessed from label length; payload label became part of the "zone" | real second-level-under-ccTLD list (`co.uk`, `com.au`...) |
| C05 | DNS tunnel over TCP/53 | UDP only | RFC 1035 length-prefixed TCP parsing |
| C06 | 60 hosts x 12 names/h to one zone | all thresholds per client | per-zone fleet view (>=200 names, >=5 clients, 1 h); analyst-facing, names no host |
| C07 | C2 pool of 20 IPs in a /24 | history per exact (src, dst) | second history per (src, dst /24 or /48), needs >=3 distinct addresses |
| C08 | beacon, +/-80 % jitter | tests assumed tight jitter | bounded "wide-jitter" test, **calibrated by Monte-Carlo test** (99 % TPR at +/-80 %, 0.1 % FPR log-normal sigma=1, 0 % exponential) |
| C09 | SQLi `%0a`, `%0b`, `%0c`, `/*!50000UNION*/`, `union all(select`, `union(select`, `union distinct select`, `union#\nselect` — **0/8 caught** | substring match; CR/LF not folded; versioned comments deleted | structural `UNION [ALL\|DISTINCT] SELECT` matcher on an SQL-only view |
| C10 | marker halves 45 s apart | partial-match TTL 30 s | 300 s |
| C11 | 120k junk flows between marker halves | 20k-entry table, tiny prefix evicted first | store state only for real partial matches; 100k cap; least-specific evicted first; pressure alert |
| C13 | `::ffff:10.0.0.1` bypassed protected list; ban of mapped attacker did nothing on its IPv4 packets; IPv6 link-local gateway bannable | raw `IpAddr` comparisons | `ban_key` canonicalisation in quarantine / protected list / never-ban; fe80::/10 never bannable |
| C14 | rotate inside a banned /64 (64-entry ban table filled by 100 addresses) | per-address bans | IPv6 bans are /64-wide; 100 addresses = 1 entry; `ban_exact` for human-approved bans inside protected prefixes |
| C15 | recon, pause > window, then anomaly: no verdict | evidence dropped 60 s after last event | hosts with Medium+ evidence retained `campaign_window` (default 6 h) |
| C16 | portscan + SQLi + beacon with ML lanes evaded: **no verdict** | all NSM detectors count as one lane | >=2 distinct Medium+ NSM detectors corroborate; lands in human review, **not** auto-ban (confidence stays < floor) |
| Q05 | (round-1 false positive) "reunion selection" fired a High SQLi alert | substring without word boundaries | word-boundary matcher |

One regression was caught and fixed along the way: the first version of the network-keyed beacon
history pushed round-1 scenario A21 (memory under 300k spoofed sources) to 72 MB vs the 64 MB budget;
that map is now capped at a quarter of the budget (56 MB). Two of my own scenarios were also
corrected after inspection (C11 was initially hash-order luck).

## Known limits (measured, not scored)

* **K02** out-of-order TCP segments: `PacketMeta` has no sequence number.
* **K03** C2 rotating across unrelated networks every beat.
* **K04** fleet-wide cloud-AV/EDR reputation lookups are indistinguishable from C06; they raise a zone-level alert that names no host and cannot reach containment. Add known vendors to the exemption list.
* >100k *equally specific* crafted flows can still evict a partial match (now alerted).
* Fleet/distributed alerts can FP on passive-FTP ranges or P2P hosts (Medium, victim-only).

## Operational notes / behaviour changes you should know about

1. **IPv6 bans are /64-granular** in rustwall's in-process quarantine. **Add your own LAN prefixes via `protect_hosts`** (protecting one address protects its /64); otherwise a ban of a compromised internal host would blackhole its segment. OS-firewall (nftables) sync still receives the exact address; making the sets prefix-aware needs `flags interval` and could not be tested here (no CAP_NET_ADMIN).
2. More verdicts reach the human queue (C15, C16). None of them auto-ban.
3. `CorrelationConfig` gained `campaign_window`; the `unified_nsm...` SQLi rule names `possible-sqli-union-all-select` was merged into `possible-sqli-union-select`.

## Not covered

Adversarial examples against the ML lanes (autoencoder / LSTM / RL policy poisoning), XDP/kernel path, real packet capture, review-queue flooding, and anything needing root. Built and tested with Cargo 1.75 on the touched crates only; the full workspace could not be checked in this sandbox (dependency needs newer Cargo).
