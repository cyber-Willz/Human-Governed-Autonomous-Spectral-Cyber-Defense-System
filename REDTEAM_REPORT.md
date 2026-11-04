# Red-team report: simulated attacks vs. the unified NSM/NDR/IPS/SIEM stack

Harness: `attack_sim` (workspace member). Run: `cargo run --release -p attack_sim -- --json out.json`.
All traffic is in-process synthetic packet metadata with simulated time (no network I/O).
Before/after JSON: `redteam/results_before.json`, `redteam/results_after.json`.
`--no-protect-list` re-runs without the operator protected-host list: only B05 then fails, so the other fixes stand on their own.

| | Before | After |
|---|---|---|
| Attacks detected | 10/24 | 24/24 |
| Benign controls quiet | 6/8 | 8/8 |
| Response-safety | 0/10 | 10/10 |
| Total scored | 16/42 | 42/42 |

## Findings and fixes

**Response path (most serious)**
- B01 SYN-flood alert is keyed on the victim; IPS auto-banned the victim for 24h (the original demo's second ban was this). Now `Attribution::VictimOnly`, capped below auto-containment.
- B08/B09 one forged UDP packet with a Log4Shell string got the gateway / DNS resolver banned. Forgeable sources (UDP/ICMP/bare SYN) are now capped (`Spoofable`); protected-host list + never-ban for loopback/multicast/broadcast.
- B02 host already given a verdict was ignored thereafter (a later honeytoken touch never banned). Engine now re-judges on strict severity escalation (max 3 re-emits).
- B03 confidence was a plain mean, so noise diluted a honeytoken (1.0 -> 0.18). Now per-lane max, mean across lanes, floor at Critical evidence confidence. Evidence eviction drops least-significant, not oldest.
- B04 forged-ban flood filled the ban table and made a real attacker unbannable. Fixed upstream (B08 fix) plus quarantine now reclaims expired bans, can displace a clearly shorter ban, and the control plane sweeps expired bans (nothing did before).
- B05 authorised scanner auto-banned: protected list routes to human review; human approval may override.
- B06 400k-event noise flood dropped 94% of events including the real Critical one. High/Critical now use a priority channel drained first; High+ verdicts are not dropped when the control-plane channel is full.
- B07 host-table eviction was O(n) per event (10k events/s). Batched eviction: ~860k events/s.

**Detection**
- Stealth scans (NULL/FIN/XMAS/illegal flags) and slow scans (1 probe/20 s) were invisible: added probe-flag logic and a 1h inbound window.
- DNS tunnel: replaced per-label entropy with distinct-subdomain cardinality per (client, zone) in 60 s/10 min/1 h tiers; fixes CDN false positive, low-and-slow, hex, split-label and dictionary-word evasions; counts queries only; CDN zones exempt.
- Beacon: key per (src,dst) not port; added outlier-robust and bounded-jitter tests (catches +/-30% jitter, port hopping).
- Signatures: normalisation (case, %-decoding, SQL comments, Log4j `${lower:j}`/`${::-j}`), cross-segment carry, 1 KiB capture window. FTP `USER` rule anchored to port 21 to avoid case-fold false positives.
- All detectors: memory bounded with eviction that protects accumulated evidence (300k forged sources: 422 MB -> 56 MB), time taken from packet timestamps (replay-safe), per-key alert cooldowns.

## Known limits (not fixed)
- Legit periodic software (keepalives/NTP) looks like a beacon; kept Medium so it cannot auto-contain (K01).
- Payload beyond 1 KiB, TLS/DoH/DoT content, full TCP/IP reassembly are invisible to this engine.
- `hg_builder` fan-in/fan-out still uses hard count thresholds (documented evasion); not modified.
- FTP detection on non-21 ports dropped. Protected-host list is config: you must populate it for your gateway/DNS/servers (demo runner protects only the demo victim).
- Simulation only: validate on real traffic/pcaps before production reliance.
