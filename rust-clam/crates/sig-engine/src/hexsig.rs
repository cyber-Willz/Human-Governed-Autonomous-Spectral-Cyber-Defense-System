//! Wildcard byte-pattern signatures, e.g.:
//!   evil.trojan.generic:6a6f??686e*{2-8}646f65
//!
//! A pattern is split on `*` (unbounded gap) or `*{min-max}` (bounded gap)
//! into a sequence of fixed-length "segments". Each segment is a run of
//! tokens that are either a concrete byte or a `??` single-byte wildcard.
//!
//! Unlike ClamAV's bytecode-VM signatures, there is no interpreted code
//! path here at all -- matching is pure data-driven segment verification,
//! which removes an entire historical CVE class (malformed-bytecode memory
//! corruption in the scanner itself).

use crate::error::{Result, SigError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Token {
    Byte(u8),
    Wildcard,
}

#[derive(Debug, Clone)]
pub struct Segment {
    pub tokens: Vec<Token>,
}

impl Segment {
    fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Does this segment match the buffer at `pos`? (pos..pos+len must be in-bounds)
    fn matches_at(&self, buf: &[u8], pos: usize) -> bool {
        if pos + self.len() > buf.len() {
            return false;
        }
        self.tokens.iter().enumerate().all(|(i, t)| match t {
            Token::Byte(b) => buf[pos + i] == *b,
            Token::Wildcard => true,
        })
    }

    /// Longest contiguous run of concrete (non-wildcard) bytes within this
    /// segment, returned as (offset_in_segment, literal_bytes).
    fn longest_literal_run(&self) -> Option<(usize, Vec<u8>)> {
        let mut best: Option<(usize, Vec<u8>)> = None;
        let mut cur_start = 0usize;
        let mut cur: Vec<u8> = Vec::new();
        for (i, t) in self.tokens.iter().enumerate() {
            match t {
                Token::Byte(b) => {
                    if cur.is_empty() {
                        cur_start = i;
                    }
                    cur.push(*b);
                }
                Token::Wildcard => {
                    if best.as_ref().map(|(_, v)| v.len()).unwrap_or(0) < cur.len() {
                        best = Some((cur_start, std::mem::take(&mut cur)));
                    } else {
                        cur.clear();
                    }
                }
            }
        }
        if best.as_ref().map(|(_, v)| v.len()).unwrap_or(0) < cur.len() {
            best = Some((cur_start, cur));
        }
        best
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Gap {
    Unbounded,
    Bounded(usize, usize),
}

/// Hard ceiling on how far we'll search for the next segment when a gap is
/// unbounded (`*`). ClamAV enforces something functionally similar; without
/// this a crafted file could force pathological backtracking.
const MAX_GAP_SEARCH: usize = 1 << 16;

#[derive(Debug, Clone)]
pub struct HexSignature {
    pub name: String,
    pub segments: Vec<Segment>,
    pub gaps: Vec<Gap>, // len == segments.len() - 1
    /// Which segment supplies the literal string fed into the shared AC automaton.
    pub anchor_segment: usize,
    pub anchor_offset: usize,
    pub anchor_literal: Vec<u8>,
}

impl HexSignature {
    /// Parse `name:pattern` into a HexSignature, choosing the longest
    /// wildcard-free run across all segments as the AC anchor.
    pub fn parse(line_no: usize, name: &str, pattern: &str) -> Result<Self> {
        let segments_raw: Vec<&str> = split_on_gaps(pattern);
        // split_on_gaps interleaves segment strings and gap markers; rebuild.
        let mut segments = Vec::new();
        let mut gaps = Vec::new();
        let mut it = segments_raw.into_iter();
        // First token is always a segment (possibly empty only if pattern starts with '*')
        if let Some(first) = it.next() {
            segments.push(parse_segment(line_no, first)?);
        }
        while let Some(gap_tok) = it.next() {
            gaps.push(parse_gap(line_no, gap_tok)?);
            if let Some(seg_tok) = it.next() {
                segments.push(parse_segment(line_no, seg_tok)?);
            } else {
                return Err(SigError::BadHexSig {
                    line: line_no,
                    reason: "trailing gap with no following segment".into(),
                });
            }
        }

        // Choose the best (longest) literal anchor across all segments.
        let mut best: Option<(usize, usize, Vec<u8>)> = None;
        for (seg_idx, seg) in segments.iter().enumerate() {
            if let Some((off, lit)) = seg.longest_literal_run() {
                let better = best
                    .as_ref()
                    .map(|(_, _, b)| lit.len() > b.len())
                    .unwrap_or(true);
                if better && !lit.is_empty() {
                    best = Some((seg_idx, off, lit));
                }
            }
        }
        // Require at least a 2-byte literal anchor -- anything shorter makes
        // the shared automaton useless as a selectivity filter and would
        // flood every scan with candidate positions.
        let (anchor_segment, anchor_offset, anchor_literal) = match best {
            Some((s, o, l)) if l.len() >= 2 => (s, o, l),
            _ => return Err(SigError::NoAnchor),
        };

        Ok(HexSignature {
            name: name.to_string(),
            segments,
            gaps,
            anchor_segment,
            anchor_offset,
            anchor_literal,
        })
    }

    /// Given a confirmed AC match of `anchor_literal` at buffer offset
    /// `anchor_match_pos`, verify the *entire* signature against `buf`,
    /// returning the absolute start offset of the whole match if it holds.
    pub fn verify(&self, buf: &[u8], anchor_match_pos: usize) -> Option<usize> {
        if anchor_match_pos < self.anchor_offset {
            return None;
        }
        let anchor_seg_start = anchor_match_pos - self.anchor_offset;
        let anchor_seg = &self.segments[self.anchor_segment];
        if !anchor_seg.matches_at(buf, anchor_seg_start) {
            return None;
        }

        // Walk forward from anchor to the end of the pattern.
        let mut cursor_end = anchor_seg_start + anchor_seg.len();
        for idx in self.anchor_segment..self.gaps.len() {
            let gap = self.gaps[idx];
            let next_seg = &self.segments[idx + 1];
            match find_forward(buf, next_seg, cursor_end, gap) {
                Some(pos) => cursor_end = pos + next_seg.len(),
                None => return None,
            }
        }

        // Walk backward from anchor to the start of the pattern.
        let mut cursor_start = anchor_seg_start;
        for idx in (0..self.anchor_segment).rev() {
            let gap = self.gaps[idx];
            let prev_seg = &self.segments[idx];
            match find_backward(buf, prev_seg, cursor_start, gap) {
                Some(pos) => cursor_start = pos,
                None => return None,
            }
        }

        Some(cursor_start)
    }
}

fn find_forward(buf: &[u8], seg: &Segment, after: usize, gap: Gap) -> Option<usize> {
    let (min, max) = gap_bounds(gap);
    let start = after.saturating_add(min);
    let end = after.saturating_add(max).min(buf.len());
    (start..=end).find(|&pos| seg.matches_at(buf, pos))
}

fn find_backward(buf: &[u8], seg: &Segment, before: usize, gap: Gap) -> Option<usize> {
    let (min, max) = gap_bounds(gap);
    let min_end = min + seg.len();
    // If there isn't even enough room before `before` to fit the minimum
    // required gap plus the segment itself, no valid position exists --
    // this must return None rather than silently clamping to 0, or a
    // minimum-gap constraint could be satisfied by a match that's actually
    // too close to the anchor.
    if before < min_end {
        return None;
    }
    let latest_start = before - min_end;
    let earliest_start = before.saturating_sub(max.saturating_add(seg.len()));
    (earliest_start..=latest_start)
        .rev()
        .find(|&pos| seg.matches_at(buf, pos))
}

fn gap_bounds(gap: Gap) -> (usize, usize) {
    match gap {
        Gap::Unbounded => (0, MAX_GAP_SEARCH),
        Gap::Bounded(min, max) => (min, max),
    }
}

/// Splits "AABB*CC{1-4}DD" into ["AABB", "*", "CC", "{1-4}", "DD"] style
/// alternating tokens (gap markers combine the `*` with an optional
/// following `{min-max}`).
fn split_on_gaps(pattern: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let bytes = pattern.as_bytes();
    let mut seg_start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'*' {
            out.push(&pattern[seg_start..i]);
            let gap_start = i;
            i += 1;
            if i < bytes.len() && bytes[i] == b'{' {
                while i < bytes.len() && bytes[i] != b'}' {
                    i += 1;
                }
                if i < bytes.len() {
                    i += 1; // consume '}'
                }
            }
            out.push(&pattern[gap_start..i]);
            seg_start = i;
        } else {
            i += 1;
        }
    }
    out.push(&pattern[seg_start..]);
    out
}

fn parse_gap(line_no: usize, tok: &str) -> Result<Gap> {
    if tok == "*" {
        return Ok(Gap::Unbounded);
    }
    // "*{min-max}"
    let inner = tok
        .strip_prefix('*')
        .and_then(|s| s.strip_prefix('{'))
        .and_then(|s| s.strip_suffix('}'))
        .ok_or_else(|| SigError::BadHexSig {
            line: line_no,
            reason: format!("bad gap token '{tok}'"),
        })?;
    let (min_s, max_s) = inner.split_once('-').ok_or_else(|| SigError::BadHexSig {
        line: line_no,
        reason: format!("bad gap range '{tok}'"),
    })?;
    let min: usize = min_s.parse().map_err(|_| SigError::BadHexSig {
        line: line_no,
        reason: format!("bad gap min in '{tok}'"),
    })?;
    let max: usize = max_s.parse().map_err(|_| SigError::BadHexSig {
        line: line_no,
        reason: format!("bad gap max in '{tok}'"),
    })?;
    Ok(Gap::Bounded(min, max))
}

fn parse_segment(line_no: usize, tok: &str) -> Result<Segment> {
    let bytes = tok.as_bytes();
    if bytes.len() % 2 != 0 {
        return Err(SigError::BadHexSig {
            line: line_no,
            reason: "odd number of hex digits".into(),
        });
    }
    let mut tokens = Vec::with_capacity(bytes.len() / 2);
    let mut i = 0;
    while i < bytes.len() {
        let pair = &tok[i..i + 2];
        if pair == "??" {
            tokens.push(Token::Wildcard);
        } else {
            let b = u8::from_str_radix(pair, 16).map_err(|_| SigError::BadHexSig {
                line: line_no,
                reason: format!("invalid hex byte '{pair}'"),
            })?;
            tokens.push(Token::Byte(b));
        }
        i += 2;
    }
    Ok(Segment { tokens })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_literal() {
        let sig = HexSignature::parse(1, "test.sig", "68656c6c6f").unwrap();
        assert_eq!(sig.segments.len(), 1);
        assert_eq!(sig.anchor_literal, b"hello".to_vec());
    }

    #[test]
    fn matches_with_wildcard_gap() {
        let sig = HexSignature::parse(1, "test.sig", "68656c6c6f*{0-8}776f726c64").unwrap();
        let buf = b"XX hello ---- world YY";
        // find AC-anchor position manually: "hello" at index 3
        let pos = buf.windows(5).position(|w| w == b"hello").unwrap();
        assert!(sig.verify(buf, pos).is_some());
    }

    #[test]
    fn rejects_out_of_range_gap() {
        let sig = HexSignature::parse(1, "test.sig", "68656c6c6f*{0-2}776f726c64").unwrap();
        let buf = b"hello ---- world"; // gap of 4 bytes, max allowed 2
        let pos = 0usize;
        assert!(sig.verify(buf, pos).is_none());
    }

    #[test]
    fn backward_min_gap_is_enforced() {
        // anchor segment "hello" (5 bytes, longer literal) so the "aa"
        // segment must be located via find_backward. Required min gap is
        // 20 bytes but the actual gap in the buffer is only 6 bytes -- this
        // must NOT match.
        let sig = HexSignature::parse(1, "t", "6161*{20-30}68656c6c6f").unwrap();
        let buf = b"aa XXXX hello"; // gap between "aa" and "hello" is 6 bytes
        let anchor_pos = buf.windows(5).position(|w| w == b"hello").unwrap();
        assert!(
            sig.verify(buf, anchor_pos).is_none(),
            "matched despite violating the minimum gap constraint"
        );
    }

    #[test]
    fn backward_search_matches_within_valid_gap_range() {
        // Same shape as above, but the gap (6 bytes) is within {0-30}, so
        // this one legitimately should match.
        let sig = HexSignature::parse(1, "t", "6161*{0-30}68656c6c6f").unwrap();
        let buf = b"aa XXXX hello";
        let anchor_pos = buf.windows(5).position(|w| w == b"hello").unwrap();
        assert_eq!(sig.verify(buf, anchor_pos), Some(0));
    }
}
