//! Minimal, dependency-free PE parser used purely for heuristic scoring
//! (entropy-based packer detection, suspicious section flag combos).
//!
//! Every field access goes through bounds-checked slice reads (`get_*`
//! helpers below) rather than raw pointer casts over the file buffer --
//! this is precisely the class of code where ClamAV's C parsers have
//! historically had OOB-read CVEs against malformed PE input. A malformed
//! or truncated PE here can only produce `Err(PeError::..)`, never UB.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PeError {
    #[error("buffer too small for DOS header")]
    TruncatedDos,
    #[error("bad DOS magic (not MZ)")]
    BadDosMagic,
    #[error("e_lfanew points outside buffer")]
    BadPeOffset,
    #[error("bad PE magic (not 'PE\\0\\0')")]
    BadPeMagic,
    #[error("buffer too small for COFF/optional header")]
    TruncatedHeaders,
    #[error("section table extends past buffer end")]
    TruncatedSections,
}

type Result<T> = std::result::Result<T, PeError>;

fn u16_at(buf: &[u8], off: usize) -> Option<u16> {
    buf.get(off..off + 2)
        .map(|s| u16::from_le_bytes([s[0], s[1]]))
}
fn u32_at(buf: &[u8], off: usize) -> Option<u32> {
    buf.get(off..off + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

#[derive(Debug, Clone)]
pub struct SectionInfo {
    pub name: String,
    pub virtual_size: u32,
    pub raw_size: u32,
    pub raw_offset: u32,
    pub characteristics: u32,
    pub entropy: f64,
}

impl SectionInfo {
    pub fn is_executable(&self) -> bool {
        self.characteristics & 0x2000_0000 != 0 // IMAGE_SCN_MEM_EXECUTE
    }
    pub fn is_writable(&self) -> bool {
        self.characteristics & 0x8000_0000 != 0 // IMAGE_SCN_MEM_WRITE
    }
}

#[derive(Debug, Clone)]
pub struct PeInfo {
    pub machine: u16,
    pub number_of_sections: u16,
    pub entry_point_rva: u32,
    pub sections: Vec<SectionInfo>,
}

#[derive(Debug, Clone, Default)]
pub struct Heuristics {
    pub high_entropy_sections: Vec<String>,
    pub writable_and_executable_sections: Vec<String>,
    pub entry_point_outside_sections: bool,
    pub suspiciously_few_sections: bool,
    pub score: u32,
}

const SECTION_HEADER_SIZE: usize = 40;

pub fn parse(buf: &[u8]) -> Result<PeInfo> {
    if buf.len() < 64 {
        return Err(PeError::TruncatedDos);
    }
    if &buf[0..2] != b"MZ" {
        return Err(PeError::BadDosMagic);
    }
    let e_lfanew = u32_at(buf, 0x3C).ok_or(PeError::TruncatedDos)? as usize;
    // Use checked arithmetic throughout this function rather than plain `+`.
    // `e_lfanew` is attacker-controlled (read straight from the file), and on
    // a 32-bit target `usize` is only 32 bits wide, so `e_lfanew + N` can wrap
    // around for a large enough crafted offset. A wrapped value could slip
    // past the bounds check below and then get used as a valid-looking (but
    // actually out-of-bounds) slice start. Every offset derived from
    // attacker-controlled fields is `checked_add`ed instead.
    let pe_header_end = e_lfanew
        .checked_add(4 + 20 + 2)
        .ok_or(PeError::BadPeOffset)?;
    if pe_header_end > buf.len() {
        return Err(PeError::BadPeOffset);
    }
    if &buf[e_lfanew..e_lfanew + 4] != b"PE\0\0" {
        return Err(PeError::BadPeMagic);
    }

    let coff_off = e_lfanew.checked_add(4).ok_or(PeError::BadPeOffset)?;
    let machine = u16_at(buf, coff_off).ok_or(PeError::TruncatedHeaders)?;
    let number_of_sections = u16_at(buf, coff_off + 2).ok_or(PeError::TruncatedHeaders)?;
    let size_of_optional_header =
        u16_at(buf, coff_off + 16).ok_or(PeError::TruncatedHeaders)? as usize;

    let opt_header_off = coff_off.checked_add(20).ok_or(PeError::TruncatedHeaders)?;
    // AddressOfEntryPoint lives at offset 16 within the optional header for
    // both PE32 and PE32+, so anything claiming a smaller optional header
    // doesn't actually have room for it -- treat that as truncated rather
    // than reading past the declared header into whatever follows (which
    // would still be bounds-checked and thus safe, but could silently
    // misinterpret section-table bytes as an entry point).
    if size_of_optional_header < 20 {
        return Err(PeError::TruncatedHeaders);
    }
    if opt_header_off
        .checked_add(size_of_optional_header)
        .map_or(true, |end| end > buf.len())
    {
        return Err(PeError::TruncatedHeaders);
    }
    // AddressOfEntryPoint sits at offset 16 within the optional header for
    // both PE32 and PE32+.
    let entry_point_rva = u32_at(buf, opt_header_off + 16).ok_or(PeError::TruncatedHeaders)?;

    let section_table_off = opt_header_off
        .checked_add(size_of_optional_header)
        .ok_or(PeError::TruncatedSections)?;
    let table_bytes = (number_of_sections as usize)
        .checked_mul(SECTION_HEADER_SIZE)
        .ok_or(PeError::TruncatedSections)?;
    let sections_end = section_table_off
        .checked_add(table_bytes)
        .ok_or(PeError::TruncatedSections)?;
    if sections_end > buf.len() {
        return Err(PeError::TruncatedSections);
    }

    let mut sections = Vec::with_capacity(number_of_sections as usize);
    for i in 0..number_of_sections as usize {
        let base = section_table_off + i * SECTION_HEADER_SIZE;
        let name_bytes = &buf[base..base + 8];
        let name = String::from_utf8_lossy(name_bytes)
            .trim_end_matches('\0')
            .to_string();
        let virtual_size = u32_at(buf, base + 8).unwrap_or(0);
        let raw_size = u32_at(buf, base + 16).unwrap_or(0);
        let raw_offset = u32_at(buf, base + 20).unwrap_or(0);
        let characteristics = u32_at(buf, base + 36).unwrap_or(0);

        let entropy = {
            let start = raw_offset as usize;
            let len = raw_size as usize;
            match buf.get(start..start.saturating_add(len).min(buf.len())) {
                Some(slice) if !slice.is_empty() => shannon_entropy(slice),
                _ => 0.0,
            }
        };

        sections.push(SectionInfo {
            name,
            virtual_size,
            raw_size,
            raw_offset,
            characteristics,
            entropy,
        });
    }

    Ok(PeInfo {
        machine,
        number_of_sections,
        entry_point_rva,
        sections,
    })
}

fn shannon_entropy(data: &[u8]) -> f64 {
    let mut counts = [0u64; 256];
    for &b in data {
        counts[b as usize] += 1;
    }
    let len = data.len() as f64;
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / len;
            -p * p.log2()
        })
        .sum()
}

/// Score a parsed PE for packer-like / suspicious structural traits.
/// This mirrors ClamAV's structural heuristics (`Heuristics.Packed.*`)
/// without needing a packer signature database at all.
pub fn heuristics(info: &PeInfo) -> Heuristics {
    let mut h = Heuristics::default();

    for s in &info.sections {
        // >= 7.2 bits/byte on a reasonably sized section is a strong signal
        // of compression or encryption (i.e. packing).
        if s.entropy >= 7.2 && s.raw_size >= 256 {
            h.high_entropy_sections.push(s.name.clone());
            h.score += 30;
        }
        if s.is_executable() && s.is_writable() {
            h.writable_and_executable_sections.push(s.name.clone());
            h.score += 40;
        }
    }

    let entry_in_section = info.sections.iter().any(|s| {
        let start = s.raw_offset;
        let end = s.raw_offset.saturating_add(s.virtual_size.max(s.raw_size));
        info.entry_point_rva >= start && info.entry_point_rva < end
    });
    if !entry_in_section && !info.sections.is_empty() {
        h.entry_point_outside_sections = true;
        h.score += 25;
    }

    if info.number_of_sections <= 2 {
        h.suspiciously_few_sections = true;
        h.score += 10;
    }

    h
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal, well-formed PE buffer with one section so the
    /// happy-path parser logic has something real to chew on.
    fn build_pe(section_data: &[u8], characteristics: u32) -> Vec<u8> {
        let mut buf = vec![0u8; 64];
        buf[0] = b'M';
        buf[1] = b'Z';
        let e_lfanew: u32 = 64;
        buf[0x3C..0x40].copy_from_slice(&e_lfanew.to_le_bytes());

        buf.extend_from_slice(b"PE\0\0");

        let machine: u16 = 0x8664;
        let number_of_sections: u16 = 1;
        let size_of_optional_header: u16 = 24;
        let coff_characteristics: u16 = 0x0102;
        buf.extend_from_slice(&machine.to_le_bytes());
        buf.extend_from_slice(&number_of_sections.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // time date stamp
        buf.extend_from_slice(&0u32.to_le_bytes()); // ptr to symtab
        buf.extend_from_slice(&0u32.to_le_bytes()); // num symtab
        buf.extend_from_slice(&size_of_optional_header.to_le_bytes());
        buf.extend_from_slice(&coff_characteristics.to_le_bytes());

        let mut opt = vec![0u8; size_of_optional_header as usize];
        opt[0..2].copy_from_slice(&0x20bu16.to_le_bytes());
        opt[16..20].copy_from_slice(&0x1000u32.to_le_bytes()); // entry point RVA
        buf.extend_from_slice(&opt);

        let raw_offset = buf.len() as u32 + SECTION_HEADER_SIZE as u32;
        let mut name = [0u8; 8];
        name[..5].copy_from_slice(b".text");
        buf.extend_from_slice(&name);
        buf.extend_from_slice(&(section_data.len() as u32).to_le_bytes()); // virtual size
        buf.extend_from_slice(&0x1000u32.to_le_bytes()); // virtual addr
        buf.extend_from_slice(&(section_data.len() as u32).to_le_bytes()); // raw size
        buf.extend_from_slice(&raw_offset.to_le_bytes()); // raw ptr
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&characteristics.to_le_bytes());

        buf.extend_from_slice(section_data);
        buf
    }

    #[test]
    fn parses_valid_minimal_pe() {
        let data = vec![0x90u8; 512]; // NOP sled: low entropy
        let buf = build_pe(&data, 0x6000_0020);
        let info = parse(&buf).expect("should parse");
        assert_eq!(info.number_of_sections, 1);
        assert_eq!(info.sections[0].name, ".text");
        assert_eq!(info.entry_point_rva, 0x1000);
    }

    #[test]
    fn flags_high_entropy_section_as_packed() {
        // Pseudo-random bytes to simulate packed/encrypted content.
        let mut data = vec![0u8; 4096];
        let mut x: u32 = 0x12345678;
        for b in data.iter_mut() {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *b = (x & 0xFF) as u8;
        }
        let buf = build_pe(&data, 0x6000_0020);
        let info = parse(&buf).unwrap();
        let h = heuristics(&info);
        assert!(
            h.high_entropy_sections.contains(&".text".to_string()),
            "expected high-entropy section to be flagged, got {h:?}"
        );
        assert!(h.score > 0);
    }

    #[test]
    fn low_entropy_section_not_flagged_as_packed() {
        let data = vec![0x90u8; 4096];
        let buf = build_pe(&data, 0x6000_0020);
        let info = parse(&buf).unwrap();
        let h = heuristics(&info);
        assert!(h.high_entropy_sections.is_empty());
    }

    #[test]
    fn flags_writable_executable_section() {
        let data = vec![0x90u8; 512];
        // MEM_EXECUTE | MEM_WRITE | MEM_READ
        let buf = build_pe(&data, 0x2000_0000 | 0x8000_0000 | 0x4000_0000);
        let info = parse(&buf).unwrap();
        let h = heuristics(&info);
        assert_eq!(
            h.writable_and_executable_sections,
            vec![".text".to_string()]
        );
    }

    #[test]
    fn rejects_maximal_e_lfanew_without_overflow_panic() {
        // e_lfanew = u32::MAX exercises the checked-arithmetic path in the
        // offset math (would wrap a 32-bit usize on a 32-bit target); this
        // must cleanly error rather than panic or, worse, wrap around to a
        // small in-bounds-looking offset.
        let mut buf = vec![0u8; 64];
        buf[0] = b'M';
        buf[1] = b'Z';
        buf[0x3C..0x40].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(parse(&buf), Err(PeError::BadPeOffset)));
    }

    #[test]
    fn rejects_truncated_buffer() {
        assert!(matches!(parse(&[]), Err(PeError::TruncatedDos)));
        assert!(matches!(parse(&[0u8; 10]), Err(PeError::TruncatedDos)));
    }

    #[test]
    fn rejects_bad_dos_magic() {
        let buf = vec![0u8; 64];
        assert!(matches!(parse(&buf), Err(PeError::BadDosMagic)));
    }

    #[test]
    fn rejects_bad_pe_offset() {
        let mut buf = vec![0u8; 64];
        buf[0] = b'M';
        buf[1] = b'Z';
        // e_lfanew points way past the end of a tiny buffer.
        buf[0x3C..0x40].copy_from_slice(&0x00FF_FFFFu32.to_le_bytes());
        assert!(matches!(parse(&buf), Err(PeError::BadPeOffset)));
    }

    #[test]
    fn rejects_undersized_optional_header_without_panicking() {
        // A crafted header that declares an optional-header size too small
        // to contain AddressOfEntryPoint must error, not read adjacent
        // bytes or panic.
        let mut buf = vec![0u8; 64];
        buf[0] = b'M';
        buf[1] = b'Z';
        let e_lfanew: u32 = 64;
        buf[0x3C..0x40].copy_from_slice(&e_lfanew.to_le_bytes());
        buf.extend_from_slice(b"PE\0\0");
        buf.extend_from_slice(&0x8664u16.to_le_bytes()); // machine
        buf.extend_from_slice(&1u16.to_le_bytes()); // sections
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&4u16.to_le_bytes()); // size_of_optional_header = 4 (too small)
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&[0u8; 4]); // the undersized "optional header"
        assert!(matches!(parse(&buf), Err(PeError::TruncatedHeaders)));
    }
}
