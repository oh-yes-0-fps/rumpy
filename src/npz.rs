//! Reader/writer for the numpy `.npz` format.
//!
//! `.npz` is a plain (or DEFLATEd) ZIP archive whose members are individual
//! `.npy` files. `np.savez` uses STORED (no compression); we do the same.
//! `np.load` on a `.npz` returns a lazy `NpzFile`; for embedded use we
//! return a `dict[str, ndarray]`.
//!
//! We implement just enough of the ZIP spec to round-trip with CPython:
//!
//! ```text
//!   per file:
//!     local file header     (30 bytes + filename)
//!     file data             (raw .npy bytes)
//!   end:
//!     central directory     (one entry per file, 46 bytes + filename each)
//!     end-of-central-directory record (22 bytes)
//! ```
//!
//! No compression, no encryption, no zip64, no extra fields.

use crate::dtype::ArraysD;
use crate::npy;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

const LFH_SIG: u32 = 0x04034b50;
const CDH_SIG: u32 = 0x02014b50;
const EOCD_SIG: u32 = 0x06054b50;

// =====================================================================
// Write
// =====================================================================

pub fn save(path: &Path, named: &[(String, ArraysD)]) -> std::io::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    write_to(&mut f, named)?;
    f.flush()?;
    Ok(())
}

/// Write an .npz to any `Write + Seek`-like target by buffering into a Vec
/// internally (so we don't need Seek).
pub fn write_to<W: Write>(w: &mut W, named: &[(String, ArraysD)]) -> std::io::Result<()> {
    let mut offset: u32 = 0;
    let mut cd_entries: Vec<Vec<u8>> = Vec::with_capacity(named.len());

    for (name, arr) in named {
        let payload = npy::save_to_bytes(arr);
        let fname = format!("{name}.npy");
        let crc = crc32_ieee(&payload);
        let local_offset = offset;

        // Local file header
        let fname_bytes = fname.as_bytes();
        w.write_all(&LFH_SIG.to_le_bytes())?;
        w.write_all(&20u16.to_le_bytes())?; // version needed
        w.write_all(&0u16.to_le_bytes())?; // gp flag
        w.write_all(&0u16.to_le_bytes())?; // method = STORED
        w.write_all(&0u16.to_le_bytes())?; // mtime
        w.write_all(&0u16.to_le_bytes())?; // mdate
        w.write_all(&crc.to_le_bytes())?;
        w.write_all(&(payload.len() as u32).to_le_bytes())?; // compressed
        w.write_all(&(payload.len() as u32).to_le_bytes())?; // uncompressed
        w.write_all(&(fname_bytes.len() as u16).to_le_bytes())?;
        w.write_all(&0u16.to_le_bytes())?; // extra len
        w.write_all(fname_bytes)?;
        w.write_all(&payload)?;
        offset += 30 + fname_bytes.len() as u32 + payload.len() as u32;

        // Stash the corresponding central directory entry for the trailer.
        let mut cd = Vec::with_capacity(46 + fname_bytes.len());
        cd.extend_from_slice(&CDH_SIG.to_le_bytes());
        cd.extend_from_slice(&20u16.to_le_bytes()); // version made by
        cd.extend_from_slice(&20u16.to_le_bytes()); // version needed
        cd.extend_from_slice(&0u16.to_le_bytes()); // gp flag
        cd.extend_from_slice(&0u16.to_le_bytes()); // method
        cd.extend_from_slice(&0u16.to_le_bytes()); // mtime
        cd.extend_from_slice(&0u16.to_le_bytes()); // mdate
        cd.extend_from_slice(&crc.to_le_bytes());
        cd.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        cd.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        cd.extend_from_slice(&(fname_bytes.len() as u16).to_le_bytes());
        cd.extend_from_slice(&0u16.to_le_bytes()); // extra len
        cd.extend_from_slice(&0u16.to_le_bytes()); // comment len
        cd.extend_from_slice(&0u16.to_le_bytes()); // disk #
        cd.extend_from_slice(&0u16.to_le_bytes()); // internal attrs
        cd.extend_from_slice(&0u32.to_le_bytes()); // external attrs
        cd.extend_from_slice(&local_offset.to_le_bytes());
        cd.extend_from_slice(fname_bytes);
        cd_entries.push(cd);
    }

    let cd_offset = offset;
    let mut cd_size: u32 = 0;
    for entry in &cd_entries {
        w.write_all(entry)?;
        cd_size += entry.len() as u32;
    }

    // End of central directory record.
    w.write_all(&EOCD_SIG.to_le_bytes())?;
    w.write_all(&0u16.to_le_bytes())?; // disk #
    w.write_all(&0u16.to_le_bytes())?; // disk start
    w.write_all(&(cd_entries.len() as u16).to_le_bytes())?;
    w.write_all(&(cd_entries.len() as u16).to_le_bytes())?;
    w.write_all(&cd_size.to_le_bytes())?;
    w.write_all(&cd_offset.to_le_bytes())?;
    w.write_all(&0u16.to_le_bytes())?; // comment length

    Ok(())
}

// =====================================================================
// Read
// =====================================================================

#[derive(Debug)]
pub enum NpzError {
    Io(std::io::Error),
    Format(String),
    Compression,
}

impl From<std::io::Error> for NpzError {
    fn from(e: std::io::Error) -> Self {
        NpzError::Io(e)
    }
}

impl std::fmt::Display for NpzError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NpzError::Io(e) => write!(f, "io: {e}"),
            NpzError::Format(s) => write!(f, "format: {s}"),
            NpzError::Compression => write!(f, "compressed .npz not supported"),
        }
    }
}

pub fn load(path: &Path) -> Result<Vec<(String, ArraysD)>, NpzError> {
    let mut data = Vec::new();
    File::open(path)?.read_to_end(&mut data)?;
    read_from(&data)
}

pub fn read_from(data: &[u8]) -> Result<Vec<(String, ArraysD)>, NpzError> {
    let eocd = find_eocd(data)
        .ok_or_else(|| NpzError::Format("no end-of-central-directory".to_string()))?;
    let entries = u16::from_le_bytes([data[eocd + 10], data[eocd + 11]]) as usize;
    let cd_offset = u32::from_le_bytes([
        data[eocd + 16],
        data[eocd + 17],
        data[eocd + 18],
        data[eocd + 19],
    ]) as usize;
    let mut out = Vec::with_capacity(entries);

    let mut p = cd_offset;
    for _ in 0..entries {
        if data[p..p + 4] != CDH_SIG.to_le_bytes() {
            return Err(NpzError::Format("bad central directory header".to_string()));
        }
        let method = u16::from_le_bytes([data[p + 10], data[p + 11]]);
        if method != 0 {
            return Err(NpzError::Compression);
        }
        let comp_size =
            u32::from_le_bytes([data[p + 20], data[p + 21], data[p + 22], data[p + 23]]) as usize;
        let name_len = u16::from_le_bytes([data[p + 28], data[p + 29]]) as usize;
        let extra_len = u16::from_le_bytes([data[p + 30], data[p + 31]]) as usize;
        let comment_len = u16::from_le_bytes([data[p + 32], data[p + 33]]) as usize;
        let local_offset = u32::from_le_bytes([
            data[p + 42],
            data[p + 43],
            data[p + 44],
            data[p + 45],
        ]) as usize;
        let name = std::str::from_utf8(&data[p + 46..p + 46 + name_len])
            .map_err(|_| NpzError::Format("non-utf8 name".to_string()))?
            .to_owned();
        p += 46 + name_len + extra_len + comment_len;

        // Local file header: 30 bytes + filename + extra (skip), then data.
        if data[local_offset..local_offset + 4] != LFH_SIG.to_le_bytes() {
            return Err(NpzError::Format("bad local file header".to_string()));
        }
        let l_name_len = u16::from_le_bytes([data[local_offset + 26], data[local_offset + 27]]) as usize;
        let l_extra_len =
            u16::from_le_bytes([data[local_offset + 28], data[local_offset + 29]]) as usize;
        let data_start = local_offset + 30 + l_name_len + l_extra_len;
        let blob = &data[data_start..data_start + comp_size];

        let arr = crate::npy::load_from_bytes(blob)
            .map_err(|e| NpzError::Format(format!("inner .npy: {e}")))?;

        let stripped = name.trim_end_matches(".npy").to_owned();
        out.push((stripped, arr));
    }

    Ok(out)
}

fn find_eocd(data: &[u8]) -> Option<usize> {
    // EOCD is at least 22 bytes; comment can extend it. Scan backwards.
    if data.len() < 22 {
        return None;
    }
    let max = data.len();
    let start = max.saturating_sub(22 + 0xFFFF);
    for i in (start..=max - 22).rev() {
        if data[i..i + 4] == EOCD_SIG.to_le_bytes() {
            return Some(i);
        }
    }
    None
}

// =====================================================================
// CRC-32 (IEEE polynomial 0xedb88320)
// =====================================================================

fn crc32_ieee(buf: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for n in 0..256u32 {
            let mut c = n;
            for _ in 0..8 {
                c = if c & 1 == 1 {
                    0xedb88320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
            t[n as usize] = c;
        }
        t
    });
    let mut crc = 0xFFFFFFFFu32;
    for &b in buf {
        let idx = ((crc ^ b as u32) & 0xFF) as usize;
        crc = table[idx] ^ (crc >> 8);
    }
    crc ^ 0xFFFFFFFF
}
