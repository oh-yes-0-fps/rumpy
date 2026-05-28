//! Reader/writer for the numpy `.npy` format, version 1.0.
//!
//! Format spec (https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html):
//!
//! ```text
//!   bytes  0..6      "\x93NUMPY"          magic
//!   bytes  6..8      \x01\x00             version 1.0 (major, minor)
//!   bytes  8..10     header length (LE u16)
//!   bytes  10..      ASCII header (Python dict literal), padded with spaces,
//!                    last byte is '\n'. Total header offset must be a multiple
//!                    of 64 bytes (numpy 1.10+).
//!   bytes ..         raw element data, C-contiguous unless fortran_order=True.
//! ```
//!
//! All numeric data is written little-endian. Strings inside the header use
//! the numpy dtype "descr" form (e.g. `'<f8'`, `'|b1'`).
//!
//! Version 2.0 (header length as u32) and 3.0 (UTF-8 header) are *read*
//! transparently; we only ever *write* 1.0 since our headers are ASCII and
//! fit in 16 bits.

use crate::dtype::{ArraysD, C32, C64, DType};
use half::f16;
use ndarray::{ArrayD, IxDyn};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 6] = b"\x93NUMPY";

// =====================================================================
// Save
// =====================================================================

pub fn save(path: &Path, arr: &ArraysD) -> std::io::Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    write_to(&mut w, arr)?;
    w.flush()?;
    Ok(())
}

pub fn write_to<W: Write>(w: &mut W, arr: &ArraysD) -> std::io::Result<()> {
    let descr = descr_of(arr.dtype()).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                ".npy v1 does not support dtype {}",
                arr.dtype().name_owned()
            ),
        )
    })?;
    let shape_part = shape_string(arr.shape());
    let header_body = format!(
        "{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_part}, }}"
    );

    // Total preamble = 10 bytes (magic + version + len) + header bytes
    // must end in '\n' and total length must be a multiple of 64.
    let raw_len = header_body.len() + 1; // for trailing newline
    let total = 10 + raw_len;
    let pad = (64 - (total % 64)) % 64;
    let header_len = raw_len + pad;
    debug_assert_eq!((10 + header_len) % 64, 0);

    w.write_all(MAGIC)?;
    w.write_all(&[0x01, 0x00])?; // version 1.0
    w.write_all(&(header_len as u16).to_le_bytes())?;
    w.write_all(header_body.as_bytes())?;
    for _ in 0..pad {
        w.write_all(b" ")?;
    }
    w.write_all(b"\n")?;

    write_data(w, arr)
}

fn descr_of(dt: DType) -> Option<&'static str> {
    Some(match dt {
        DType::Bool => "|b1",
        DType::I8 => "|i1",
        DType::I16 => "<i2",
        DType::I32 => "<i4",
        DType::I64 => "<i8",
        DType::U8 => "|u1",
        DType::U16 => "<u2",
        DType::U32 => "<u4",
        DType::U64 => "<u8",
        DType::F16 => "<f2",
        DType::F32 => "<f4",
        DType::F64 => "<f8",
        DType::C64 => "<c8",
        DType::C128 => "<c16",
        // .npy v1 only encodes the fixed-width numeric dtypes; non-numeric
        // arrays don't have a representable descr in this format. Caller
        // surfaces the error.
        _ => return None,
    })
}

fn shape_string(shape: &[usize]) -> String {
    match shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", shape[0]),
        _ => {
            let parts: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            format!("({})", parts.join(", "))
        }
    }
}

fn write_data<W: Write>(w: &mut W, arr: &ArraysD) -> std::io::Result<()> {
    match arr {
        ArraysD::Bool(a) => {
            for &v in a.iter() {
                w.write_all(&[v as u8])?;
            }
        }
        ArraysD::I8(a) => write_iter(w, a.iter().map(|v| v.to_le_bytes()))?,
        ArraysD::I16(a) => write_iter(w, a.iter().map(|v| v.to_le_bytes()))?,
        ArraysD::I32(a) => write_iter(w, a.iter().map(|v| v.to_le_bytes()))?,
        ArraysD::I64(a) => write_iter(w, a.iter().map(|v| v.to_le_bytes()))?,
        ArraysD::U8(a) => write_iter(w, a.iter().map(|v| v.to_le_bytes()))?,
        ArraysD::U16(a) => write_iter(w, a.iter().map(|v| v.to_le_bytes()))?,
        ArraysD::U32(a) => write_iter(w, a.iter().map(|v| v.to_le_bytes()))?,
        ArraysD::U64(a) => write_iter(w, a.iter().map(|v| v.to_le_bytes()))?,
        ArraysD::F16(a) => {
            for &v in a.iter() {
                w.write_all(&v.to_le_bytes())?;
            }
        }
        ArraysD::F32(a) => write_iter(w, a.iter().map(|v| v.to_le_bytes()))?,
        ArraysD::F64(a) => write_iter(w, a.iter().map(|v| v.to_le_bytes()))?,
        ArraysD::C64(a) => {
            for &v in a.iter() {
                w.write_all(&v.re.to_le_bytes())?;
                w.write_all(&v.im.to_le_bytes())?;
            }
        }
        ArraysD::C128(a) => {
            for &v in a.iter() {
                w.write_all(&v.re.to_le_bytes())?;
                w.write_all(&v.im.to_le_bytes())?;
            }
        }
        // Non-numeric dtypes can't be serialised by the .npy v1 binary
        // format. Caller already checks `descr_of`; this branch is a
        // belt-and-braces sanity error.
        other => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    ".npy v1 write: dtype {} not supported",
                    other.dtype().name_owned()
                ),
            ));
        }
    }
    Ok(())
}

fn write_iter<W, I, const N: usize>(w: &mut W, it: I) -> std::io::Result<()>
where
    W: Write,
    I: Iterator<Item = [u8; N]>,
{
    for bytes in it {
        w.write_all(&bytes)?;
    }
    Ok(())
}

// =====================================================================
// Load
// =====================================================================

#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    Format(String),
}

impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        LoadError::Io(e)
    }
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "io error: {e}"),
            LoadError::Format(s) => write!(f, "format error: {s}"),
        }
    }
}

pub fn load(path: &Path) -> Result<ArraysD, LoadError> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);
    read_from(&mut r)
}

pub fn read_from<R: Read>(r: &mut R) -> Result<ArraysD, LoadError> {
    let mut magic = [0u8; 6];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(LoadError::Format("not a .npy file (bad magic)".to_string()));
    }
    let mut ver = [0u8; 2];
    r.read_exact(&mut ver)?;
    let header_len = match ver {
        [1, 0] => {
            let mut b = [0u8; 2];
            r.read_exact(&mut b)?;
            u16::from_le_bytes(b) as usize
        }
        [2, 0] | [3, 0] => {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            u32::from_le_bytes(b) as usize
        }
        v => {
            return Err(LoadError::Format(format!(
                "unsupported .npy version {}.{}",
                v[0], v[1]
            )));
        }
    };

    let mut header = vec![0u8; header_len];
    r.read_exact(&mut header)?;
    let header_str = std::str::from_utf8(&header)
        .map_err(|e| LoadError::Format(format!("header utf-8: {e}")))?;

    let (descr, fortran_order, shape) = parse_header(header_str)?;
    if fortran_order {
        return Err(LoadError::Format(
            "fortran_order=True .npy files are not supported yet".to_string(),
        ));
    }
    let dtype = parse_descr(&descr)?;
    let nelem: usize = shape.iter().product::<usize>().max(if shape.is_empty() { 1 } else { 0 });
    let nelem = if shape.is_empty() { 1 } else { nelem };
    let arr = read_data(r, dtype, nelem, &shape)?;
    Ok(arr)
}

fn parse_descr(s: &str) -> Result<DType, LoadError> {
    DType::parse(s).ok_or_else(|| LoadError::Format(format!("unknown dtype descr {s:?}")))
}

/// Lightweight parser for the three header values we care about.
fn parse_header(s: &str) -> Result<(String, bool, Vec<usize>), LoadError> {
    let descr = get_str_value(s, "descr")?;
    let fortran = get_bool_value(s, "fortran_order")?;
    let shape = get_tuple_value(s, "shape")?;
    Ok((descr, fortran, shape))
}

fn get_str_value(s: &str, key: &str) -> Result<String, LoadError> {
    let needle = format!("'{key}'");
    let i = s
        .find(&needle)
        .ok_or_else(|| LoadError::Format(format!("missing key {key}")))?;
    let rest = &s[i + needle.len()..];
    let colon = rest.find(':').ok_or_else(|| LoadError::Format("bad header".into()))?;
    let rest = &rest[colon + 1..];
    let q = rest
        .find('\'')
        .ok_or_else(|| LoadError::Format("descr quote".into()))?;
    let after = &rest[q + 1..];
    let end = after
        .find('\'')
        .ok_or_else(|| LoadError::Format("descr end quote".into()))?;
    Ok(after[..end].to_string())
}

fn get_bool_value(s: &str, key: &str) -> Result<bool, LoadError> {
    let needle = format!("'{key}'");
    let i = s
        .find(&needle)
        .ok_or_else(|| LoadError::Format(format!("missing key {key}")))?;
    let after = &s[i + needle.len()..];
    // `contains("True")` proves a match exists, but defensive-style: pull
    // the offset via `map_or` so a future refactor of `.contains/.find`
    // semantics can't introduce a panic.
    if let Some(true_pos) = after.find("True") {
        if true_pos < after.find(',').unwrap_or(after.len()) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn get_tuple_value(s: &str, key: &str) -> Result<Vec<usize>, LoadError> {
    let needle = format!("'{key}'");
    let i = s
        .find(&needle)
        .ok_or_else(|| LoadError::Format(format!("missing key {key}")))?;
    let rest = &s[i + needle.len()..];
    let open = rest
        .find('(')
        .ok_or_else(|| LoadError::Format("shape ( missing".into()))?;
    let close = rest[open..]
        .find(')')
        .ok_or_else(|| LoadError::Format("shape ) missing".into()))?;
    let inner = &rest[open + 1..open + close];
    let parts: Vec<&str> = inner
        .split(',')
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .collect();
    let mut out = Vec::with_capacity(parts.len());
    for p in parts {
        let n: usize = p
            .parse()
            .map_err(|_| LoadError::Format(format!("bad shape dim {p:?}")))?;
        out.push(n);
    }
    Ok(out)
}

fn read_data<R: Read>(
    r: &mut R,
    dt: DType,
    nelem: usize,
    shape: &[usize],
) -> Result<ArraysD, LoadError> {
    let shape_dyn = IxDyn(shape);
    Ok(match dt {
        DType::Bool => {
            let mut v = vec![0u8; nelem];
            r.read_exact(&mut v)?;
            let bools: Vec<bool> = v.into_iter().map(|b| b != 0).collect();
            ArraysD::Bool(ArrayD::from_shape_vec(shape_dyn, bools).unwrap_or_default())
        }
        DType::I8 => {
            let mut v = vec![0u8; nelem];
            r.read_exact(&mut v)?;
            let cast: Vec<i8> = v.into_iter().map(|b| b as i8).collect();
            ArraysD::I8(ArrayD::from_shape_vec(shape_dyn, cast).unwrap_or_default())
        }
        DType::U8 => {
            let mut v = vec![0u8; nelem];
            r.read_exact(&mut v)?;
            ArraysD::U8(ArrayD::from_shape_vec(shape_dyn, v).unwrap_or_default())
        }
        DType::I16 => read_le::<R, i16, 2>(r, nelem, shape_dyn, i16::from_le_bytes, ArraysD::I16)?,
        DType::U16 => read_le::<R, u16, 2>(r, nelem, shape_dyn, u16::from_le_bytes, ArraysD::U16)?,
        DType::F16 => read_le::<R, f16, 2>(r, nelem, shape_dyn, f16::from_le_bytes, ArraysD::F16)?,
        DType::I32 => read_le::<R, i32, 4>(r, nelem, shape_dyn, i32::from_le_bytes, ArraysD::I32)?,
        DType::U32 => read_le::<R, u32, 4>(r, nelem, shape_dyn, u32::from_le_bytes, ArraysD::U32)?,
        DType::F32 => read_le::<R, f32, 4>(r, nelem, shape_dyn, f32::from_le_bytes, ArraysD::F32)?,
        DType::I64 => read_le::<R, i64, 8>(r, nelem, shape_dyn, i64::from_le_bytes, ArraysD::I64)?,
        DType::U64 => read_le::<R, u64, 8>(r, nelem, shape_dyn, u64::from_le_bytes, ArraysD::U64)?,
        DType::F64 => read_le::<R, f64, 8>(r, nelem, shape_dyn, f64::from_le_bytes, ArraysD::F64)?,
        DType::C64 => {
            let mut v = Vec::with_capacity(nelem);
            for _ in 0..nelem {
                let mut b = [0u8; 8];
                r.read_exact(&mut b)?;
                let re = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                let im = f32::from_le_bytes([b[4], b[5], b[6], b[7]]);
                v.push(C32::new(re, im));
            }
            ArraysD::C64(ArrayD::from_shape_vec(shape_dyn, v).unwrap_or_default())
        }
        DType::C128 => {
            let mut v = Vec::with_capacity(nelem);
            for _ in 0..nelem {
                let mut b = [0u8; 16];
                r.read_exact(&mut b)?;
                let re = f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
                let im = f64::from_le_bytes([b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]]);
                v.push(C64::new(re, im));
            }
            ArraysD::C128(ArrayD::from_shape_vec(shape_dyn, v).unwrap_or_default())
        }
        // Non-numeric dtypes aren't supported by .npy v1 — surface a clean
        // format error.
        other => {
            return Err(LoadError::Format(format!(
                ".npy v1 read: dtype {} not supported",
                other.name_owned()
            )));
        }
    })
}

fn read_le<R, T, const N: usize>(
    r: &mut R,
    nelem: usize,
    shape: IxDyn,
    decode: fn([u8; N]) -> T,
    wrap: fn(ArrayD<T>) -> ArraysD,
) -> Result<ArraysD, LoadError>
where
    R: Read,
{
    let mut v = Vec::with_capacity(nelem);
    let mut buf = [0u8; N];
    for _ in 0..nelem {
        r.read_exact(&mut buf)?;
        v.push(decode(buf));
    }
    let arr = ArrayD::from_shape_vec(shape, v)
        .map_err(|e| LoadError::Format(format!("from_shape_vec: {e}")))?;
    Ok(wrap(arr))
}

// =====================================================================
// In-memory variants (handy for .npz support and tests)
// =====================================================================

pub fn save_to_bytes(arr: &ArraysD) -> Vec<u8> {
    let mut buf = Vec::new();
    // Writing to a Vec<u8> is infallible in practice, but if it ever does
    // fail we drop the buffer rather than panic — callers see an empty
    // payload, which `load_from_bytes` will reject with LoadError.
    let _ = write_to(&mut buf, arr);
    buf
}

pub fn load_from_bytes(buf: &[u8]) -> Result<ArraysD, LoadError> {
    let mut cursor = std::io::Cursor::new(buf);
    read_from(&mut cursor)
}
