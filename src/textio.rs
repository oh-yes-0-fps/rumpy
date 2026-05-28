//! `numpy.savetxt` / `loadtxt` (text format) and `tofile` / `fromfile`
//! (raw binary).

use crate::dtype::{ArraysD, DType};
use ndarray::{ArrayD, IxDyn};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

// =====================================================================
// savetxt / loadtxt
// =====================================================================

pub fn savetxt(
    path: &Path,
    a: &ArraysD,
    delimiter: &str,
    header: Option<&str>,
    comments: &str,
    fmt: &str,
) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    if let Some(h) = header {
        for line in h.split('\n') {
            writeln!(w, "{comments}{line}")?;
        }
    }
    // Numpy supports 1-D (one row, "fmt" per value) or 2-D (rows × cols).
    let f = match a.cast(DType::F64) {
        ArraysD::F64(x) => x,
        _ => return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "savetxt: cast to F64 failed",
        )),
    };
    match f.ndim() {
        0 => {
            writeln!(w, "{}", fmt_value(f[IxDyn(&[])], fmt))?;
        }
        1 => {
            // numpy writes 1-D as a single column.
            for &v in f.iter() {
                writeln!(w, "{}", fmt_value(v, fmt))?;
            }
        }
        2 => {
            let m = f.shape()[0];
            let n = f.shape()[1];
            for i in 0..m {
                let row: Vec<String> =
                    (0..n).map(|j| fmt_value(f[IxDyn(&[i, j])], fmt)).collect();
                writeln!(w, "{}", row.join(delimiter))?;
            }
        }
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "savetxt: only 0/1/2-D arrays supported",
            ));
        }
    }
    w.flush()
}

fn fmt_value(v: f64, fmt: &str) -> String {
    // We honour a small useful subset of numpy's fmt strings:
    //   - "%.<n>e"   → scientific, n decimals
    //   - "%.<n>f"   → fixed, n decimals
    //   - "%g" or "%.<n>g" → shortest reasonable
    //   - "%d" → integer-rounded
    //   - default fall-through: Rust Display
    let s = fmt;
    if !s.starts_with('%') {
        return format!("{v}");
    }
    let body = &s[1..];
    // Pull the precision if it's `.<n>`.
    let mut prec: Option<usize> = None;
    let body = if let Some(rest) = body.strip_prefix('.') {
        let dig_end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
        prec = rest[..dig_end].parse().ok();
        &rest[dig_end..]
    } else {
        body
    };
    match body {
        "e" => format!("{v:.*e}", prec.unwrap_or(6)),
        "E" => format!("{v:.*E}", prec.unwrap_or(6)),
        "f" => format!("{v:.*}", prec.unwrap_or(6)),
        "g" | "G" => match prec {
            Some(p) => {
                let s1 = format!("{v:.*e}", p);
                let s2 = format!("{v:.*}", p);
                if s2.len() <= s1.len() { s2 } else { s1 }
            }
            None => format!("{v}"),
        },
        "d" | "i" => format!("{}", v.round() as i64),
        _ => format!("{v}"),
    }
}

pub fn loadtxt(
    path: &Path,
    delimiter: Option<&str>,
    comments: &str,
    skiprows: usize,
) -> std::io::Result<ArraysD> {
    let f = File::open(path)?;
    let r = BufReader::new(f);
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for (i, line) in r.lines().enumerate() {
        let line = line?;
        if i < skiprows {
            continue;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with(comments) {
            continue;
        }
        let parts: Vec<&str> = match delimiter {
            Some(d) => trimmed.split(d).collect(),
            None => trimmed.split_ascii_whitespace().collect(),
        };
        let mut row = Vec::with_capacity(parts.len());
        for p in parts {
            let p = p.trim();
            if p.is_empty() {
                continue;
            }
            let v: f64 = p.parse().map_err(|e: std::num::ParseFloatError| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
            })?;
            row.push(v);
        }
        if !row.is_empty() {
            rows.push(row);
        }
    }
    if rows.is_empty() {
        return Ok(ArraysD::F64(
            ArrayD::from_shape_vec(IxDyn(&[0]), Vec::<f64>::new()).unwrap_or_default(),
        ));
    }
    let cols = rows[0].len();
    if rows.iter().all(|r| r.len() == cols) {
        if cols == 1 {
            // Single column → 1-D array (matches numpy).
            let flat: Vec<f64> = rows.into_iter().flatten().collect();
            Ok(ArraysD::F64(
                ArrayD::from_shape_vec(IxDyn(&[flat.len()]), flat).unwrap_or_default(),
            ))
        } else {
            let n = rows.len();
            let flat: Vec<f64> = rows.into_iter().flatten().collect();
            Ok(ArraysD::F64(
                ArrayD::from_shape_vec(IxDyn(&[n, cols]), flat).unwrap_or_default(),
            ))
        }
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "loadtxt: rows have inconsistent column counts",
        ))
    }
}

// =====================================================================
// tofile / fromfile  (raw little-endian binary, no header)
// =====================================================================

pub fn tofile(path: &Path, a: &ArraysD) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    write_raw(&mut w, a)?;
    w.flush()
}

fn write_raw<W: Write>(w: &mut W, a: &ArraysD) -> std::io::Result<()> {
    match a {
        ArraysD::Bool(arr) => {
            for &v in arr.iter() {
                w.write_all(&[v as u8])?;
            }
        }
        ArraysD::I8(arr) => {
            for &v in arr.iter() {
                w.write_all(&v.to_le_bytes())?;
            }
        }
        ArraysD::I16(arr) => {
            for &v in arr.iter() {
                w.write_all(&v.to_le_bytes())?;
            }
        }
        ArraysD::I32(arr) => {
            for &v in arr.iter() {
                w.write_all(&v.to_le_bytes())?;
            }
        }
        ArraysD::I64(arr) => {
            for &v in arr.iter() {
                w.write_all(&v.to_le_bytes())?;
            }
        }
        ArraysD::U8(arr) => {
            if let Some(s) = arr.as_slice() {
                w.write_all(s)?;
            } else {
                for &v in arr.iter() {
                    w.write_all(&[v])?;
                }
            }
        }
        ArraysD::U16(arr) => {
            for &v in arr.iter() {
                w.write_all(&v.to_le_bytes())?;
            }
        }
        ArraysD::U32(arr) => {
            for &v in arr.iter() {
                w.write_all(&v.to_le_bytes())?;
            }
        }
        ArraysD::U64(arr) => {
            for &v in arr.iter() {
                w.write_all(&v.to_le_bytes())?;
            }
        }
        ArraysD::F16(arr) => {
            for &v in arr.iter() {
                w.write_all(&v.to_le_bytes())?;
            }
        }
        ArraysD::F32(arr) => {
            for &v in arr.iter() {
                w.write_all(&v.to_le_bytes())?;
            }
        }
        ArraysD::F64(arr) => {
            for &v in arr.iter() {
                w.write_all(&v.to_le_bytes())?;
            }
        }
        ArraysD::C64(arr) => {
            for &v in arr.iter() {
                w.write_all(&v.re.to_le_bytes())?;
                w.write_all(&v.im.to_le_bytes())?;
            }
        }
        ArraysD::C128(arr) => {
            for &v in arr.iter() {
                w.write_all(&v.re.to_le_bytes())?;
                w.write_all(&v.im.to_le_bytes())?;
            }
        }
        _ => { return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "non-numeric dtype not supported by write_raw")) },
    }
    Ok(())
}

/// `np.fromfile(path, dtype=...)` — raw little-endian, no header. Always
/// returns a 1-D array of `count` elements (or the entire file if `count`
/// is `-1`).
pub fn fromfile(
    path: &Path,
    dtype: DType,
    count: isize,
) -> std::io::Result<ArraysD> {
    let mut f = File::open(path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    read_raw(&buf, dtype, count)
}

fn read_raw(buf: &[u8], dtype: DType, count: isize) -> std::io::Result<ArraysD> {
    let item = dtype.itemsize();
    let available = buf.len() / item;
    let n = if count < 0 {
        available
    } else {
        (count as usize).min(available)
    };
    macro_rules! le {
        ($ty:ty, $bytes:expr) => {{
            let mut v = Vec::<$ty>::with_capacity(n);
            for i in 0..n {
                let s = i * $bytes;
                let b: [u8; $bytes] = buf
                    .get(s..s + $bytes)
                    .and_then(|sl| sl.try_into().ok())
                    .unwrap_or([0; $bytes]);
                v.push(<$ty>::from_le_bytes(b));
            }
            v
        }};
    }
    Ok(match dtype {
        DType::Bool => ArraysD::Bool(
            ArrayD::from_shape_vec(
                IxDyn(&[n]),
                buf.get(..n).unwrap_or(&[]).iter().map(|&b| b != 0).collect::<Vec<_>>(),
            )
            .unwrap_or_default(),
        ),
        DType::I8 => {
            let v: Vec<i8> = buf.get(..n).unwrap_or(&[]).iter().map(|&b| b as i8).collect();
            ArraysD::I8(ArrayD::from_shape_vec(IxDyn(&[n]), v).unwrap_or_default())
        }
        DType::U8 => ArraysD::U8(
            ArrayD::from_shape_vec(IxDyn(&[n]), buf.get(..n).unwrap_or(&[]).to_vec()).unwrap_or_default(),
        ),
        DType::I16 => ArraysD::I16(ArrayD::from_shape_vec(IxDyn(&[n]), le!(i16, 2)).unwrap_or_default()),
        DType::U16 => ArraysD::U16(ArrayD::from_shape_vec(IxDyn(&[n]), le!(u16, 2)).unwrap_or_default()),
        DType::F16 => {
            let mut v = Vec::<half::f16>::with_capacity(n);
            for i in 0..n {
                let s = i * 2;
                let b: [u8; 2] = buf
                    .get(s..s + 2)
                    .and_then(|sl| sl.try_into().ok())
                    .unwrap_or([0; 2]);
                v.push(half::f16::from_le_bytes(b));
            }
            ArraysD::F16(ArrayD::from_shape_vec(IxDyn(&[n]), v).unwrap_or_default())
        }
        DType::I32 => ArraysD::I32(ArrayD::from_shape_vec(IxDyn(&[n]), le!(i32, 4)).unwrap_or_default()),
        DType::U32 => ArraysD::U32(ArrayD::from_shape_vec(IxDyn(&[n]), le!(u32, 4)).unwrap_or_default()),
        DType::F32 => ArraysD::F32(ArrayD::from_shape_vec(IxDyn(&[n]), le!(f32, 4)).unwrap_or_default()),
        DType::I64 => ArraysD::I64(ArrayD::from_shape_vec(IxDyn(&[n]), le!(i64, 8)).unwrap_or_default()),
        DType::U64 => ArraysD::U64(ArrayD::from_shape_vec(IxDyn(&[n]), le!(u64, 8)).unwrap_or_default()),
        DType::F64 => ArraysD::F64(ArrayD::from_shape_vec(IxDyn(&[n]), le!(f64, 8)).unwrap_or_default()),
        DType::C64 => {
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                let s = i * 8;
                let re_b: [u8; 4] = buf.get(s..s + 4).and_then(|sl| sl.try_into().ok()).unwrap_or([0; 4]);
                let im_b: [u8; 4] = buf.get(s + 4..s + 8).and_then(|sl| sl.try_into().ok()).unwrap_or([0; 4]);
                v.push(crate::dtype::C32::new(f32::from_le_bytes(re_b), f32::from_le_bytes(im_b)));
            }
            ArraysD::C64(ArrayD::from_shape_vec(IxDyn(&[n]), v).unwrap_or_default())
        }
        DType::C128 => {
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                let s = i * 16;
                let re_b: [u8; 8] = buf.get(s..s + 8).and_then(|sl| sl.try_into().ok()).unwrap_or([0; 8]);
                let im_b: [u8; 8] = buf.get(s + 8..s + 16).and_then(|sl| sl.try_into().ok()).unwrap_or([0; 8]);
                v.push(crate::dtype::C64::new(f64::from_le_bytes(re_b), f64::from_le_bytes(im_b)));
            }
            ArraysD::C128(ArrayD::from_shape_vec(IxDyn(&[n]), v).unwrap_or_default())
        }
        _ => { return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "non-numeric dtype not supported by read_raw")) },
    })
}
