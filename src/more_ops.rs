//! Third-tier numpy operations: flip/roll/rot90, column_stack/dstack,
//! diag/triu/tril/diagflat, atleast_Nd, count_nonzero/bincount/histogram,
//! nan-aware reductions, searchsorted, meshgrid, interp/trapz/gradient.

use crate::dtype::{ArraysD, DType};
use ndarray::{ArrayD, IxDyn};
use rustpython_vm::{PyResult, VirtualMachine};

/// Cast to F64 and unwrap into the underlying ArrayD, or return an empty array
/// if the cast somehow yielded a different variant (logically dead, but kept
/// as a panic-free fallback).
fn cast_f64_or_empty(a: &ArraysD) -> ArrayD<f64> {
    match a.cast(DType::F64) {
        ArraysD::F64(x) => x,
        _ => ArrayD::<f64>::zeros(IxDyn(&[0])),
    }
}

// =====================================================================
// flip / flipud / fliplr / roll / rot90
// =====================================================================

pub fn flip(a: &ArraysD, axis: Option<isize>) -> ArraysD {
    let nd = a.ndim();
    if nd == 0 {
        return a.clone();
    }
    let axes: Vec<usize> = match axis {
        None => (0..nd).collect(),
        Some(ax) => vec![if ax < 0 { (ax + nd as isize) as usize } else { ax as usize }],
    };
    let mut out = a.clone();
    for ax in axes {
        out = flip_axis(&out, ax);
    }
    out
}

fn flip_axis(a: &ArraysD, axis: usize) -> ArraysD {
    let nd = a.ndim();
    let mut info: Vec<ndarray::SliceInfoElem> = (0..nd)
        .map(|_| ndarray::SliceInfoElem::Slice {
            start: 0,
            end: None,
            step: 1,
        })
        .collect();
    info[axis] = ndarray::SliceInfoElem::Slice {
        start: 0,
        end: None,
        step: -1,
    };
    let si = match ndarray::SliceInfo::<_, IxDyn, IxDyn>::try_from(info) {
        Ok(s) => s,
        // Logically dead — `info` is built with the array's own ndim.
        Err(_) => return a.clone(),
    };
    macro_rules! per {
        ($var:ident, $arr:ident) => {
            ArraysD::$var($arr.slice(si.as_ref()).to_owned())
        };
    }
    match a {
        ArraysD::Bool(x) => per!(Bool, x),
        ArraysD::I8(x) => per!(I8, x),
        ArraysD::I16(x) => per!(I16, x),
        ArraysD::I32(x) => per!(I32, x),
        ArraysD::I64(x) => per!(I64, x),
        ArraysD::U8(x) => per!(U8, x),
        ArraysD::U16(x) => per!(U16, x),
        ArraysD::U32(x) => per!(U32, x),
        ArraysD::U64(x) => per!(U64, x),
        ArraysD::F16(x) => per!(F16, x),
        ArraysD::F32(x) => per!(F32, x),
        ArraysD::F64(x) => per!(F64, x),
        ArraysD::C64(x) => per!(C64, x),
        ArraysD::C128(x) => per!(C128, x),
        // Non-numeric data: same slicing, but the inner array isn't Copy so
        // we use a clone-based variant builder.
        ArraysD::Object(x) => ArraysD::Object(x.slice(si.as_ref()).to_owned()),
        ArraysD::Str { itemsize_chars, data } => ArraysD::Str {
            itemsize_chars: *itemsize_chars,
            data: data.slice(si.as_ref()).to_owned(),
        },
        ArraysD::Bytes { itemsize, data } => ArraysD::Bytes {
            itemsize: *itemsize,
            data: data.slice(si.as_ref()).to_owned(),
        },
        ArraysD::Datetime64 { unit, data } => ArraysD::Datetime64 {
            unit: *unit,
            data: data.slice(si.as_ref()).to_owned(),
        },
        ArraysD::Timedelta64 { unit, data } => ArraysD::Timedelta64 {
            unit: *unit,
            data: data.slice(si.as_ref()).to_owned(),
        },
        ArraysD::Void { layout, data } => ArraysD::Void {
            layout: layout.clone(),
            data: data.slice(si.as_ref()).to_owned(),
        },
    }
}

pub fn flipud(a: &ArraysD) -> ArraysD {
    flip(a, Some(0))
}

pub fn fliplr(a: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    if a.ndim() < 2 {
        return Err(vm.new_value_error("fliplr requires array of dimension >= 2".to_string()));
    }
    Ok(flip(a, Some(1)))
}

pub fn roll(a: &ArraysD, shift: isize, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let f = crate::linalg::flatten(a);
    let n = f.len() as isize;
    if n == 0 {
        return Ok(f);
    }
    let s = ((shift % n) + n) % n;
    if s == 0 {
        return Ok(f);
    }
    let _ = vm;
    let k = (n - s) as usize;
    // Concatenate the two halves: a[k:] + a[:k]
    macro_rules! per {
        ($var:ident, $ty:ty, $arr:ident) => {{
            let v: Vec<$ty> = $arr.iter().copied().collect();
            let mut out: Vec<$ty> = Vec::with_capacity(v.len());
            out.extend_from_slice(&v[k..]);
            out.extend_from_slice(&v[..k]);
            ArraysD::$var(ArrayD::from_shape_vec(IxDyn(&[out.len()]), out).unwrap_or_default())
        }};
    }
    Ok(match f {
        ArraysD::Bool(arr) => per!(Bool, bool, arr),
        ArraysD::I8(arr) => per!(I8, i8, arr),
        ArraysD::I16(arr) => per!(I16, i16, arr),
        ArraysD::I32(arr) => per!(I32, i32, arr),
        ArraysD::I64(arr) => per!(I64, i64, arr),
        ArraysD::U8(arr) => per!(U8, u8, arr),
        ArraysD::U16(arr) => per!(U16, u16, arr),
        ArraysD::U32(arr) => per!(U32, u32, arr),
        ArraysD::U64(arr) => per!(U64, u64, arr),
        ArraysD::F16(arr) => per!(F16, half::f16, arr),
        ArraysD::F32(arr) => per!(F32, f32, arr),
        ArraysD::F64(arr) => per!(F64, f64, arr),
        ArraysD::C64(arr) => per!(C64, crate::dtype::C32, arr),
        ArraysD::C128(arr) => per!(C128, crate::dtype::C64, arr),
        // Non-numeric: use clone-based concatenation.
        ref other => {
            // Walk the storage element-wise via a closure that handles each
            // variant. For roll on non-numeric we cycle the flat array.
            macro_rules! per_clone {
                ($wrap:expr, $arr:expr) => {{
                    let v: Vec<_> = $arr.iter().cloned().collect();
                    let mut new_v = Vec::with_capacity(v.len());
                    new_v.extend_from_slice(&v[k..]);
                    new_v.extend_from_slice(&v[..k]);
                    let n = new_v.len();
                    match ArrayD::from_shape_vec(IxDyn(&[n]), new_v) {
                        Ok(d) => $wrap(d),
                        Err(_) => return Ok(other.clone()),
                    }
                }};
            }
            let out: ArraysD = match other {
                ArraysD::Object(arr) => per_clone!(ArraysD::Object, arr),
                ArraysD::Str { itemsize_chars, data } => {
                    let n = *itemsize_chars;
                    per_clone!(|d| ArraysD::Str { itemsize_chars: n, data: d }, data)
                }
                ArraysD::Bytes { itemsize, data } => {
                    let n = *itemsize;
                    per_clone!(|d| ArraysD::Bytes { itemsize: n, data: d }, data)
                }
                ArraysD::Datetime64 { unit, data } => {
                    let u = *unit;
                    per_clone!(|d| ArraysD::Datetime64 { unit: u, data: d }, data)
                }
                ArraysD::Timedelta64 { unit, data } => {
                    let u = *unit;
                    per_clone!(|d| ArraysD::Timedelta64 { unit: u, data: d }, data)
                }
                ArraysD::Void { layout, data } => {
                    let l = layout.clone();
                    per_clone!(|d| ArraysD::Void { layout: l.clone(), data: d }, data)
                }
                _ => other.clone(),
            };
            out
        }
    })
}

/// rot90 rotates the array 90 degrees counter-clockwise k times in the
/// (0, 1) plane (numpy default).
pub fn rot90(a: &ArraysD, k: isize, vm: &VirtualMachine) -> PyResult<ArraysD> {
    if a.ndim() < 2 {
        return Err(vm.new_value_error("rot90 requires >= 2-D array".to_string()));
    }
    let k = k.rem_euclid(4);
    let mut out = a.clone();
    for _ in 0..k {
        out = crate::linalg::transpose(&flip(&out, Some(1)));
    }
    Ok(out)
}

// =====================================================================
// column_stack / dstack
// =====================================================================

pub fn column_stack(arrays: &[ArraysD], vm: &VirtualMachine) -> PyResult<ArraysD> {
    // 1-D arrays become column vectors then concatenate along axis 1.
    let normalized: Vec<ArraysD> = arrays
        .iter()
        .map(|a| {
            if a.ndim() == 1 {
                let n = a.len();
                crate::linalg::reshape(a, &[n, 1]).unwrap_or_else(|| a.clone())
            } else {
                a.clone()
            }
        })
        .collect();
    crate::linalg::concatenate(&normalized, 1, vm)
}

pub fn dstack(arrays: &[ArraysD], vm: &VirtualMachine) -> PyResult<ArraysD> {
    // Stack along third axis. 1-D -> (1, n, 1); 2-D -> (m, n, 1); then concat.
    let normalized: Vec<ArraysD> = arrays
        .iter()
        .map(|a| match a.ndim() {
            0 => crate::linalg::reshape(a, &[1, 1, 1]).unwrap_or_else(|| a.clone()),
            1 => {
                let n = a.len();
                crate::linalg::reshape(a, &[1, n, 1]).unwrap_or_else(|| a.clone())
            }
            2 => {
                let s = a.shape();
                let new_shape = vec![s[0], s[1], 1];
                crate::linalg::reshape(a, &new_shape).unwrap_or_else(|| a.clone())
            }
            _ => a.clone(),
        })
        .collect();
    crate::linalg::concatenate(&normalized, 2, vm)
}

// =====================================================================
// diag / triu / tril / diagflat
// =====================================================================

pub fn diag(a: &ArraysD, k: isize, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let nd = a.ndim();
    if nd == 1 {
        // 1-D → build a square diagonal matrix.
        let n = a.len() as isize;
        let size = (n + k.abs()) as usize;
        let mut out = crate::create::zeros(&[size, size], a.dtype());
        let f = a.cast(a.dtype());
        for i in 0..(n as usize) {
            let (r, c) = if k >= 0 {
                (i, i + k as usize)
            } else {
                (i + (-k) as usize, i)
            };
            copy_scalar(&f, i, &mut out, &[r, c]);
        }
        Ok(out)
    } else if nd == 2 {
        // 2-D → extract the k-th diagonal.
        let (m, n) = (a.shape()[0] as isize, a.shape()[1] as isize);
        let start_r = if k < 0 { -k } else { 0 };
        let start_c = if k >= 0 { k } else { 0 };
        let len = (m - start_r).min(n - start_c).max(0) as usize;
        let mut out = crate::create::zeros(&[len], a.dtype());
        for i in 0..len {
            let r = (start_r as usize) + i;
            let c = (start_c as usize) + i;
            copy_scalar_2d(a, r, c, &mut out, i);
        }
        Ok(out)
    } else {
        Err(vm.new_value_error("diag requires 1-D or 2-D array".to_string()))
    }
}

fn copy_scalar(src: &ArraysD, src_idx: usize, dst: &mut ArraysD, dst_idx: &[usize]) {
    let si = IxDyn(&[src_idx]);
    let di = IxDyn(dst_idx);
    match (src, dst) {
        (ArraysD::Bool(s), ArraysD::Bool(d)) => d[di] = s[si],
        (ArraysD::I8(s), ArraysD::I8(d)) => d[di] = s[si],
        (ArraysD::I16(s), ArraysD::I16(d)) => d[di] = s[si],
        (ArraysD::I32(s), ArraysD::I32(d)) => d[di] = s[si],
        (ArraysD::I64(s), ArraysD::I64(d)) => d[di] = s[si],
        (ArraysD::U8(s), ArraysD::U8(d)) => d[di] = s[si],
        (ArraysD::U16(s), ArraysD::U16(d)) => d[di] = s[si],
        (ArraysD::U32(s), ArraysD::U32(d)) => d[di] = s[si],
        (ArraysD::U64(s), ArraysD::U64(d)) => d[di] = s[si],
        (ArraysD::F16(s), ArraysD::F16(d)) => d[di] = s[si],
        (ArraysD::F32(s), ArraysD::F32(d)) => d[di] = s[si],
        (ArraysD::F64(s), ArraysD::F64(d)) => d[di] = s[si],
        (ArraysD::C64(s), ArraysD::C64(d)) => d[di] = s[si],
        (ArraysD::C128(s), ArraysD::C128(d)) => d[di] = s[si],
        // dtype-mismatched arms are unreachable: callers always pass dst with
        // the same dtype as src. Silent no-op is preferable to a panic.
        _ => (),
    }
}
fn copy_scalar_2d(src: &ArraysD, r: usize, c: usize, dst: &mut ArraysD, di: usize) {
    let si = IxDyn(&[r, c]);
    let di = IxDyn(&[di]);
    match (src, dst) {
        (ArraysD::Bool(s), ArraysD::Bool(d)) => d[di] = s[si],
        (ArraysD::I8(s), ArraysD::I8(d)) => d[di] = s[si],
        (ArraysD::I16(s), ArraysD::I16(d)) => d[di] = s[si],
        (ArraysD::I32(s), ArraysD::I32(d)) => d[di] = s[si],
        (ArraysD::I64(s), ArraysD::I64(d)) => d[di] = s[si],
        (ArraysD::U8(s), ArraysD::U8(d)) => d[di] = s[si],
        (ArraysD::U16(s), ArraysD::U16(d)) => d[di] = s[si],
        (ArraysD::U32(s), ArraysD::U32(d)) => d[di] = s[si],
        (ArraysD::U64(s), ArraysD::U64(d)) => d[di] = s[si],
        (ArraysD::F16(s), ArraysD::F16(d)) => d[di] = s[si],
        (ArraysD::F32(s), ArraysD::F32(d)) => d[di] = s[si],
        (ArraysD::F64(s), ArraysD::F64(d)) => d[di] = s[si],
        (ArraysD::C64(s), ArraysD::C64(d)) => d[di] = s[si],
        (ArraysD::C128(s), ArraysD::C128(d)) => d[di] = s[si],
        _ => (),
    }
}

pub fn diagflat(a: &ArraysD) -> ArraysD {
    let f = crate::linalg::flatten(a);
    let _vm: Option<&VirtualMachine> = None; // we know it's 1-D so diag won't error
    let n = f.len();
    let mut out = crate::create::zeros(&[n, n], f.dtype());
    for i in 0..n {
        copy_scalar(&f, i, &mut out, &[i, i]);
    }
    out
}

/// `triu` — zero out below the k-th diagonal.
pub fn triu(a: &ArraysD, k: isize, vm: &VirtualMachine) -> PyResult<ArraysD> {
    if a.ndim() != 2 {
        return Err(vm.new_value_error("triu requires 2-D array".to_string()));
    }
    let mut out = a.clone();
    let m = a.shape()[0];
    let n = a.shape()[1];
    for i in 0..m {
        for j in 0..n {
            if (j as isize) < (i as isize) + k {
                zero_at(&mut out, &[i, j]);
            }
        }
    }
    Ok(out)
}

pub fn tril(a: &ArraysD, k: isize, vm: &VirtualMachine) -> PyResult<ArraysD> {
    if a.ndim() != 2 {
        return Err(vm.new_value_error("tril requires 2-D array".to_string()));
    }
    let mut out = a.clone();
    let m = a.shape()[0];
    let n = a.shape()[1];
    for i in 0..m {
        for j in 0..n {
            if (j as isize) > (i as isize) + k {
                zero_at(&mut out, &[i, j]);
            }
        }
    }
    Ok(out)
}

fn zero_at(a: &mut ArraysD, idx: &[usize]) {
    let i = IxDyn(idx);
    match a {
        ArraysD::Bool(x) => x[i] = false,
        ArraysD::I8(x) => x[i] = 0,
        ArraysD::I16(x) => x[i] = 0,
        ArraysD::I32(x) => x[i] = 0,
        ArraysD::I64(x) => x[i] = 0,
        ArraysD::U8(x) => x[i] = 0,
        ArraysD::U16(x) => x[i] = 0,
        ArraysD::U32(x) => x[i] = 0,
        ArraysD::U64(x) => x[i] = 0,
        ArraysD::F16(x) => x[i] = half::f16::ZERO,
        ArraysD::F32(x) => x[i] = 0.0,
        ArraysD::F64(x) => x[i] = 0.0,
        ArraysD::C64(x) => x[i] = crate::dtype::C32::new(0.0, 0.0),
        ArraysD::C128(x) => x[i] = crate::dtype::C64::new(0.0, 0.0),
        // Non-numeric zero: set the cell to its dtype's natural empty value.
        ArraysD::Str { data, .. } => data[i] = String::new(),
        ArraysD::Bytes { itemsize, data } => data[i] = vec![0u8; *itemsize as usize],
        ArraysD::Datetime64 { data, .. } | ArraysD::Timedelta64 { data, .. } => data[i] = 0,
        ArraysD::Void { layout, data } => data[i] = vec![0u8; layout.itemsize],
        // Object: no vm available to construct a None ref; leave the cell as-is.
        ArraysD::Object(_) => {}
    }
}

/// `tri(N, M, k)` — lower-triangular ones.
pub fn tri(n: usize, m: usize, k: isize, dtype: DType) -> ArraysD {
    let mut out = crate::create::zeros(&[n, m], dtype);
    for i in 0..n {
        for j in 0..m {
            if (j as isize) <= (i as isize) + k {
                set_one(&mut out, &[i, j]);
            }
        }
    }
    out
}

fn set_one(a: &mut ArraysD, idx: &[usize]) {
    let i = IxDyn(idx);
    match a {
        ArraysD::Bool(x) => x[i] = true,
        ArraysD::I8(x) => x[i] = 1,
        ArraysD::I16(x) => x[i] = 1,
        ArraysD::I32(x) => x[i] = 1,
        ArraysD::I64(x) => x[i] = 1,
        ArraysD::U8(x) => x[i] = 1,
        ArraysD::U16(x) => x[i] = 1,
        ArraysD::U32(x) => x[i] = 1,
        ArraysD::U64(x) => x[i] = 1,
        ArraysD::F16(x) => x[i] = half::f16::from_f32(1.0),
        ArraysD::F32(x) => x[i] = 1.0,
        ArraysD::F64(x) => x[i] = 1.0,
        ArraysD::C64(x) => x[i] = crate::dtype::C32::new(1.0, 0.0),
        ArraysD::C128(x) => x[i] = crate::dtype::C64::new(1.0, 0.0),
        // set_one is only meaningful for numeric variants; ignore otherwise.
        _ => {}
    }
}

// =====================================================================
// atleast_1d / 2d / 3d
// =====================================================================

pub fn atleast_1d(a: &ArraysD) -> ArraysD {
    if a.ndim() >= 1 {
        a.clone()
    } else {
        crate::linalg::reshape(a, &[1]).unwrap_or_else(|| a.clone())
    }
}
pub fn atleast_2d(a: &ArraysD) -> ArraysD {
    match a.ndim() {
        0 => crate::linalg::reshape(a, &[1, 1]).unwrap_or_else(|| a.clone()),
        1 => {
            let n = a.len();
            crate::linalg::reshape(a, &[1, n]).unwrap_or_else(|| a.clone())
        }
        _ => a.clone(),
    }
}
pub fn atleast_3d(a: &ArraysD) -> ArraysD {
    match a.ndim() {
        0 => crate::linalg::reshape(a, &[1, 1, 1]).unwrap_or_else(|| a.clone()),
        1 => {
            let n = a.len();
            crate::linalg::reshape(a, &[1, n, 1]).unwrap_or_else(|| a.clone())
        }
        2 => {
            let s = a.shape();
            let new = vec![s[0], s[1], 1];
            crate::linalg::reshape(a, &new).unwrap_or_else(|| a.clone())
        }
        _ => a.clone(),
    }
}

// =====================================================================
// count_nonzero / bincount / histogram
// =====================================================================

pub fn count_nonzero(a: &ArraysD) -> ArraysD {
    let b = match a.cast(DType::Bool) {
        ArraysD::Bool(x) => x,
        _ => return ArraysD::I64(ArrayD::from_elem(IxDyn(&[]), 0)),
    };
    let c = b.iter().filter(|&&v| v).count() as i64;
    ArraysD::I64(ArrayD::from_elem(IxDyn(&[]), c))
}

pub fn bincount(a: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    if a.dtype().is_float() || a.dtype().is_complex() {
        return Err(vm.new_type_error("bincount: integer input required".to_string()));
    }
    let f = a.cast(DType::I64);
    let ArraysD::I64(arr) = f else {
        return Err(crate::internal::internal(vm, "bincount: cast to I64 failed"));
    };
    let max = arr.iter().copied().max().unwrap_or(-1);
    if max < 0 {
        return Err(vm.new_value_error(
            "bincount: array must not contain negative values".to_string(),
        ));
    }
    let n = (max + 1) as usize;
    let mut counts = vec![0i64; n];
    for &v in arr.iter() {
        counts[v as usize] += 1;
    }
    Ok(ArraysD::I64(
        ArrayD::from_shape_vec(IxDyn(&[n]), counts).unwrap_or_default(),
    ))
}

/// Simple equal-width histogram.
pub fn histogram(
    a: &ArraysD,
    bins: usize,
    range: Option<(f64, f64)>,
    vm: &VirtualMachine,
) -> PyResult<(ArraysD, ArraysD)> {
    if bins == 0 {
        return Err(vm.new_value_error("histogram: bins must be > 0".to_string()));
    }
    let f = cast_f64_or_empty(a);
    let (lo, hi) = match range {
        Some(r) => r,
        None => {
            let mn = f.iter().copied().fold(f64::INFINITY, f64::min);
            let mx = f.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (mn, mx)
        }
    };
    let width = (hi - lo) / bins as f64;
    let mut counts = vec![0i64; bins];
    for &v in f.iter() {
        if v < lo || v > hi {
            continue;
        }
        let mut idx = ((v - lo) / width).floor() as isize;
        if idx >= bins as isize {
            idx = bins as isize - 1;
        }
        if idx < 0 {
            idx = 0;
        }
        counts[idx as usize] += 1;
    }
    let edges: Vec<f64> = (0..=bins).map(|i| lo + width * i as f64).collect();
    Ok((
        ArraysD::I64(ArrayD::from_shape_vec(IxDyn(&[bins]), counts).unwrap_or_default()),
        ArraysD::F64(ArrayD::from_shape_vec(IxDyn(&[bins + 1]), edges).unwrap_or_default()),
    ))
}

// =====================================================================
// nan-aware reductions
// =====================================================================

fn finite_f64(a: &ArraysD) -> Vec<f64> {
    use crate::dtype::CoerceArray as _CA;
    let arr = a.coerce::<f64>();
    arr.iter().copied().filter(|v| !v.is_nan()).collect()
}

pub fn nansum(a: &ArraysD) -> ArraysD {
    let v = finite_f64(a);
    ArraysD::F64(ArrayD::from_elem(IxDyn(&[]), v.iter().sum()))
}
pub fn nanmean(a: &ArraysD) -> ArraysD {
    let v = finite_f64(a);
    let m = if v.is_empty() {
        f64::NAN
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    };
    ArraysD::F64(ArrayD::from_elem(IxDyn(&[]), m))
}
pub fn nanmin(a: &ArraysD) -> ArraysD {
    let v = finite_f64(a);
    let m = v.iter().copied().fold(f64::INFINITY, f64::min);
    ArraysD::F64(ArrayD::from_elem(IxDyn(&[]), m))
}
pub fn nanmax(a: &ArraysD) -> ArraysD {
    let v = finite_f64(a);
    let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    ArraysD::F64(ArrayD::from_elem(IxDyn(&[]), m))
}
pub fn nanstd(a: &ArraysD, ddof: usize) -> ArraysD {
    let v = finite_f64(a);
    let n = v.len();
    if n <= ddof {
        return ArraysD::F64(ArrayD::from_elem(IxDyn(&[]), f64::NAN));
    }
    let mean = v.iter().sum::<f64>() / n as f64;
    let ss: f64 = v.iter().map(|x| (x - mean).powi(2)).sum();
    ArraysD::F64(ArrayD::from_elem(IxDyn(&[]), (ss / (n - ddof) as f64).sqrt()))
}
pub fn nanvar(a: &ArraysD, ddof: usize) -> ArraysD {
    let v = finite_f64(a);
    let n = v.len();
    if n <= ddof {
        return ArraysD::F64(ArrayD::from_elem(IxDyn(&[]), f64::NAN));
    }
    let mean = v.iter().sum::<f64>() / n as f64;
    let ss: f64 = v.iter().map(|x| (x - mean).powi(2)).sum();
    ArraysD::F64(ArrayD::from_elem(IxDyn(&[]), ss / (n - ddof) as f64))
}
pub fn nanmedian(a: &ArraysD) -> ArraysD {
    let mut v = finite_f64(a);
    if v.is_empty() {
        return ArraysD::F64(ArrayD::from_elem(IxDyn(&[]), f64::NAN));
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    let m = if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) * 0.5
    };
    ArraysD::F64(ArrayD::from_elem(IxDyn(&[]), m))
}

// =====================================================================
// searchsorted
// =====================================================================

pub fn searchsorted(a: &ArraysD, v: &ArraysD) -> ArraysD {
    let af = cast_f64_or_empty(a);
    let vf = cast_f64_or_empty(v);
    let sorted: Vec<f64> = af.iter().copied().collect();
    let result: Vec<i64> = vf
        .iter()
        .map(|&x| {
            // left side
            sorted.partition_point(|&t| t < x) as i64
        })
        .collect();
    ArraysD::I64(
        ArrayD::from_shape_vec(IxDyn(vf.shape()), result).unwrap_or_default(),
    )
}

// =====================================================================
// meshgrid (2-D 'xy' indexing — most common)
// =====================================================================

pub fn meshgrid(x: &ArraysD, y: &ArraysD) -> (ArraysD, ArraysD) {
    let xf = cast_f64_or_empty(x);
    let yf = cast_f64_or_empty(y);
    let nx = xf.len();
    let ny = yf.len();
    let mut xx = ArrayD::<f64>::zeros(IxDyn(&[ny, nx]));
    let mut yy = ArrayD::<f64>::zeros(IxDyn(&[ny, nx]));
    for j in 0..ny {
        for i in 0..nx {
            xx[IxDyn(&[j, i])] = xf[IxDyn(&[i])];
            yy[IxDyn(&[j, i])] = yf[IxDyn(&[j])];
        }
    }
    let out_dt = crate::promote::promote(x.dtype(), y.dtype());
    (
        ArraysD::F64(xx).cast(out_dt),
        ArraysD::F64(yy).cast(out_dt),
    )
}

// =====================================================================
// interp / trapz / gradient
// =====================================================================

pub fn interp(x: &ArraysD, xp: &ArraysD, fp: &ArraysD) -> ArraysD {
    let xv = cast_f64_or_empty(x);
    let xpv = cast_f64_or_empty(xp);
    let fpv = cast_f64_or_empty(fp);
    let xp_slice: Vec<f64> = xpv.iter().copied().collect();
    let fp_slice: Vec<f64> = fpv.iter().copied().collect();
    // If xp/fp are empty, interp is undefined — return zeros of x's shape.
    if xp_slice.is_empty() || fp_slice.is_empty() {
        return ArraysD::F64(ArrayD::<f64>::zeros(IxDyn(xv.shape())));
    }
    let last_xp = xp_slice[xp_slice.len() - 1];
    let last_fp = fp_slice[fp_slice.len() - 1];
    let out: Vec<f64> = xv
        .iter()
        .map(|&q| {
            if q <= xp_slice[0] {
                return fp_slice[0];
            }
            if q >= last_xp {
                return last_fp;
            }
            let i = xp_slice.partition_point(|&t| t <= q).saturating_sub(1);
            if i + 1 >= xp_slice.len() {
                return last_fp;
            }
            let denom = xp_slice[i + 1] - xp_slice[i];
            if denom == 0.0 {
                return fp_slice[i];
            }
            let t = (q - xp_slice[i]) / denom;
            fp_slice[i] + t * (fp_slice[i + 1] - fp_slice[i])
        })
        .collect();
    ArraysD::F64(ArrayD::from_shape_vec(IxDyn(xv.shape()), out).unwrap_or_default())
}

pub fn trapz(y: &ArraysD, dx: f64) -> ArraysD {
    let yf = cast_f64_or_empty(y);
    let n = yf.len();
    if n < 2 {
        return ArraysD::F64(ArrayD::from_elem(IxDyn(&[]), 0.0));
    }
    let mut acc = 0.0;
    // Fall back to iter() if the array isn't contiguous — yf is owned and
    // built from `cast(F64)` so contiguity is the common case, but we don't
    // want to panic on the rare non-contiguous one.
    if let Some(s) = yf.as_slice() {
        for i in 0..(n - 1) {
            acc += (s[i] + s[i + 1]) * 0.5 * dx;
        }
    } else {
        let v: Vec<f64> = yf.iter().copied().collect();
        for i in 0..(n - 1) {
            acc += (v[i] + v[i + 1]) * 0.5 * dx;
        }
    }
    ArraysD::F64(ArrayD::from_elem(IxDyn(&[]), acc))
}

pub fn gradient(a: &ArraysD) -> ArraysD {
    // 1-D central differences (edges use one-sided).
    let f = cast_f64_or_empty(a);
    let n = f.len();
    if n < 2 {
        return ArraysD::F64(ArrayD::from_elem(IxDyn(&[n]), 0.0));
    }
    let mut out = vec![0.0f64; n];
    out[0] = f[IxDyn(&[1])] - f[IxDyn(&[0])];
    out[n - 1] = f[IxDyn(&[n - 1])] - f[IxDyn(&[n - 2])];
    for i in 1..(n - 1) {
        out[i] = (f[IxDyn(&[i + 1])] - f[IxDyn(&[i - 1])]) * 0.5;
    }
    ArraysD::F64(ArrayD::from_shape_vec(IxDyn(&[n]), out).unwrap_or_default())
}

/// `numpy.delete` along axis 0 (flat).
pub fn delete(a: &ArraysD, idx: usize, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let f = crate::linalg::flatten(a);
    if idx >= f.len() {
        return Err(vm.new_index_error(format!("delete: index {idx} out of range")));
    }
    macro_rules! per {
        ($var:ident, $ty:ty, $arr:ident) => {{
            let mut v: Vec<$ty> = $arr.iter().copied().collect();
            v.remove(idx);
            ArraysD::$var(ArrayD::from_shape_vec(IxDyn(&[v.len()]), v).unwrap_or_default())
        }};
    }
    Ok(match f {
        ArraysD::Bool(arr) => per!(Bool, bool, arr),
        ArraysD::I8(arr) => per!(I8, i8, arr),
        ArraysD::I16(arr) => per!(I16, i16, arr),
        ArraysD::I32(arr) => per!(I32, i32, arr),
        ArraysD::I64(arr) => per!(I64, i64, arr),
        ArraysD::U8(arr) => per!(U8, u8, arr),
        ArraysD::U16(arr) => per!(U16, u16, arr),
        ArraysD::U32(arr) => per!(U32, u32, arr),
        ArraysD::U64(arr) => per!(U64, u64, arr),
        ArraysD::F16(arr) => per!(F16, half::f16, arr),
        ArraysD::F32(arr) => per!(F32, f32, arr),
        ArraysD::F64(arr) => per!(F64, f64, arr),
        ArraysD::C64(arr) => per!(C64, crate::dtype::C32, arr),
        ArraysD::C128(arr) => per!(C128, crate::dtype::C64, arr),
        // Non-numeric path: clone-based delete.
        ref other => {
            macro_rules! per_clone {
                ($wrap:expr, $arr:expr) => {{
                    let mut v: Vec<_> = $arr.iter().cloned().collect();
                    v.remove(idx);
                    let n = v.len();
                    match ArrayD::from_shape_vec(IxDyn(&[n]), v) {
                        Ok(d) => $wrap(d),
                        Err(_) => return Ok(other.clone()),
                    }
                }};
            }
            match other {
                ArraysD::Object(arr) => per_clone!(ArraysD::Object, arr),
                ArraysD::Str { itemsize_chars, data } => {
                    let n = *itemsize_chars;
                    per_clone!(|d| ArraysD::Str { itemsize_chars: n, data: d }, data)
                }
                ArraysD::Bytes { itemsize, data } => {
                    let n = *itemsize;
                    per_clone!(|d| ArraysD::Bytes { itemsize: n, data: d }, data)
                }
                ArraysD::Datetime64 { unit, data } => {
                    let u = *unit;
                    per_clone!(|d| ArraysD::Datetime64 { unit: u, data: d }, data)
                }
                ArraysD::Timedelta64 { unit, data } => {
                    let u = *unit;
                    per_clone!(|d| ArraysD::Timedelta64 { unit: u, data: d }, data)
                }
                ArraysD::Void { layout, data } => {
                    let l = layout.clone();
                    per_clone!(|d| ArraysD::Void { layout: l.clone(), data: d }, data)
                }
                _ => other.clone(),
            }
        }
    })
}

pub fn append(a: &ArraysD, b: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let af = crate::linalg::flatten(a);
    let bf = crate::linalg::flatten(b);
    crate::linalg::concatenate(&[af, bf], 0, vm)
}
