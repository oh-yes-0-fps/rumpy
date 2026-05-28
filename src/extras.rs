//! Second-tier numpy operations: logical/bitwise/finite predicates, clip,
//! cumulative ops, where/nonzero, sort/argsort/unique, stack/squeeze/expand,
//! repeat/tile, median/ptp.

use crate::dtype::{ArraysD, C32, C64, DType};
use crate::internal::{OptionExt, internal};
use crate::promote::promote;
use half::f16;
use ndarray::{ArrayD, Axis, IxDyn, Zip};
use rustpython_vm::{PyResult, VirtualMachine};

// =====================================================================
// Logical (treat any non-zero as True)
// =====================================================================

fn to_bool_array(a: &ArraysD) -> ArrayD<bool> {
    // `cast(Bool)` always yields the Bool variant; the fallback arm is
    // logically dead but kept as a safe alternative to `unreachable!()`.
    match a.cast(DType::Bool) {
        ArraysD::Bool(b) => b,
        _ => ArrayD::from_elem(IxDyn(&[]), false),
    }
}

pub fn logical_and(a: &ArraysD, b: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let ba = to_bool_array(a);
    let bb = to_bool_array(b);
    bool_binary(&ba, &bb, |x, y| x && y, vm)
}
pub fn logical_or(a: &ArraysD, b: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let ba = to_bool_array(a);
    let bb = to_bool_array(b);
    bool_binary(&ba, &bb, |x, y| x || y, vm)
}
pub fn logical_xor(a: &ArraysD, b: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let ba = to_bool_array(a);
    let bb = to_bool_array(b);
    bool_binary(&ba, &bb, |x, y| x ^ y, vm)
}
pub fn logical_not(a: &ArraysD) -> ArraysD {
    let ba = to_bool_array(a);
    ArraysD::Bool(ba.mapv(|v| !v))
}

fn bool_binary(
    a: &ArrayD<bool>,
    b: &ArrayD<bool>,
    op: impl Fn(bool, bool) -> bool,
    vm: &VirtualMachine,
) -> PyResult<ArraysD> {
    let s = broadcast(a.shape(), b.shape()).ok_or_else(|| {
        vm.new_value_error(format!(
            "broadcast {:?} vs {:?}",
            a.shape(),
            b.shape()
        ))
    })?;
    let av = a.broadcast(IxDyn(&s)).or_internal(vm, "bool_binary lhs")?;
    let bv = b.broadcast(IxDyn(&s)).or_internal(vm, "bool_binary rhs")?;
    let mut out = ArrayD::<bool>::from_elem(IxDyn(&s), false);
    Zip::from(&mut out).and(&av).and(&bv).for_each(|o, &p, &q| *o = op(p, q));
    Ok(ArraysD::Bool(out))
}

pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    broadcast(a, b)
}

fn broadcast(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let nd = a.len().max(b.len());
    let mut out = vec![1usize; nd];
    for i in 0..nd {
        let da = if i + a.len() >= nd { a[i + a.len() - nd] } else { 1 };
        let db = if i + b.len() >= nd { b[i + b.len() - nd] } else { 1 };
        out[i] = match (da, db) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => return None,
        };
    }
    Some(out)
}

// =====================================================================
// Bitwise (integers + bool)
// =====================================================================

pub fn bitwise_and(a: &ArraysD, b: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    bitwise(a, b, vm, |x, y| x & y, |x, y| x & y, |x, y| x & y)
}
pub fn bitwise_or(a: &ArraysD, b: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    bitwise(a, b, vm, |x, y| x | y, |x, y| x | y, |x, y| x | y)
}
pub fn bitwise_xor(a: &ArraysD, b: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    bitwise(a, b, vm, |x, y| x ^ y, |x, y| x ^ y, |x, y| x ^ y)
}

fn bitwise(
    a: &ArraysD,
    b: &ArraysD,
    vm: &VirtualMachine,
    bop: impl Fn(bool, bool) -> bool,
    sop: impl Fn(i64, i64) -> i64,
    uop: impl Fn(u64, u64) -> u64,
) -> PyResult<ArraysD> {
    let out_dtype = promote(a.dtype(), b.dtype());
    if out_dtype.is_float() || out_dtype.is_complex() {
        return Err(vm.new_type_error(format!(
            "bitwise op not supported for {} and {}",
            a.dtype().name(),
            b.dtype().name()
        )));
    }
    let a = a.cast(out_dtype);
    let b = b.cast(out_dtype);
    let s = broadcast(a.shape(), b.shape()).ok_or_else(|| {
        vm.new_value_error(format!("broadcast {:?} vs {:?}", a.shape(), b.shape()))
    })?;
    let sd = IxDyn(&s);
    macro_rules! sint {
        ($var:ident, $ty:ty) => {{
            let (x, y) = match (&a, &b) {
                (ArraysD::$var(x), ArraysD::$var(y)) => (x, y),
                _ => return Err(internal(vm, "bitwise: dtype mismatch after cast")),
            };
            let xv = x.broadcast(sd.clone()).or_internal(vm, "bitwise lhs")?;
            let yv = y.broadcast(sd.clone()).or_internal(vm, "bitwise rhs")?;
            let mut out = ArrayD::<$ty>::zeros(sd.clone());
            Zip::from(&mut out).and(&xv).and(&yv).for_each(|o, &p, &q| {
                *o = sop(p as i64, q as i64) as $ty;
            });
            ArraysD::$var(out)
        }};
    }
    macro_rules! uint {
        ($var:ident, $ty:ty) => {{
            let (x, y) = match (&a, &b) {
                (ArraysD::$var(x), ArraysD::$var(y)) => (x, y),
                _ => return Err(internal(vm, "bitwise: dtype mismatch after cast")),
            };
            let xv = x.broadcast(sd.clone()).or_internal(vm, "bitwise lhs")?;
            let yv = y.broadcast(sd.clone()).or_internal(vm, "bitwise rhs")?;
            let mut out = ArrayD::<$ty>::zeros(sd.clone());
            Zip::from(&mut out).and(&xv).and(&yv).for_each(|o, &p, &q| {
                *o = uop(p as u64, q as u64) as $ty;
            });
            ArraysD::$var(out)
        }};
    }
    Ok(match out_dtype {
        DType::Bool => {
            let (x, y) = match (&a, &b) {
                (ArraysD::Bool(x), ArraysD::Bool(y)) => (x, y),
                _ => return Err(internal(vm, "bitwise bool: dtype mismatch after cast")),
            };
            let xv = x.broadcast(sd.clone()).or_internal(vm, "bitwise bool lhs")?;
            let yv = y.broadcast(sd.clone()).or_internal(vm, "bitwise bool rhs")?;
            let mut out = ArrayD::<bool>::from_elem(sd.clone(), false);
            Zip::from(&mut out).and(&xv).and(&yv).for_each(|o, &p, &q| *o = bop(p, q));
            ArraysD::Bool(out)
        }
        DType::I8 => sint!(I8, i8),
        DType::I16 => sint!(I16, i16),
        DType::I32 => sint!(I32, i32),
        DType::I64 => sint!(I64, i64),
        DType::U8 => uint!(U8, u8),
        DType::U16 => uint!(U16, u16),
        DType::U32 => uint!(U32, u32),
        DType::U64 => uint!(U64, u64),
        _ => return Err(internal(vm, "bitwise: unexpected dtype after promotion")),
    })
}

pub fn invert(a: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    Ok(match a {
        ArraysD::Bool(x) => ArraysD::Bool(x.mapv(|v| !v)),
        ArraysD::I8(x) => ArraysD::I8(x.mapv(|v| !v)),
        ArraysD::I16(x) => ArraysD::I16(x.mapv(|v| !v)),
        ArraysD::I32(x) => ArraysD::I32(x.mapv(|v| !v)),
        ArraysD::I64(x) => ArraysD::I64(x.mapv(|v| !v)),
        ArraysD::U8(x) => ArraysD::U8(x.mapv(|v| !v)),
        ArraysD::U16(x) => ArraysD::U16(x.mapv(|v| !v)),
        ArraysD::U32(x) => ArraysD::U32(x.mapv(|v| !v)),
        ArraysD::U64(x) => ArraysD::U64(x.mapv(|v| !v)),
        _ => {
            return Err(vm.new_type_error(format!(
                "bitwise_not / invert not defined for {}",
                a.dtype().name()
            )));
        }
    })
}

// =====================================================================
// Finite predicates
// =====================================================================

pub fn isnan(a: &ArraysD) -> ArraysD {
    match a {
        ArraysD::F16(x) => ArraysD::Bool(x.mapv(|v| f32::from(v).is_nan())),
        ArraysD::F32(x) => ArraysD::Bool(x.mapv(|v| v.is_nan())),
        ArraysD::F64(x) => ArraysD::Bool(x.mapv(|v| v.is_nan())),
        ArraysD::C64(x) => ArraysD::Bool(x.mapv(|v| v.re.is_nan() || v.im.is_nan())),
        ArraysD::C128(x) => ArraysD::Bool(x.mapv(|v| v.re.is_nan() || v.im.is_nan())),
        _ => {
            let s = a.raw_dim();
            ArraysD::Bool(ArrayD::from_elem(s, false))
        }
    }
}

pub fn isinf(a: &ArraysD) -> ArraysD {
    match a {
        ArraysD::F16(x) => ArraysD::Bool(x.mapv(|v| f32::from(v).is_infinite())),
        ArraysD::F32(x) => ArraysD::Bool(x.mapv(|v| v.is_infinite())),
        ArraysD::F64(x) => ArraysD::Bool(x.mapv(|v| v.is_infinite())),
        ArraysD::C64(x) => ArraysD::Bool(x.mapv(|v| v.re.is_infinite() || v.im.is_infinite())),
        ArraysD::C128(x) => ArraysD::Bool(x.mapv(|v| v.re.is_infinite() || v.im.is_infinite())),
        _ => {
            let s = a.raw_dim();
            ArraysD::Bool(ArrayD::from_elem(s, false))
        }
    }
}

pub fn isfinite(a: &ArraysD) -> ArraysD {
    match a {
        ArraysD::F16(x) => ArraysD::Bool(x.mapv(|v| f32::from(v).is_finite())),
        ArraysD::F32(x) => ArraysD::Bool(x.mapv(|v| v.is_finite())),
        ArraysD::F64(x) => ArraysD::Bool(x.mapv(|v| v.is_finite())),
        ArraysD::C64(x) => ArraysD::Bool(x.mapv(|v| v.re.is_finite() && v.im.is_finite())),
        ArraysD::C128(x) => ArraysD::Bool(x.mapv(|v| v.re.is_finite() && v.im.is_finite())),
        _ => {
            let s = a.raw_dim();
            ArraysD::Bool(ArrayD::from_elem(s, true))
        }
    }
}

pub fn isclose(
    a: &ArraysD,
    b: &ArraysD,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
    vm: &VirtualMachine,
) -> PyResult<ArraysD> {
    let pt = promote(a.dtype(), b.dtype());
    let a = a.cast(pt).cast(if pt.is_complex() { DType::C128 } else { DType::F64 });
    let b = b.cast(pt).cast(if pt.is_complex() { DType::C128 } else { DType::F64 });
    let s = broadcast(a.shape(), b.shape())
        .ok_or_else(|| vm.new_value_error("broadcast failure".to_string()))?;
    let sd = IxDyn(&s);
    let mut out = ArrayD::<bool>::from_elem(sd.clone(), false);
    match (&a, &b) {
        (ArraysD::F64(x), ArraysD::F64(y)) => {
            let xv = x.broadcast(sd.clone()).or_internal(vm, "isclose lhs")?;
            let yv = y.broadcast(sd.clone()).or_internal(vm, "isclose rhs")?;
            Zip::from(&mut out).and(&xv).and(&yv).for_each(|o, &p, &q| {
                *o = close_real(p, q, rtol, atol, equal_nan);
            });
        }
        (ArraysD::C128(x), ArraysD::C128(y)) => {
            let xv = x.broadcast(sd.clone()).or_internal(vm, "isclose lhs c")?;
            let yv = y.broadcast(sd.clone()).or_internal(vm, "isclose rhs c")?;
            Zip::from(&mut out).and(&xv).and(&yv).for_each(|o, &p, &q| {
                *o = close_real(p.re, q.re, rtol, atol, equal_nan)
                    && close_real(p.im, q.im, rtol, atol, equal_nan);
            });
        }
        _ => return Err(internal(vm, "isclose: dtype unification failed")),
    }
    Ok(ArraysD::Bool(out))
}

fn close_real(a: f64, b: f64, rtol: f64, atol: f64, equal_nan: bool) -> bool {
    if equal_nan && a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_nan() || b.is_nan() {
        return false;
    }
    (a - b).abs() <= atol + rtol * b.abs()
}

pub fn allclose(
    a: &ArraysD,
    b: &ArraysD,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
    vm: &VirtualMachine,
) -> PyResult<bool> {
    let close = isclose(a, b, rtol, atol, equal_nan, vm)?;
    match close {
        ArraysD::Bool(c) => Ok(c.iter().all(|&x| x)),
        _ => Err(internal(vm, "allclose: isclose returned non-bool")),
    }
}

pub fn array_equal(a: &ArraysD, b: &ArraysD) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    let pt = promote(a.dtype(), b.dtype());
    let a = a.cast(pt);
    let b = b.cast(pt);
    match (&a, &b) {
        (ArraysD::Bool(x), ArraysD::Bool(y)) => x == y,
        (ArraysD::I8(x), ArraysD::I8(y)) => x == y,
        (ArraysD::I16(x), ArraysD::I16(y)) => x == y,
        (ArraysD::I32(x), ArraysD::I32(y)) => x == y,
        (ArraysD::I64(x), ArraysD::I64(y)) => x == y,
        (ArraysD::U8(x), ArraysD::U8(y)) => x == y,
        (ArraysD::U16(x), ArraysD::U16(y)) => x == y,
        (ArraysD::U32(x), ArraysD::U32(y)) => x == y,
        (ArraysD::U64(x), ArraysD::U64(y)) => x == y,
        (ArraysD::F16(x), ArraysD::F16(y)) => x == y,
        (ArraysD::F32(x), ArraysD::F32(y)) => x == y,
        (ArraysD::F64(x), ArraysD::F64(y)) => x == y,
        (ArraysD::C64(x), ArraysD::C64(y)) => x == y,
        (ArraysD::C128(x), ArraysD::C128(y)) => x == y,
        // Promotion brings both operands to the same dtype, so this arm is
        // logically unreachable; treat any mismatch as "not equal".
        _ => false,
    }
}

// =====================================================================
// any / all
// =====================================================================

pub fn any(a: &ArraysD, axis: Option<isize>, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let b = to_bool_array(a);
    boolean_reduce(&b, axis, |v| v.iter().any(|&x| x), vm)
}
pub fn all(a: &ArraysD, axis: Option<isize>, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let b = to_bool_array(a);
    boolean_reduce(&b, axis, |v| v.iter().all(|&x| x), vm)
}

fn boolean_reduce(
    a: &ArrayD<bool>,
    axis: Option<isize>,
    op: impl Fn(&[bool]) -> bool,
    vm: &VirtualMachine,
) -> PyResult<ArraysD> {
    let nd = a.ndim() as isize;
    let axis = axis.map(|ax| if ax < 0 { ax + nd } else { ax });
    if let Some(ax) = axis
        && (ax < 0 || ax >= nd) {
            return Err(vm.new_value_error(format!("axis {ax} out of range")));
        }
    Ok(match axis {
        None => {
            let v: Vec<bool> = a.iter().copied().collect();
            ArraysD::Bool(ArrayD::from_elem(IxDyn(&[]), op(&v)))
        }
        Some(ax) => {
            let ax = ax as usize;
            let mut out_shape: Vec<usize> = a.shape().to_vec();
            let axis_len = out_shape.remove(ax);
            if out_shape.is_empty() {
                let v: Vec<bool> = a.iter().copied().collect();
                return Ok(ArraysD::Bool(ArrayD::from_elem(IxDyn(&[]), op(&v))));
            }
            let out_n: usize = out_shape.iter().product();
            let mut buckets: Vec<Vec<bool>> =
                (0..out_n).map(|_| Vec::with_capacity(axis_len)).collect();
            for sub in a.axis_iter(Axis(ax)) {
                for (i, &x) in sub.iter().enumerate() {
                    buckets[i].push(x);
                }
            }
            let data: Vec<bool> = buckets.iter().map(|v| op(v)).collect();
            ArraysD::Bool(ArrayD::from_shape_vec(IxDyn(&out_shape), data).unwrap_or_default())
        }
    })
}

// =====================================================================
// cumsum / cumprod / diff
// =====================================================================

/// numpy `cumsum` — flattens with `axis=None` (default), or accumulates along
/// the given axis.
pub fn cumsum_axis(
    a: &ArraysD,
    axis: Option<isize>,
    vm: &VirtualMachine,
) -> PyResult<ArraysD> {
    match axis {
        None => cumsum(a, vm),
        Some(ax) => {
            let nd = a.ndim() as isize;
            let r = if ax < 0 { ax + nd } else { ax };
            if r < 0 || r >= nd {
                return Err(vm.new_value_error(format!("axis {ax} out of range")));
            }
            cumulate_along_axis(a, r as usize, true, vm)
        }
    }
}

pub fn cumprod_axis(
    a: &ArraysD,
    axis: Option<isize>,
    vm: &VirtualMachine,
) -> PyResult<ArraysD> {
    match axis {
        None => cumprod(a, vm),
        Some(ax) => {
            let nd = a.ndim() as isize;
            let r = if ax < 0 { ax + nd } else { ax };
            if r < 0 || r >= nd {
                return Err(vm.new_value_error(format!("axis {ax} out of range")));
            }
            cumulate_along_axis(a, r as usize, false, vm)
        }
    }
}

fn cumulate_along_axis(
    a: &ArraysD,
    axis: usize,
    is_sum: bool,
    vm: &VirtualMachine,
) -> PyResult<ArraysD> {
    // Widen ints to i64/u64 to match numpy's cumsum/cumprod accumulator dtype.
    let widened = match a.dtype() {
        DType::Bool | DType::I8 | DType::I16 | DType::I32 | DType::I64 => a.cast(DType::I64),
        DType::U8 | DType::U16 | DType::U32 | DType::U64 => a.cast(DType::U64),
        _ => a.clone(),
    };
    macro_rules! per_int {
        ($var:ident, $arr:ident) => {{
            let mut out = $arr.clone();
            for mut lane in out.lanes_mut(Axis(axis)) {
                let mut acc = if is_sum { 0 } else { 1 };
                for slot in lane.iter_mut() {
                    if is_sum {
                        acc += *slot;
                    } else {
                        acc *= *slot;
                    }
                    *slot = acc;
                }
            }
            ArraysD::$var(out)
        }};
    }
    macro_rules! per_float {
        ($var:ident, $arr:ident, $zero:expr, $one:expr) => {{
            let mut out = $arr.clone();
            for mut lane in out.lanes_mut(Axis(axis)) {
                let mut acc = if is_sum { $zero } else { $one };
                for slot in lane.iter_mut() {
                    if is_sum {
                        acc += *slot;
                    } else {
                        acc *= *slot;
                    }
                    *slot = acc;
                }
            }
            ArraysD::$var(out)
        }};
    }
    Ok(match widened {
        ArraysD::I64(arr) => per_int!(I64, arr),
        ArraysD::U64(arr) => per_int!(U64, arr),
        ArraysD::F32(arr) => per_float!(F32, arr, 0.0f32, 1.0f32),
        ArraysD::F64(arr) => per_float!(F64, arr, 0.0f64, 1.0f64),
        ArraysD::F16(arr) => {
            let mut out = arr.clone();
            for mut lane in out.lanes_mut(Axis(axis)) {
                let mut acc: f32 = if is_sum { 0.0 } else { 1.0 };
                for slot in lane.iter_mut() {
                    if is_sum {
                        acc += f32::from(*slot);
                    } else {
                        acc *= f32::from(*slot);
                    }
                    *slot = half::f16::from_f32(acc);
                }
            }
            ArraysD::F16(out)
        }
        ArraysD::C64(arr) => per_float!(C64, arr, C32::new(0.0, 0.0), C32::new(1.0, 0.0)),
        ArraysD::C128(arr) => per_float!(C128, arr, C64::new(0.0, 0.0), C64::new(1.0, 0.0)),
        _ => return Err(internal(vm, "cumulate: unexpected widened dtype")),
    })
}

pub fn cumsum(a: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    // numpy: cumulative along flattened array by default; we expose flat only.
    let widened = match a.dtype() {
        DType::Bool
        | DType::I8
        | DType::I16
        | DType::I32
        | DType::I64 => a.cast(DType::I64),
        DType::U8 | DType::U16 | DType::U32 | DType::U64 => a.cast(DType::U64),
        _ => a.clone(),
    };
    Ok(match widened {
        ArraysD::I64(arr) => {
            let mut acc = 0i64;
            let data: Vec<i64> = arr
                .iter()
                .map(|&v| {
                    acc = acc.wrapping_add(v);
                    acc
                })
                .collect();
            ArraysD::I64(ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap_or_default())
        }
        ArraysD::U64(arr) => {
            let mut acc = 0u64;
            let data: Vec<u64> = arr
                .iter()
                .map(|&v| {
                    acc = acc.wrapping_add(v);
                    acc
                })
                .collect();
            ArraysD::U64(ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap_or_default())
        }
        ArraysD::F32(arr) => {
            let mut acc = 0f32;
            let data: Vec<f32> = arr
                .iter()
                .map(|&v| {
                    acc += v;
                    acc
                })
                .collect();
            ArraysD::F32(ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap_or_default())
        }
        ArraysD::F64(arr) => {
            let mut acc = 0f64;
            let data: Vec<f64> = arr
                .iter()
                .map(|&v| {
                    acc += v;
                    acc
                })
                .collect();
            ArraysD::F64(ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap_or_default())
        }
        ArraysD::F16(arr) => {
            let mut acc = 0f32;
            let data: Vec<f16> = arr
                .iter()
                .map(|&v| {
                    acc += f32::from(v);
                    f16::from_f32(acc)
                })
                .collect();
            ArraysD::F16(ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap_or_default())
        }
        ArraysD::C64(arr) => {
            let mut acc = C32::new(0.0, 0.0);
            let data: Vec<C32> = arr
                .iter()
                .map(|&v| {
                    acc += v;
                    acc
                })
                .collect();
            ArraysD::C64(ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap_or_default())
        }
        ArraysD::C128(arr) => {
            let mut acc = C64::new(0.0, 0.0);
            let data: Vec<C64> = arr
                .iter()
                .map(|&v| {
                    acc += v;
                    acc
                })
                .collect();
            ArraysD::C128(ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap_or_default())
        }
        _ => return Err(vm.new_type_error("cumsum: unexpected dtype".to_string())),
    })
}

pub fn cumprod(a: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let widened = match a.dtype() {
        DType::Bool
        | DType::I8
        | DType::I16
        | DType::I32
        | DType::I64 => a.cast(DType::I64),
        DType::U8 | DType::U16 | DType::U32 | DType::U64 => a.cast(DType::U64),
        _ => a.clone(),
    };
    Ok(match widened {
        ArraysD::I64(arr) => {
            let mut acc = 1i64;
            let data: Vec<i64> = arr
                .iter()
                .map(|&v| {
                    acc = acc.wrapping_mul(v);
                    acc
                })
                .collect();
            ArraysD::I64(ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap_or_default())
        }
        ArraysD::U64(arr) => {
            let mut acc = 1u64;
            let data: Vec<u64> = arr
                .iter()
                .map(|&v| {
                    acc = acc.wrapping_mul(v);
                    acc
                })
                .collect();
            ArraysD::U64(ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap_or_default())
        }
        ArraysD::F64(arr) => {
            let mut acc = 1f64;
            let data: Vec<f64> = arr
                .iter()
                .map(|&v| {
                    acc *= v;
                    acc
                })
                .collect();
            ArraysD::F64(ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap_or_default())
        }
        ArraysD::F32(arr) => {
            let mut acc = 1f32;
            let data: Vec<f32> = arr
                .iter()
                .map(|&v| {
                    acc *= v;
                    acc
                })
                .collect();
            ArraysD::F32(ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap_or_default())
        }
        ArraysD::F16(arr) => {
            let mut acc = 1f32;
            let data: Vec<f16> = arr
                .iter()
                .map(|&v| {
                    acc *= f32::from(v);
                    f16::from_f32(acc)
                })
                .collect();
            ArraysD::F16(ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap_or_default())
        }
        _ => return Err(vm.new_type_error("cumprod: unexpected dtype".to_string())),
    })
}

pub fn diff_axis(
    a: &ArraysD,
    axis: Option<isize>,
    vm: &VirtualMachine,
) -> PyResult<ArraysD> {
    let ax = axis.unwrap_or(-1);
    let nd = a.ndim() as isize;
    if nd == 0 {
        return Err(vm.new_value_error("diff: scalar input has no axis".to_string()));
    }
    let r = if ax < 0 { ax + nd } else { ax };
    if r < 0 || r >= nd {
        return Err(vm.new_value_error(format!("axis {ax} out of range")));
    }
    let axis_idx = r as usize;
    let mut shape: Vec<usize> = a.shape().to_vec();
    if shape[axis_idx] == 0 {
        shape[axis_idx] = 0;
        return Ok(crate::create::zeros(&shape, a.dtype()));
    }
    shape[axis_idx] -= 1;
    macro_rules! diff_per {
        ($var:ident, $arr:ident, $sub:expr) => {{
            let n = $arr.shape()[axis_idx];
            let mut out = ndarray::ArrayD::<_>::default(IxDyn(&shape));
            for k in 1..n {
                let lhs = $arr.index_axis(Axis(axis_idx), k);
                let rhs = $arr.index_axis(Axis(axis_idx), k - 1);
                let mut out_slice = out.index_axis_mut(Axis(axis_idx), k - 1);
                ndarray::Zip::from(&mut out_slice).and(&lhs).and(&rhs).for_each(
                    |o, &a, &b| *o = $sub(a, b),
                );
            }
            ArraysD::$var(out)
        }};
    }
    // numpy promotes the diff dtype so negative results fit:
    //   Bool -> i8;  U<n> -> I<2n> (capped at I64).
    let widened = match a.dtype() {
        DType::Bool => a.cast(DType::I8),
        DType::U8 => a.cast(DType::I16),
        DType::U16 => a.cast(DType::I32),
        DType::U32 | DType::U64 => a.cast(DType::I64),
        _ => a.clone(),
    };
    Ok(match widened {
        ArraysD::I8(arr) => diff_per!(I8, arr, |a: i8, b: i8| a.wrapping_sub(b)),
        ArraysD::I16(arr) => diff_per!(I16, arr, |a: i16, b: i16| a.wrapping_sub(b)),
        ArraysD::I32(arr) => diff_per!(I32, arr, |a: i32, b: i32| a.wrapping_sub(b)),
        ArraysD::I64(arr) => diff_per!(I64, arr, |a: i64, b: i64| a.wrapping_sub(b)),
        ArraysD::U8(arr) => diff_per!(U8, arr, |a: u8, b: u8| a.wrapping_sub(b)),
        ArraysD::U16(arr) => diff_per!(U16, arr, |a: u16, b: u16| a.wrapping_sub(b)),
        ArraysD::U32(arr) => diff_per!(U32, arr, |a: u32, b: u32| a.wrapping_sub(b)),
        ArraysD::U64(arr) => diff_per!(U64, arr, |a: u64, b: u64| a.wrapping_sub(b)),
        ArraysD::F16(arr) => diff_per!(F16, arr, |a: f16, b: f16| f16::from_f32(
            f32::from(a) - f32::from(b)
        )),
        ArraysD::F32(arr) => diff_per!(F32, arr, |a: f32, b: f32| a - b),
        ArraysD::F64(arr) => diff_per!(F64, arr, |a: f64, b: f64| a - b),
        ArraysD::C64(arr) => diff_per!(C64, arr, |a: C32, b: C32| a - b),
        ArraysD::C128(arr) => diff_per!(C128, arr, |a: C64, b: C64| a - b),
        ArraysD::Bool(_) => return Err(internal(vm, "diff: bool slipped past widening")),
        _ => { return Err(crate::internal::unsupported_dtype(vm, "diff", a.dtype())) },
    })
}

pub fn diff(a: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    // numpy `diff` defaults to axis -1; we operate on the flat array.
    let f = crate::linalg::flatten(a);
    let n = f.len();
    if n == 0 {
        return Ok(f);
    }
    // Unsigned subtraction would wrap (uint8(1) - uint8(2) == 255), so
    // widen to a signed type that can express negative diffs. numpy's
    // rule: U<n> -> I<2n> (capped at I64).
    let widened = match f.dtype() {
        DType::Bool => f.cast(DType::I8),
        DType::U8 => f.cast(DType::I16),
        DType::U16 => f.cast(DType::I32),
        DType::U32 => f.cast(DType::I64),
        DType::U64 => f.cast(DType::I64),
        _ => f,
    };
    Ok(match widened {
        ArraysD::I8(x) => ArraysD::I8(diff_ints(&x)),
        ArraysD::I16(x) => ArraysD::I16(diff_ints(&x)),
        ArraysD::I32(x) => ArraysD::I32(diff_ints(&x)),
        ArraysD::I64(x) => ArraysD::I64(diff_ints(&x)),
        ArraysD::F16(x) => ArraysD::F16(diff_seq(&x, |a, b| f16::from_f32(f32::from(b) - f32::from(a)))),
        ArraysD::F32(x) => ArraysD::F32(diff_seq(&x, |a, b| b - a)),
        ArraysD::F64(x) => ArraysD::F64(diff_seq(&x, |a, b| b - a)),
        ArraysD::C64(x) => ArraysD::C64(diff_seq(&x, |a, b| b - a)),
        ArraysD::C128(x) => ArraysD::C128(diff_seq(&x, |a, b| b - a)),
        _ => return Err(vm.new_type_error("diff: unexpected dtype".to_string())),
    })
}

fn diff_ints<T: Copy + Default + std::ops::Sub<Output = T>>(
    x: &ArrayD<T>,
) -> ArrayD<T> {
    let n = x.len();
    let data: Vec<T> = (1..n).map(|i| x[IxDyn(&[i])] - x[IxDyn(&[i - 1])]).collect();
    ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap_or_default()
}

fn diff_seq<T: Copy + Default, F: Fn(T, T) -> T>(x: &ArrayD<T>, f: F) -> ArrayD<T> {
    let n = x.len();
    let data: Vec<T> = (1..n).map(|i| f(x[IxDyn(&[i - 1])], x[IxDyn(&[i])])).collect();
    ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap_or_default()
}

// =====================================================================
// clip / round / trunc
// =====================================================================

pub fn clip(a: &ArraysD, lo: Option<f64>, hi: Option<f64>) -> ArraysD {
    use crate::dtype::CoerceArray as _CA;
    let arr = a.coerce::<f64>();
    let mut out = arr.clone();
    out.mapv_inplace(|v| {
        let mut v = v;
        if let Some(l) = lo
            && v < l { v = l; }
        if let Some(h) = hi
            && v > h { v = h; }
        v
    });
    ArraysD::F64(out).cast(a.dtype())
}

pub fn round_half_even(a: &ArraysD) -> ArraysD {
    crate::ops::unary_real_or_complex(a, |x| {
        // numpy round → banker's rounding
        
        x.round_ties_even()
    }, |c| c)
}

pub fn trunc(a: &ArraysD) -> ArraysD {
    crate::ops::unary_real_or_complex(a, f64::trunc, |c| c)
}

// =====================================================================
// where / nonzero
// =====================================================================

pub fn where_op(
    cond: &ArraysD,
    a: &ArraysD,
    b: &ArraysD,
    vm: &VirtualMachine,
) -> PyResult<ArraysD> {
    let c = to_bool_array(cond);
    let out_dt = promote(a.dtype(), b.dtype());
    let a = a.cast(out_dt);
    let b = b.cast(out_dt);
    let s1 = broadcast(c.shape(), a.shape())
        .ok_or_else(|| vm.new_value_error("broadcast".to_string()))?;
    let s = broadcast(&s1, b.shape())
        .ok_or_else(|| vm.new_value_error("broadcast".to_string()))?;
    let sd = IxDyn(&s);
    let cv = c.broadcast(sd.clone()).or_internal(vm, "where: broadcast cond")?;
    macro_rules! per {
        ($var:ident, $ty:ty) => {{
            let (ArraysD::$var(x), ArraysD::$var(y)) = (&a, &b) else {
                return Err(internal(vm, "where: dtype mismatch after cast"));
            };
            let xv = x.broadcast(sd.clone()).or_internal(vm, "where: broadcast a")?;
            let yv = y.broadcast(sd.clone()).or_internal(vm, "where: broadcast b")?;
            let mut out = ArrayD::<$ty>::from_elem(sd.clone(), <$ty as Default>::default());
            Zip::from(&mut out)
                .and(&cv)
                .and(&xv)
                .and(&yv)
                .for_each(|o, &cc, &xx, &yy| {
                    *o = if cc { xx } else { yy };
                });
            ArraysD::$var(out)
        }};
    }
    Ok(match out_dt {
        DType::Bool => per!(Bool, bool),
        DType::I8 => per!(I8, i8),
        DType::I16 => per!(I16, i16),
        DType::I32 => per!(I32, i32),
        DType::I64 => per!(I64, i64),
        DType::U8 => per!(U8, u8),
        DType::U16 => per!(U16, u16),
        DType::U32 => per!(U32, u32),
        DType::U64 => per!(U64, u64),
        DType::F16 => {
            let (ArraysD::F16(x), ArraysD::F16(y)) = (&a, &b) else {
                return Err(internal(vm, "where: dtype mismatch after cast"));
            };
            let xv = x.broadcast(sd.clone()).or_internal(vm, "where: broadcast a")?;
            let yv = y.broadcast(sd.clone()).or_internal(vm, "where: broadcast b")?;
            let mut out = ArrayD::<f16>::from_elem(sd.clone(), f16::ZERO);
            Zip::from(&mut out).and(&cv).and(&xv).and(&yv).for_each(|o, &cc, &xx, &yy| {
                *o = if cc { xx } else { yy };
            });
            ArraysD::F16(out)
        }
        DType::F32 => per!(F32, f32),
        DType::F64 => per!(F64, f64),
        DType::C64 => {
            let (ArraysD::C64(x), ArraysD::C64(y)) = (&a, &b) else {
                return Err(internal(vm, "where: dtype mismatch after cast"));
            };
            let xv = x.broadcast(sd.clone()).or_internal(vm, "where: broadcast a")?;
            let yv = y.broadcast(sd.clone()).or_internal(vm, "where: broadcast b")?;
            let mut out = ArrayD::<C32>::from_elem(sd.clone(), C32::new(0.0, 0.0));
            Zip::from(&mut out).and(&cv).and(&xv).and(&yv).for_each(|o, &cc, &xx, &yy| {
                *o = if cc { xx } else { yy };
            });
            ArraysD::C64(out)
        }
        DType::C128 => {
            let (ArraysD::C128(x), ArraysD::C128(y)) = (&a, &b) else {
                return Err(internal(vm, "where: dtype mismatch after cast"));
            };
            let xv = x.broadcast(sd.clone()).or_internal(vm, "where: broadcast a")?;
            let yv = y.broadcast(sd.clone()).or_internal(vm, "where: broadcast b")?;
            let mut out = ArrayD::<C64>::from_elem(sd.clone(), C64::new(0.0, 0.0));
            Zip::from(&mut out).and(&cv).and(&xv).and(&yv).for_each(|o, &cc, &xx, &yy| {
                *o = if cc { xx } else { yy };
            });
            ArraysD::C128(out)
        }
        _ => { return Err(crate::internal::unsupported_dtype(vm, "where", out_dt)) },
    })
}

/// `np.nonzero` on a 1-D array — returns indices where value is truthy. For
/// multi-dim arrays numpy returns a tuple of N arrays; we flatten and return
/// the flat indices.
pub fn nonzero(a: &ArraysD) -> ArraysD {
    let b = to_bool_array(a);
    let idx: Vec<i64> = b
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v { Some(i as i64) } else { None })
        .collect();
    ArraysD::I64(ArrayD::from_shape_vec(IxDyn(&[idx.len()]), idx).unwrap_or_default())
}

// =====================================================================
// sort / argsort / unique
// =====================================================================

pub fn sort(a: &ArraysD, axis: Option<isize>, vm: &VirtualMachine) -> PyResult<ArraysD> {
    if matches!(a, ArraysD::C64(_) | ArraysD::C128(_)) {
        return Err(vm.new_type_error("sort not defined for complex".to_string()));
    }
    let resolved = match axis {
        None => return sort_flat(&crate::linalg::flatten(a), vm),
        Some(ax) => {
            let nd = a.ndim() as isize;
            let r = if ax < 0 { ax + nd } else { ax };
            if r < 0 || r >= nd {
                return Err(vm.new_value_error(format!("axis {ax} out of range")));
            }
            r as usize
        }
    };
    sort_along_axis(a, resolved, vm)
}

fn sort_flat(f: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    Ok(match f.clone() {
        ArraysD::Bool(x) => {
            let mut v: Vec<bool> = x.iter().copied().collect();
            v.sort();
            ArraysD::Bool(ArrayD::from_shape_vec(IxDyn(&[v.len()]), v).unwrap_or_default())
        }
        ArraysD::I8(x) => ArraysD::I8(sorted_int(x)),
        ArraysD::I16(x) => ArraysD::I16(sorted_int(x)),
        ArraysD::I32(x) => ArraysD::I32(sorted_int(x)),
        ArraysD::I64(x) => ArraysD::I64(sorted_int(x)),
        ArraysD::U8(x) => ArraysD::U8(sorted_int(x)),
        ArraysD::U16(x) => ArraysD::U16(sorted_int(x)),
        ArraysD::U32(x) => ArraysD::U32(sorted_int(x)),
        ArraysD::U64(x) => ArraysD::U64(sorted_int(x)),
        ArraysD::F16(x) => ArraysD::F16(sorted_float(x)),
        ArraysD::F32(x) => ArraysD::F32(sorted_float(x)),
        ArraysD::F64(x) => ArraysD::F64(sorted_float(x)),
        ArraysD::Str { itemsize_chars, data } => {
            let mut v: Vec<String> = data.iter().cloned().collect();
            v.sort();
            ArraysD::Str {
                itemsize_chars,
                data: ArrayD::from_shape_vec(IxDyn(&[v.len()]), v).unwrap_or_default(),
            }
        }
        ArraysD::Bytes { itemsize, data } => {
            let mut v: Vec<Vec<u8>> = data.iter().cloned().collect();
            v.sort();
            ArraysD::Bytes {
                itemsize,
                data: ArrayD::from_shape_vec(IxDyn(&[v.len()]), v).unwrap_or_default(),
            }
        }
        _ => return Err(internal(vm, "sort_flat: unexpected dtype")),
    })
}

fn sort_along_axis(a: &ArraysD, axis: usize, vm: &VirtualMachine) -> PyResult<ArraysD> {
    macro_rules! per_int {
        ($var:ident, $arr:ident) => {{
            let mut out = $arr.clone();
            for mut lane in out.lanes_mut(Axis(axis)) {
                let mut v: Vec<_> = lane.iter().copied().collect();
                v.sort();
                for (slot, val) in lane.iter_mut().zip(v.into_iter()) {
                    *slot = val;
                }
            }
            ArraysD::$var(out)
        }};
    }
    macro_rules! per_float {
        ($var:ident, $arr:ident) => {{
            let mut out = $arr.clone();
            for mut lane in out.lanes_mut(Axis(axis)) {
                let mut v: Vec<_> = lane.iter().copied().collect();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                for (slot, val) in lane.iter_mut().zip(v.into_iter()) {
                    *slot = val;
                }
            }
            ArraysD::$var(out)
        }};
    }
    Ok(match a {
        ArraysD::Bool(arr) => per_int!(Bool, arr),
        ArraysD::I8(arr) => per_int!(I8, arr),
        ArraysD::I16(arr) => per_int!(I16, arr),
        ArraysD::I32(arr) => per_int!(I32, arr),
        ArraysD::I64(arr) => per_int!(I64, arr),
        ArraysD::U8(arr) => per_int!(U8, arr),
        ArraysD::U16(arr) => per_int!(U16, arr),
        ArraysD::U32(arr) => per_int!(U32, arr),
        ArraysD::U64(arr) => per_int!(U64, arr),
        ArraysD::F16(arr) => per_float!(F16, arr),
        ArraysD::F32(arr) => per_float!(F32, arr),
        ArraysD::F64(arr) => per_float!(F64, arr),
        _ => return Err(internal(vm, "sort: unexpected dtype")),
    })
}

fn sorted_int<T: Copy + Default + Ord>(x: ArrayD<T>) -> ArrayD<T> {
    let mut v: Vec<T> = x.iter().copied().collect();
    v.sort();
    ArrayD::from_shape_vec(IxDyn(&[v.len()]), v).unwrap_or_default()
}

fn sorted_float<T: Copy + Default + PartialOrd>(x: ArrayD<T>) -> ArrayD<T> {
    let mut v: Vec<T> = x.iter().copied().collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ArrayD::from_shape_vec(IxDyn(&[v.len()]), v).unwrap_or_default()
}

pub fn argsort(a: &ArraysD, axis: Option<isize>, vm: &VirtualMachine) -> PyResult<ArraysD> {
    if matches!(a, ArraysD::C64(_) | ArraysD::C128(_)) {
        return Err(vm.new_type_error("argsort not defined for complex".to_string()));
    }
    let resolved = match axis {
        None => return argsort_flat(&crate::linalg::flatten(a), vm),
        Some(ax) => {
            let nd = a.ndim() as isize;
            let r = if ax < 0 { ax + nd } else { ax };
            if r < 0 || r >= nd {
                return Err(vm.new_value_error(format!("axis {ax} out of range")));
            }
            r as usize
        }
    };
    argsort_along_axis(a, resolved, vm)
}

fn argsort_flat(f: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    fn order<T: Copy + PartialOrd>(a: &ArrayD<T>) -> Vec<i64> {
        let mut idx: Vec<i64> = (0..a.len() as i64).collect();
        idx.sort_by(|&i, &j| {
            a[IxDyn(&[i as usize])]
                .partial_cmp(&a[IxDyn(&[j as usize])])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx
    }
    let v: Vec<i64> = match f {
        ArraysD::Bool(x) => {
            let mut idx: Vec<i64> = (0..x.len() as i64).collect();
            idx.sort_by(|&i, &j| {
                x[IxDyn(&[i as usize])].cmp(&x[IxDyn(&[j as usize])])
            });
            idx
        }
        ArraysD::I8(x) => order(x),
        ArraysD::I16(x) => order(x),
        ArraysD::I32(x) => order(x),
        ArraysD::I64(x) => order(x),
        ArraysD::U8(x) => order(x),
        ArraysD::U16(x) => order(x),
        ArraysD::U32(x) => order(x),
        ArraysD::U64(x) => order(x),
        ArraysD::F16(x) => order(x),
        ArraysD::F32(x) => order(x),
        ArraysD::F64(x) => order(x),
        ArraysD::Str { data, .. } => {
            let mut idx: Vec<i64> = (0..data.len() as i64).collect();
            idx.sort_by(|&i, &j| {
                data[IxDyn(&[i as usize])].cmp(&data[IxDyn(&[j as usize])])
            });
            idx
        }
        ArraysD::Bytes { data, .. } => {
            let mut idx: Vec<i64> = (0..data.len() as i64).collect();
            idx.sort_by(|&i, &j| {
                data[IxDyn(&[i as usize])].cmp(&data[IxDyn(&[j as usize])])
            });
            idx
        }
        _ => return Err(internal(vm, "argsort_flat: unexpected dtype")),
    };
    Ok(ArraysD::I64(
        ArrayD::from_shape_vec(IxDyn(&[v.len()]), v).unwrap_or_default(),
    ))
}

fn argsort_along_axis(a: &ArraysD, axis: usize, vm: &VirtualMachine) -> PyResult<ArraysD> {
    fn order_lane<T: Copy + PartialOrd>(slice: &[T]) -> Vec<i64> {
        let n = slice.len();
        let mut idx: Vec<i64> = (0..n as i64).collect();
        idx.sort_by(|&i, &j| {
            slice[i as usize]
                .partial_cmp(&slice[j as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx
    }
    macro_rules! per {
        ($arr:ident) => {{
            let shape = $arr.shape().to_vec();
            let mut out = ArrayD::<i64>::zeros(IxDyn(&shape));
            for (lane, mut out_lane) in $arr
                .lanes(Axis(axis))
                .into_iter()
                .zip(out.lanes_mut(Axis(axis)))
            {
                let buf: Vec<_> = lane.iter().copied().collect();
                let v = order_lane(&buf);
                for (slot, val) in out_lane.iter_mut().zip(v.into_iter()) {
                    *slot = val;
                }
            }
            ArraysD::I64(out)
        }};
    }
    // Clone-based variant for non-Copy element types.
    macro_rules! per_clone {
        ($arr:ident) => {{
            let shape = $arr.shape().to_vec();
            let mut out = ArrayD::<i64>::zeros(IxDyn(&shape));
            for (lane, mut out_lane) in $arr
                .lanes(Axis(axis))
                .into_iter()
                .zip(out.lanes_mut(Axis(axis)))
            {
                let n = lane.len();
                let mut idx: Vec<i64> = (0..n as i64).collect();
                idx.sort_by(|&i, &j| lane[i as usize].cmp(&lane[j as usize]));
                for (slot, val) in out_lane.iter_mut().zip(idx.into_iter()) {
                    *slot = val;
                }
            }
            ArraysD::I64(out)
        }};
    }
    Ok(match a {
        ArraysD::Bool(arr) => per!(arr),
        ArraysD::I8(arr) => per!(arr),
        ArraysD::I16(arr) => per!(arr),
        ArraysD::I32(arr) => per!(arr),
        ArraysD::I64(arr) => per!(arr),
        ArraysD::U8(arr) => per!(arr),
        ArraysD::U16(arr) => per!(arr),
        ArraysD::U32(arr) => per!(arr),
        ArraysD::U64(arr) => per!(arr),
        ArraysD::F16(arr) => per!(arr),
        ArraysD::F32(arr) => per!(arr),
        ArraysD::F64(arr) => per!(arr),
        ArraysD::Str { data, .. } => per_clone!(data),
        ArraysD::Bytes { data, .. } => per_clone!(data),
        _ => return Err(internal(vm, "argsort: unexpected dtype")),
    })
}

pub fn unique(a: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    // numpy.unique with no axis flattens, sorts, and dedups.
    let sorted = sort(a, None, vm)?;
    dedup_after_sort(sorted, vm)
}

fn dedup_after_sort(arr: ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    Ok(match arr {
        ArraysD::Bool(x) => {
            let mut v: Vec<bool> = x.iter().copied().collect();
            v.dedup();
            ArraysD::Bool(ArrayD::from_shape_vec(IxDyn(&[v.len()]), v).unwrap_or_default())
        }
        ArraysD::I8(x) => ArraysD::I8(dedup_int(x)),
        ArraysD::I16(x) => ArraysD::I16(dedup_int(x)),
        ArraysD::I32(x) => ArraysD::I32(dedup_int(x)),
        ArraysD::I64(x) => ArraysD::I64(dedup_int(x)),
        ArraysD::U8(x) => ArraysD::U8(dedup_int(x)),
        ArraysD::U16(x) => ArraysD::U16(dedup_int(x)),
        ArraysD::U32(x) => ArraysD::U32(dedup_int(x)),
        ArraysD::U64(x) => ArraysD::U64(dedup_int(x)),
        ArraysD::F16(x) => ArraysD::F16(dedup_float(x)),
        ArraysD::F32(x) => ArraysD::F32(dedup_float(x)),
        ArraysD::F64(x) => ArraysD::F64(dedup_float(x)),
        ArraysD::Str { itemsize_chars, data } => {
            let mut v: Vec<String> = data.iter().cloned().collect();
            v.dedup();
            ArraysD::Str {
                itemsize_chars,
                data: ArrayD::from_shape_vec(IxDyn(&[v.len()]), v).unwrap_or_default(),
            }
        }
        ArraysD::Bytes { itemsize, data } => {
            let mut v: Vec<Vec<u8>> = data.iter().cloned().collect();
            v.dedup();
            ArraysD::Bytes {
                itemsize,
                data: ArrayD::from_shape_vec(IxDyn(&[v.len()]), v).unwrap_or_default(),
            }
        }
        _ => return Err(internal(vm, "unique/dedup: unexpected dtype")),
    })
}

fn dedup_int<T: Copy + Default + PartialEq>(x: ArrayD<T>) -> ArrayD<T> {
    let mut v: Vec<T> = x.iter().copied().collect();
    v.dedup();
    ArrayD::from_shape_vec(IxDyn(&[v.len()]), v).unwrap_or_default()
}
fn dedup_float<T: Copy + Default + PartialEq>(x: ArrayD<T>) -> ArrayD<T> {
    dedup_int(x)
}

// =====================================================================
// stack / squeeze / expand_dims / broadcast_to / repeat / tile / moveaxis / swapaxes
// =====================================================================

pub fn stack(arrays: &[ArraysD], axis: usize, vm: &VirtualMachine) -> PyResult<ArraysD> {
    if arrays.is_empty() {
        return Err(vm.new_value_error("need at least one array to stack".to_string()));
    }
    let s0 = arrays[0].shape().to_vec();
    for a in &arrays[1..] {
        if a.shape() != s0.as_slice() {
            return Err(vm.new_value_error(format!(
                "all input arrays must have the same shape: {:?} vs {:?}",
                s0,
                a.shape()
            )));
        }
    }
    // Insert axis of size 1 in each array, then concatenate.
    let with_axis: Vec<ArraysD> = arrays
        .iter()
        .map(|a| expand_dims(a, axis, vm))
        .collect::<PyResult<_>>()?;
    crate::linalg::concatenate(&with_axis, axis, vm)
}

pub fn hstack(arrays: &[ArraysD], vm: &VirtualMachine) -> PyResult<ArraysD> {
    if arrays.iter().all(|a| a.ndim() == 1) {
        return crate::linalg::concatenate(arrays, 0, vm);
    }
    crate::linalg::concatenate(arrays, 1, vm)
}

pub fn vstack(arrays: &[ArraysD], vm: &VirtualMachine) -> PyResult<ArraysD> {
    // numpy: 1-D arrays are treated as rows.
    let normalized: Vec<ArraysD> = arrays
        .iter()
        .map(|a| {
            if a.ndim() == 1 {
                let mut s = vec![1usize];
                s.extend(a.shape());
                crate::linalg::reshape(a, &s).unwrap_or_else(|| a.clone())
            } else {
                a.clone()
            }
        })
        .collect();
    crate::linalg::concatenate(&normalized, 0, vm)
}

pub fn squeeze(a: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let new_shape: Vec<usize> = a.shape().iter().copied().filter(|&d| d != 1).collect();
    crate::linalg::reshape(a, &new_shape)
        .ok_or_else(|| vm.new_value_error("squeeze failed".to_string()))
}

pub fn expand_dims(a: &ArraysD, axis: usize, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let mut s = a.shape().to_vec();
    if axis > s.len() {
        return Err(vm.new_value_error(format!("axis {axis} out of range")));
    }
    s.insert(axis, 1);
    crate::linalg::reshape(a, &s)
        .ok_or_else(|| vm.new_value_error("expand_dims failed".to_string()))
}

pub fn broadcast_to(a: &ArraysD, shape: &[usize], vm: &VirtualMachine) -> PyResult<ArraysD> {
    let sd = IxDyn(shape);
    macro_rules! per {
        ($var:ident, $arr:ident) => {{
            $arr.broadcast(sd.clone())
                .map(|v| ArraysD::$var(v.to_owned()))
                .ok_or_else(|| {
                    vm.new_value_error(format!(
                        "cannot broadcast array of shape {:?} to {:?}",
                        $arr.shape(),
                        shape
                    ))
                })
        }};
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
        _ => { return Err(crate::internal::unsupported_dtype(vm, "broadcast_to", a.dtype())) },
    }
}

pub fn repeat(a: &ArraysD, count: usize) -> ArraysD {
    // numpy.repeat flattens first when no axis is given.
    let f = crate::linalg::flatten(a);
    macro_rules! per {
        ($var:ident, $ty:ty, $arr:ident) => {{
            let n = $arr.len();
            let mut data: Vec<$ty> = Vec::with_capacity(n * count);
            for &v in $arr.iter() {
                for _ in 0..count {
                    data.push(v);
                }
            }
            ArraysD::$var(ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap_or_default())
        }};
    }
    match f {
        ArraysD::Bool(x) => per!(Bool, bool, x),
        ArraysD::I8(x) => per!(I8, i8, x),
        ArraysD::I16(x) => per!(I16, i16, x),
        ArraysD::I32(x) => per!(I32, i32, x),
        ArraysD::I64(x) => per!(I64, i64, x),
        ArraysD::U8(x) => per!(U8, u8, x),
        ArraysD::U16(x) => per!(U16, u16, x),
        ArraysD::U32(x) => per!(U32, u32, x),
        ArraysD::U64(x) => per!(U64, u64, x),
        ArraysD::F16(x) => per!(F16, f16, x),
        ArraysD::F32(x) => per!(F32, f32, x),
        ArraysD::F64(x) => per!(F64, f64, x),
        ArraysD::C64(x) => per!(C64, C32, x),
        ArraysD::C128(x) => per!(C128, C64, x),
        _ => { a.clone() },
    }
}

pub fn tile(a: &ArraysD, reps: usize) -> ArraysD {
    // 1-D tile: repeats the whole array `reps` times.
    let f = crate::linalg::flatten(a);
    macro_rules! per {
        ($var:ident, $ty:ty, $arr:ident) => {{
            let mut data: Vec<$ty> = Vec::with_capacity($arr.len() * reps);
            for _ in 0..reps {
                for &v in $arr.iter() {
                    data.push(v);
                }
            }
            ArraysD::$var(ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap_or_default())
        }};
    }
    match f {
        ArraysD::Bool(x) => per!(Bool, bool, x),
        ArraysD::I8(x) => per!(I8, i8, x),
        ArraysD::I16(x) => per!(I16, i16, x),
        ArraysD::I32(x) => per!(I32, i32, x),
        ArraysD::I64(x) => per!(I64, i64, x),
        ArraysD::U8(x) => per!(U8, u8, x),
        ArraysD::U16(x) => per!(U16, u16, x),
        ArraysD::U32(x) => per!(U32, u32, x),
        ArraysD::U64(x) => per!(U64, u64, x),
        ArraysD::F16(x) => per!(F16, f16, x),
        ArraysD::F32(x) => per!(F32, f32, x),
        ArraysD::F64(x) => per!(F64, f64, x),
        ArraysD::C64(x) => per!(C64, C32, x),
        ArraysD::C128(x) => per!(C128, C64, x),
        _ => { a.clone() },
    }
}

// =====================================================================
// ptp / median
// =====================================================================

pub fn ptp(a: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    use crate::dtype::CoerceArray as _CA;
    let arr = a.coerce::<f64>();
    if arr.is_empty() {
        return Err(vm.new_value_error("ptp of empty array".to_string()));
    }
    let mn = arr.iter().copied().fold(f64::INFINITY, f64::min);
    let mx = arr.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Ok(ArraysD::F64(ArrayD::from_elem(IxDyn(&[]), mx - mn)).cast(a.dtype()))
}

pub fn median(a: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    use crate::dtype::CoerceArray as _CA;
    let arr = a.coerce::<f64>();
    if arr.is_empty() {
        return Err(vm.new_value_error("median of empty array".to_string()));
    }
    let mut v: Vec<f64> = arr.iter().copied().collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    let m = if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) * 0.5
    };
    Ok(ArraysD::F64(ArrayD::from_elem(IxDyn(&[]), m)))
}
