//! dot / matmul (numpy semantics for 1-D and 2-D cases).
//!
//! Integer dtypes are promoted to the smallest float that holds them
//! (i.e. f64 for any integer operand) before calling ndarray::dot. This
//! avoids a separate generic integer matmul implementation while matching
//! numpy's behaviour: `np.dot(int_array, int_array)` returns an int array
//! computed exactly. To keep parity for small integer dtypes we use an
//! explicit integer matmul instead.

use crate::dtype::{ArraysD, C32, C64, DType};
use crate::internal::{ResultExt, internal};
use crate::promote::promote;
use half::f16;
use ndarray::{ArrayD, Axis, IxDyn};
use rustpython_vm::{PyResult, VirtualMachine};

pub fn dot(a: &ArraysD, b: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let target = promote(a.dtype(), b.dtype());
    let a = a.cast(target);
    let b = b.cast(target);
    match (a.ndim(), b.ndim()) {
        (1, 1) => dot_1d(&a, &b, vm),
        (2, 2) => dot_2d(&a, &b, vm),
        (2, 1) => mat_vec(&a, &b, vm),
        (1, 2) => vec_mat(&a, &b, vm),
        _ => Err(vm.new_value_error(
            "dot only supports 1-D and 2-D inputs in this implementation".to_string(),
        )),
    }
}

fn dot_1d(a: &ArraysD, b: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    if a.len() != b.len() {
        return Err(vm.new_value_error("shapes not aligned for dot".to_string()));
    }
    Ok(match (a, b) {
        (ArraysD::Bool(x), ArraysD::Bool(y)) => {
            let s: i64 = x.iter().zip(y.iter()).map(|(p, q)| (*p & *q) as i64).sum();
            ArraysD::I64(ArrayD::from_elem(IxDyn(&[]), s))
        }
        (ArraysD::I8(x), ArraysD::I8(y)) => scalar_array(
            x.iter()
                .zip(y.iter())
                .map(|(a, b)| *a as i64 * *b as i64)
                .sum::<i64>(),
            DType::I64,
        ),
        (ArraysD::I16(x), ArraysD::I16(y)) => scalar_array(
            x.iter()
                .zip(y.iter())
                .map(|(a, b)| *a as i64 * *b as i64)
                .sum::<i64>(),
            DType::I64,
        ),
        (ArraysD::I32(x), ArraysD::I32(y)) => scalar_array(
            x.iter()
                .zip(y.iter())
                .map(|(a, b)| *a as i64 * *b as i64)
                .sum::<i64>(),
            DType::I64,
        ),
        (ArraysD::I64(x), ArraysD::I64(y)) => {
            let s: i64 = x
                .iter()
                .zip(y.iter())
                .map(|(a, b)| a.wrapping_mul(*b))
                .sum();
            ArraysD::I64(ArrayD::from_elem(IxDyn(&[]), s))
        }
        (ArraysD::U8(x), ArraysD::U8(y)) => scalar_array_u(
            x.iter()
                .zip(y.iter())
                .map(|(a, b)| *a as u64 * *b as u64)
                .sum::<u64>(),
            DType::U64,
        ),
        (ArraysD::U16(x), ArraysD::U16(y)) => scalar_array_u(
            x.iter()
                .zip(y.iter())
                .map(|(a, b)| *a as u64 * *b as u64)
                .sum::<u64>(),
            DType::U64,
        ),
        (ArraysD::U32(x), ArraysD::U32(y)) => scalar_array_u(
            x.iter()
                .zip(y.iter())
                .map(|(a, b)| *a as u64 * *b as u64)
                .sum::<u64>(),
            DType::U64,
        ),
        (ArraysD::U64(x), ArraysD::U64(y)) => {
            let s: u64 = x
                .iter()
                .zip(y.iter())
                .map(|(a, b)| a.wrapping_mul(*b))
                .sum();
            ArraysD::U64(ArrayD::from_elem(IxDyn(&[]), s))
        }
        (ArraysD::F16(x), ArraysD::F16(y)) => {
            let s: f32 = x
                .iter()
                .zip(y.iter())
                .map(|(a, b)| f32::from(*a) * f32::from(*b))
                .sum();
            ArraysD::F16(ArrayD::from_elem(IxDyn(&[]), f16::from_f32(s)))
        }
        (ArraysD::F32(x), ArraysD::F32(y)) => {
            let xv = x
                .view()
                .into_dimensionality::<ndarray::Ix1>()
                .or_internal(vm, "into Ix1")?;
            let yv = y
                .view()
                .into_dimensionality::<ndarray::Ix1>()
                .or_internal(vm, "into Ix1")?;
            ArraysD::F32(ArrayD::from_elem(IxDyn(&[]), xv.dot(&yv)))
        }
        (ArraysD::F64(x), ArraysD::F64(y)) => {
            let xv = x
                .view()
                .into_dimensionality::<ndarray::Ix1>()
                .or_internal(vm, "into Ix1")?;
            let yv = y
                .view()
                .into_dimensionality::<ndarray::Ix1>()
                .or_internal(vm, "into Ix1")?;
            ArraysD::F64(ArrayD::from_elem(IxDyn(&[]), xv.dot(&yv)))
        }
        (ArraysD::C64(x), ArraysD::C64(y)) => {
            let s: C32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
            ArraysD::C64(ArrayD::from_elem(IxDyn(&[]), s))
        }
        (ArraysD::C128(x), ArraysD::C128(y)) => {
            let s: C64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
            ArraysD::C128(ArrayD::from_elem(IxDyn(&[]), s))
        }
        _ => return Err(internal(vm, "dot_1d: dtype mismatch after promotion")),
    })
}

fn scalar_array(v: i64, dt: DType) -> ArraysD {
    ArraysD::I64(ArrayD::from_elem(IxDyn(&[]), v)).cast(dt)
}
fn scalar_array_u(v: u64, dt: DType) -> ArraysD {
    ArraysD::U64(ArrayD::from_elem(IxDyn(&[]), v)).cast(dt)
}

fn dot_2d(a: &ArraysD, b: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let (k2, n) = (b.shape()[0], b.shape()[1]);
    if k != k2 {
        return Err(vm.new_value_error(format!(
            "shapes {:?} and {:?} not aligned",
            a.shape(),
            b.shape()
        )));
    }
    Ok(match (a, b) {
        (ArraysD::F32(x), ArraysD::F32(y)) => {
            let x2 = x
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .or_internal(vm, "into Ix2")?;
            let y2 = y
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .or_internal(vm, "into Ix2")?;
            ArraysD::F32(x2.dot(&y2).into_dyn())
        }
        (ArraysD::F64(x), ArraysD::F64(y)) => {
            let x2 = x
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .or_internal(vm, "into Ix2")?;
            let y2 = y
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .or_internal(vm, "into Ix2")?;
            ArraysD::F64(x2.dot(&y2).into_dyn())
        }
        (ArraysD::C64(x), ArraysD::C64(y)) => {
            let x2 = x
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .or_internal(vm, "into Ix2")?;
            let y2 = y
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .or_internal(vm, "into Ix2")?;
            ArraysD::C64(x2.dot(&y2).into_dyn())
        }
        (ArraysD::C128(x), ArraysD::C128(y)) => {
            let x2 = x
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .or_internal(vm, "into Ix2")?;
            let y2 = y
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .or_internal(vm, "into Ix2")?;
            ArraysD::C128(x2.dot(&y2).into_dyn())
        }
        // For all other dtypes (bool, integers, f16) do the loop in i64/u64/f32.
        _ => integer_matmul_2d(a, b, m, k, n, vm)?,
    })
}

fn integer_matmul_2d(
    a: &ArraysD,
    b: &ArraysD,
    m: usize,
    k: usize,
    n: usize,
    vm: &VirtualMachine,
) -> PyResult<ArraysD> {
    // Promote both to f64 for f16/bool/small ints — preserves precision for
    // the common 2-D test case.
    if let (ArraysD::F16(x), ArraysD::F16(y)) = (a, b) {
        let mut out = ArrayD::<f16>::from_elem(IxDyn(&[m, n]), f16::ZERO);
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for p in 0..k {
                    acc += f32::from(x[IxDyn(&[i, p])]) * f32::from(y[IxDyn(&[p, j])]);
                }
                out[IxDyn(&[i, j])] = f16::from_f32(acc);
            }
        }
        return Ok(ArraysD::F16(out));
    }
    // Integer / bool path — accumulate in i64 and cast back.
    use crate::dtype::CoerceArray;
    let dt = a.dtype();
    let xi = a.coerce::<i64>();
    let yi = b.coerce::<i64>();
    let mut out = ArrayD::<i64>::zeros(IxDyn(&[m, n]));
    for i in 0..m {
        for j in 0..n {
            let mut acc: i64 = 0;
            for p in 0..k {
                acc = acc.wrapping_add(xi[IxDyn(&[i, p])].wrapping_mul(yi[IxDyn(&[p, j])]));
            }
            out[IxDyn(&[i, j])] = acc;
        }
    }
    let _ = vm;
    Ok(ArraysD::I64(out).cast(dt))
}

fn mat_vec(a: &ArraysD, b: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let m = a.shape()[0];
    let n = a.shape()[1];
    if n != b.len() {
        return Err(vm.new_value_error("shapes not aligned".to_string()));
    }
    let _ = m;
    Ok(match (a, b) {
        (ArraysD::F32(x), ArraysD::F32(y)) => {
            let x2 = x
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .or_internal(vm, "into Ix2")?;
            let y1 = y
                .view()
                .into_dimensionality::<ndarray::Ix1>()
                .or_internal(vm, "into Ix1")?;
            ArraysD::F32(x2.dot(&y1).into_dyn())
        }
        (ArraysD::F64(x), ArraysD::F64(y)) => {
            let x2 = x
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .or_internal(vm, "into Ix2")?;
            let y1 = y
                .view()
                .into_dimensionality::<ndarray::Ix1>()
                .or_internal(vm, "into Ix1")?;
            ArraysD::F64(x2.dot(&y1).into_dyn())
        }
        (ArraysD::C64(x), ArraysD::C64(y)) => {
            let x2 = x
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .or_internal(vm, "into Ix2")?;
            let y1 = y
                .view()
                .into_dimensionality::<ndarray::Ix1>()
                .or_internal(vm, "into Ix1")?;
            ArraysD::C64(x2.dot(&y1).into_dyn())
        }
        (ArraysD::C128(x), ArraysD::C128(y)) => {
            let x2 = x
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .or_internal(vm, "into Ix2")?;
            let y1 = y
                .view()
                .into_dimensionality::<ndarray::Ix1>()
                .or_internal(vm, "into Ix1")?;
            ArraysD::C128(x2.dot(&y1).into_dyn())
        }
        _ => integer_mat_vec(a, b),
    })
}

fn integer_mat_vec(a: &ArraysD, b: &ArraysD) -> ArraysD {
    use crate::dtype::CoerceArray;
    let m = a.shape()[0];
    let n = a.shape()[1];
    let dt = a.dtype();
    let xi = a.coerce::<i64>();
    let yi = b.coerce::<i64>();
    let mut out = ArrayD::<i64>::zeros(IxDyn(&[m]));
    for i in 0..m {
        let mut acc = 0i64;
        for p in 0..n {
            acc = acc.wrapping_add(xi[IxDyn(&[i, p])].wrapping_mul(yi[IxDyn(&[p])]));
        }
        out[IxDyn(&[i])] = acc;
    }
    ArraysD::I64(out).cast(dt)
}

fn vec_mat(a: &ArraysD, b: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let k = a.len();
    let (k2, n) = (b.shape()[0], b.shape()[1]);
    if k != k2 {
        return Err(vm.new_value_error("shapes not aligned".to_string()));
    }
    let _ = n;
    Ok(match (a, b) {
        (ArraysD::F32(x), ArraysD::F32(y)) => {
            let x1 = x
                .view()
                .into_dimensionality::<ndarray::Ix1>()
                .or_internal(vm, "into Ix1")?;
            let y2 = y
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .or_internal(vm, "into Ix2")?;
            ArraysD::F32(x1.dot(&y2).into_dyn())
        }
        (ArraysD::F64(x), ArraysD::F64(y)) => {
            let x1 = x
                .view()
                .into_dimensionality::<ndarray::Ix1>()
                .or_internal(vm, "into Ix1")?;
            let y2 = y
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .or_internal(vm, "into Ix2")?;
            ArraysD::F64(x1.dot(&y2).into_dyn())
        }
        (ArraysD::C64(x), ArraysD::C64(y)) => {
            let x1 = x
                .view()
                .into_dimensionality::<ndarray::Ix1>()
                .or_internal(vm, "into Ix1")?;
            let y2 = y
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .or_internal(vm, "into Ix2")?;
            ArraysD::C64(x1.dot(&y2).into_dyn())
        }
        (ArraysD::C128(x), ArraysD::C128(y)) => {
            let x1 = x
                .view()
                .into_dimensionality::<ndarray::Ix1>()
                .or_internal(vm, "into Ix1")?;
            let y2 = y
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .or_internal(vm, "into Ix2")?;
            ArraysD::C128(x1.dot(&y2).into_dyn())
        }
        _ => integer_vec_mat(a, b),
    })
}

fn integer_vec_mat(a: &ArraysD, b: &ArraysD) -> ArraysD {
    use crate::dtype::CoerceArray;
    let k = a.len();
    let n = b.shape()[1];
    let dt = a.dtype();
    let xi = a.coerce::<i64>();
    let yi = b.coerce::<i64>();
    let mut out = ArrayD::<i64>::zeros(IxDyn(&[n]));
    for j in 0..n {
        let mut acc = 0i64;
        for p in 0..k {
            acc = acc.wrapping_add(xi[IxDyn(&[p])].wrapping_mul(yi[IxDyn(&[p, j])]));
        }
        out[IxDyn(&[j])] = acc;
    }
    ArraysD::I64(out).cast(dt)
}

/// Transpose: reverse axis order (numpy default).
pub fn transpose(a: &ArraysD) -> ArraysD {
    match a {
        ArraysD::Bool(arr) => ArraysD::Bool(arr.t().to_owned()),
        ArraysD::I8(arr) => ArraysD::I8(arr.t().to_owned()),
        ArraysD::I16(arr) => ArraysD::I16(arr.t().to_owned()),
        ArraysD::I32(arr) => ArraysD::I32(arr.t().to_owned()),
        ArraysD::I64(arr) => ArraysD::I64(arr.t().to_owned()),
        ArraysD::U8(arr) => ArraysD::U8(arr.t().to_owned()),
        ArraysD::U16(arr) => ArraysD::U16(arr.t().to_owned()),
        ArraysD::U32(arr) => ArraysD::U32(arr.t().to_owned()),
        ArraysD::U64(arr) => ArraysD::U64(arr.t().to_owned()),
        ArraysD::F16(arr) => ArraysD::F16(arr.t().to_owned()),
        ArraysD::F32(arr) => ArraysD::F32(arr.t().to_owned()),
        ArraysD::F64(arr) => ArraysD::F64(arr.t().to_owned()),
        ArraysD::C64(arr) => ArraysD::C64(arr.t().to_owned()),
        ArraysD::C128(arr) => ArraysD::C128(arr.t().to_owned()),
        // Transpose is a pure data-rearrangement that works for any element
        // type — handle the non-numeric variants by extending the same
        // operation to their inner ArrayD.
        ArraysD::Object(arr) => ArraysD::Object(arr.t().to_owned()),
        ArraysD::Str {
            itemsize_chars,
            data,
        } => ArraysD::Str {
            itemsize_chars: *itemsize_chars,
            data: data.t().to_owned(),
        },
        ArraysD::Bytes { itemsize, data } => ArraysD::Bytes {
            itemsize: *itemsize,
            data: data.t().to_owned(),
        },
        ArraysD::Datetime64 { unit, data } => ArraysD::Datetime64 {
            unit: *unit,
            data: data.t().to_owned(),
        },
        ArraysD::Timedelta64 { unit, data } => ArraysD::Timedelta64 {
            unit: *unit,
            data: data.t().to_owned(),
        },
        ArraysD::Void { layout, data } => ArraysD::Void {
            layout: layout.clone(),
            data: data.t().to_owned(),
        },
    }
}

/// flatten / ravel — copy to a 1-D array.
pub fn flatten(a: &ArraysD) -> ArraysD {
    macro_rules! per {
        ($var:ident, $ty:ty, $arr:ident) => {{
            let data: Vec<$ty> = $arr.iter().copied().collect();
            ArraysD::$var(ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap_or_default())
        }};
    }
    // Non-Copy element types (PyObjectRef, String, Vec<u8>) need `.cloned()`
    // instead. We split the dispatch into the copy-friendly numeric arms and
    // the non-Copy arms.
    macro_rules! per_clone {
        ($var:expr, $arr:ident) => {{
            let data: Vec<_> = $arr.iter().cloned().collect();
            let len = data.len();
            match ArrayD::from_shape_vec(IxDyn(&[len]), data) {
                Ok(d) => $var(d),
                Err(_) => return a.clone(),
            }
        }};
    }
    match a {
        ArraysD::Bool(arr) => per!(Bool, bool, arr),
        ArraysD::I8(arr) => per!(I8, i8, arr),
        ArraysD::I16(arr) => per!(I16, i16, arr),
        ArraysD::I32(arr) => per!(I32, i32, arr),
        ArraysD::I64(arr) => per!(I64, i64, arr),
        ArraysD::U8(arr) => per!(U8, u8, arr),
        ArraysD::U16(arr) => per!(U16, u16, arr),
        ArraysD::U32(arr) => per!(U32, u32, arr),
        ArraysD::U64(arr) => per!(U64, u64, arr),
        ArraysD::F16(arr) => per!(F16, f16, arr),
        ArraysD::F32(arr) => per!(F32, f32, arr),
        ArraysD::F64(arr) => per!(F64, f64, arr),
        ArraysD::C64(arr) => per!(C64, C32, arr),
        ArraysD::C128(arr) => per!(C128, C64, arr),
        ArraysD::Object(arr) => per_clone!(ArraysD::Object, arr),
        ArraysD::Str {
            itemsize_chars,
            data,
        } => {
            let n = *itemsize_chars;
            per_clone!(
                |d| ArraysD::Str {
                    itemsize_chars: n,
                    data: d
                },
                data
            )
        }
        ArraysD::Bytes { itemsize, data } => {
            let n = *itemsize;
            per_clone!(
                |d| ArraysD::Bytes {
                    itemsize: n,
                    data: d
                },
                data
            )
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
            per_clone!(
                |d| ArraysD::Void {
                    layout: l.clone(),
                    data: d
                },
                data
            )
        }
    }
}

/// reshape into a new shape (must match total size). If the array is not
/// contiguous (e.g., after a transpose), we materialize a fresh C-order copy
/// first — matching numpy's `reshape` semantics, which behaves *as if*
/// it ravels into row-major and then re-shapes.
pub fn reshape(a: &ArraysD, shape: &[usize]) -> Option<ArraysD> {
    let s = IxDyn(shape);
    // Verify size match — for input shapes that don't match the target total,
    // bail out early rather than walking elements wastefully.
    let target_size: usize = shape.iter().product();
    if a.len() != target_size {
        return None;
    }
    macro_rules! per {
        ($var:ident, $arr:ident, $ty:ty) => {{
            match $arr.clone().into_shape_with_order(s.clone()) {
                Ok(new) => Some(ArraysD::$var(new)),
                Err(_) => {
                    // Non-contiguous (e.g., transposed) — copy into a fresh
                    // row-major array, then reshape.
                    let data: Vec<$ty> = $arr.iter().copied().collect();
                    ndarray::ArrayD::from_shape_vec(s.clone(), data)
                        .ok()
                        .map(ArraysD::$var)
                }
            }
        }};
    }
    macro_rules! per_clone {
        ($wrap:expr, $arr:ident) => {{
            match $arr.clone().into_shape_with_order(s.clone()) {
                Ok(new) => Some($wrap(new)),
                Err(_) => {
                    let data: Vec<_> = $arr.iter().cloned().collect();
                    ndarray::ArrayD::from_shape_vec(s.clone(), data)
                        .ok()
                        .map($wrap)
                }
            }
        }};
    }
    match a {
        ArraysD::Bool(arr) => per!(Bool, arr, bool),
        ArraysD::I8(arr) => per!(I8, arr, i8),
        ArraysD::I16(arr) => per!(I16, arr, i16),
        ArraysD::I32(arr) => per!(I32, arr, i32),
        ArraysD::I64(arr) => per!(I64, arr, i64),
        ArraysD::U8(arr) => per!(U8, arr, u8),
        ArraysD::U16(arr) => per!(U16, arr, u16),
        ArraysD::U32(arr) => per!(U32, arr, u32),
        ArraysD::U64(arr) => per!(U64, arr, u64),
        ArraysD::F16(arr) => per!(F16, arr, half::f16),
        ArraysD::F32(arr) => per!(F32, arr, f32),
        ArraysD::F64(arr) => per!(F64, arr, f64),
        ArraysD::C64(arr) => per!(C64, arr, C32),
        ArraysD::C128(arr) => per!(C128, arr, C64),
        ArraysD::Object(arr) => per_clone!(ArraysD::Object, arr),
        ArraysD::Str {
            itemsize_chars,
            data,
        } => {
            let n = *itemsize_chars;
            per_clone!(
                |d| ArraysD::Str {
                    itemsize_chars: n,
                    data: d
                },
                data
            )
        }
        ArraysD::Bytes { itemsize, data } => {
            let n = *itemsize;
            per_clone!(
                |d| ArraysD::Bytes {
                    itemsize: n,
                    data: d
                },
                data
            )
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
            per_clone!(
                |d| ArraysD::Void {
                    layout: l.clone(),
                    data: d
                },
                data
            )
        }
    }
}

/// `concatenate` along axis 0 of arrays already cast to the same dtype.
pub fn concatenate(arrays: &[ArraysD], axis: usize, vm: &VirtualMachine) -> PyResult<ArraysD> {
    if arrays.is_empty() {
        return Err(vm.new_value_error("need at least one array to concatenate".to_string()));
    }
    let promoted_dtype = arrays
        .iter()
        .map(|a| a.dtype())
        .fold(arrays[0].dtype(), promote);
    let cast: Vec<ArraysD> = arrays.iter().map(|a| a.cast(promoted_dtype)).collect();
    // ndarray::concatenate takes views; use a typed dispatch.
    macro_rules! cat {
        ($var:ident, $($pat:tt)*) => {{
            // Filter on the matching variant: post-condition of `cast`
            // guarantees every element is the same variant, but if a future
            // bug breaks that we get a clean Python error rather than panic.
            let views: Vec<_> = cast
                .iter()
                .filter_map(|a| match a {
                    ArraysD::$var(x) => Some(x.view()),
                    _ => None,
                })
                .collect();
            if views.len() != cast.len() {
                return Err(internal(vm, "concatenate: dtype mismatch after promotion"));
            }
            let res = ndarray::concatenate(Axis(axis), &views)
                .map_err(|e| vm.new_value_error(e.to_string()))?;
            Ok(ArraysD::$var(res))
        }};
    }
    match promoted_dtype {
        DType::Bool => cat!(Bool,),
        DType::I8 => cat!(I8,),
        DType::I16 => cat!(I16,),
        DType::I32 => cat!(I32,),
        DType::I64 => cat!(I64,),
        DType::U8 => cat!(U8,),
        DType::U16 => cat!(U16,),
        DType::U32 => cat!(U32,),
        DType::U64 => cat!(U64,),
        DType::F16 => cat!(F16,),
        DType::F32 => cat!(F32,),
        DType::F64 => cat!(F64,),
        DType::C64 => cat!(C64,),
        DType::C128 => cat!(C128,),
        // Non-numeric dtypes: gather views from the matching struct-variant
        // arms. Each path uses cloned views (the underlying ArrayD<T> isn't
        // Copy here, but ndarray::concatenate works fine with cloned data).
        DType::Object
        | DType::Str(_)
        | DType::Bytes(_)
        | DType::Datetime64(_)
        | DType::Timedelta64(_)
        | DType::Void(_) => concat_nonnumeric(&cast, axis, promoted_dtype, vm),
    }
}

fn concat_nonnumeric(
    cast: &[ArraysD],
    axis: usize,
    target: DType,
    vm: &VirtualMachine,
) -> PyResult<ArraysD> {
    // Convert each array to typed views of the same element type.
    match target {
        DType::Object => {
            let views: Vec<_> = cast
                .iter()
                .filter_map(|a| match a {
                    ArraysD::Object(x) => Some(x.view()),
                    _ => None,
                })
                .collect();
            let res = ndarray::concatenate(Axis(axis), &views)
                .map_err(|e| vm.new_value_error(e.to_string()))?;
            Ok(ArraysD::Object(res))
        }
        DType::Str(n) => {
            let views: Vec<_> = cast
                .iter()
                .filter_map(|a| match a {
                    ArraysD::Str { data, .. } => Some(data.view()),
                    _ => None,
                })
                .collect();
            let res = ndarray::concatenate(Axis(axis), &views)
                .map_err(|e| vm.new_value_error(e.to_string()))?;
            Ok(ArraysD::Str {
                itemsize_chars: n,
                data: res,
            })
        }
        DType::Bytes(n) => {
            let views: Vec<_> = cast
                .iter()
                .filter_map(|a| match a {
                    ArraysD::Bytes { data, .. } => Some(data.view()),
                    _ => None,
                })
                .collect();
            let res = ndarray::concatenate(Axis(axis), &views)
                .map_err(|e| vm.new_value_error(e.to_string()))?;
            Ok(ArraysD::Bytes {
                itemsize: n,
                data: res,
            })
        }
        DType::Datetime64(u) => {
            let views: Vec<_> = cast
                .iter()
                .filter_map(|a| match a {
                    ArraysD::Datetime64 { data, .. } => Some(data.view()),
                    _ => None,
                })
                .collect();
            let res = ndarray::concatenate(Axis(axis), &views)
                .map_err(|e| vm.new_value_error(e.to_string()))?;
            Ok(ArraysD::Datetime64 { unit: u, data: res })
        }
        DType::Timedelta64(u) => {
            let views: Vec<_> = cast
                .iter()
                .filter_map(|a| match a {
                    ArraysD::Timedelta64 { data, .. } => Some(data.view()),
                    _ => None,
                })
                .collect();
            let res = ndarray::concatenate(Axis(axis), &views)
                .map_err(|e| vm.new_value_error(e.to_string()))?;
            Ok(ArraysD::Timedelta64 { unit: u, data: res })
        }
        DType::Void(_) => {
            let layout = cast
                .iter()
                .find_map(|a| match a {
                    ArraysD::Void { layout, .. } => Some(layout.clone()),
                    _ => None,
                })
                .ok_or_else(|| crate::internal::internal(vm, "concat: void layout missing"))?;
            let views: Vec<_> = cast
                .iter()
                .filter_map(|a| match a {
                    ArraysD::Void { data, .. } => Some(data.view()),
                    _ => None,
                })
                .collect();
            let res = ndarray::concatenate(Axis(axis), &views)
                .map_err(|e| vm.new_value_error(e.to_string()))?;
            Ok(ArraysD::Void { layout, data: res })
        }
        _ => Err(crate::internal::internal(
            vm,
            "concat_nonnumeric: numeric dtype routed here",
        )),
    }
}
