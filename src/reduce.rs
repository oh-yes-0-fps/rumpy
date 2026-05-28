//! Reductions: sum / prod / mean / min / max / std / var.
//!
//! Numpy promotion rules for accumulator dtype:
//!
//!   sum, prod   — integer inputs widen to platform int (we use i64/u64);
//!                 bool widens to int64; float / complex keep their width.
//!   mean        — always returns float (≥ f64 for int/bool input).
//!   var, std    — same as mean.
//!   min, max    — keep input dtype.

use crate::dtype::{ArraysD, C32, C64, DType};
use crate::internal::{OptionExt, internal};
use half::f16;
use ndarray::{ArrayD, Axis, IxDyn};
use num_complex::Complex;
use rustpython_vm::{PyResult, VirtualMachine};

#[derive(Copy, Clone)]
pub enum Reduce {
    Sum,
    Prod,
    Mean,
    Min,
    Max,
    Var(usize),
    Std(usize),
}

/// Output type for a reduction.
///
/// numpy.sum/prod widen int/bool to i64/u64; numpy.mean/var/std always
/// produce float64 (for non-complex input). We follow those rules.
fn acc_dtype(input: DType, op: Reduce) -> DType {
    match op {
        Reduce::Sum | Reduce::Prod => match input {
            DType::Bool | DType::I8 | DType::I16 | DType::I32 | DType::I64 => DType::I64,
            DType::U8 | DType::U16 | DType::U32 | DType::U64 => DType::U64,
            other => other,
        },
        Reduce::Mean | Reduce::Var(_) | Reduce::Std(_) => match input {
            DType::C64 | DType::C128 => input,
            DType::F32 => DType::F64, // numpy: mean of float32 still returns float32 actually
            _ => DType::F64,
        },
        Reduce::Min | Reduce::Max => input,
    }
}

/// Reduce along one or more axes, optionally keeping the reduced dimensions
/// as size-1 (`keepdims=True`). `axes=None` reduces the entire array.
///
/// Numpy semantics:
///   * tuple-of-axes is equivalent to reducing along each axis in turn
///     (the result is independent of order for sum/prod/min/max/any/all
///     and *almost* identical for mean — for mean we reweight by the
///     combined size of the reduced axes).
pub fn reduce_multi(
    a: &ArraysD,
    axes: Option<&[isize]>,
    keepdims: bool,
    op: Reduce,
    vm: &VirtualMachine,
) -> PyResult<ArraysD> {
    let nd = a.ndim();
    // Normalize & sort axes (descending so removals don't shift later indices).
    let mut norm_axes: Vec<usize> = match axes {
        None => (0..nd).collect(),
        Some(list) => {
            let mut v: Vec<usize> = Vec::with_capacity(list.len());
            for &ax in list {
                let na = if ax < 0 { ax + nd as isize } else { ax };
                if na < 0 || na >= nd as isize {
                    return Err(vm.new_value_error(format!(
                        "axis {ax} is out of bounds for array of dimension {nd}"
                    )));
                }
                if v.contains(&(na as usize)) {
                    return Err(vm.new_value_error(format!(
                        "duplicate axis {ax} in reduction"
                    )));
                }
                v.push(na as usize);
            }
            v
        }
    };
    norm_axes.sort_by(|x, y| y.cmp(x));
    // Mean/var/std require special handling for multi-axis: we reduce along
    // a single transient axis after permuting/flattening the target axes
    // together. For now, take the simpler approach: for these ops over more
    // than one axis, flatten the target axes into one before reducing.
    let multi_mean_like = norm_axes.len() > 1
        && matches!(op, Reduce::Mean | Reduce::Var(_) | Reduce::Std(_));
    if multi_mean_like {
        // Move all target axes to the front, then merge them into a single
        // axis, then reduce along that single axis.
        let mut perm: Vec<usize> = norm_axes.iter().rev().copied().collect();
        for ax in 0..nd {
            if !perm.contains(&ax) {
                perm.push(ax);
            }
        }
        let transposed = transpose_axes(a, &perm);
        let merged_len: usize = norm_axes.iter().map(|&i| a.shape()[i]).product();
        let mut new_shape: Vec<usize> = vec![merged_len];
        for &ax in &perm[norm_axes.len()..] {
            new_shape.push(a.shape()[ax]);
        }
        let merged = crate::linalg::reshape(&transposed, &new_shape)
            .ok_or_else(|| crate::internal::internal(vm, "reduce_multi: reshape failed"))?;
        let reduced = reduce(&merged, Some(0), op, vm)?;
        return apply_keepdims(a, &norm_axes, reduced, keepdims, vm);
    }
    // Iteratively reduce one axis at a time (descending order keeps indices stable).
    let mut current = a.clone();
    let mut had_axis = false;
    match axes {
        None => {
            // Single full reduction.
            let reduced = reduce(&current, None, op, vm)?;
            return apply_keepdims(a, &norm_axes, reduced, keepdims, vm);
        }
        Some(_) => {}
    }
    for &ax in &norm_axes {
        current = reduce(&current, Some(ax as isize), op, vm)?;
        had_axis = true;
    }
    let _ = had_axis;
    apply_keepdims(a, &norm_axes, current, keepdims, vm)
}

/// Transpose `a` according to a full permutation `perm` and return a freshly
/// contiguous (row-major) array — the permuted result is materialized so a
/// follow-up `reshape` succeeds.
fn transpose_axes(a: &ArraysD, perm: &[usize]) -> ArraysD {
    macro_rules! per {
        ($var:ident, $arr:ident, $ty:ty) => {{
            let permuted = $arr.view().permuted_axes(ndarray::IxDyn(perm));
            let shape: Vec<usize> = permuted.shape().to_vec();
            let data: Vec<$ty> = permuted.iter().copied().collect();
            let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), data)
                .unwrap_or_else(|_| ndarray::ArrayD::default(ndarray::IxDyn(&[0])));
            ArraysD::$var(arr)
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
        ArraysD::C64(arr) => per!(C64, arr, crate::dtype::C32),
        ArraysD::C128(arr) => per!(C128, arr, crate::dtype::C64),
        _ => { a.clone() },
    }
}

/// Re-insert size-1 axes at the originally-reduced positions, if requested.
fn apply_keepdims(
    original: &ArraysD,
    reduced_axes_desc: &[usize],
    reduced: ArraysD,
    keepdims: bool,
    vm: &VirtualMachine,
) -> PyResult<ArraysD> {
    if !keepdims {
        return Ok(reduced);
    }
    let nd = original.ndim();
    let mut full_shape: Vec<usize> = original.shape().to_vec();
    for ax in 0..nd {
        if reduced_axes_desc.contains(&ax) {
            full_shape[ax] = 1;
        }
    }
    crate::linalg::reshape(&reduced, &full_shape).ok_or_else(|| {
        crate::internal::internal(
            vm,
            format!(
                "keepdims reshape failed: cannot reshape size {} into {:?}",
                reduced.len(),
                full_shape
            ),
        )
    })
}

/// Reduce a whole array (axis=None) or along a single axis.
pub fn reduce(
    a: &ArraysD,
    axis: Option<isize>,
    op: Reduce,
    vm: &VirtualMachine,
) -> PyResult<ArraysD> {
    // For min/max on empty: raise.
    if a.is_empty() && matches!(op, Reduce::Min | Reduce::Max) {
        return Err(vm.new_value_error(
            "zero-size array to reduction operation min/max which has no identity".to_string(),
        ));
    }
    let target = acc_dtype(a.dtype(), op);
    let widened = if matches!(op, Reduce::Min | Reduce::Max) {
        a.clone()
    } else {
        a.cast(target)
    };
    let nd = widened.ndim() as isize;
    let axis = axis.map(|ax| if ax < 0 { ax + nd } else { ax });
    if let Some(ax) = axis
        && (ax < 0 || ax >= nd) {
            return Err(vm.new_value_error(format!("axis {ax} out of range")));
        }
    Ok(match &widened {
        ArraysD::I64(a) => ArraysD::I64(reduce_int_axis(a, axis.map(|v| v as usize), op)),
        ArraysD::U64(a) => ArraysD::U64(reduce_int_axis(a, axis.map(|v| v as usize), op)),
        // Float min/max paths can keep their dtype.
        ArraysD::F16(a) => ArraysD::F16(reduce_float_axis(a, axis.map(|v| v as usize), op, f16::ZERO, f16::from_f32(1.0))),
        ArraysD::F32(a) => ArraysD::F32(reduce_float_axis(a, axis.map(|v| v as usize), op, 0.0, 1.0)),
        ArraysD::F64(a) => ArraysD::F64(reduce_float_axis(a, axis.map(|v| v as usize), op, 0.0, 1.0)),
        ArraysD::C64(a) => ArraysD::C64(reduce_complex_axis(a, axis.map(|v| v as usize), op, C32::new(0.0, 0.0), C32::new(1.0, 0.0), vm)?),
        ArraysD::C128(a) => ArraysD::C128(reduce_complex_axis(a, axis.map(|v| v as usize), op, C64::new(0.0, 0.0), C64::new(1.0, 0.0), vm)?),
        // For min/max of smaller integer types, dtype is kept.
        ArraysD::Bool(a) => ArraysD::Bool(reduce_bool_axis(a, axis.map(|v| v as usize), op)),
        ArraysD::I8(a) => ArraysD::I8(reduce_int_axis(a, axis.map(|v| v as usize), op)),
        ArraysD::I16(a) => ArraysD::I16(reduce_int_axis(a, axis.map(|v| v as usize), op)),
        ArraysD::I32(a) => ArraysD::I32(reduce_int_axis(a, axis.map(|v| v as usize), op)),
        ArraysD::U8(a) => ArraysD::U8(reduce_int_axis(a, axis.map(|v| v as usize), op)),
        ArraysD::U16(a) => ArraysD::U16(reduce_int_axis(a, axis.map(|v| v as usize), op)),
        ArraysD::U32(a) => ArraysD::U32(reduce_int_axis(a, axis.map(|v| v as usize), op)),
        _ => { return Err(crate::internal::unsupported_dtype(vm, "reduce", a.dtype())) },
    })
}

fn reduce_int_axis<T>(a: &ArrayD<T>, axis: Option<usize>, op: Reduce) -> ArrayD<T>
where
    T: Copy
        + Default
        + Ord
        + num_traits::Zero
        + num_traits::One
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>,
{
    // `reduce()` validates that the array is non-empty for Min/Max before
    // dispatching here, so the `min/max` calls below can never observe an
    // empty iterator. Use `.unwrap_or_else(T::default)` rather than
    // `.unwrap()` so a future refactor can't introduce a panic.
    match axis {
        None => {
            let v = match op {
                Reduce::Sum => a.iter().copied().fold(T::zero(), |acc, x| acc + x),
                Reduce::Prod => a.iter().copied().fold(T::one(), |acc, x| acc * x),
                Reduce::Min => a.iter().copied().min().unwrap_or_else(T::default),
                Reduce::Max => a.iter().copied().max().unwrap_or_else(T::default),
                // mean/var/std never route here — they go through
                // `reduce_float_axis`. Falling through to zero is harmless.
                Reduce::Mean | Reduce::Var(_) | Reduce::Std(_) => T::zero(),
            };
            ArrayD::from_elem(IxDyn(&[]), v)
        }
        Some(ax) => match op {
            Reduce::Sum => a.fold_axis(Axis(ax), T::zero(), |&acc, &x| acc + x),
            Reduce::Prod => a.fold_axis(Axis(ax), T::one(), |&acc, &x| acc * x),
            Reduce::Min => a.map_axis(Axis(ax), |row| {
                row.iter().copied().min().unwrap_or_else(T::default)
            }),
            Reduce::Max => a.map_axis(Axis(ax), |row| {
                row.iter().copied().max().unwrap_or_else(T::default)
            }),
            Reduce::Mean | Reduce::Var(_) | Reduce::Std(_) => {
                ArrayD::from_elem(IxDyn(&[]), T::zero())
            }
        },
    }
}

fn reduce_bool_axis(a: &ArrayD<bool>, axis: Option<usize>, op: Reduce) -> ArrayD<bool> {
    match axis {
        None => {
            let v = match op {
                Reduce::Min => a.iter().all(|x| *x),
                Reduce::Max => a.iter().any(|x| *x),
                // Other reductions don't keep bool dtype — `reduce()`
                // promotes the dtype before reaching us, so this arm is
                // unreachable in normal use.
                _ => false,
            };
            ArrayD::from_elem(IxDyn(&[]), v)
        }
        Some(ax) => match op {
            Reduce::Min => a.fold_axis(Axis(ax), true, |&acc, &x| acc & x),
            Reduce::Max => a.fold_axis(Axis(ax), false, |&acc, &x| acc | x),
            _ => ArrayD::from_elem(IxDyn(&[]), false),
        },
    }
}

fn reduce_float_axis<T>(
    a: &ArrayD<T>,
    axis: Option<usize>,
    op: Reduce,
    zero: T,
    one: T,
) -> ArrayD<T>
where
    T: Copy
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + num_traits::FromPrimitive
        + PartialOrd
        + FloatMath,
{
    let combine = |v: &[T]| -> T {
        // `T::from_usize(n)` returns None only when `n` cannot be exactly
        // represented as `T`. For the float types we use (f16/f32/f64) this
        // can only happen for absurdly large `n`; fall back to NaN rather
        // than panic in that pathological case.
        let from_usize = |n: usize| T::from_usize(n).unwrap_or_else(T::nan_);
        match op {
            Reduce::Sum => v.iter().copied().fold(zero, |a, x| a + x),
            Reduce::Prod => v.iter().copied().fold(one, |a, x| a * x),
            Reduce::Mean => {
                if v.is_empty() {
                    return T::nan_();
                }
                let s = v.iter().copied().fold(zero, |a, x| a + x);
                s / from_usize(v.len())
            }
            Reduce::Min => v
                .iter()
                .copied()
                .reduce(|a, b| {
                    // numpy: any NaN in input -> NaN result (NaN propagates).
                    if a.is_nan_() || b.is_nan_() {
                        T::nan_()
                    } else if a < b {
                        a
                    } else {
                        b
                    }
                })
                .unwrap_or_else(T::nan_),
            Reduce::Max => v
                .iter()
                .copied()
                .reduce(|a, b| {
                    if a.is_nan_() || b.is_nan_() {
                        T::nan_()
                    } else if a > b {
                        a
                    } else {
                        b
                    }
                })
                .unwrap_or_else(T::nan_),
            Reduce::Var(ddof) => {
                let n = v.len();
                if n <= ddof {
                    return T::nan_();
                }
                let s = v.iter().copied().fold(zero, |a, x| a + x);
                let mean = s / from_usize(n);
                let ss = v
                    .iter()
                    .copied()
                    .map(|x| {
                        let d = x - mean;
                        d * d
                    })
                    .fold(zero, |a, x| a + x);
                ss / from_usize(n - ddof)
            }
            Reduce::Std(ddof) => {
                let n = v.len();
                if n <= ddof {
                    return T::nan_();
                }
                let s = v.iter().copied().fold(zero, |a, x| a + x);
                let mean = s / from_usize(n);
                let ss = v
                    .iter()
                    .copied()
                    .map(|x| {
                        let d = x - mean;
                        d * d
                    })
                    .fold(zero, |a, x| a + x);
                (ss / from_usize(n - ddof)).sqrt_()
            }
        }
    };

    match axis {
        None => {
            let v: Vec<T> = a.iter().copied().collect();
            ArrayD::from_elem(IxDyn(&[]), combine(&v))
        }
        Some(ax) => {
            // map_axis walks each lane in place; no bucket allocation.
            a.map_axis(Axis(ax), |row| {
                let v: Vec<T> = row.iter().copied().collect();
                combine(&v)
            })
        }
    }
}

fn reduce_complex_axis<T>(
    a: &ArrayD<T>,
    axis: Option<usize>,
    op: Reduce,
    zero: T,
    one: T,
    vm: &VirtualMachine,
) -> PyResult<ArrayD<T>>
where
    T: Copy
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + ComplexFromUsize,
{
    let combine = |v: &[T]| -> PyResult<T> {
        match op {
            Reduce::Sum => Ok(v.iter().copied().fold(zero, |a, x| a + x)),
            Reduce::Prod => Ok(v.iter().copied().fold(one, |a, x| a * x)),
            Reduce::Mean => {
                let s = v.iter().copied().fold(zero, |a, x| a + x);
                Ok(s / T::from_usize_(v.len()))
            }
            Reduce::Var(_) | Reduce::Std(_) => Err(vm
                .new_type_error("var/std not implemented for complex".to_string())),
            Reduce::Min | Reduce::Max => Err(vm
                .new_type_error("min/max not defined for complex".to_string())),
        }
    };
    match axis {
        None => {
            let v: Vec<T> = a.iter().copied().collect();
            Ok(ArrayD::from_elem(IxDyn(&[]), combine(&v)?))
        }
        Some(ax) => {
            // Walk each lane once; collect into a flat Vec then reshape.
            let mut out_shape: Vec<usize> = a.shape().to_vec();
            out_shape.remove(ax);
            let data: Vec<T> = a
                .lanes(Axis(ax))
                .into_iter()
                .map(|lane| {
                    let v: Vec<T> = lane.iter().copied().collect();
                    combine(&v)
                })
                .collect::<PyResult<_>>()?;
            let shape = if out_shape.is_empty() {
                IxDyn(&[])
            } else {
                IxDyn(&out_shape)
            };
            ArrayD::from_shape_vec(shape, data)
                .map_err(|e| internal(vm, format!("reduce shape: {e}")))
        }
    }
}

// ---- helper traits so the generic float reduction works on f16/f32/f64 ----
trait FloatMath {
    fn sqrt_(self) -> Self;
    fn nan_() -> Self;
    fn is_nan_(self) -> bool;
}
impl FloatMath for f32 {
    fn sqrt_(self) -> Self {
        self.sqrt()
    }
    fn nan_() -> Self {
        f32::NAN
    }
    fn is_nan_(self) -> bool {
        self.is_nan()
    }
}
impl FloatMath for f64 {
    fn sqrt_(self) -> Self {
        self.sqrt()
    }
    fn nan_() -> Self {
        f64::NAN
    }
    fn is_nan_(self) -> bool {
        self.is_nan()
    }
}
impl FloatMath for f16 {
    fn sqrt_(self) -> Self {
        f16::from_f32(f32::from(self).sqrt())
    }
    fn nan_() -> Self {
        f16::NAN
    }
    fn is_nan_(self) -> bool {
        self.is_nan()
    }
}

trait ComplexFromUsize {
    fn from_usize_(n: usize) -> Self;
}
impl ComplexFromUsize for C32 {
    fn from_usize_(n: usize) -> Self {
        Complex::new(n as f32, 0.0)
    }
}
impl ComplexFromUsize for C64 {
    fn from_usize_(n: usize) -> Self {
        Complex::new(n as f64, 0.0)
    }
}

/// argmin / argmax (always returns the flat index).
pub fn arg_extremum(a: &ArraysD, want_max: bool, vm: &VirtualMachine) -> PyResult<usize> {
    if a.is_empty() {
        return Err(vm.new_value_error(
            "attempt to get arg{min,max} of empty array".to_string(),
        ));
    }
    if matches!(a, ArraysD::C64(_) | ArraysD::C128(_)) {
        return Err(vm.new_type_error(
            "argmin/argmax not defined for complex".to_string(),
        ));
    }
    fn scan<T: Copy + PartialOrd>(arr: &ArrayD<T>, want_max: bool) -> Option<usize> {
        // Caller has guaranteed non-empty; the Option<...> here just removes
        // the `unwrap()` from the iterator advance.
        let mut iter = arr.iter().enumerate();
        let (_, first) = iter.next()?;
        let mut best = 0usize;
        let mut bv = *first;
        for (i, &v) in iter {
            let pick = if want_max { v > bv } else { v < bv };
            if pick {
                bv = v;
                best = i;
            }
        }
        Some(best)
    }
    let result: Option<usize> = match a {
        ArraysD::Bool(arr) => {
            let mut iter = arr.iter().enumerate();
            iter.next().map(|(_, first)| {
                let mut best = 0usize;
                let mut bv = *first;
                for (i, &v) in iter {
                    let pick = if want_max { v && !bv } else { bv && !v };
                    if pick {
                        bv = v;
                        best = i;
                    }
                }
                best
            })
        }
        ArraysD::I8(arr) => scan(arr, want_max),
        ArraysD::I16(arr) => scan(arr, want_max),
        ArraysD::I32(arr) => scan(arr, want_max),
        ArraysD::I64(arr) => scan(arr, want_max),
        ArraysD::U8(arr) => scan(arr, want_max),
        ArraysD::U16(arr) => scan(arr, want_max),
        ArraysD::U32(arr) => scan(arr, want_max),
        ArraysD::U64(arr) => scan(arr, want_max),
        ArraysD::F16(arr) => scan(arr, want_max),
        ArraysD::F32(arr) => scan(arr, want_max),
        ArraysD::F64(arr) => scan(arr, want_max),
        ArraysD::C64(_) | ArraysD::C128(_) => {
            return Err(internal(vm, "arg_extremum reached complex arm"));
        }
        _ => { None },
    };
    result.or_internal(vm, "arg_extremum: empty after non-empty check")
}
