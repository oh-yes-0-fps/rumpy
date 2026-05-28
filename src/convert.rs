//! Python ↔ ArraysD conversion.
//!
//! These functions are called from the `numpy_module` mod inside `lib.rs`.

use crate::dtype::{ArraysD, C32, C64, DType};
use crate::internal::{OptionExt, ResultExt, internal};
use half::f16;
use ndarray::{ArrayD, Axis, IxDyn};
use num_complex::Complex;
use rustpython_vm::{
    AsObject, PyObjectRef, PyPayload, PyResult, VirtualMachine,
    builtins::{PyComplex, PyFloat, PyInt, PyList, PyStr, PyTuple},
};

/// Try to extract a typed scalar value from a Python object.
#[derive(Debug, Clone, Copy)]
pub enum Scalar {
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Complex(C64),
}

impl Scalar {
    #[cfg_attr(feature = "no-panic", no_panic::no_panic)]
    #[inline]
    pub fn inferred_dtype(self) -> DType {
        match self {
            Scalar::Bool(_) => DType::Bool,
            Scalar::Int(_) => DType::I64,
            Scalar::UInt(_) => DType::U64,
            Scalar::Float(_) => DType::F64,
            Scalar::Complex(_) => DType::C128,
        }
    }

    #[cfg_attr(feature = "no-panic", no_panic::no_panic)]
    #[inline]
    pub fn as_f64(self) -> f64 {
        match self {
            Scalar::Bool(b) => if b { 1.0 } else { 0.0 },
            Scalar::Int(i) => i as f64,
            Scalar::UInt(u) => u as f64,
            Scalar::Float(f) => f,
            Scalar::Complex(c) => c.re,
        }
    }

    #[cfg_attr(feature = "no-panic", no_panic::no_panic)]
    #[inline]
    pub fn as_c128(self) -> C64 {
        match self {
            Scalar::Complex(c) => c,
            _ => Complex::new(self.as_f64(), 0.0),
        }
    }

    /// Stringify the scalar for non-numeric dtype storage (e.g. building a
    /// `Str`/`Bytes` array from a list of Python ints/floats).
    pub fn to_display_string(&self) -> String {
        match self {
            Scalar::Bool(b) => if *b { "True".to_string() } else { "False".to_string() },
            Scalar::Int(i) => i.to_string(),
            Scalar::UInt(u) => u.to_string(),
            Scalar::Float(f) => f.to_string(),
            Scalar::Complex(c) => format!("({}+{}j)", c.re, c.im),
        }
    }
}

/// Try to interpret a Python object as a scalar of one of our supported types.
pub fn obj_as_scalar(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<Scalar> {
    // A 0-D ndarray is itself a scalar — extract its single value.
    if let Some(arr) = obj.downcast_ref::<crate::PyNdArray>()
        && arr.view().ndim() == 0
    {
        return Ok(zero_d_to_scalar(&arr.view()));
    }
    // bool is a subclass of int; check for the exact True / False singletons.
    if obj.is(&vm.ctx.true_value) {
        return Ok(Scalar::Bool(true));
    }
    if obj.is(&vm.ctx.false_value) {
        return Ok(Scalar::Bool(false));
    }
    if let Some(c) = obj.downcast_ref::<PyComplex>() {
        let v = c.to_complex64();
        return Ok(Scalar::Complex(Complex::new(v.re, v.im)));
    }
    if let Some(f) = obj.downcast_ref::<PyFloat>() {
        return Ok(Scalar::Float(f.to_f64()));
    }
    if let Some(i) = obj.downcast_ref::<PyInt>() {
        if let Ok(v) = i.try_to_primitive::<i64>(vm) {
            return Ok(Scalar::Int(v));
        }
        if let Ok(v) = i.try_to_primitive::<u64>(vm) {
            return Ok(Scalar::UInt(v));
        }
        // Fall back to float (numpy widens overflowing python ints to f64).
        let f = obj.try_float(vm)?.to_f64();
        return Ok(Scalar::Float(f));
    }
    Err(vm.new_type_error(format!(
        "cannot convert {} to a numeric scalar",
        obj.class().name()
    )))
}

/// Convert a Python object into an `ArraysD`.
///
///  * an existing `ndarray` is returned (cloned)
///  * scalars become 0-D arrays of their inferred dtype
///  * nested lists/tuples are flattened; the dtype is inferred as the
///    promoted result of all leaf scalars
///  * if `target_dtype` is `Some`, the value is cast.
pub fn obj_to_array(
    obj: &PyObjectRef,
    target_dtype: Option<DType>,
    vm: &VirtualMachine,
) -> PyResult<ArraysD> {
    if let Some(arr) = obj.downcast_ref::<crate::PyNdArray>() {
        let a = arr.view().clone();
        return Ok(match target_dtype {
            Some(dt) => a.cast(dt),
            None => a,
        });
    }
    // Fast paths for non-numeric target dtypes. These handle data that
    // `obj_as_scalar` would reject (Python strings, bytes, arbitrary
    // objects). The path that runs depends on what numpy *actually* does for
    // the target dtype:
    //
    //   dtype=object → wrap every leaf as a PyObjectRef, preserving identity
    //   dtype=str / U… → str() each leaf and store as a Python str
    //   dtype=bytes / S… → bytes() each leaf
    //   dtype=datetime64 / timedelta64 → parse ISO-8601 strings or accept
    //     integers
    if let Some(t) = target_dtype {
        match t {
            DType::Object => return obj_to_object_array(obj, vm),
            DType::Str(n) => return obj_to_str_array(obj, n, vm),
            DType::Bytes(n) => return obj_to_bytes_array(obj, n, vm),
            DType::Datetime64(u) | DType::Timedelta64(u) => {
                return obj_to_time_array(obj, t, u, vm);
            }
            _ => {}
        }
    }
    // Scalar?
    if let Ok(s) = obj_as_scalar(obj, vm) {
        let dt = target_dtype.unwrap_or_else(|| s.inferred_dtype());
        return Ok(scalar_to_zero_d(s, dt));
    }
    // No target dtype, but it might still be a Python string we should keep
    // as a U… array (numpy's `np.array(['hi','yo'])` infers U).
    if target_dtype.is_none() {
        if let Some(s) = obj.downcast_ref::<PyStr>() {
            let v = s.as_wtf8().to_string_lossy().into_owned();
            let n = v.chars().count() as u32;
            return Ok(ArraysD::Str {
                itemsize_chars: n,
                data: ArrayD::from_elem(IxDyn(&[]), v),
            });
        }
        if let Some(b) = obj.downcast_ref::<rustpython_vm::builtins::PyBytes>() {
            let bytes = b.as_bytes().to_vec();
            let n = bytes.len() as u32;
            return Ok(ArraysD::Bytes {
                itemsize: n,
                data: ArrayD::from_elem(IxDyn(&[]), bytes),
            });
        }
    }
    // Nested sequence
    if obj.downcast_ref::<PyList>().is_some() || obj.downcast_ref::<PyTuple>().is_some() {
        // If every element is itself an ndarray (or nested list of arrays),
        // stack instead of flatten-as-scalars.
        if let Some(items) = seq_items(obj) {
            // Try the "list of ndarrays" fast path: every entry is itself
            // an ndarray of equal shape → stack along a new axis 0.
            let arrs: Option<Vec<crate::dtype::ArraysD>> = if !items.is_empty() {
                items
                    .iter()
                    .map(|it| {
                        it.downcast_ref::<crate::PyNdArray>()
                            .map(|p| p.view().clone())
                    })
                    .collect()
            } else {
                None
            };
            if let Some(arrs) = arrs {
                let s0 = arrs[0].shape().to_vec();
                if arrs.iter().all(|a| a.shape() == s0.as_slice()) {
                    // Stack along axis 0 — promote dtypes.
                    let promoted = arrs
                        .iter()
                        .map(|a| a.dtype())
                        .fold(arrs[0].dtype(), crate::promote::promote);
                    let cast: Vec<crate::dtype::ArraysD> =
                        arrs.iter().map(|a| a.cast(promoted)).collect();
                    let mut with_axis: Vec<crate::dtype::ArraysD> =
                        Vec::with_capacity(cast.len());
                    for a in &cast {
                        let mut s = vec![1usize];
                        s.extend(a.shape());
                        // reshape returns None only if the total size disagrees;
                        // here `s` adds a leading 1 to the original shape so the
                        // total element count is unchanged.
                        let r = crate::linalg::reshape(a, &s)
                            .or_internal(vm, "stack-reshape")?;
                        with_axis.push(r);
                    }
                    let stacked = crate::linalg::concatenate(&with_axis, 0, vm)?;
                    return Ok(match target_dtype {
                        Some(dt) => stacked.cast(dt),
                        None => stacked,
                    });
                }
            }
        }
        // If no explicit dtype was given, detect a homogeneous list of
        // strings / bytes (numpy infers `U…` / `S…`) before falling back to
        // numeric scalar collection. The collector below would otherwise
        // raise "cannot convert str to a numeric scalar".
        if target_dtype.is_none() {
            let (shape, objs) = collect_objects(obj, vm)?;
            if !objs.is_empty()
                && objs.iter().all(|o| o.downcast_ref::<PyStr>().is_some())
            {
                let strings: Vec<String> = objs
                    .iter()
                    .map(|o| {
                        o.downcast_ref::<PyStr>()
                            .map(|s| s.as_wtf8().to_string_lossy().into_owned())
                            .unwrap_or_default()
                    })
                    .collect();
                let widest = strings
                    .iter()
                    .map(|s| s.chars().count())
                    .max()
                    .unwrap_or(0) as u32;
                let arr = ArrayD::from_shape_vec(IxDyn(&shape), strings)
                    .or_internal(vm, "list-of-str inference")?;
                return Ok(ArraysD::Str { itemsize_chars: widest, data: arr });
            }
            if !objs.is_empty()
                && objs
                    .iter()
                    .all(|o| o.downcast_ref::<rustpython_vm::builtins::PyBytes>().is_some())
            {
                let raw: Vec<Vec<u8>> = objs
                    .iter()
                    .map(|o| {
                        o.downcast_ref::<rustpython_vm::builtins::PyBytes>()
                            .map(|b| b.as_bytes().to_vec())
                            .unwrap_or_default()
                    })
                    .collect();
                let widest = raw.iter().map(|v| v.len()).max().unwrap_or(0) as u32;
                let padded: Vec<Vec<u8>> = raw
                    .into_iter()
                    .map(|mut v| {
                        v.resize(widest as usize, 0);
                        v
                    })
                    .collect();
                let arr = ArrayD::from_shape_vec(IxDyn(&shape), padded)
                    .or_internal(vm, "list-of-bytes inference")?;
                return Ok(ArraysD::Bytes { itemsize: widest, data: arr });
            }
        }
        let (shape, scalars) = collect_nested(obj, vm)?;
        let dtype = match target_dtype {
            Some(dt) => dt,
            None => infer_dtype_from_scalars(&scalars),
        };
        return scalars_to_array(&shape, scalars, dtype, vm);
    }
    Err(vm.new_type_error(format!(
        "cannot convert {} to numpy.ndarray",
        obj.class().name()
    )))
}

// =====================================================================
// Non-numeric construction helpers
// =====================================================================

/// Walk a nested Python list/tuple, accumulating per-cell `PyObjectRef`s
/// into a flat vector while inferring the surrounding `shape`. Strings and
/// bytes count as leaves (not iterables) for purposes of shape inference,
/// matching numpy's `dtype=object` behavior.
fn collect_objects(
    obj: &PyObjectRef,
    vm: &VirtualMachine,
) -> PyResult<(Vec<usize>, Vec<PyObjectRef>)> {
    fn is_seq(o: &PyObjectRef) -> bool {
        o.downcast_ref::<PyList>().is_some() || o.downcast_ref::<PyTuple>().is_some()
    }

    let mut shape = Vec::new();
    let mut cur = obj.clone();
    while is_seq(&cur) {
        let Some(items) = seq_items(&cur) else {
            // `is_seq` was true on the previous loop iteration but a foreign
            // sequence type (or a future refactor) could in principle make
            // `seq_items` return None. Treat that as a leaf rather than panic.
            break;
        };
        shape.push(items.len());
        if items.is_empty() {
            break;
        }
        cur = items[0].clone();
    }

    fn walk(
        obj: &PyObjectRef,
        depth: usize,
        shape: &[usize],
        out: &mut Vec<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<()> {
        if depth == shape.len() {
            out.push(obj.clone());
            return Ok(());
        }
        let items = seq_items(obj).ok_or_else(|| {
            vm.new_value_error("inhomogeneous nested sequence".to_string())
        })?;
        if items.len() != shape[depth] {
            return Err(vm.new_value_error(
                "inhomogeneous nested sequence".to_string(),
            ));
        }
        for it in items {
            walk(&it, depth + 1, shape, out, vm)?;
        }
        Ok(())
    }
    let mut data = Vec::new();
    walk(obj, 0, &shape, &mut data, vm)?;
    Ok((shape, data))
}

fn obj_to_object_array(
    obj: &PyObjectRef,
    vm: &VirtualMachine,
) -> PyResult<ArraysD> {
    // Scalar (non-iterable) → 0-D array holding the object itself.
    if !(obj.downcast_ref::<PyList>().is_some() || obj.downcast_ref::<PyTuple>().is_some()) {
        return Ok(ArraysD::Object(
            ArrayD::<PyObjectRef>::from_elem(IxDyn(&[]), obj.clone()),
        ));
    }
    let (shape, data) = collect_objects(obj, vm)?;
    let arr = ArrayD::from_shape_vec(IxDyn(&shape), data)
        .or_internal(vm, "obj_to_object_array shape")?;
    Ok(ArraysD::Object(arr))
}

fn obj_to_str_array(
    obj: &PyObjectRef,
    target_n: u32,
    vm: &VirtualMachine,
) -> PyResult<ArraysD> {
    let coerce = |o: &PyObjectRef| -> PyResult<String> {
        if let Some(s) = o.downcast_ref::<PyStr>() {
            Ok(s.as_wtf8().to_string_lossy().into_owned())
        } else {
            // Use builtin str() coercion.
            let s = o.str(vm)?;
            Ok(s.as_wtf8().to_string_lossy().into_owned())
        }
    };
    if !(obj.downcast_ref::<PyList>().is_some() || obj.downcast_ref::<PyTuple>().is_some()) {
        let v = coerce(obj)?;
        let n = if target_n == 0 { v.chars().count() as u32 } else { target_n };
        return Ok(ArraysD::Str {
            itemsize_chars: n,
            data: ArrayD::from_elem(IxDyn(&[]), v),
        });
    }
    let (shape, data) = collect_objects(obj, vm)?;
    let strings: Vec<String> = data.iter().map(coerce).collect::<PyResult<_>>()?;
    let widest = strings.iter().map(|s| s.chars().count()).max().unwrap_or(0) as u32;
    let n = if target_n == 0 { widest } else { target_n };
    let arr = ArrayD::from_shape_vec(IxDyn(&shape), strings)
        .or_internal(vm, "obj_to_str_array shape")?;
    Ok(ArraysD::Str { itemsize_chars: n, data: arr })
}

fn obj_to_bytes_array(
    obj: &PyObjectRef,
    target_n: u32,
    vm: &VirtualMachine,
) -> PyResult<ArraysD> {
    let coerce = |o: &PyObjectRef| -> PyResult<Vec<u8>> {
        if let Some(b) = o.downcast_ref::<rustpython_vm::builtins::PyBytes>() {
            Ok(b.as_bytes().to_vec())
        } else if let Some(s) = o.downcast_ref::<PyStr>() {
            Ok(s.as_wtf8().to_string_lossy().as_bytes().to_vec())
        } else {
            // Fall through to str() then encode latin-1.
            let s = o.str(vm)?;
            Ok(s.as_wtf8().to_string_lossy().as_bytes().to_vec())
        }
    };
    if !(obj.downcast_ref::<PyList>().is_some() || obj.downcast_ref::<PyTuple>().is_some()) {
        let v = coerce(obj)?;
        let n = if target_n == 0 { v.len() as u32 } else { target_n };
        let mut padded = v;
        padded.resize(n as usize, 0);
        return Ok(ArraysD::Bytes {
            itemsize: n,
            data: ArrayD::from_elem(IxDyn(&[]), padded),
        });
    }
    let (shape, data) = collect_objects(obj, vm)?;
    let raw: Vec<Vec<u8>> = data.iter().map(coerce).collect::<PyResult<_>>()?;
    let widest = raw.iter().map(|v| v.len()).max().unwrap_or(0) as u32;
    let n = if target_n == 0 { widest } else { target_n };
    let padded: Vec<Vec<u8>> = raw
        .into_iter()
        .map(|mut v| {
            v.resize(n as usize, 0);
            v
        })
        .collect();
    let arr = ArrayD::from_shape_vec(IxDyn(&shape), padded)
        .or_internal(vm, "obj_to_bytes_array shape")?;
    Ok(ArraysD::Bytes { itemsize: n, data: arr })
}

fn obj_to_time_array(
    obj: &PyObjectRef,
    target: DType,
    unit: crate::dtype::TimeUnit,
    vm: &VirtualMachine,
) -> PyResult<ArraysD> {
    // For now we accept ints / floats / ISO strings (numpy parses these too).
    // Strings are interpreted as ISO-8601 dates/datetimes via `chrono`.
    let coerce = |o: &PyObjectRef| -> PyResult<i64> {
        if let Ok(s) = obj_as_scalar(o, vm) {
            return Ok(s.as_f64() as i64);
        }
        if let Some(s) = o.downcast_ref::<PyStr>() {
            let text = s.as_wtf8().to_string_lossy().into_owned();
            return parse_iso_to_unit(&text, unit).ok_or_else(|| {
                vm.new_value_error(format!("could not parse datetime: {text:?}"))
            });
        }
        Err(vm.new_type_error(
            "could not interpret element as datetime/timedelta".to_string(),
        ))
    };
    let build = |unit: crate::dtype::TimeUnit, data: ArrayD<i64>| -> ArraysD {
        match target {
            DType::Datetime64(_) => ArraysD::Datetime64 { unit, data },
            DType::Timedelta64(_) => ArraysD::Timedelta64 { unit, data },
            // Caller routes only the two time dtypes here, but the match
            // must still be total. Default to Datetime64 — bug-class invariant.
            _ => ArraysD::Datetime64 { unit, data },
        }
    };
    if !(obj.downcast_ref::<PyList>().is_some() || obj.downcast_ref::<PyTuple>().is_some()) {
        let v = coerce(obj)?;
        return Ok(build(unit, ArrayD::from_elem(IxDyn(&[]), v)));
    }
    let (shape, data) = collect_objects(obj, vm)?;
    let ints: Vec<i64> = data.iter().map(coerce).collect::<PyResult<_>>()?;
    let arr = ArrayD::from_shape_vec(IxDyn(&shape), ints)
        .or_internal(vm, "obj_to_time_array shape")?;
    Ok(build(unit, arr))
}

/// Parse an ISO-8601 date or datetime string into the requested time unit.
pub(crate) fn parse_iso_to_unit(s: &str, unit: crate::dtype::TimeUnit) -> Option<i64> {
    use chrono::{NaiveDate, NaiveDateTime};
    use crate::dtype::TimeUnit;
    // Try datetime first, then date.
    let dt: NaiveDateTime = if let Ok(d) = s.parse::<NaiveDateTime>() {
        d
    } else if let Ok(d) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        d.and_hms_opt(0, 0, 0)?
    } else if let Ok(d) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S") {
        d
    } else if let Ok(d) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
        d
    } else {
        return None;
    };
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1)?.and_hms_opt(0, 0, 0)?;
    let dur = dt.signed_duration_since(epoch);
    Some(match unit {
        TimeUnit::Y => dt.date().signed_duration_since(epoch.date()).num_days() / 365,
        TimeUnit::M => dt.date().signed_duration_since(epoch.date()).num_days() / 30,
        TimeUnit::W => dur.num_weeks(),
        TimeUnit::D => dur.num_days(),
        TimeUnit::H => dur.num_hours(),
        TimeUnit::Min => dur.num_minutes(),
        TimeUnit::S => dur.num_seconds(),
        TimeUnit::Ms => dur.num_milliseconds(),
        TimeUnit::Us => dur.num_microseconds()?,
        TimeUnit::Ns => dur.num_nanoseconds()?,
        TimeUnit::Ps => dur.num_nanoseconds()?.checked_mul(1000)?,
        TimeUnit::Fs => dur.num_nanoseconds()?.checked_mul(1_000_000)?,
        TimeUnit::As => dur.num_nanoseconds()?.checked_mul(1_000_000_000)?,
    })
}

/// Coerce any Python object into an `ndarray::ArrayD<T>` by first running
/// it through `obj_to_array` (which handles ndarrays, scalars, lists,
/// tuples, mixed nesting) and then casting to `T`. The high-level
/// generic-driven entry point for Rust callers:
///
/// ```ignore
/// use rumpy::convert::obj_to_typed;
/// // any python object → ArrayD<f32>
/// let arr: ndarray::ArrayD<f32> = obj_to_typed::<f32>(&py_obj, vm)?;
/// ```
pub fn obj_to_typed<T: crate::dtype::ArrayElement>(
    obj: &PyObjectRef,
    vm: &VirtualMachine,
) -> PyResult<ndarray::ArrayD<T>> {
    let a = obj_to_array(obj, Some(T::DTYPE), vm)?;
    use crate::dtype::CoerceArray;
    Ok(a.into_coerced::<T>())
}

/// Reduce a 0-D ArraysD (or a scalar-shaped one with a single element) to an
/// `f64`. Used by `index::set_via_index` for scalar assignment.
pub fn obj_as_scalar_from_array(
    a: &ArraysD,
    vm: &VirtualMachine,
) -> PyResult<f64> {
    if a.len() != 1 {
        return Err(vm.new_value_error(format!(
            "expected a scalar value for assignment, got shape {:?}",
            a.shape()
        )));
    }
    match a.cast(DType::F64) {
        ArraysD::F64(x) => x
            .iter()
            .next()
            .copied()
            .or_internal(vm, "obj_as_scalar_from_array: empty F64 view"),
        _ => Err(internal(
            vm,
            "obj_as_scalar_from_array: cast(F64) returned non-F64",
        )),
    }
}

fn zero_d_to_scalar(a: &crate::dtype::ArraysD) -> Scalar {
    use crate::dtype::ArraysD::*;
    let ix = ndarray::IxDyn(&[]);
    match a {
        Bool(x) => Scalar::Bool(x[ix]),
        I8(x) => Scalar::Int(x[ix] as i64),
        I16(x) => Scalar::Int(x[ix] as i64),
        I32(x) => Scalar::Int(x[ix] as i64),
        I64(x) => Scalar::Int(x[ix]),
        U8(x) => Scalar::UInt(x[ix] as u64),
        U16(x) => Scalar::UInt(x[ix] as u64),
        U32(x) => Scalar::UInt(x[ix] as u64),
        U64(x) => Scalar::UInt(x[ix]),
        F16(x) => Scalar::Float(f32::from(x[ix]) as f64),
        F32(x) => Scalar::Float(x[ix] as f64),
        F64(x) => Scalar::Float(x[ix]),
        C64(x) => {
            let v = x[ix];
            Scalar::Complex(Complex::new(v.re as f64, v.im as f64))
        }
        C128(x) => Scalar::Complex(x[ix]),
        // Non-numeric 0-d arrays: stringify or fall back to 0. Numpy itself
        // raises here for some dtypes; we return a default-ish scalar so
        // existing call sites don't panic.
        Datetime64 { data, .. } | Timedelta64 { data, .. } => Scalar::Int(data[ix]),
        Object(_) | Str { .. } | Bytes { .. } | Void { .. } => Scalar::Float(0.0),
    }
}

fn scalar_to_zero_d(s: Scalar, dt: DType) -> ArraysD {
    let shape = IxDyn(&[]);
    macro_rules! one {
        ($var:ident, $ty:ty, $val:expr) => {
            ArraysD::$var(ArrayD::<$ty>::from_elem(shape.clone(), $val))
        };
    }
    match dt {
        DType::Bool => one!(Bool, bool, matches!(s, Scalar::Bool(true)) || s.as_f64() != 0.0),
        DType::I8 => one!(I8, i8, s.as_f64() as i8),
        DType::I16 => one!(I16, i16, s.as_f64() as i16),
        DType::I32 => one!(I32, i32, s.as_f64() as i32),
        DType::I64 => match s {
            Scalar::UInt(u) => one!(I64, i64, u as i64),
            _ => one!(I64, i64, s.as_f64() as i64),
        },
        DType::U8 => one!(U8, u8, s.as_f64() as u8),
        DType::U16 => one!(U16, u16, s.as_f64() as u16),
        DType::U32 => one!(U32, u32, s.as_f64() as u32),
        DType::U64 => match s {
            Scalar::UInt(u) => one!(U64, u64, u),
            _ => one!(U64, u64, s.as_f64() as u64),
        },
        DType::F16 => one!(F16, f16, f16::from_f64(s.as_f64())),
        DType::F32 => one!(F32, f32, s.as_f64() as f32),
        DType::F64 => one!(F64, f64, s.as_f64()),
        DType::C64 => one!(C64, C32, {
            let c = s.as_c128();
            C32::new(c.re as f32, c.im as f32)
        }),
        DType::C128 => one!(C128, C64, s.as_c128()),
        DType::Datetime64(u) => ArraysD::Datetime64 {
            unit: u,
            data: ArrayD::from_elem(shape, s.as_f64() as i64),
        },
        DType::Timedelta64(u) => ArraysD::Timedelta64 {
            unit: u,
            data: ArrayD::from_elem(shape, s.as_f64() as i64),
        },
        DType::Str(n) => ArraysD::Str {
            itemsize_chars: n,
            data: ArrayD::from_elem(shape, String::new()),
        },
        DType::Bytes(n) => ArraysD::Bytes {
            itemsize: n,
            data: ArrayD::from_elem(shape, vec![0u8; n as usize]),
        },
        DType::Object => ArraysD::Object(
            crate::internal::empty_array(),
        ),
        DType::Void(n) => ArraysD::Void {
            layout: std::sync::Arc::new(crate::dtype::StructLayout::new(Vec::new(), n as usize)),
            data: ArrayD::from_elem(shape, vec![0u8; n as usize]),
        },
    }
}

fn seq_items(obj: &PyObjectRef) -> Option<Vec<PyObjectRef>> {
    if let Some(l) = obj.downcast_ref::<PyList>() {
        return Some(l.borrow_vec().to_vec());
    }
    if let Some(t) = obj.downcast_ref::<PyTuple>() {
        return Some(t.as_slice().to_vec());
    }
    None
}

fn collect_nested(
    obj: &PyObjectRef,
    vm: &VirtualMachine,
) -> PyResult<(Vec<usize>, Vec<Scalar>)> {
    let mut shape = Vec::new();
    let mut cur = obj.clone();
    while let Some(items) = seq_items(&cur) {
        shape.push(items.len());
        if items.is_empty() {
            break;
        }
        cur = items[0].clone();
    }

    fn walk(
        obj: &PyObjectRef,
        depth: usize,
        shape: &[usize],
        out: &mut Vec<Scalar>,
        vm: &VirtualMachine,
    ) -> PyResult<()> {
        if depth == shape.len() {
            out.push(obj_as_scalar(obj, vm)?);
            return Ok(());
        }
        let items = seq_items(obj)
            .ok_or_else(|| vm.new_value_error("inhomogeneous nested sequence".to_string()))?;
        if items.len() != shape[depth] {
            return Err(vm.new_value_error("inhomogeneous nested sequence".to_string()));
        }
        for it in items {
            walk(&it, depth + 1, shape, out, vm)?;
        }
        Ok(())
    }

    let mut data = Vec::new();
    walk(obj, 0, &shape, &mut data, vm)?;
    Ok((shape, data))
}

fn infer_dtype_from_scalars(scalars: &[Scalar]) -> DType {
    if scalars.is_empty() {
        return DType::F64;
    }
    let mut acc = scalars[0].inferred_dtype();
    for s in &scalars[1..] {
        acc = crate::promote::promote(acc, s.inferred_dtype());
    }
    acc
}

fn scalars_to_array(
    shape: &[usize],
    scalars: Vec<Scalar>,
    dt: DType,
    vm: &VirtualMachine,
) -> PyResult<ArraysD> {
    macro_rules! build {
        ($var:ident, $ty:ty, $conv:expr) => {{
            let data: Vec<$ty> = scalars.iter().copied().map($conv).collect();
            ArrayD::<$ty>::from_shape_vec(IxDyn(shape), data)
                .or_internal(vm, "scalars_to_array")
                .map(ArraysD::$var)
        }};
    }
    match dt {
        DType::Bool => build!(Bool, bool, |s: Scalar| {
            matches!(s, Scalar::Bool(true)) || (!matches!(s, Scalar::Bool(_)) && s.as_f64() != 0.0)
        }),
        DType::I8 => build!(I8, i8, |s: Scalar| s.as_f64() as i8),
        DType::I16 => build!(I16, i16, |s: Scalar| s.as_f64() as i16),
        DType::I32 => build!(I32, i32, |s: Scalar| s.as_f64() as i32),
        DType::I64 => build!(I64, i64, |s: Scalar| match s {
            Scalar::UInt(u) => u as i64,
            _ => s.as_f64() as i64,
        }),
        DType::U8 => build!(U8, u8, |s: Scalar| s.as_f64() as u8),
        DType::U16 => build!(U16, u16, |s: Scalar| s.as_f64() as u16),
        DType::U32 => build!(U32, u32, |s: Scalar| s.as_f64() as u32),
        DType::U64 => build!(U64, u64, |s: Scalar| match s {
            Scalar::UInt(u) => u,
            _ => s.as_f64() as u64,
        }),
        DType::F16 => build!(F16, f16, |s: Scalar| f16::from_f64(s.as_f64())),
        DType::F32 => build!(F32, f32, |s: Scalar| s.as_f64() as f32),
        DType::F64 => build!(F64, f64, |s: Scalar| s.as_f64()),
        DType::C64 => build!(C64, C32, |s: Scalar| {
            let c = s.as_c128();
            C32::new(c.re as f32, c.im as f32)
        }),
        DType::C128 => build!(C128, C64, |s: Scalar| s.as_c128()),
        DType::Datetime64(u) => {
            let data: Vec<i64> = scalars.iter().map(|s| s.as_f64() as i64).collect();
            ArrayD::<i64>::from_shape_vec(IxDyn(shape), data)
                .or_internal(vm, "scalars_to_array dt")
                .map(|d| ArraysD::Datetime64 { unit: u, data: d })
        }
        DType::Timedelta64(u) => {
            let data: Vec<i64> = scalars.iter().map(|s| s.as_f64() as i64).collect();
            ArrayD::<i64>::from_shape_vec(IxDyn(shape), data)
                .or_internal(vm, "scalars_to_array td")
                .map(|d| ArraysD::Timedelta64 { unit: u, data: d })
        }
        DType::Str(n) => {
            let data: Vec<String> = scalars.iter().map(|s| s.to_display_string()).collect();
            ArrayD::<String>::from_shape_vec(IxDyn(shape), data)
                .or_internal(vm, "scalars_to_array str")
                .map(|d| ArraysD::Str { itemsize_chars: n, data: d })
        }
        DType::Bytes(n) => {
            let data: Vec<Vec<u8>> = scalars
                .iter()
                .map(|s| {
                    let mut b = s.to_display_string().into_bytes();
                    b.resize(n as usize, 0);
                    b
                })
                .collect();
            ArrayD::<Vec<u8>>::from_shape_vec(IxDyn(shape), data)
                .or_internal(vm, "scalars_to_array bytes")
                .map(|d| ArraysD::Bytes { itemsize: n, data: d })
        }
        DType::Object => Err(vm.new_type_error(
            "cannot build object array from scalars without vm-aware path".to_string(),
        )),
        DType::Void(n) => {
            let data: Vec<Vec<u8>> = scalars.iter().map(|_| vec![0u8; n as usize]).collect();
            ArrayD::<Vec<u8>>::from_shape_vec(IxDyn(shape), data)
                .or_internal(vm, "scalars_to_array void")
                .map(|d| ArraysD::Void {
                    layout: std::sync::Arc::new(crate::dtype::StructLayout::new(Vec::new(), n as usize)),
                    data: d,
                })
        }
    }
}

/// Build a nested Python list mirroring the array's shape and values.
pub fn array_to_pylist(arr: &ArraysD, vm: &VirtualMachine) -> PyObjectRef {
    fn rec_f64(
        a: &ndarray::ArrayBase<ndarray::ViewRepr<&f64>, IxDyn>,
        vm: &VirtualMachine,
    ) -> PyObjectRef {
        if a.ndim() == 0 {
            return vm.ctx.new_float(a[IxDyn(&[])]).into();
        }
        if a.ndim() == 1 {
            let v: Vec<PyObjectRef> = a.iter().map(|&x| vm.ctx.new_float(x).into()).collect();
            return PyList::from(v).into_ref(&vm.ctx).into();
        }
        let mut subs = Vec::with_capacity(a.shape()[0]);
        for sub in a.axis_iter(Axis(0)) {
            subs.push(rec_f64(&sub, vm));
        }
        PyList::from(subs).into_ref(&vm.ctx).into()
    }
    // Universal path: cast to f64 (real) or to a Python complex tree for complex.
    if arr.dtype().is_complex() {
        return complex_array_to_pylist(arr, vm);
    }
    if arr.dtype() == DType::Bool {
        return bool_array_to_pylist(arr, vm);
    }
    if arr.dtype().is_integer() {
        return int_array_to_pylist(arr, vm);
    }
    // Non-numeric: render to whatever Python value the cell already holds.
    match arr {
        ArraysD::Object(a) => return object_array_to_pylist(a, vm),
        ArraysD::Str { data, .. } => return str_array_to_pylist(data, vm),
        ArraysD::Bytes { data, .. } => return bytes_array_to_pylist(data, vm),
        ArraysD::Datetime64 { data, .. } | ArraysD::Timedelta64 { data, .. } => {
            // Datetime/Timedelta surface as integers (the raw unit counter).
            return raw_i64_array_to_pylist(data, vm);
        }
        ArraysD::Void { data, .. } => return bytes_array_to_pylist(data, vm),
        _ => {}
    }
    // `coerce::<f64>` always returns an owned `ArrayD<f64>` (clones if dtype
    // already matches), so no panic-prone variant unwrap is needed.
    use crate::dtype::CoerceArray;
    let owned = arr.coerce::<f64>();
    rec_f64(&owned.view(), vm)
}

fn object_array_to_pylist(
    arr: &ArrayD<PyObjectRef>,
    vm: &VirtualMachine,
) -> PyObjectRef {
    if arr.ndim() == 0 {
        return arr[IxDyn(&[])].clone();
    }
    if arr.ndim() == 1 {
        let v: Vec<PyObjectRef> = arr.iter().cloned().collect();
        return PyList::from(v).into_ref(&vm.ctx).into();
    }
    let mut subs = Vec::with_capacity(arr.shape()[0]);
    for sub in arr.axis_iter(Axis(0)) {
        let owned = sub.to_owned();
        subs.push(object_array_to_pylist(&owned, vm));
    }
    PyList::from(subs).into_ref(&vm.ctx).into()
}

fn str_array_to_pylist(
    arr: &ArrayD<String>,
    vm: &VirtualMachine,
) -> PyObjectRef {
    if arr.ndim() == 0 {
        return vm.ctx.new_str(arr[IxDyn(&[])].as_str()).into();
    }
    if arr.ndim() == 1 {
        let v: Vec<PyObjectRef> = arr
            .iter()
            .map(|s| vm.ctx.new_str(s.as_str()).into())
            .collect();
        return PyList::from(v).into_ref(&vm.ctx).into();
    }
    let mut subs = Vec::with_capacity(arr.shape()[0]);
    for sub in arr.axis_iter(Axis(0)) {
        subs.push(str_array_to_pylist(&sub.to_owned(), vm));
    }
    PyList::from(subs).into_ref(&vm.ctx).into()
}

fn bytes_array_to_pylist(
    arr: &ArrayD<Vec<u8>>,
    vm: &VirtualMachine,
) -> PyObjectRef {
    if arr.ndim() == 0 {
        return vm.ctx.new_bytes(arr[IxDyn(&[])].clone()).into();
    }
    if arr.ndim() == 1 {
        let v: Vec<PyObjectRef> = arr
            .iter()
            .map(|b| vm.ctx.new_bytes(b.clone()).into())
            .collect();
        return PyList::from(v).into_ref(&vm.ctx).into();
    }
    let mut subs = Vec::with_capacity(arr.shape()[0]);
    for sub in arr.axis_iter(Axis(0)) {
        subs.push(bytes_array_to_pylist(&sub.to_owned(), vm));
    }
    PyList::from(subs).into_ref(&vm.ctx).into()
}

fn raw_i64_array_to_pylist(
    arr: &ArrayD<i64>,
    vm: &VirtualMachine,
) -> PyObjectRef {
    if arr.ndim() == 0 {
        return vm.ctx.new_int(arr[IxDyn(&[])]).into();
    }
    if arr.ndim() == 1 {
        let v: Vec<PyObjectRef> = arr.iter().map(|&x| vm.ctx.new_int(x).into()).collect();
        return PyList::from(v).into_ref(&vm.ctx).into();
    }
    let mut subs = Vec::with_capacity(arr.shape()[0]);
    for sub in arr.axis_iter(Axis(0)) {
        subs.push(raw_i64_array_to_pylist(&sub.to_owned(), vm));
    }
    PyList::from(subs).into_ref(&vm.ctx).into()
}

fn int_array_to_pylist(arr: &ArraysD, vm: &VirtualMachine) -> PyObjectRef {
    fn rec(
        a: &ndarray::ArrayBase<ndarray::ViewRepr<&i64>, IxDyn>,
        vm: &VirtualMachine,
    ) -> PyObjectRef {
        if a.ndim() == 0 {
            return vm.ctx.new_int(a[IxDyn(&[])]).into();
        }
        if a.ndim() == 1 {
            let v: Vec<PyObjectRef> = a.iter().map(|&x| vm.ctx.new_int(x).into()).collect();
            return PyList::from(v).into_ref(&vm.ctx).into();
        }
        let mut subs = Vec::with_capacity(a.shape()[0]);
        for sub in a.axis_iter(Axis(0)) {
            subs.push(rec(&sub, vm));
        }
        PyList::from(subs).into_ref(&vm.ctx).into()
    }
    use crate::dtype::CoerceArray;
    let owned = arr.coerce::<i64>();
    rec(&owned.view(), vm)
}

fn bool_array_to_pylist(arr: &ArraysD, vm: &VirtualMachine) -> PyObjectRef {
    fn rec(
        a: &ndarray::ArrayBase<ndarray::ViewRepr<&bool>, IxDyn>,
        vm: &VirtualMachine,
    ) -> PyObjectRef {
        if a.ndim() == 0 {
            return vm.ctx.new_bool(a[IxDyn(&[])]).into();
        }
        if a.ndim() == 1 {
            let v: Vec<PyObjectRef> = a.iter().map(|&x| vm.ctx.new_bool(x).into()).collect();
            return PyList::from(v).into_ref(&vm.ctx).into();
        }
        let mut subs = Vec::with_capacity(a.shape()[0]);
        for sub in a.axis_iter(Axis(0)) {
            subs.push(rec(&sub, vm));
        }
        PyList::from(subs).into_ref(&vm.ctx).into()
    }
    use crate::dtype::CoerceArray;
    let owned = arr.coerce::<bool>();
    rec(&owned.view(), vm)
}

fn complex_array_to_pylist(arr: &ArraysD, vm: &VirtualMachine) -> PyObjectRef {
    fn rec(
        a: &ndarray::ArrayBase<ndarray::ViewRepr<&C64>, IxDyn>,
        vm: &VirtualMachine,
    ) -> PyObjectRef {
        if a.ndim() == 0 {
            let v = a[IxDyn(&[])];
            return vm.ctx.new_complex(num_complex::Complex64::new(v.re, v.im)).into();
        }
        if a.ndim() == 1 {
            let v: Vec<PyObjectRef> = a
                .iter()
                .map(|&x| {
                    vm.ctx
                        .new_complex(num_complex::Complex64::new(x.re, x.im))
                        .into()
                })
                .collect();
            return PyList::from(v).into_ref(&vm.ctx).into();
        }
        let mut subs = Vec::with_capacity(a.shape()[0]);
        for sub in a.axis_iter(Axis(0)) {
            subs.push(rec(&sub, vm));
        }
        PyList::from(subs).into_ref(&vm.ctx).into()
    }
    use crate::dtype::CoerceArray;
    let owned = arr.coerce::<C64>();
    rec(&owned.view(), vm)
}

/// Parse a shape argument: int → 1-D, sequence-of-ints → N-D.
pub fn parse_shape(
    obj: &PyObjectRef,
    vm: &VirtualMachine,
) -> PyResult<Vec<usize>> {
    if let Some(i) = obj.downcast_ref::<PyInt>() {
        let n = i.try_to_primitive::<i64>(vm)?;
        if n < 0 {
            return Err(vm.new_value_error("negative dimension".to_string()));
        }
        return Ok(vec![n as usize]);
    }
    if let Some(items) = seq_items(obj) {
        let mut out = Vec::with_capacity(items.len());
        for it in items {
            let n = it.try_int(vm)?.try_to_primitive::<i64>(vm)?;
            if n < 0 {
                return Err(vm.new_value_error("negative dimension".to_string()));
            }
            out.push(n as usize);
        }
        return Ok(out);
    }
    Err(vm.new_type_error("shape must be int or sequence of ints".to_string()))
}

/// Same as `parse_shape` but allows `-1` (the placeholder used by `reshape`).
pub fn parse_shape_signed(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<Vec<i64>> {
    if let Some(i) = obj.downcast_ref::<PyInt>() {
        return Ok(vec![i.try_to_primitive::<i64>(vm)?]);
    }
    if let Some(items) = seq_items(obj) {
        let mut out = Vec::with_capacity(items.len());
        for it in items {
            out.push(it.try_int(vm)?.try_to_primitive::<i64>(vm)?);
        }
        return Ok(out);
    }
    Err(vm.new_type_error("shape must be int or sequence of ints".to_string()))
}

/// Resolve a (possibly contains-`-1`) shape against a known total size.
pub fn resolve_neg_one(
    shape: &[i64],
    total: usize,
    vm: &VirtualMachine,
) -> PyResult<Vec<usize>> {
    let mut neg = None;
    let mut prod: i64 = 1;
    for (i, &d) in shape.iter().enumerate() {
        if d == -1 {
            if neg.is_some() {
                return Err(vm.new_value_error("can only specify one -1 in shape".to_string()));
            }
            neg = Some(i);
        } else if d < 0 {
            return Err(vm.new_value_error("negative dimensions not allowed".to_string()));
        } else {
            prod *= d;
        }
    }
    let mut out: Vec<usize> = shape
        .iter()
        .map(|&d| if d == -1 { 0 } else { d as usize })
        .collect();
    if let Some(i) = neg {
        if prod == 0 {
            return Err(
                vm.new_value_error("cannot reshape: zero element in other dims".to_string()),
            );
        }
        if (total as i64) % prod != 0 {
            return Err(vm.new_value_error(format!(
                "cannot reshape array of size {total} into shape {shape:?}"
            )));
        }
        out[i] = (total as i64 / prod) as usize;
    }
    Ok(out)
}

/// Parse a `dtype=...` Python argument (None / str / Python builtin type) into
/// a DType or `None`. Accepts numpy-style strings (`"float32"`, `"f4"`, `"<f8"`,
/// `"?"`), Python builtins (`int`, `float`, `bool`, `complex`), and a handful of
/// numpy aliases bound on the module.
pub fn parse_dtype_arg(
    obj: &Option<PyObjectRef>,
    vm: &VirtualMachine,
) -> PyResult<Option<DType>> {
    let Some(o) = obj else { return Ok(None) };
    if o.is(&vm.ctx.none) {
        return Ok(None);
    }
    if let Some(d) = o.downcast_ref::<crate::numpy_module::PyDType>() {
        return Ok(Some(d.inner));
    }
    if let Some(s) = o.downcast_ref::<PyStr>() {
        let bytes = s.as_wtf8().to_string_lossy();
        return match DType::parse(&bytes) {
            Some(d) => Ok(Some(d)),
            None => Err(vm.new_type_error(format!("unknown dtype string {bytes:?}"))),
        };
    }
    // Python built-in type objects map to default numpy dtypes.
    if let Some(t) = o.downcast_ref::<rustpython_vm::builtins::PyType>() {
        let name = t.name();
        let mapped = match &*name {
            "int" => Some(DType::I64),
            "float" => Some(DType::F64),
            "bool" => Some(DType::Bool),
            "complex" => Some(DType::C128),
            // np.array([...], dtype=object) / np.array(..., dtype=str)
            "object" | "object_" => Some(DType::Object),
            "str" | "str_" => Some(DType::Str(0)),
            "bytes" | "bytes_" => Some(DType::Bytes(0)),
            other => DType::parse(other),
        };
        if let Some(d) = mapped {
            return Ok(Some(d));
        }
    }
    Err(vm.new_type_error(format!(
        "dtype= must be a string, None, or a recognized type, got {}",
        o.class().name()
    )))
}
