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
    #[no_panic::no_panic]
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

    #[no_panic::no_panic]
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

    #[no_panic::no_panic]
    #[inline]
    pub fn as_c128(self) -> C64 {
        match self {
            Scalar::Complex(c) => c,
            _ => Complex::new(self.as_f64(), 0.0),
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
    // Scalar?
    if let Ok(s) = obj_as_scalar(obj, vm) {
        let dt = target_dtype.unwrap_or_else(|| s.inferred_dtype());
        return Ok(scalar_to_zero_d(s, dt));
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
    // `coerce::<f64>` always returns an owned `ArrayD<f64>` (clones if dtype
    // already matches), so no panic-prone variant unwrap is needed.
    use crate::dtype::CoerceArray;
    let owned = arr.coerce::<f64>();
    rec_f64(&owned.view(), vm)
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
