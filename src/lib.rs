//! rumpy — a numpy-feature-compatible Python module implemented in Rust on top
//! of [`ndarray`], exposed to [`rustpython_vm`] as the module `numpy`.
//!
//! Supports the full numpy numeric dtype set:
//!
//! - `bool`
//! - signed integers: `int8 / int16 / int32 / int64`
//! - unsigned integers: `uint8 / uint16 / uint32 / uint64`
//! - floats: `float16 / float32 / float64`
//! - complex: `complex64 / complex128`
//!
//! Element type promotion follows `numpy.result_type` (see `promote.rs`).
//! The on-array data is a plain `ndarray::ArrayD<T>` per dtype — there is no
//! internal locking; synchronization is the embedder's responsibility.

pub mod convert;
pub mod create;
pub mod dtype;
pub mod einsum;
pub mod extras;
pub mod extras2;
pub mod fft;
pub mod fmt;
pub mod index;
pub mod internal;
pub mod linalg;
pub mod linalg_extra;
pub mod more_ops;
pub mod npy;
pub mod npz;
pub mod ops;
pub mod poly;
pub mod promote;
pub mod random;
pub mod reduce;
pub mod textio;

pub use dtype::{ArrayElement, ArraysD, CoerceArray, DType};
pub use numpy_module::PyNdArray;

/// `CoerceArray` for `&PyNdArray` — delegates through the inner [`ArraysD`].
impl CoerceArray for PyNdArray {
    #[inline]
    fn coerce<T: ArrayElement>(&self) -> ndarray::ArrayD<T> {
        self.view().coerce::<T>()
    }
    #[inline]
    fn try_borrow_as<T: ArrayElement>(&self) -> Option<&ndarray::ArrayD<T>> {
        // SAFETY: see `PyNdArray::raw_ref` — the embedder must not call
        // `view_mut()` concurrently with this borrow.
        unsafe { self.raw_ref() }.try_borrow_as::<T>()
    }
    #[inline]
    fn into_coerced<T: ArrayElement>(self) -> ndarray::ArrayD<T> {
        self.inner.into_inner().into_coerced::<T>()
    }
}

// ----- Ergonomic Rust-side traits for PyNdArray -----
//
// These mirror the same SAFETY contract as `view()` (no concurrent mutation
// while the borrow is live). In `safe-locks` mode, `view()`/`view_mut()`
// still take the real lock — these "fast lane" accessors bypass it for the
// same ergonomics across both features.

impl AsRef<ArraysD> for PyNdArray {
    #[inline]
    fn as_ref(&self) -> &ArraysD {
        // SAFETY: see `PyNdArray::raw_ref`.
        unsafe { self.raw_ref() }
    }
}

impl AsMut<ArraysD> for PyNdArray {
    #[inline]
    fn as_mut(&mut self) -> &mut ArraysD {
        // We have `&mut self`, so exclusive access is statically guaranteed.
        self.raw_mut()
    }
}

impl std::ops::Deref for PyNdArray {
    type Target = ArraysD;
    #[inline]
    fn deref(&self) -> &ArraysD {
        // SAFETY: see `PyNdArray::raw_ref`.
        unsafe { self.raw_ref() }
    }
}

impl std::ops::DerefMut for PyNdArray {
    #[inline]
    fn deref_mut(&mut self) -> &mut ArraysD {
        self.raw_mut()
    }
}

impl<T: ArrayElement> From<ndarray::ArrayD<T>> for PyNdArray {
    #[inline]
    fn from(a: ndarray::ArrayD<T>) -> Self {
        PyNdArray::from_arrays(T::from_array(a))
    }
}

impl From<ArraysD> for PyNdArray {
    #[inline]
    fn from(inner: ArraysD) -> Self {
        PyNdArray::from_arrays(inner)
    }
}

impl AsRef<PyNdArray> for PyNdArray {
    #[inline]
    fn as_ref(&self) -> &PyNdArray {
        self
    }
}

impl AsMut<PyNdArray> for PyNdArray {
    #[inline]
    fn as_mut(&mut self) -> &mut PyNdArray {
        self
    }
}

/// Return the `numpy` module definition for embedding into a
/// [`rustpython_vm::Interpreter`].
pub fn module_def(ctx: &rustpython_vm::Context) -> &'static rustpython_vm::builtins::PyModuleDef {
    numpy_module::module_def(ctx)
}

#[rustpython_vm::pymodule(name = "numpy")]
pub(crate) mod numpy_module {
    use crate::convert::{
        array_to_pylist, obj_to_array, parse_dtype_arg, parse_shape, parse_shape_signed,
        resolve_neg_one,
    };
    use crate::dtype::{ArraysD, DType};
    use crate::ops::CmpOp;
    use crate::reduce::Reduce;
    use crate::{create, fmt as repr_fmt, index, linalg, ops, reduce};
    use rustpython_vm::AsObject;
    use rustpython_vm::function::{Either, PyComparisonValue};
    use rustpython_vm::{
        FromArgs, Py, PyObject, PyObjectRef, PyPayload, PyResult, VirtualMachine,
        builtins::{PyTuple, PyType},
        function::{ArgIntoFloat, FuncArgs, OptionalArg},
        protocol::{PyMappingMethods, PyNumberMethods},
        pyclass, pymodule,
        types::{
            AsMapping, AsNumber, Comparable, Constructor, Iterable, PyComparisonOp, Representable,
        },
    };
    // The `pyattr(once)` macro expands to a `rustpython_common::static_cell!`
    // call. rustpython-vm re-exports the common crate as `vm::common`, so we
    // make it available under its bare name here for the macro's benefit.
    use rustpython_vm::common as rustpython_common;

    // -----------------------------------------------------------------
    // FromArgs structs for keyword arguments
    // -----------------------------------------------------------------

    #[derive(FromArgs)]
    pub(crate) struct AxisArg {
        #[pyarg(any, optional)]
        axis: OptionalArg<Option<isize>>,
    }

    /// Reduction kwargs supporting `axis=int|tuple|None` plus `keepdims=`.
    #[derive(FromArgs)]
    pub(crate) struct ReduceArgs {
        #[pyarg(any, optional)]
        axis: OptionalArg<PyObjectRef>,
        #[pyarg(any, optional)]
        keepdims: OptionalArg<bool>,
    }

    /// Parse the `axis=` kwarg into `Option<Vec<isize>>`. `None` means
    /// "reduce the entire array" (numpy's `axis=None` default).
    fn parse_axes(
        arg: &OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<Option<Vec<isize>>> {
        match arg {
            OptionalArg::Missing => Ok(None),
            OptionalArg::Present(o) if o.is(&vm.ctx.none) => Ok(None),
            OptionalArg::Present(o) => {
                if let Some(t) = o.downcast_ref::<PyTuple>() {
                    let v: PyResult<Vec<isize>> = t
                        .as_slice()
                        .iter()
                        .map(|x| x.try_int(vm)?.try_to_primitive::<isize>(vm))
                        .collect();
                    return Ok(Some(v?));
                }
                if let Some(l) = o.downcast_ref::<rustpython_vm::builtins::PyList>() {
                    let v: PyResult<Vec<isize>> = l
                        .borrow_vec()
                        .iter()
                        .map(|x| x.try_int(vm)?.try_to_primitive::<isize>(vm))
                        .collect();
                    return Ok(Some(v?));
                }
                let i = o.try_int(vm)?.try_to_primitive::<isize>(vm)?;
                Ok(Some(vec![i]))
            }
        }
    }

    /// Single-arg reduction with axis/keepdims kwargs.
    fn do_reduce(
        arr: &ArraysD,
        args: ReduceArgs,
        op: Reduce,
        vm: &VirtualMachine,
    ) -> PyResult<ArraysD> {
        let axes = parse_axes(&args.axis, vm)?;
        let keepdims = args.keepdims.unwrap_or(false);
        reduce::reduce_multi(arr, axes.as_deref(), keepdims, op, vm)
    }

    #[derive(FromArgs)]
    pub(crate) struct VarianceArg {
        #[pyarg(any, optional)]
        axis: OptionalArg<PyObjectRef>,
        #[pyarg(any, optional)]
        ddof: OptionalArg<usize>,
        #[pyarg(any, optional)]
        keepdims: OptionalArg<bool>,
    }

    #[derive(FromArgs)]
    pub(crate) struct DTypeArg {
        #[pyarg(any, optional)]
        dtype: OptionalArg<PyObjectRef>,
    }

    #[derive(FromArgs)]
    pub(crate) struct FullArgs {
        #[pyarg(positional)]
        shape: PyObjectRef,
        #[pyarg(positional)]
        fill_value: ArgIntoFloat,
        #[pyarg(any, optional)]
        dtype: OptionalArg<PyObjectRef>,
    }

    /// `np.concatenate(arrays, axis=0)` — `axis=None` flattens.
    #[derive(FromArgs)]
    pub(crate) struct ConcatenateArgs {
        #[pyarg(any, optional)]
        axis: OptionalArg<Option<isize>>,
    }

    // -----------------------------------------------------------------
    // The ndarray class
    // -----------------------------------------------------------------

    /// Python-visible ndarray. Wraps an [`ArraysD`] — a tagged union over all
    /// numpy element types.
    ///
    /// The element store sits behind interior mutability because Python's
    /// `arr[i] = v` mutates through a shared reference. Two implementations
    /// are available, picked by Cargo feature:
    ///
    /// * Default (no features): [`std::cell::UnsafeCell`] with a manual
    ///   `unsafe impl Sync` — the embedder is responsible for not mutating
    ///   the same array from multiple threads concurrently.
    /// * `safe-locks`: rustpython's [`PyRwLock`](rustpython_vm::common::lock::PyRwLock).
    ///   Fully safe at the cost of read/write lock overhead per access.
    #[pyattr]
    #[pyclass(module = "numpy", name = "ndarray")]
    #[derive(Debug, PyPayload)]
    pub struct PyNdArray {
        #[cfg(not(feature = "safe-locks"))]
        pub(crate) inner: std::cell::UnsafeCell<ArraysD>,
        #[cfg(feature = "safe-locks")]
        pub(crate) inner: rustpython_vm::common::lock::PyRwLock<ArraysD>,
    }

    // SAFETY: in lock-free mode we trade Sync correctness under concurrent
    // mutation for the simplicity of a lock-free design. With `safe-locks`
    // the inner lock is itself Sync, so the auto-derived impl applies.
    #[cfg(not(feature = "safe-locks"))]
    unsafe impl Sync for PyNdArray {}
    #[cfg(not(feature = "safe-locks"))]
    unsafe impl Send for PyNdArray {}

    /// Read borrow returned by [`PyNdArray::view`]. Derefs to `&ArraysD`.
    /// In `safe-locks` mode this holds a read guard that releases on drop.
    pub struct ArrayView<'a> {
        #[cfg(not(feature = "safe-locks"))]
        inner: &'a ArraysD,
        #[cfg(feature = "safe-locks")]
        inner: rustpython_vm::common::lock::PyRwLockReadGuard<'a, ArraysD>,
    }

    impl std::ops::Deref for ArrayView<'_> {
        type Target = ArraysD;
        #[inline]
        fn deref(&self) -> &ArraysD {
            #[cfg(not(feature = "safe-locks"))]
            {
                self.inner
            }
            #[cfg(feature = "safe-locks")]
            {
                &self.inner
            }
        }
    }

    /// Write borrow returned by [`PyNdArray::view_mut`]. Derefs to
    /// `&mut ArraysD`. In `safe-locks` mode this holds a write guard.
    pub struct ArrayViewMut<'a> {
        #[cfg(not(feature = "safe-locks"))]
        inner: &'a mut ArraysD,
        #[cfg(feature = "safe-locks")]
        inner: rustpython_vm::common::lock::PyRwLockWriteGuard<'a, ArraysD>,
    }

    impl std::ops::Deref for ArrayViewMut<'_> {
        type Target = ArraysD;
        #[inline]
        fn deref(&self) -> &ArraysD {
            #[cfg(not(feature = "safe-locks"))]
            {
                self.inner
            }
            #[cfg(feature = "safe-locks")]
            {
                &self.inner
            }
        }
    }

    impl std::ops::DerefMut for ArrayViewMut<'_> {
        #[inline]
        fn deref_mut(&mut self) -> &mut ArraysD {
            #[cfg(not(feature = "safe-locks"))]
            {
                self.inner
            }
            #[cfg(feature = "safe-locks")]
            {
                &mut self.inner
            }
        }
    }

    impl PyNdArray {
        #[cfg(not(feature = "safe-locks"))]
        pub fn from_arrays(a: ArraysD) -> Self {
            Self {
                inner: std::cell::UnsafeCell::new(a),
            }
        }

        #[cfg(feature = "safe-locks")]
        pub fn from_arrays(a: ArraysD) -> Self {
            Self {
                inner: rustpython_vm::common::lock::PyRwLock::new(a),
            }
        }

        /// Read-only borrow of the inner array. The returned wrapper derefs
        /// to `&ArraysD`; in `safe-locks` mode it also holds a read guard
        /// that releases when dropped.
        #[cfg(not(feature = "safe-locks"))]
        #[inline]
        pub fn view(&self) -> ArrayView<'_> {
            // SAFETY: see struct doc-comment.
            ArrayView {
                inner: unsafe { &*self.inner.get() },
            }
        }

        #[cfg(feature = "safe-locks")]
        #[inline]
        pub fn view(&self) -> ArrayView<'_> {
            ArrayView {
                inner: self.inner.read(),
            }
        }

        /// Mutable borrow of the inner array. Lock-free mode dereferences
        /// a raw pointer (embedder's contract: no concurrent access).
        /// `safe-locks` mode returns a write-guard wrapper.
        #[cfg(not(feature = "safe-locks"))]
        #[inline]
        pub fn view_mut(&self) -> ArrayViewMut<'_> {
            ArrayViewMut {
                inner: unsafe { &mut *self.inner.get() },
            }
        }

        #[cfg(feature = "safe-locks")]
        #[inline]
        pub fn view_mut(&self) -> ArrayViewMut<'_> {
            ArrayViewMut {
                inner: self.inner.write(),
            }
        }

        /// Lock-bypassing read borrow used by AsRef/Deref. Returns a reference
        /// tied to `&self` so the borrow's lifetime is the same in both
        /// feature configurations.
        ///
        /// # Safety
        /// The caller must not invoke `view_mut()` or hold an `ArrayViewMut`
        /// concurrently with this borrow. In `safe-locks` mode this bypasses
        /// the read lock; lock-checked access goes through `view()`.
        #[inline]
        pub(crate) unsafe fn raw_ref(&self) -> &ArraysD {
            #[cfg(not(feature = "safe-locks"))]
            unsafe {
                &*self.inner.get()
            }
            #[cfg(feature = "safe-locks")]
            unsafe {
                &*self.inner.data_ptr()
            }
        }

        /// `&mut self`-rooted access to the inner array — statically race-free.
        #[inline]
        pub(crate) fn raw_mut(&mut self) -> &mut ArraysD {
            self.inner.get_mut()
        }
    }

    impl Constructor for PyNdArray {
        type Args = FuncArgs;
        fn py_new(_cls: &Py<PyType>, args: FuncArgs, vm: &VirtualMachine) -> PyResult<Self> {
            let arr = match args.args.into_iter().next() {
                None => crate::create::zeros(&[0], DType::F64),
                Some(o) => obj_to_array(&o, None, vm)?,
            };
            Ok(PyNdArray::from_arrays(arr))
        }
    }

    impl Representable for PyNdArray {
        #[inline]
        fn repr_str(zelf: &Py<Self>, _vm: &VirtualMachine) -> PyResult<String> {
            Ok(repr_fmt::repr(&zelf.view()))
        }
    }

    impl Comparable for PyNdArray {
        fn slot_richcompare(
            zelf: &PyObject,
            other: &PyObject,
            op: PyComparisonOp,
            vm: &VirtualMachine,
        ) -> PyResult<Either<PyObjectRef, PyComparisonValue>> {
            let z = zelf
                .downcast_ref::<PyNdArray>()
                .ok_or_else(|| vm.new_type_error("comparison: unexpected payload".to_string()))?;
            let rhs = match obj_to_array(&other.to_owned(), None, vm) {
                Ok(v) => v,
                Err(_) => {
                    return Ok(Either::B(PyComparisonValue::NotImplemented));
                }
            };
            let cmp = match op {
                PyComparisonOp::Eq => CmpOp::Eq,
                PyComparisonOp::Ne => CmpOp::Ne,
                PyComparisonOp::Lt => CmpOp::Lt,
                PyComparisonOp::Le => CmpOp::Le,
                PyComparisonOp::Gt => CmpOp::Gt,
                PyComparisonOp::Ge => CmpOp::Ge,
            };
            let res = ops::compare(&z.view(), &rhs, cmp, vm)?;
            Ok(Either::A(PyNdArray::from_arrays(res).into_pyobject(vm)))
        }

        fn cmp(
            _zelf: &Py<Self>,
            _other: &PyObject,
            _op: PyComparisonOp,
            _vm: &VirtualMachine,
        ) -> PyResult<PyComparisonValue> {
            // Unreachable: slot_richcompare is overridden above.
            Ok(PyComparisonValue::NotImplemented)
        }
    }

    impl Iterable for PyNdArray {
        /// Iterate over the first axis, matching numpy. 1-D arrays yield
        /// scalars; n-D (n ≥ 2) arrays yield (n-1)-D sub-arrays. 0-D arrays
        /// raise ``TypeError``, again matching numpy.
        fn iter(zelf: rustpython_vm::PyRef<Self>, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
            let nd = zelf.view().ndim();
            if nd == 0 {
                return Err(vm.new_type_error("iteration over a 0-d array".to_string()));
            }
            let n = zelf.view().shape()[0];
            let mut items: Vec<PyObjectRef> = Vec::with_capacity(n);
            for i in 0..n {
                let key: PyObjectRef = vm.ctx.new_int(i as isize).into();
                let parsed = index::parse_index(&key, nd, vm)?;
                let sub = index::apply_index(&zelf.view(), &parsed, vm)?;
                items.push(scalar_or_array(sub, vm));
            }
            let list: PyObjectRef = vm.ctx.new_list(items).into();
            Ok(list.get_iter(vm)?.into())
        }
    }

    impl AsMapping for PyNdArray {
        fn as_mapping() -> &'static PyMappingMethods {
            static M: PyMappingMethods = PyMappingMethods {
                length: Some(|m, _vm| {
                    let z = PyNdArray::mapping_downcast(m);
                    Ok(if z.view().ndim() == 0 {
                        0
                    } else {
                        z.view().shape()[0]
                    })
                }),
                subscript: Some(|m, key, vm| {
                    let z = PyNdArray::mapping_downcast(m);
                    let parsed = index::parse_index(&key.to_owned(), z.view().ndim(), vm)?;
                    let sub = index::apply_index(&z.view(), &parsed, vm)?;
                    Ok(scalar_or_array(sub, vm))
                }),
                ass_subscript: Some(|m, key, value, vm| {
                    let z = PyNdArray::mapping_downcast(m);
                    let value = match value {
                        Some(v) => v,
                        None => {
                            return Err(
                                vm.new_type_error("cannot delete ndarray elements".to_string())
                            );
                        }
                    };
                    // Hint with the destination dtype so non-numeric assignments
                    // (object/str/bytes/datetime) take their tailored conversion
                    // path instead of failing in the generic obj_to_array.
                    let target_dt = z.view().dtype();
                    let v_arr = obj_to_array(&value, Some(target_dt), vm)?;
                    let parsed = index::parse_index(&key.to_owned(), z.view().ndim(), vm)?;
                    let mut inner = z.view_mut();
                    index::set_via_index(&mut inner, &parsed, &v_arr, vm)
                }),
            };
            &M
        }
    }

    impl AsNumber for PyNdArray {
        fn as_number() -> &'static PyNumberMethods {
            static N: PyNumberMethods = PyNumberMethods {
                add: Some(|a, b, vm| {
                    binary_slot(a, b, vm, |x, y, vm| ops::binary_op(x, y, vm, ops::Add))
                }),
                subtract: Some(|a, b, vm| {
                    binary_slot(a, b, vm, |x, y, vm| ops::binary_op(x, y, vm, ops::Sub))
                }),
                multiply: Some(|a, b, vm| {
                    binary_slot(a, b, vm, |x, y, vm| ops::binary_op(x, y, vm, ops::Mul))
                }),
                true_divide: Some(|a, b, vm| binary_slot(a, b, vm, ops::true_divide)),
                floor_divide: Some(|a, b, vm| binary_slot(a, b, vm, ops::floor_divide)),
                remainder: Some(|a, b, vm| binary_slot(a, b, vm, ops::remainder)),
                power: Some(|a, b, _modulus, vm| binary_slot(a, b, vm, ops::power)),
                matrix_multiply: Some(|a, b, vm| binary_slot(a, b, vm, linalg::dot)),
                and: Some(|a, b, vm| binary_slot(a, b, vm, crate::extras::bitwise_and)),
                or: Some(|a, b, vm| binary_slot(a, b, vm, crate::extras::bitwise_or)),
                xor: Some(|a, b, vm| binary_slot(a, b, vm, crate::extras::bitwise_xor)),
                // ----- inplace ops (numpy semantics: mutate lhs, return lhs) -----
                inplace_add: Some(|a, b, vm| {
                    inplace_slot(a, b, vm, |x, y, vm| ops::binary_op(x, y, vm, ops::Add))
                }),
                inplace_subtract: Some(|a, b, vm| {
                    inplace_slot(a, b, vm, |x, y, vm| ops::binary_op(x, y, vm, ops::Sub))
                }),
                inplace_multiply: Some(|a, b, vm| {
                    inplace_slot(a, b, vm, |x, y, vm| ops::binary_op(x, y, vm, ops::Mul))
                }),
                inplace_true_divide: Some(|a, b, vm| inplace_slot(a, b, vm, ops::true_divide)),
                inplace_floor_divide: Some(|a, b, vm| inplace_slot(a, b, vm, ops::floor_divide)),
                inplace_remainder: Some(|a, b, vm| inplace_slot(a, b, vm, ops::remainder)),
                inplace_power: Some(|a, b, _modulus, vm| inplace_slot(a, b, vm, ops::power)),
                inplace_matrix_multiply: Some(|a, b, vm| inplace_slot(a, b, vm, linalg::dot)),
                inplace_and: Some(|a, b, vm| inplace_slot(a, b, vm, crate::extras::bitwise_and)),
                inplace_or: Some(|a, b, vm| inplace_slot(a, b, vm, crate::extras::bitwise_or)),
                inplace_xor: Some(|a, b, vm| inplace_slot(a, b, vm, crate::extras::bitwise_xor)),
                invert: Some(|num, vm| {
                    let z = PyNdArray::number_downcast(num);
                    Ok(
                        PyNdArray::from_arrays(crate::extras::invert(&z.view(), vm)?)
                            .into_pyobject(vm),
                    )
                }),
                negative: Some(|num, vm| {
                    let z = PyNdArray::number_downcast(num);
                    Ok(PyNdArray::from_arrays(ops::negate(&z.view(), vm)?).into_pyobject(vm))
                }),
                positive: Some(|num, vm| {
                    let z = PyNdArray::number_downcast(num);
                    Ok(PyNdArray::from_arrays(z.view().clone()).into_pyobject(vm))
                }),
                absolute: Some(|num, vm| {
                    let z = PyNdArray::number_downcast(num);
                    Ok(PyNdArray::from_arrays(ops::absolute(&z.view())).into_pyobject(vm))
                }),
                float: Some(|num, vm| {
                    let z = PyNdArray::number_downcast(num);
                    if z.view().len() != 1 {
                        return Err(vm.new_type_error(format!(
                            "only size-1 arrays can be converted to float; got shape {:?}",
                            z.view().shape()
                        )));
                    }
                    let v = match z.view().cast(DType::F64) {
                        ArraysD::F64(x) => x.iter().next().copied().unwrap_or(0.0),
                        _ => return Err(crate::internal::internal(vm, "__float__: cast failed")),
                    };
                    Ok(vm.ctx.new_float(v).into())
                }),
                int: Some(|num, vm| {
                    let z = PyNdArray::number_downcast(num);
                    if z.view().len() != 1 {
                        return Err(vm.new_type_error(format!(
                            "only size-1 arrays can be converted to int; got shape {:?}",
                            z.view().shape()
                        )));
                    }
                    let v = match z.view().cast(DType::I64) {
                        ArraysD::I64(x) => x.iter().next().copied().unwrap_or(0),
                        _ => return Err(crate::internal::internal(vm, "__int__: cast failed")),
                    };
                    Ok(vm.ctx.new_int(v).into())
                }),
                boolean: Some(|num, vm| {
                    let z = PyNdArray::number_downcast(num);
                    if z.view().len() != 1 {
                        return Err(vm.new_value_error(format!(
                            "the truth value of an array with more than one element is ambiguous; got shape {:?}",
                            z.view().shape()
                        )));
                    }
                    let v = match z.view().cast(DType::Bool) {
                        ArraysD::Bool(x) => x.iter().next().copied().unwrap_or(false),
                        _ => return Err(crate::internal::internal(vm, "__bool__: cast failed")),
                    };
                    Ok(v)
                }),
                ..PyNumberMethods::NOT_IMPLEMENTED
            };
            &N
        }
    }

    fn binary_slot(
        a: &PyObject,
        b: &PyObject,
        vm: &VirtualMachine,
        f: impl FnOnce(&ArraysD, &ArraysD, &VirtualMachine) -> PyResult<ArraysD>,
    ) -> PyResult {
        let ax = obj_to_array(&a.to_owned(), None, vm);
        let bx = obj_to_array(&b.to_owned(), None, vm);
        match (ax, bx) {
            (Ok(x), Ok(y)) => Ok(PyNdArray::from_arrays(f(&x, &y, vm)?).into_pyobject(vm)),
            _ => Ok(vm.ctx.not_implemented()),
        }
    }

    /// `__i<op>__` slot: mutate the lhs ndarray in place and return it.
    /// Falls back to NotImplemented if either argument can't be coerced.
    fn inplace_slot(
        a: &PyObject,
        b: &PyObject,
        vm: &VirtualMachine,
        f: impl FnOnce(&ArraysD, &ArraysD, &VirtualMachine) -> PyResult<ArraysD>,
    ) -> PyResult {
        let Some(lhs) = a.downcast_ref::<PyNdArray>() else {
            return Ok(vm.ctx.not_implemented());
        };
        let bx = match obj_to_array(&b.to_owned(), None, vm) {
            Ok(v) => v,
            Err(_) => return Ok(vm.ctx.not_implemented()),
        };
        // Compute against a borrow of lhs, drop the borrow before mutating.
        let result = {
            let view = lhs.view();
            f(&view, &bx, vm)?
        };
        // numpy semantics: an inplace op preserves the lhs dtype — coerce
        // the result back into the original dtype.
        let dst_dtype = lhs.view().dtype();
        let result = if result.dtype() != dst_dtype {
            result.cast(dst_dtype)
        } else {
            result
        };
        *lhs.view_mut() = result;
        Ok(a.to_owned())
    }

    /// Wrap an `ArraysD` result, unwrapping 0-D into a Python scalar.
    fn scalar_or_array(arr: ArraysD, vm: &VirtualMachine) -> PyObjectRef {
        if arr.ndim() == 0 {
            return array_to_pylist(&arr, vm);
        }
        PyNdArray::from_arrays(arr).into_pyobject(vm)
    }

    #[pyclass(
        with(Constructor, Representable, AsMapping, AsNumber, Comparable, Iterable),
        flags(BASETYPE)
    )]
    impl PyNdArray {
        // ---- attributes ----
        #[pygetset]
        fn shape(&self, vm: &VirtualMachine) -> PyObjectRef {
            let items: Vec<PyObjectRef> = self
                .view()
                .shape()
                .iter()
                .map(|&d| vm.ctx.new_int(d).into())
                .collect();
            PyTuple::new_ref(items, &vm.ctx).into()
        }

        #[pygetset]
        fn ndim(&self) -> usize {
            self.view().ndim()
        }

        #[pygetset]
        fn size(&self) -> usize {
            self.view().len()
        }

        #[pygetset]
        fn nbytes(&self) -> usize {
            self.view().nbytes()
        }

        #[pygetset]
        fn itemsize(&self) -> usize {
            self.view().dtype().itemsize()
        }

        #[pygetset]
        fn dtype(&self, vm: &VirtualMachine) -> PyObjectRef {
            PyDType::from_dtype(self.view().dtype()).into_pyobject(vm)
        }

        #[pygetset(name = "T")]
        fn transpose_attr(&self) -> PyNdArray {
            PyNdArray::from_arrays(linalg::transpose(&self.view()))
        }

        #[pygetset(name = "real")]
        fn real(&self) -> PyNdArray {
            PyNdArray::from_arrays(ops::real_part(&self.view()))
        }

        #[pygetset(name = "imag")]
        fn imag(&self) -> PyNdArray {
            PyNdArray::from_arrays(ops::imag_part(&self.view()))
        }

        /// `__array_interface__` — numpy's interop protocol (v3). Returns a
        /// dict with `shape`, `typestr`, `data` (None — rumpy doesn't expose
        /// a stable buffer pointer to embedded Python), `strides` (None for
        /// contiguous arrays) and `version`. External libraries that read
        /// `__array_interface__` use the `typestr` + `shape` keys to
        /// understand the dtype layout.
        #[pygetset(name = "__array_interface__")]
        fn array_interface(&self, vm: &VirtualMachine) -> PyObjectRef {
            let arr = self.view();
            let dt = arr.dtype();
            // numpy's "typestr": kind code + itemsize, prefixed with byteorder.
            // Use '|' for byte-order-insensitive (1-byte / object / string),
            // otherwise '<' for little-endian (matches every supported host).
            let kind = dt.kind();
            let typestr = match dt {
                DType::Object => "|O".to_string(),
                DType::Bool => "|b1".to_string(),
                DType::I8 | DType::U8 => format!("|{kind}1"),
                DType::Datetime64(u) => format!("<M8[{}]", u.code()),
                DType::Timedelta64(u) => format!("<m8[{}]", u.code()),
                DType::Str(n) => format!("<U{n}"),
                DType::Bytes(n) => format!("|S{n}"),
                DType::Void(n) => format!("|V{n}"),
                _ => format!("<{kind}{}", dt.itemsize()),
            };
            let dict = vm.ctx.new_dict();
            let _ = dict.set_item("version", vm.ctx.new_int(3).into(), vm);
            let _ = dict.set_item("typestr", vm.ctx.new_str(typestr).into(), vm);
            let shape: Vec<PyObjectRef> = arr
                .shape()
                .iter()
                .map(|&n| vm.ctx.new_int(n as i64).into())
                .collect();
            let _ = dict.set_item("shape", PyTuple::new_ref(shape, &vm.ctx).into(), vm);
            let _ = dict.set_item("data", vm.ctx.none(), vm);
            let _ = dict.set_item("strides", vm.ctx.none(), vm);
            dict.into()
        }

        /// `__array__()` — return self. Lets other libraries that call
        /// `np.asarray(x)` see a rumpy ndarray as-is.
        #[pymethod(name = "__array__")]
        fn array_protocol(
            zelf: rustpython_vm::PyRef<Self>,
            _args: rustpython_vm::function::FuncArgs,
            _vm: &VirtualMachine,
        ) -> rustpython_vm::PyRef<Self> {
            zelf
        }

        // ---- conversion ----
        #[pymethod]
        fn tolist(&self, vm: &VirtualMachine) -> PyObjectRef {
            array_to_pylist(&self.view(), vm)
        }

        #[pymethod]
        fn astype(&self, dtype: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
            let dt = parse_dtype_arg(&Some(dtype), vm)?
                .ok_or_else(|| vm.new_type_error("dtype required".to_string()))?;
            Ok(PyNdArray::from_arrays(self.view().cast(dt)))
        }

        #[pymethod]
        fn conj(&self) -> PyNdArray {
            PyNdArray::from_arrays(ops::conj(&self.view()))
        }

        #[pymethod]
        fn conjugate(&self) -> PyNdArray {
            self.conj()
        }

        // ---- shape ops ----
        #[pymethod]
        fn reshape(&self, args: FuncArgs, vm: &VirtualMachine) -> PyResult<PyNdArray> {
            let shape_signed = parse_shape_from_args(&args, vm)?;
            let total = self.view().len();
            let resolved = resolve_neg_one(&shape_signed, total, vm)?;
            let prod: usize = resolved.iter().product();
            if prod != total {
                return Err(vm.new_value_error(format!(
                    "cannot reshape array of size {total} into shape {resolved:?}"
                )));
            }
            let res = linalg::reshape(&self.view(), &resolved)
                .ok_or_else(|| vm.new_value_error("reshape failed".to_string()))?;
            Ok(PyNdArray::from_arrays(res))
        }

        #[pymethod]
        fn transpose(&self) -> PyNdArray {
            PyNdArray::from_arrays(linalg::transpose(&self.view()))
        }

        #[pymethod]
        fn flatten(&self) -> PyNdArray {
            PyNdArray::from_arrays(linalg::flatten(&self.view()))
        }

        #[pymethod]
        fn ravel(&self) -> PyNdArray {
            self.flatten()
        }

        #[pymethod]
        fn copy(&self) -> PyNdArray {
            PyNdArray::from_arrays(self.view().clone())
        }

        // ---- reductions ----
        #[pymethod]
        fn sum(&self, args: ReduceArgs, vm: &VirtualMachine) -> PyResult {
            let r = do_reduce(&self.view(), args, Reduce::Sum, vm)?;
            Ok(scalar_or_array(r, vm))
        }

        #[pymethod]
        fn prod(&self, args: ReduceArgs, vm: &VirtualMachine) -> PyResult {
            let r = do_reduce(&self.view(), args, Reduce::Prod, vm)?;
            Ok(scalar_or_array(r, vm))
        }

        #[pymethod]
        fn mean(&self, args: ReduceArgs, vm: &VirtualMachine) -> PyResult {
            let r = do_reduce(&self.view(), args, Reduce::Mean, vm)?;
            Ok(scalar_or_array(r, vm))
        }

        #[pymethod]
        fn min(&self, args: ReduceArgs, vm: &VirtualMachine) -> PyResult {
            let r = do_reduce(&self.view(), args, Reduce::Min, vm)?;
            Ok(scalar_or_array(r, vm))
        }

        #[pymethod]
        fn max(&self, args: ReduceArgs, vm: &VirtualMachine) -> PyResult {
            let r = do_reduce(&self.view(), args, Reduce::Max, vm)?;
            Ok(scalar_or_array(r, vm))
        }

        #[pymethod]
        fn argmin(&self, vm: &VirtualMachine) -> PyResult<usize> {
            reduce::arg_extremum(&self.view(), false, vm)
        }

        #[pymethod]
        fn argmax(&self, vm: &VirtualMachine) -> PyResult<usize> {
            reduce::arg_extremum(&self.view(), true, vm)
        }

        #[pymethod]
        fn std(&self, args: VarianceArg, vm: &VirtualMachine) -> PyResult {
            let ddof = args.ddof.unwrap_or(0);
            let axes = parse_axes(&args.axis, vm)?;
            let keepdims = args.keepdims.unwrap_or(false);
            let r = reduce::reduce_multi(
                &self.view(),
                axes.as_deref(),
                keepdims,
                Reduce::Std(ddof),
                vm,
            )?;
            Ok(scalar_or_array(r, vm))
        }

        #[pymethod]
        fn var(&self, args: VarianceArg, vm: &VirtualMachine) -> PyResult {
            let ddof = args.ddof.unwrap_or(0);
            let axes = parse_axes(&args.axis, vm)?;
            let keepdims = args.keepdims.unwrap_or(false);
            let r = reduce::reduce_multi(
                &self.view(),
                axes.as_deref(),
                keepdims,
                Reduce::Var(ddof),
                vm,
            )?;
            Ok(scalar_or_array(r, vm))
        }

        #[pymethod]
        fn dot(&self, other: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
            let b = obj_to_array(&other, None, vm)?;
            Ok(PyNdArray::from_arrays(linalg::dot(&self.view(), &b, vm)?))
        }

        // ---- shape manipulation methods ----

        #[pymethod]
        fn squeeze(&self, vm: &VirtualMachine) -> PyResult<PyNdArray> {
            Ok(PyNdArray::from_arrays(crate::extras::squeeze(
                &self.view(),
                vm,
            )?))
        }

        #[pymethod(name = "swapaxes")]
        fn method_swapaxes(
            &self,
            axis1: isize,
            axis2: isize,
            vm: &VirtualMachine,
        ) -> PyResult<PyNdArray> {
            let nd = self.view().ndim();
            let n1 = normalize_axis_arg(axis1, nd, vm)?;
            let n2 = normalize_axis_arg(axis2, nd, vm)?;
            let mut perm: Vec<usize> = (0..nd).collect();
            perm.swap(n1, n2);
            Ok(PyNdArray::from_arrays(transpose_with_perm(
                &self.view(),
                &perm,
            )))
        }

        #[pymethod]
        fn diagonal(&self, k: OptionalArg<isize>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
            Ok(PyNdArray::from_arrays(crate::more_ops::diag(
                &self.view(),
                k.unwrap_or(0),
                vm,
            )?))
        }

        #[pymethod(name = "trace")]
        fn method_trace(&self, vm: &VirtualMachine) -> PyResult<PyNdArray> {
            Ok(PyNdArray::from_arrays(crate::linalg_extra::trace(
                &self.view(),
                vm,
            )?))
        }

        // ---- elementwise methods ----

        #[pymethod]
        fn clip(
            &self,
            min: OptionalArg<PyObjectRef>,
            max: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<PyNdArray> {
            use crate::dtype::CoerceArray;
            let arr = self.view().clone();
            let f = arr.coerce::<f64>();
            let lo = match min {
                OptionalArg::Missing => None,
                OptionalArg::Present(o) if o.is(&vm.ctx.none) => None,
                OptionalArg::Present(o) => Some(o.try_float(vm)?.to_f64()),
            };
            let hi = match max {
                OptionalArg::Missing => None,
                OptionalArg::Present(o) if o.is(&vm.ctx.none) => None,
                OptionalArg::Present(o) => Some(o.try_float(vm)?.to_f64()),
            };
            let mapped = f.mapv(|x| {
                let mut v = x;
                if let Some(l) = lo {
                    if v < l {
                        v = l;
                    }
                }
                if let Some(h) = hi {
                    if v > h {
                        v = h;
                    }
                }
                v
            });
            Ok(PyNdArray::from_arrays(
                ArraysD::F64(mapped).cast(arr.dtype()),
            ))
        }

        #[pymethod]
        fn round(&self) -> PyNdArray {
            use crate::dtype::CoerceArray;
            let arr = self.view().clone();
            let f = arr.coerce::<f64>();
            // numpy uses round-half-to-even (banker's rounding).
            let mapped = f.mapv(|x| x.round_ties_even());
            PyNdArray::from_arrays(ArraysD::F64(mapped).cast(arr.dtype()))
        }

        // ---- cumulative reductions ----

        #[pymethod]
        fn cumsum(&self, args: AxisArg, vm: &VirtualMachine) -> PyResult<PyNdArray> {
            Ok(PyNdArray::from_arrays(crate::extras::cumsum_axis(
                &self.view(),
                args.axis.flatten(),
                vm,
            )?))
        }
        #[pymethod]
        fn cumprod(&self, args: AxisArg, vm: &VirtualMachine) -> PyResult<PyNdArray> {
            Ok(PyNdArray::from_arrays(crate::extras::cumprod_axis(
                &self.view(),
                args.axis.flatten(),
                vm,
            )?))
        }

        #[pymethod]
        fn ptp(&self, args: ReduceArgs, vm: &VirtualMachine) -> PyResult {
            let max = do_reduce(
                &self.view(),
                ReduceArgs {
                    axis: args.axis.clone(),
                    keepdims: args.keepdims,
                },
                Reduce::Max,
                vm,
            )?;
            let min = do_reduce(
                &self.view(),
                ReduceArgs {
                    axis: args.axis,
                    keepdims: args.keepdims,
                },
                Reduce::Min,
                vm,
            )?;
            let r = ops::binary_op(&max, &min, vm, ops::Sub)?;
            Ok(scalar_or_array(r, vm))
        }

        // ---- logical reductions ----

        #[pymethod]
        fn any(&self, args: ReduceArgs, vm: &VirtualMachine) -> PyResult {
            let r = any_all_kw(&self.view(), args, false, vm)?;
            Ok(scalar_or_array(r, vm))
        }
        #[pymethod]
        fn all(&self, args: ReduceArgs, vm: &VirtualMachine) -> PyResult {
            let r = any_all_kw(&self.view(), args, true, vm)?;
            Ok(scalar_or_array(r, vm))
        }

        // ---- indexing helpers ----

        #[pymethod]
        fn nonzero(&self) -> PyNdArray {
            PyNdArray::from_arrays(crate::extras::nonzero(&self.view()))
        }

        #[pymethod]
        fn sort(&self, axis: OptionalArg<isize>, vm: &VirtualMachine) -> PyResult<()> {
            // In-place sort along axis.
            let sorted = crate::extras::sort(&self.view(), Some(axis.unwrap_or(-1)), vm)?;
            *self.view_mut() = sorted;
            Ok(())
        }

        #[pymethod]
        fn argsort(&self, axis: OptionalArg<isize>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
            Ok(PyNdArray::from_arrays(crate::extras::argsort(
                &self.view(),
                Some(axis.unwrap_or(-1)),
                vm,
            )?))
        }

        #[pymethod(name = "searchsorted")]
        fn method_searchsorted(&self, v: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
            let other = obj_to_array(&v, None, vm)?;
            Ok(PyNdArray::from_arrays(crate::more_ops::searchsorted(
                &self.view(),
                &other,
            )))
        }

        #[pymethod(name = "repeat")]
        fn method_repeat(&self, n: usize) -> PyNdArray {
            PyNdArray::from_arrays(crate::extras::repeat(&self.view(), n))
        }

        #[pymethod(name = "tile")]
        fn method_tile(&self, n: usize) -> PyNdArray {
            PyNdArray::from_arrays(crate::extras::tile(&self.view(), n))
        }

        #[pymethod]
        fn take(&self, indices: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
            // Treat as fancy index along axis 0.
            let idx = obj_to_array(&indices, None, vm)?;
            use crate::dtype::CoerceArray;
            let idx_i: Vec<isize> = idx.coerce::<i64>().iter().map(|&v| v as isize).collect();
            let flat = if self.view().ndim() == 1 {
                self.view().clone()
            } else {
                crate::linalg::flatten(&self.view())
            };
            let mut parts: Vec<ArraysD> = Vec::with_capacity(idx_i.len());
            for i in &idx_i {
                let n = if *i < 0 {
                    (i + flat.len() as isize) as usize
                } else {
                    *i as usize
                };
                let p = index::apply_index(&flat, &[index::IdxItem::Int(n as isize)], vm)?;
                parts.push(p);
            }
            crate::extras::stack(&parts, 0, vm).map(PyNdArray::from_arrays)
        }

        // ---- scalar / buffer access ----

        #[pymethod]
        fn fill(&self, value: PyObjectRef, vm: &VirtualMachine) -> PyResult<()> {
            // Fill every element with `value` (cast to the dtype).
            let v = obj_to_array(&value, None, vm)?;
            let dst_dtype = self.view().dtype();
            let v = v.cast(dst_dtype);
            // Broadcast a 0-D scalar to the array's shape, then assign.
            let target_shape = self.view().shape().to_vec();
            let broadcast = if v.ndim() == 0 {
                crate::extras::broadcast_to(&v, &target_shape, vm)?
            } else {
                v
            };
            *self.view_mut() = broadcast;
            Ok(())
        }

        #[pymethod]
        fn item(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
            if self.view().len() != 1 {
                return Err(vm.new_value_error(format!(
                    "can only convert an array of size 1 to a Python scalar; got shape {:?}",
                    self.view().shape()
                )));
            }
            // Pick the best scalar type for the dtype.
            let dt = self.view().dtype();
            if dt == DType::Bool {
                use crate::dtype::CoerceArray;
                let v = self.view().coerce::<bool>();
                let b = v.iter().next().copied().unwrap_or(false);
                return Ok(vm.ctx.new_bool(b).into());
            }
            if dt.is_integer() {
                use crate::dtype::CoerceArray;
                let v = self.view().coerce::<i64>();
                let i = v.iter().next().copied().unwrap_or(0);
                return Ok(vm.ctx.new_int(i).into());
            }
            use crate::dtype::CoerceArray;
            let v = self.view().coerce::<f64>();
            let f = v.iter().next().copied().unwrap_or(0.0);
            Ok(vm.ctx.new_float(f).into())
        }

        #[pymethod]
        fn tobytes(&self, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
            let view = self.view();
            let bytes: Vec<u8> = match &*view {
                ArraysD::Bool(arr) => arr.iter().map(|&b| if b { 1u8 } else { 0u8 }).collect(),
                ArraysD::I8(arr) => arr.iter().flat_map(|v| v.to_ne_bytes()).collect(),
                ArraysD::I16(arr) => arr.iter().flat_map(|v| v.to_ne_bytes()).collect(),
                ArraysD::I32(arr) => arr.iter().flat_map(|v| v.to_ne_bytes()).collect(),
                ArraysD::I64(arr) => arr.iter().flat_map(|v| v.to_ne_bytes()).collect(),
                ArraysD::U8(arr) => arr.iter().copied().collect(),
                ArraysD::U16(arr) => arr.iter().flat_map(|v| v.to_ne_bytes()).collect(),
                ArraysD::U32(arr) => arr.iter().flat_map(|v| v.to_ne_bytes()).collect(),
                ArraysD::U64(arr) => arr.iter().flat_map(|v| v.to_ne_bytes()).collect(),
                ArraysD::F16(arr) => arr.iter().flat_map(|v| v.to_bits().to_ne_bytes()).collect(),
                ArraysD::F32(arr) => arr.iter().flat_map(|v| v.to_bits().to_ne_bytes()).collect(),
                ArraysD::F64(arr) => arr.iter().flat_map(|v| v.to_bits().to_ne_bytes()).collect(),
                ArraysD::C64(arr) => arr
                    .iter()
                    .flat_map(|c| {
                        c.re.to_bits()
                            .to_ne_bytes()
                            .into_iter()
                            .chain(c.im.to_bits().to_ne_bytes())
                    })
                    .collect(),
                ArraysD::C128(arr) => arr
                    .iter()
                    .flat_map(|c| {
                        c.re.to_bits()
                            .to_ne_bytes()
                            .into_iter()
                            .chain(c.im.to_bits().to_ne_bytes())
                    })
                    .collect(),
                _ => {
                    return Err(crate::internal::unsupported_dtype(
                        vm,
                        "tobytes",
                        view.dtype(),
                    ));
                }
            };
            Ok(vm.ctx.new_bytes(bytes).into())
        }

        #[pymethod(name = "view")]
        fn method_view(&self) -> PyNdArray {
            // We don't yet have true views; return a copy.
            PyNdArray::from_arrays(self.view().clone())
        }

        #[pygetset]
        fn flat(&self) -> PyNdArray {
            // numpy `.flat` is a 1-D iterator object; we return a flat copy as
            // a 1-D ndarray which supports indexing/iteration.
            PyNdArray::from_arrays(crate::linalg::flatten(&self.view()))
        }
    }

    // -----------------------------------------------------------------
    // Module-level functions
    // -----------------------------------------------------------------

    #[pyfunction]
    fn array(obj: PyObjectRef, dtype: DTypeArg, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let dt = parse_dtype_arg(&dtype.dtype.into_option(), vm)?;
        Ok(PyNdArray::from_arrays(obj_to_array(&obj, dt, vm)?))
    }

    #[pyfunction]
    fn asarray(obj: PyObjectRef, dtype: DTypeArg, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        array(obj, dtype, vm)
    }

    #[pyfunction]
    fn zeros(shape: PyObjectRef, dtype: DTypeArg, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let s = parse_shape(&shape, vm)?;
        let dt = parse_dtype_arg(&dtype.dtype.into_option(), vm)?.unwrap_or(DType::F64);
        Ok(PyNdArray::from_arrays(create::zeros(&s, dt)))
    }

    #[pyfunction]
    fn ones(shape: PyObjectRef, dtype: DTypeArg, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let s = parse_shape(&shape, vm)?;
        let dt = parse_dtype_arg(&dtype.dtype.into_option(), vm)?.unwrap_or(DType::F64);
        Ok(PyNdArray::from_arrays(create::ones(&s, dt)))
    }

    #[pyfunction]
    fn full(args: FullArgs, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let s = parse_shape(&args.shape, vm)?;
        let v: f64 = args.fill_value.into();
        let dt = parse_dtype_arg(&args.dtype.into_option(), vm)?.unwrap_or(DType::F64);
        Ok(PyNdArray::from_arrays(create::full_f64(&s, v, dt)))
    }

    #[pyfunction]
    fn empty(shape: PyObjectRef, dtype: DTypeArg, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        zeros(shape, dtype, vm)
    }

    #[derive(FromArgs)]
    pub(crate) struct LikeArgs {
        #[pyarg(any, optional)]
        dtype: OptionalArg<PyObjectRef>,
        #[pyarg(any, optional)]
        shape: OptionalArg<PyObjectRef>,
    }

    #[pyfunction]
    fn zeros_like(a: PyObjectRef, args: LikeArgs, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        let dt = parse_dtype_arg(&args.dtype.into_option(), vm)?.unwrap_or_else(|| arr.dtype());
        let shape = match args.shape {
            OptionalArg::Missing => arr.shape().to_vec(),
            OptionalArg::Present(o) if o.is(&vm.ctx.none) => arr.shape().to_vec(),
            OptionalArg::Present(o) => parse_shape(&o, vm)?,
        };
        Ok(PyNdArray::from_arrays(create::zeros(&shape, dt)))
    }

    #[pyfunction]
    fn ones_like(a: PyObjectRef, args: LikeArgs, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        let dt = parse_dtype_arg(&args.dtype.into_option(), vm)?.unwrap_or_else(|| arr.dtype());
        let shape = match args.shape {
            OptionalArg::Missing => arr.shape().to_vec(),
            OptionalArg::Present(o) if o.is(&vm.ctx.none) => arr.shape().to_vec(),
            OptionalArg::Present(o) => parse_shape(&o, vm)?,
        };
        Ok(PyNdArray::from_arrays(create::ones(&shape, dt)))
    }

    #[pyfunction]
    fn arange(mut args: FuncArgs, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        // dtype= kwarg, popped manually so the remaining positional list maps
        // cleanly onto numpy's (start, stop, step) signature.
        let dtype_obj = args.take_keyword("dtype");
        if !args.kwargs.is_empty() {
            let keys: Vec<&str> = args.kwargs.keys().map(|s| s.as_str()).collect();
            return Err(vm.new_type_error(format!(
                "arange() got unexpected keyword arguments: {keys:?}"
            )));
        }
        let dt_user = parse_dtype_arg(&dtype_obj, vm)?;
        // Record whether *any* positional was a Python float (vs. int/bool).
        // numpy uses this to default to float64 even when all values are
        // integer-valued (e.g., `np.arange(0.0, 10.0, 2.0)` → float64).
        let any_float = args.args.iter().any(|o| {
            o.downcast_ref::<rustpython_vm::builtins::PyFloat>()
                .is_some()
        });
        let positional: Vec<f64> = args
            .args
            .iter()
            .map(|o| o.try_float(vm).map(|f| f.to_f64()))
            .collect::<PyResult<Vec<_>>>()?;
        let (start, stop, step) = match positional.len() {
            1 => (0.0, positional[0], 1.0),
            2 => (positional[0], positional[1], 1.0),
            3 => (positional[0], positional[1], positional[2]),
            _ => {
                return Err(
                    vm.new_type_error("arange() requires 1 to 3 numeric arguments".to_string())
                );
            }
        };
        if step == 0.0 {
            return Err(vm.new_value_error("arange() arg 3 must not be zero".to_string()));
        }
        // Pass an explicit float dtype if the caller used float literals; the
        // helper otherwise infers from value-fractionality, which would
        // collapse `arange(0.0, 10.0, 2.0)` to int64.
        let dt = match dt_user {
            Some(d) => Some(d),
            None if any_float => Some(DType::F64),
            None => None,
        };
        Ok(PyNdArray::from_arrays(create::arange(
            start, stop, step, dt,
        )))
    }

    #[derive(FromArgs)]
    pub(crate) struct LinspaceArgs {
        #[pyarg(positional)]
        start: ArgIntoFloat,
        #[pyarg(positional)]
        stop: ArgIntoFloat,
        #[pyarg(any, optional)]
        num: OptionalArg<usize>,
        #[pyarg(any, optional)]
        endpoint: OptionalArg<bool>,
        #[pyarg(any, optional)]
        retstep: OptionalArg<bool>,
        #[pyarg(any, optional)]
        dtype: OptionalArg<PyObjectRef>,
    }
    #[pyfunction]
    fn linspace(args: LinspaceArgs, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let num = args.num.unwrap_or(50);
        let endpoint = args.endpoint.unwrap_or(true);
        let retstep = args.retstep.unwrap_or(false);
        let dt = parse_dtype_arg(&args.dtype.into_option(), vm)?;
        let (arr, step) =
            crate::extras2::linspace_full(args.start.into(), args.stop.into(), num, endpoint);
        let arr = match dt {
            Some(d) => arr.cast(d),
            None => arr,
        };
        let py_arr = PyNdArray::from_arrays(arr).into_pyobject(vm);
        if retstep {
            let tup = PyTuple::new_ref(vec![py_arr, vm.ctx.new_float(step).into()], &vm.ctx);
            Ok(tup.into())
        } else {
            Ok(py_arr)
        }
    }

    #[derive(FromArgs)]
    pub(crate) struct LogspaceArgs {
        #[pyarg(positional)]
        start: ArgIntoFloat,
        #[pyarg(positional)]
        stop: ArgIntoFloat,
        #[pyarg(any, optional)]
        num: OptionalArg<usize>,
        #[pyarg(any, optional)]
        endpoint: OptionalArg<bool>,
        #[pyarg(any, optional)]
        base: OptionalArg<ArgIntoFloat>,
        #[pyarg(any, optional)]
        dtype: OptionalArg<PyObjectRef>,
    }
    #[pyfunction]
    fn logspace(args: LogspaceArgs, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let num = args.num.unwrap_or(50);
        let endpoint = args.endpoint.unwrap_or(true);
        let base: f64 = args.base.map(|b| b.into()).unwrap_or(10.0);
        let dt = parse_dtype_arg(&args.dtype.into_option(), vm)?;
        let arr =
            crate::extras2::logspace(args.start.into(), args.stop.into(), num, base, endpoint);
        Ok(PyNdArray::from_arrays(match dt {
            Some(d) => arr.cast(d),
            None => arr,
        }))
    }

    #[derive(FromArgs)]
    pub(crate) struct GeomspaceArgs {
        #[pyarg(positional)]
        start: ArgIntoFloat,
        #[pyarg(positional)]
        stop: ArgIntoFloat,
        #[pyarg(any, optional)]
        num: OptionalArg<usize>,
        #[pyarg(any, optional)]
        endpoint: OptionalArg<bool>,
        #[pyarg(any, optional)]
        dtype: OptionalArg<PyObjectRef>,
    }
    #[pyfunction]
    fn geomspace(args: GeomspaceArgs, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let num = args.num.unwrap_or(50);
        let endpoint = args.endpoint.unwrap_or(true);
        let dt = parse_dtype_arg(&args.dtype.into_option(), vm)?;
        let arr = crate::extras2::geomspace(args.start.into(), args.stop.into(), num, endpoint)
            .ok_or_else(|| {
                vm.new_value_error(
                    "geomspace: start and stop must be non-zero with the same sign".to_string(),
                )
            })?;
        Ok(PyNdArray::from_arrays(match dt {
            Some(d) => arr.cast(d),
            None => arr,
        }))
    }

    #[pyfunction]
    fn full_like(
        a: PyObjectRef,
        fill_value: ArgIntoFloat,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::extras2::full_like(
            &arr,
            fill_value.into(),
        )))
    }

    #[pyfunction]
    fn empty_like(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::extras2::empty_like(&arr)))
    }

    // ---------------- dtype promotion / introspection ----------------

    /// Map a Python object (dtype string, builtin type, ndarray, or scalar)
    /// to a `DType`. Used by `result_type`, `promote_types`, `can_cast`,
    /// `min_scalar_type`.
    fn dtype_of_arg(o: &PyObjectRef, vm: &VirtualMachine) -> PyResult<DType> {
        use rustpython_vm::builtins::{PyComplex, PyFloat, PyInt, PyStr};
        // PyDType — pass through.
        if let Some(d) = o.downcast_ref::<PyDType>() {
            return Ok(d.inner);
        }
        // ndarray → its dtype.
        if let Some(arr) = o.downcast_ref::<PyNdArray>() {
            return Ok(arr.view().dtype());
        }
        // String / builtin type — reuse parse_dtype_arg's logic.
        if o.downcast_ref::<PyStr>().is_some()
            || o.downcast_ref::<rustpython_vm::builtins::PyType>()
                .is_some()
        {
            return parse_dtype_arg(&Some(o.clone()), vm)?.ok_or_else(|| {
                vm.new_type_error("could not interpret dtype argument".to_string())
            });
        }
        // bool first (it's a subclass of int).
        if o.is(&vm.ctx.true_value) || o.is(&vm.ctx.false_value) {
            return Ok(DType::Bool);
        }
        if let Some(_c) = o.downcast_ref::<PyComplex>() {
            return Ok(DType::C128);
        }
        if let Some(_f) = o.downcast_ref::<PyFloat>() {
            return Ok(DType::F64);
        }
        if let Some(_i) = o.downcast_ref::<PyInt>() {
            return Ok(DType::I64);
        }
        Err(vm.new_type_error(format!(
            "cannot interpret '{}' as a dtype",
            o.class().name()
        )))
    }

    /// `np.result_type(*types)` — promote a sequence of dtypes/arrays/scalars.
    #[pyfunction]
    fn result_type(args: FuncArgs, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        if args.args.is_empty() {
            return Err(
                vm.new_value_error("result_type requires at least one argument".to_string())
            );
        }
        let dtypes: Vec<DType> = args
            .args
            .iter()
            .map(|o| dtype_of_arg(o, vm))
            .collect::<PyResult<_>>()?;
        let out = crate::promote::promote_many(&dtypes);
        Ok(vm.ctx.new_str(out.name()).into())
    }

    /// `np.promote_types(t1, t2)`.
    #[pyfunction]
    fn promote_types(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let x = dtype_of_arg(&a, vm)?;
        let y = dtype_of_arg(&b, vm)?;
        Ok(vm.ctx.new_str(crate::promote::promote(x, y).name()).into())
    }

    // ---------------- np.dtype ----------------

    /// Real `np.dtype` object. `arr.dtype` returns this; it supports the
    /// usual `.kind`, `.itemsize`, `.name`, `.char`, `.num`, `.str`,
    /// `.byteorder` attributes and compares equal to a numpy dtype string of
    /// the same name (so `arr.dtype == "float64"` keeps working).
    #[pyattr]
    #[pyclass(module = "numpy", name = "dtype")]
    #[derive(Debug, PyPayload)]
    pub struct PyDType {
        pub(crate) inner: DType,
    }

    impl Constructor for PyDType {
        type Args = FuncArgs;
        fn py_new(_cls: &Py<PyType>, args: FuncArgs, vm: &VirtualMachine) -> PyResult<Self> {
            let arg = args
                .args
                .into_iter()
                .next()
                .ok_or_else(|| vm.new_type_error("dtype() takes 1 argument".to_string()))?;
            let dt = dtype_of_arg(&arg, vm)?;
            Ok(PyDType { inner: dt })
        }
    }

    impl Representable for PyDType {
        fn repr_str(zelf: &Py<Self>, _vm: &VirtualMachine) -> PyResult<String> {
            Ok(format!("dtype('{}')", zelf.inner.name()))
        }
    }

    impl Comparable for PyDType {
        fn slot_richcompare(
            zelf: &PyObject,
            other: &PyObject,
            op: PyComparisonOp,
            vm: &VirtualMachine,
        ) -> PyResult<Either<PyObjectRef, PyComparisonValue>> {
            let z = zelf
                .downcast_ref::<PyDType>()
                .ok_or_else(|| vm.new_type_error("dtype comparison: bad payload".to_string()))?;
            // Coerce `other` into a DType if possible (string, type, dtype).
            let other_dt = match dtype_of_arg(&other.to_owned(), vm) {
                Ok(d) => d,
                Err(_) => return Ok(Either::B(PyComparisonValue::NotImplemented)),
            };
            let eq = z.inner == other_dt;
            let result = match op {
                PyComparisonOp::Eq => eq,
                PyComparisonOp::Ne => !eq,
                _ => {
                    return Ok(Either::B(PyComparisonValue::NotImplemented));
                }
            };
            Ok(Either::A(vm.ctx.new_bool(result).into()))
        }
        fn cmp(
            _zelf: &Py<Self>,
            _other: &PyObject,
            _op: PyComparisonOp,
            _vm: &VirtualMachine,
        ) -> PyResult<PyComparisonValue> {
            Ok(PyComparisonValue::NotImplemented)
        }
    }

    #[pyclass(
        with(Constructor, Representable, Comparable, rustpython_vm::types::Hashable),
        flags(BASETYPE)
    )]
    impl PyDType {
        #[pygetset]
        fn kind(&self, vm: &VirtualMachine) -> PyObjectRef {
            vm.ctx.new_str(self.inner.kind().to_string()).into()
        }
        #[pygetset]
        fn itemsize(&self) -> usize {
            self.inner.itemsize()
        }
        #[pygetset]
        fn name(&self, vm: &VirtualMachine) -> PyObjectRef {
            // Parameterized dtypes (`Str`, `Bytes`, `Datetime64(unit)`, …)
            // require the dynamic `name_owned` form. The unparameterized
            // variants still produce the same static string.
            vm.ctx.new_str(self.inner.name_owned()).into()
        }
        /// Single-character type code (numpy `dtype.char`).
        #[pygetset]
        fn char(&self, vm: &VirtualMachine) -> PyObjectRef {
            // Numeric dtypes have the traditional one-letter codes; for the
            // non-numeric variants we fall back to the dtype's kind code
            // (O / U / S / M / m / V) — same as numpy.
            let c: std::borrow::Cow<'static, str> = match self.inner {
                DType::Bool => "?".into(),
                DType::I8 => "b".into(),
                DType::I16 => "h".into(),
                DType::I32 => "i".into(),
                DType::I64 => "l".into(),
                DType::U8 => "B".into(),
                DType::U16 => "H".into(),
                DType::U32 => "I".into(),
                DType::U64 => "L".into(),
                DType::F16 => "e".into(),
                DType::F32 => "f".into(),
                DType::F64 => "d".into(),
                DType::C64 => "F".into(),
                DType::C128 => "D".into(),
                _ => self.inner.kind().to_string().into(),
            };
            vm.ctx.new_str(c.as_ref()).into()
        }
        /// Numpy's `dtype.num` (internal type number).
        #[pygetset]
        fn num(&self) -> i64 {
            match self.inner {
                DType::Bool => 0,
                DType::I8 => 1,
                DType::U8 => 2,
                DType::I16 => 3,
                DType::U16 => 4,
                DType::I32 => 5,
                DType::U32 => 6,
                DType::I64 => 7,
                DType::U64 => 8,
                DType::F16 => 23,
                DType::F32 => 11,
                DType::F64 => 12,
                DType::C64 => 14,
                DType::C128 => 15,
                _ => {
                    use std::hash::{Hash, Hasher};
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    self.inner.hash(&mut h);
                    h.finish() as i64
                }
            }
        }
        /// `'='` on most modern systems (native byteorder).
        #[pygetset]
        fn byteorder(&self, vm: &VirtualMachine) -> PyObjectRef {
            let b = if self.inner == DType::Bool || self.inner.itemsize() == 1 {
                "|"
            } else {
                "="
            };
            vm.ctx.new_str(b).into()
        }
        /// numpy `dtype.str` is `byteorder + char + itemsize_or_size`.
        #[pygetset]
        fn str(&self, vm: &VirtualMachine) -> PyObjectRef {
            let prefix = if self.inner == DType::Bool || self.inner.itemsize() == 1 {
                "|"
            } else if cfg!(target_endian = "little") {
                "<"
            } else {
                ">"
            };
            // Two-letter type code with the byte width — numpy uses 'b1', 'f8', 'c16', etc.
            // Non-numeric dtypes use their kind code + itemsize (e.g. "U10",
            // "M8" — though numpy adds "[unit]" for the time variants).
            let body: std::borrow::Cow<'static, str> = match self.inner {
                DType::Bool => "?".into(),
                DType::I8 => "i1".into(),
                DType::I16 => "i2".into(),
                DType::I32 => "i4".into(),
                DType::I64 => "i8".into(),
                DType::U8 => "u1".into(),
                DType::U16 => "u2".into(),
                DType::U32 => "u4".into(),
                DType::U64 => "u8".into(),
                DType::F16 => "f2".into(),
                DType::F32 => "f4".into(),
                DType::F64 => "f8".into(),
                DType::C64 => "c8".into(),
                DType::C128 => "c16".into(),
                DType::Datetime64(u) => format!("M8[{}]", u.code()).into(),
                DType::Timedelta64(u) => format!("m8[{}]", u.code()).into(),
                _ => format!("{}{}", self.inner.kind(), self.inner.itemsize()).into(),
            };
            vm.ctx.new_str(format!("{prefix}{body}")).into()
        }
        #[pygetset]
        fn alignment(&self) -> usize {
            self.inner.itemsize()
        }
        #[pygetset]
        fn isnative(&self) -> bool {
            true
        }
        #[pygetset]
        fn hasobject(&self) -> bool {
            false
        }
        #[pygetset]
        fn fields(&self, vm: &VirtualMachine) -> PyObjectRef {
            vm.ctx.none()
        }
        #[pygetset]
        fn names(&self, vm: &VirtualMachine) -> PyObjectRef {
            vm.ctx.none()
        }
        #[pygetset]
        fn shape(&self, vm: &VirtualMachine) -> PyObjectRef {
            PyTuple::new_ref(vec![], &vm.ctx).into()
        }
        #[pygetset]
        fn ndim(&self) -> usize {
            0
        }
        #[pymethod(name = "__str__")]
        fn str_magic(&self) -> String {
            self.inner.name().to_string()
        }
    }

    impl rustpython_vm::types::Hashable for PyDType {
        fn hash(
            zelf: &Py<Self>,
            _vm: &VirtualMachine,
        ) -> PyResult<rustpython_vm::common::hash::PyHash> {
            // DType is now a non-field-less enum (Str(u32), Datetime64(unit),
            // etc.), so we can't `as i64` cast. Hash via std::hash::Hash.
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            zelf.inner.hash(&mut h);
            Ok(h.finish() as rustpython_vm::common::hash::PyHash)
        }
    }

    impl PyDType {
        pub fn from_dtype(d: DType) -> Self {
            PyDType { inner: d }
        }
    }

    // ---------------- iinfo / finfo ----------------

    #[pyattr]
    #[pyclass(module = "numpy", name = "iinfo")]
    #[derive(Debug, PyPayload)]
    pub struct PyIinfo {
        pub(crate) dtype: DType,
    }

    impl Constructor for PyIinfo {
        type Args = FuncArgs;
        fn py_new(_cls: &Py<PyType>, args: FuncArgs, vm: &VirtualMachine) -> PyResult<Self> {
            let arg = args
                .args
                .into_iter()
                .next()
                .ok_or_else(|| vm.new_type_error("iinfo() needs an argument".to_string()))?;
            let dt = dtype_of_arg(&arg, vm)?;
            if !dt.is_integer() || dt == DType::Bool {
                return Err(vm.new_value_error(format!(
                    "iinfo only supports integer dtypes; got {}",
                    dt.name()
                )));
            }
            Ok(PyIinfo { dtype: dt })
        }
    }

    impl Representable for PyIinfo {
        fn repr_str(zelf: &Py<Self>, _vm: &VirtualMachine) -> PyResult<String> {
            let d = zelf.dtype;
            Ok(format!(
                "iinfo(min={}, max={}, dtype={})",
                iinfo_min(d),
                iinfo_max(d),
                d.name()
            ))
        }
    }

    #[pyclass(with(Constructor, Representable))]
    impl PyIinfo {
        #[pygetset]
        fn min(&self) -> i64 {
            iinfo_min(self.dtype)
        }
        #[pygetset]
        fn max(&self) -> i64 {
            iinfo_max(self.dtype)
        }
        #[pygetset]
        fn bits(&self) -> u32 {
            (self.dtype.itemsize() as u32) * 8
        }
        #[pygetset]
        fn dtype(&self, vm: &VirtualMachine) -> PyObjectRef {
            vm.ctx.new_str(self.dtype.name()).into()
        }
    }

    fn iinfo_min(d: DType) -> i64 {
        match d {
            DType::I8 => i8::MIN as i64,
            DType::I16 => i16::MIN as i64,
            DType::I32 => i32::MIN as i64,
            DType::I64 => i64::MIN,
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::Bool => 0,
            _ => 0,
        }
    }

    fn iinfo_max(d: DType) -> i64 {
        match d {
            DType::I8 => i8::MAX as i64,
            DType::I16 => i16::MAX as i64,
            DType::I32 => i32::MAX as i64,
            DType::I64 => i64::MAX,
            DType::U8 => u8::MAX as i64,
            DType::U16 => u16::MAX as i64,
            DType::U32 => u32::MAX as i64,
            // u64::MAX doesn't fit in i64; clamp.
            DType::U64 => i64::MAX,
            DType::Bool => 1,
            _ => 0,
        }
    }

    #[pyattr]
    #[pyclass(module = "numpy", name = "finfo")]
    #[derive(Debug, PyPayload)]
    pub struct PyFinfo {
        pub(crate) dtype: DType,
    }

    impl Constructor for PyFinfo {
        type Args = FuncArgs;
        fn py_new(_cls: &Py<PyType>, args: FuncArgs, vm: &VirtualMachine) -> PyResult<Self> {
            let arg = args
                .args
                .into_iter()
                .next()
                .ok_or_else(|| vm.new_type_error("finfo() needs an argument".to_string()))?;
            let dt = dtype_of_arg(&arg, vm)?;
            if !(dt.is_float() || dt.is_complex()) {
                return Err(vm.new_value_error(format!(
                    "finfo only supports floating-point dtypes; got {}",
                    dt.name()
                )));
            }
            Ok(PyFinfo { dtype: dt })
        }
    }

    impl Representable for PyFinfo {
        fn repr_str(zelf: &Py<Self>, _vm: &VirtualMachine) -> PyResult<String> {
            Ok(format!(
                "finfo(resolution=..., dtype={})",
                zelf.dtype.name()
            ))
        }
    }

    #[pyclass(with(Constructor, Representable))]
    impl PyFinfo {
        #[pygetset]
        fn bits(&self) -> u32 {
            (self.dtype.itemsize() as u32) * 8
        }
        #[pygetset]
        fn eps(&self) -> f64 {
            finfo_eps(self.dtype)
        }
        #[pygetset]
        fn min(&self) -> f64 {
            finfo_min(self.dtype)
        }
        #[pygetset]
        fn max(&self) -> f64 {
            finfo_max(self.dtype)
        }
        #[pygetset]
        fn tiny(&self) -> f64 {
            finfo_tiny(self.dtype)
        }
        #[pygetset]
        fn smallest_normal(&self) -> f64 {
            finfo_tiny(self.dtype)
        }
        #[pygetset]
        fn resolution(&self) -> f64 {
            finfo_eps(self.dtype) * 10.0
        }
        #[pygetset]
        fn precision(&self) -> i64 {
            match self.dtype {
                DType::F16 => 3,
                DType::F32 | DType::C64 => 6,
                _ => 15,
            }
        }
        #[pygetset]
        fn dtype(&self, vm: &VirtualMachine) -> PyObjectRef {
            vm.ctx.new_str(self.dtype.name()).into()
        }
    }

    fn finfo_eps(d: DType) -> f64 {
        match d {
            DType::F16 => f64::from(half::f16::EPSILON),
            DType::F32 | DType::C64 => f32::EPSILON as f64,
            DType::F64 | DType::C128 => f64::EPSILON,
            _ => f64::EPSILON,
        }
    }

    fn finfo_min(d: DType) -> f64 {
        match d {
            DType::F16 => f64::from(half::f16::MIN),
            DType::F32 | DType::C64 => f32::MIN as f64,
            _ => f64::MIN,
        }
    }

    fn finfo_max(d: DType) -> f64 {
        match d {
            DType::F16 => f64::from(half::f16::MAX),
            DType::F32 | DType::C64 => f32::MAX as f64,
            _ => f64::MAX,
        }
    }

    fn finfo_tiny(d: DType) -> f64 {
        match d {
            DType::F16 => f64::from(half::f16::MIN_POSITIVE),
            DType::F32 | DType::C64 => f32::MIN_POSITIVE as f64,
            _ => f64::MIN_POSITIVE,
        }
    }

    /// `np.min_scalar_type(value)`.
    #[pyfunction]
    fn min_scalar_type(value: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        use rustpython_vm::builtins::{PyComplex, PyFloat, PyInt};
        // Pass-through for non-Python-scalar types (dtypes / arrays).
        if let Some(arr) = value.downcast_ref::<PyNdArray>() {
            return Ok(vm.ctx.new_str(arr.view().dtype().name()).into());
        }
        if value.is(&vm.ctx.true_value) || value.is(&vm.ctx.false_value) {
            return Ok(vm.ctx.new_str(DType::Bool.name()).into());
        }
        if value.downcast_ref::<PyComplex>().is_some() {
            return Ok(vm.ctx.new_str(DType::C128.name()).into());
        }
        if value.downcast_ref::<PyFloat>().is_some() {
            // numpy considers ordinary Python floats as `float16` candidates
            // when they round-trip exactly; we conservatively pick float16
            // for the common case and fall back to float64.
            return Ok(vm.ctx.new_str(DType::F16.name()).into());
        }
        if let Some(i) = value.downcast_ref::<PyInt>() {
            let v = i.try_to_primitive::<i128>(vm).unwrap_or(0);
            let dt = if v >= 0 {
                if v <= u8::MAX as i128 {
                    DType::U8
                } else if v <= u16::MAX as i128 {
                    DType::U16
                } else if v <= u32::MAX as i128 {
                    DType::U32
                } else {
                    DType::U64
                }
            } else if v >= i8::MIN as i128 {
                DType::I8
            } else if v >= i16::MIN as i128 {
                DType::I16
            } else if v >= i32::MIN as i128 {
                DType::I32
            } else {
                DType::I64
            };
            return Ok(vm.ctx.new_str(dt.name()).into());
        }
        Err(vm.new_type_error("min_scalar_type: unsupported value".to_string()))
    }

    /// `np.can_cast(from_, to, casting='safe')`.
    #[pyfunction]
    fn can_cast(args: FuncArgs, vm: &VirtualMachine) -> PyResult<bool> {
        if args.args.len() < 2 {
            return Err(vm.new_type_error("can_cast requires 2 arguments".to_string()));
        }
        let from = dtype_of_arg(&args.args[0], vm)?;
        let to = dtype_of_arg(&args.args[1], vm)?;
        let casting = match args.kwargs.get("casting") {
            Some(o) => {
                let s = o
                    .downcast_ref::<rustpython_vm::builtins::PyStr>()
                    .ok_or_else(|| {
                        vm.new_type_error("can_cast: casting= must be a string".to_string())
                    })?;
                s.as_wtf8().to_string_lossy().into_owned()
            }
            None => "safe".to_string(),
        };
        Ok(can_cast_dtype(from, to, &casting))
    }

    /// Numpy-style casting rules.
    fn can_cast_dtype(from: DType, to: DType, casting: &str) -> bool {
        if from == to {
            return true;
        }
        match casting {
            "no" => false,
            "equiv" => from == to,
            "unsafe" => true,
            // same_kind: same kind, or any cast within the numeric hierarchy
            // (bool/int/uint/float/complex) — numpy 2.x treats these as
            // same_kind because they're all "number".
            "same_kind" => {
                if from.kind() == to.kind() {
                    return true;
                }
                let numeric = |d: DType| matches!(d.kind(), 'b' | 'i' | 'u' | 'f' | 'c');
                (numeric(from) && numeric(to)) || safe_cast(from, to)
            }
            // default "safe": promote(from, to) == to means `to` can absorb `from`.
            _ => safe_cast(from, to),
        }
    }

    fn safe_cast(from: DType, to: DType) -> bool {
        if from == to {
            return true;
        }
        crate::promote::promote(from, to) == to
    }

    #[pyfunction]
    fn eye(
        n: usize,
        m: OptionalArg<usize>,
        dtype: DTypeArg,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let cols = m.unwrap_or(n);
        let dt = parse_dtype_arg(&dtype.dtype.into_option(), vm)?.unwrap_or(DType::F64);
        Ok(PyNdArray::from_arrays(create::eye(n, cols, dt)))
    }

    #[pyfunction]
    fn identity(n: usize, dtype: DTypeArg, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        eye(n, OptionalArg::Missing, dtype, vm)
    }

    #[pyfunction(name = "dot")]
    fn dot_fn(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        Ok(PyNdArray::from_arrays(linalg::dot(&x, &y, vm)?))
    }

    #[pyfunction(name = "matmul")]
    fn matmul_fn(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        dot_fn(a, b, vm)
    }

    #[pyfunction(name = "transpose")]
    fn transpose_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(linalg::transpose(&arr)))
    }

    #[pyfunction(name = "reshape")]
    fn reshape_fn(
        a: PyObjectRef,
        newshape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        let shape = parse_shape_signed(&newshape, vm)?;
        let total = arr.len();
        let resolved = resolve_neg_one(&shape, total, vm)?;
        let prod: usize = resolved.iter().product();
        if prod != total {
            return Err(vm.new_value_error(format!(
                "cannot reshape array of size {total} into shape {resolved:?}"
            )));
        }
        let res = linalg::reshape(&arr, &resolved)
            .ok_or_else(|| vm.new_value_error("reshape failed".to_string()))?;
        Ok(PyNdArray::from_arrays(res))
    }

    #[pyfunction]
    fn concatenate(
        arrays: PyObjectRef,
        args: ConcatenateArgs,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let list = arrays
            .downcast_ref::<rustpython_vm::builtins::PyList>()
            .map(|l| l.borrow_vec().to_vec())
            .or_else(|| {
                arrays
                    .downcast_ref::<PyTuple>()
                    .map(|t| t.as_slice().to_vec())
            })
            .ok_or_else(|| {
                vm.new_type_error("concatenate() expects a sequence of arrays".to_string())
            })?;
        let arrs: Vec<ArraysD> = list
            .iter()
            .map(|o| obj_to_array(o, None, vm))
            .collect::<PyResult<_>>()?;
        // numpy: axis=None flattens before concatenating; default is axis=0.
        let axis_arg = args.axis.flatten();
        if axis_arg.is_none() && matches!(args.axis, OptionalArg::Present(None)) {
            let flat: Vec<ArraysD> = arrs.iter().map(crate::linalg::flatten).collect();
            return Ok(PyNdArray::from_arrays(linalg::concatenate(&flat, 0, vm)?));
        }
        let nd = arrs.first().map(|a| a.ndim() as isize).unwrap_or(0);
        let raw_axis = axis_arg.unwrap_or(0);
        let axis = if raw_axis < 0 {
            raw_axis + nd
        } else {
            raw_axis
        };
        if axis < 0 || axis >= nd.max(1) {
            return Err(vm.new_value_error(format!(
                "axis {raw_axis} is out of bounds for array of dimension {nd}"
            )));
        }
        Ok(PyNdArray::from_arrays(linalg::concatenate(
            &arrs,
            axis as usize,
            vm,
        )?))
    }

    // ---------------- unary ufuncs (real-only and real-or-complex) ----------------

    #[pyfunction(name = "sqrt")]
    fn sqrt_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::unary_real_or_complex(
            &arr,
            f64::sqrt,
            |c: num_complex::Complex<f64>| c.sqrt(),
        )))
    }
    #[pyfunction(name = "exp")]
    fn exp_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::unary_real_or_complex(
            &arr,
            f64::exp,
            |c: num_complex::Complex<f64>| c.exp(),
        )))
    }
    #[pyfunction(name = "log")]
    fn log_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::unary_real_or_complex(
            &arr,
            f64::ln,
            |c: num_complex::Complex<f64>| c.ln(),
        )))
    }
    #[pyfunction(name = "log2")]
    fn log2_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::unary_real_only(
            &arr,
            "log2",
            f64::log2,
            vm,
        )?))
    }
    #[pyfunction(name = "log10")]
    fn log10_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::unary_real_only(
            &arr,
            "log10",
            f64::log10,
            vm,
        )?))
    }
    #[pyfunction(name = "sin")]
    fn sin_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::unary_real_or_complex(
            &arr,
            f64::sin,
            |c: num_complex::Complex<f64>| c.sin(),
        )))
    }
    #[pyfunction(name = "cos")]
    fn cos_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::unary_real_or_complex(
            &arr,
            f64::cos,
            |c: num_complex::Complex<f64>| c.cos(),
        )))
    }
    #[pyfunction(name = "tan")]
    fn tan_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::unary_real_or_complex(
            &arr,
            f64::tan,
            |c: num_complex::Complex<f64>| c.tan(),
        )))
    }
    #[pyfunction(name = "arcsin")]
    fn arcsin_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::unary_real_only(
            &arr,
            "arcsin",
            f64::asin,
            vm,
        )?))
    }
    #[pyfunction(name = "arccos")]
    fn arccos_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::unary_real_only(
            &arr,
            "arccos",
            f64::acos,
            vm,
        )?))
    }
    #[pyfunction(name = "arctan")]
    fn arctan_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::unary_real_only(
            &arr,
            "arctan",
            f64::atan,
            vm,
        )?))
    }
    #[pyfunction(name = "arcsinh")]
    fn arcsinh_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::unary_real_only(
            &arr,
            "arcsinh",
            f64::asinh,
            vm,
        )?))
    }
    #[pyfunction(name = "arccosh")]
    fn arccosh_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::unary_real_only(
            &arr,
            "arccosh",
            f64::acosh,
            vm,
        )?))
    }
    #[pyfunction(name = "arctanh")]
    fn arctanh_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::unary_real_only(
            &arr,
            "arctanh",
            f64::atanh,
            vm,
        )?))
    }
    // C-style trig aliases added in numpy 2.x.
    #[pyfunction(name = "acos")]
    fn acos_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        arccos_fn(a, vm)
    }
    #[pyfunction(name = "asin")]
    fn asin_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        arcsin_fn(a, vm)
    }
    #[pyfunction(name = "atan")]
    fn atan_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        arctan_fn(a, vm)
    }
    #[pyfunction(name = "acosh")]
    fn acosh_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        arccosh_fn(a, vm)
    }
    #[pyfunction(name = "asinh")]
    fn asinh_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        arcsinh_fn(a, vm)
    }
    #[pyfunction(name = "atanh")]
    fn atanh_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        arctanh_fn(a, vm)
    }
    #[pyfunction(name = "sinh")]
    fn sinh_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::unary_real_or_complex(
            &arr,
            f64::sinh,
            |c: num_complex::Complex<f64>| c.sinh(),
        )))
    }
    #[pyfunction(name = "cosh")]
    fn cosh_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::unary_real_or_complex(
            &arr,
            f64::cosh,
            |c: num_complex::Complex<f64>| c.cosh(),
        )))
    }
    #[pyfunction(name = "tanh")]
    fn tanh_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::unary_real_or_complex(
            &arr,
            f64::tanh,
            |c: num_complex::Complex<f64>| c.tanh(),
        )))
    }
    #[pyfunction(name = "floor")]
    fn floor_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::unary_real_only(
            &arr,
            "floor",
            f64::floor,
            vm,
        )?))
    }
    #[pyfunction(name = "ceil")]
    fn ceil_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::unary_real_only(
            &arr,
            "ceil",
            f64::ceil,
            vm,
        )?))
    }
    #[pyfunction(name = "rint")]
    fn rint_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::unary_real_only(
            &arr,
            "rint",
            // numpy uses round-half-to-even (banker's rounding).
            |x: f64| x.round_ties_even(),
            vm,
        )?))
    }

    // ---- additional unary ufuncs (real-only) ----

    #[pyfunction(name = "cbrt")]
    fn cbrt_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::unary_real_only(
            &arr,
            "cbrt",
            f64::cbrt,
            vm,
        )?))
    }
    #[pyfunction(name = "reciprocal")]
    fn reciprocal_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::unary_real_only(
            &arr,
            "reciprocal",
            |x: f64| 1.0 / x,
            vm,
        )?))
    }
    #[pyfunction(name = "square")]
    fn square_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::unary_real_or_complex(
            &arr,
            |x: f64| x * x,
            |c: num_complex::Complex<f64>| c * c,
        )))
    }
    #[pyfunction(name = "expm1")]
    fn expm1_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::unary_real_only(
            &arr,
            "expm1",
            f64::exp_m1,
            vm,
        )?))
    }
    #[pyfunction(name = "log1p")]
    fn log1p_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::unary_real_only(
            &arr,
            "log1p",
            f64::ln_1p,
            vm,
        )?))
    }
    #[pyfunction(name = "exp2")]
    fn exp2_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::unary_real_only(
            &arr,
            "exp2",
            f64::exp2,
            vm,
        )?))
    }
    #[pyfunction(name = "fabs")]
    fn fabs_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::unary_real_only(
            &arr,
            "fabs",
            f64::abs,
            vm,
        )?))
    }
    #[pyfunction(name = "signbit")]
    fn signbit_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        use crate::dtype::CoerceArray;
        let f = arr.coerce::<f64>();
        let out: ndarray::ArrayD<bool> = f.mapv(|x| x.is_sign_negative());
        Ok(PyNdArray::from_arrays(ArraysD::Bool(out)))
    }

    // ---- new binary ufuncs ----

    #[pyfunction]
    fn copysign(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        use crate::dtype::CoerceArray;
        let xf = x.coerce::<f64>();
        let yf = y.coerce::<f64>();
        let broadcast = ndarray::Zip::from(&xf).and_broadcast(&yf);
        let _ = broadcast;
        let out = broadcast_binary_f64(&xf, &yf, vm, |a, b| a.abs() * b.signum())?;
        Ok(PyNdArray::from_arrays(ArraysD::F64(out)))
    }

    #[pyfunction]
    fn heaviside(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        use crate::dtype::CoerceArray;
        let xf = x.coerce::<f64>();
        let yf = y.coerce::<f64>();
        let out = broadcast_binary_f64(&xf, &yf, vm, |a, b| {
            if a < 0.0 {
                0.0
            } else if a > 0.0 {
                1.0
            } else {
                b
            }
        })?;
        Ok(PyNdArray::from_arrays(ArraysD::F64(out)))
    }

    /// gcd / lcm element-wise on integer arrays.
    #[pyfunction]
    fn gcd(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        use crate::dtype::CoerceArray;
        let xi = x.coerce::<i64>();
        let yi = y.coerce::<i64>();
        let out = broadcast_binary_i64(&xi, &yi, vm, |a, b| gcd_i64(a.abs(), b.abs()))?;
        Ok(PyNdArray::from_arrays(ArraysD::I64(out)))
    }
    #[pyfunction]
    fn lcm(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        use crate::dtype::CoerceArray;
        let xi = x.coerce::<i64>();
        let yi = y.coerce::<i64>();
        let out = broadcast_binary_i64(&xi, &yi, vm, |a, b| {
            if a == 0 || b == 0 {
                0
            } else {
                (a.abs() / gcd_i64(a.abs(), b.abs())).saturating_mul(b.abs())
            }
        })?;
        Ok(PyNdArray::from_arrays(ArraysD::I64(out)))
    }

    fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
        while b != 0 {
            let t = b;
            b = a % b;
            a = t;
        }
        a
    }

    #[pyfunction]
    fn left_shift(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        use crate::dtype::CoerceArray;
        let xi = x.coerce::<i64>();
        let yi = y.coerce::<i64>();
        let out = broadcast_binary_i64(&xi, &yi, vm, |a, b| a.wrapping_shl(b as u32))?;
        Ok(PyNdArray::from_arrays(ArraysD::I64(out)))
    }
    #[pyfunction]
    fn right_shift(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        use crate::dtype::CoerceArray;
        let xi = x.coerce::<i64>();
        let yi = y.coerce::<i64>();
        let out = broadcast_binary_i64(&xi, &yi, vm, |a, b| a.wrapping_shr(b as u32))?;
        Ok(PyNdArray::from_arrays(ArraysD::I64(out)))
    }

    fn broadcast_binary_f64(
        x: &ndarray::ArrayD<f64>,
        y: &ndarray::ArrayD<f64>,
        vm: &VirtualMachine,
        f: impl Fn(f64, f64) -> f64,
    ) -> PyResult<ndarray::ArrayD<f64>> {
        let bx = x
            .broadcast(
                crate::extras::broadcast_shape(x.shape(), y.shape())
                    .ok_or_else(|| vm.new_value_error("shapes not broadcastable".to_string()))?,
            )
            .ok_or_else(|| vm.new_value_error("broadcast failed".to_string()))?;
        let by = y
            .broadcast(
                crate::extras::broadcast_shape(x.shape(), y.shape())
                    .ok_or_else(|| vm.new_value_error("shapes not broadcastable".to_string()))?,
            )
            .ok_or_else(|| vm.new_value_error("broadcast failed".to_string()))?;
        Ok(ndarray::Zip::from(&bx)
            .and(&by)
            .map_collect(|&a, &b| f(a, b)))
    }
    fn broadcast_binary_i64(
        x: &ndarray::ArrayD<i64>,
        y: &ndarray::ArrayD<i64>,
        vm: &VirtualMachine,
        f: impl Fn(i64, i64) -> i64,
    ) -> PyResult<ndarray::ArrayD<i64>> {
        let bx = x
            .broadcast(
                crate::extras::broadcast_shape(x.shape(), y.shape())
                    .ok_or_else(|| vm.new_value_error("shapes not broadcastable".to_string()))?,
            )
            .ok_or_else(|| vm.new_value_error("broadcast failed".to_string()))?;
        let by = y
            .broadcast(
                crate::extras::broadcast_shape(x.shape(), y.shape())
                    .ok_or_else(|| vm.new_value_error("shapes not broadcastable".to_string()))?,
            )
            .ok_or_else(|| vm.new_value_error("broadcast failed".to_string()))?;
        Ok(ndarray::Zip::from(&bx)
            .and(&by)
            .map_collect(|&a, &b| f(a, b)))
    }

    // ---- predicates / classifiers ----

    #[pyfunction(name = "isreal")]
    fn isreal_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        let out: ndarray::ArrayD<bool> = match &arr {
            ArraysD::C64(x) => x.mapv(|c| c.im == 0.0),
            ArraysD::C128(x) => x.mapv(|c| c.im == 0.0),
            other => {
                let mut v = ndarray::ArrayD::<bool>::from_elem(ndarray::IxDyn(other.shape()), true);
                let _ = &mut v;
                v
            }
        };
        Ok(PyNdArray::from_arrays(ArraysD::Bool(out)))
    }

    #[pyfunction(name = "iscomplex")]
    fn iscomplex_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        let out: ndarray::ArrayD<bool> = match &arr {
            ArraysD::C64(x) => x.mapv(|c| c.im != 0.0),
            ArraysD::C128(x) => x.mapv(|c| c.im != 0.0),
            other => ndarray::ArrayD::<bool>::from_elem(ndarray::IxDyn(other.shape()), false),
        };
        Ok(PyNdArray::from_arrays(ArraysD::Bool(out)))
    }

    #[pyfunction(name = "isscalar")]
    fn isscalar_fn(o: PyObjectRef, vm: &VirtualMachine) -> bool {
        use rustpython_vm::builtins::{PyComplex, PyFloat, PyInt, PyStr};
        // numpy treats Python scalars (int, float, complex, str, bytes, bool)
        // and 0-D arrays' element types as scalars. Lists/tuples/ndarrays are not.
        if o.downcast_ref::<PyNdArray>().is_some() {
            return false;
        }
        if o.downcast_ref::<rustpython_vm::builtins::PyList>()
            .is_some()
            || o.downcast_ref::<PyTuple>().is_some()
        {
            return false;
        }
        o.downcast_ref::<PyInt>().is_some()
            || o.downcast_ref::<PyFloat>().is_some()
            || o.downcast_ref::<PyComplex>().is_some()
            || o.downcast_ref::<PyStr>().is_some()
            || o.is(&vm.ctx.true_value)
            || o.is(&vm.ctx.false_value)
    }

    /// `np.flatnonzero(a)` — indices of nonzero elements of a flattened a.
    #[pyfunction]
    fn flatnonzero(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        use crate::dtype::CoerceArray;
        let flat = crate::linalg::flatten(&arr);
        let mut idx: Vec<i64> = Vec::new();
        match &flat {
            ArraysD::Bool(x) => {
                for (i, &v) in x.iter().enumerate() {
                    if v {
                        idx.push(i as i64);
                    }
                }
            }
            _ => {
                let f = flat.coerce::<f64>();
                for (i, &v) in f.iter().enumerate() {
                    if v != 0.0 {
                        idx.push(i as i64);
                    }
                }
            }
        }
        let n = idx.len();
        let out = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n]), idx)
            .map_err(|e| crate::internal::internal(vm, format!("flatnonzero: {e}")))?;
        Ok(PyNdArray::from_arrays(ArraysD::I64(out)))
    }

    /// `np.argwhere(a)` — indices of nonzero elements (one row per nonzero).
    #[pyfunction]
    fn argwhere(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        use crate::dtype::CoerceArray;
        let nd = arr.ndim().max(1);
        let mut rows: Vec<i64> = Vec::new();
        let mut n_rows = 0usize;
        macro_rules! per {
            ($x:ident, $is_nonzero:expr) => {{
                use ndarray::Dimension;
                for (idx, val) in $x.indexed_iter() {
                    if $is_nonzero(*val) {
                        let slc = idx.slice();
                        for d in 0..nd {
                            let i = slc.get(d).copied().unwrap_or(0);
                            rows.push(i as i64);
                        }
                        n_rows += 1;
                    }
                }
            }};
        }
        match &arr {
            ArraysD::Bool(x) => per!(x, |v: bool| v),
            _ => {
                let f = arr.coerce::<f64>();
                per!(f, |v: f64| v != 0.0);
            }
        }
        let out = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n_rows, nd]), rows)
            .map_err(|e| crate::internal::internal(vm, format!("argwhere: {e}")))?;
        Ok(PyNdArray::from_arrays(ArraysD::I64(out)))
    }

    // ---- linear algebra extras (top-level) ----

    /// `np.outer(a, b)` — flattens both, then a column-vector × row-vector product.
    #[pyfunction]
    fn outer(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        let xf = crate::linalg::flatten(&x);
        let yf = crate::linalg::flatten(&y);
        let xa = match crate::linalg::reshape(&xf, &[xf.len(), 1]) {
            Some(v) => v,
            None => return Err(crate::internal::internal(vm, "outer: reshape failed")),
        };
        let ya = match crate::linalg::reshape(&yf, &[1, yf.len()]) {
            Some(v) => v,
            None => return Err(crate::internal::internal(vm, "outer: reshape failed")),
        };
        Ok(PyNdArray::from_arrays(linalg::dot(&xa, &ya, vm)?))
    }

    /// `np.inner(a, b)` — sum-product over the last axis of each.
    #[pyfunction]
    fn inner(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        // For 1-D inputs, inner == dot.
        if x.ndim() == 1 && y.ndim() == 1 {
            return Ok(PyNdArray::from_arrays(linalg::dot(&x, &y, vm)?));
        }
        // General case: flatten last axis and contract.
        // Use einsum semantics: outer-shape(x) × outer-shape(y) result.
        let xshape = x.shape().to_vec();
        let yshape = y.shape().to_vec();
        if xshape.last() != yshape.last() {
            return Err(vm.new_value_error(format!(
                "shapes {xshape:?} and {yshape:?} not aligned for inner"
            )));
        }
        // Reshape into (prod(outer_x), last) and (prod(outer_y), last).
        let last = *xshape.last().unwrap_or(&1);
        let mx: usize = xshape[..xshape.len().saturating_sub(1)].iter().product();
        let my: usize = yshape[..yshape.len().saturating_sub(1)].iter().product();
        let xr = crate::linalg::reshape(&x, &[mx.max(1), last])
            .ok_or_else(|| crate::internal::internal(vm, "inner: reshape failed"))?;
        let yr = crate::linalg::reshape(&y, &[my.max(1), last])
            .ok_or_else(|| crate::internal::internal(vm, "inner: reshape failed"))?;
        // dot(xr, yr.T): (mx, last) × (last, my) = (mx, my)
        let yt = crate::linalg::transpose(&yr);
        let result = linalg::dot(&xr, &yt, vm)?;
        // Reshape back to outer_x.shape ++ outer_y.shape
        let mut out_shape: Vec<usize> = xshape[..xshape.len() - 1].to_vec();
        out_shape.extend_from_slice(&yshape[..yshape.len() - 1]);
        let final_arr = crate::linalg::reshape(&result, &out_shape)
            .ok_or_else(|| crate::internal::internal(vm, "inner: final reshape failed"))?;
        Ok(PyNdArray::from_arrays(final_arr))
    }

    /// `np.vdot(a, b)` — flatten both, conjugate first, dot product.
    #[pyfunction]
    fn vdot(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        let xf = crate::linalg::flatten(&x);
        let yf = crate::linalg::flatten(&y);
        let xc = ops::conj(&xf);
        Ok(PyNdArray::from_arrays(linalg::dot(&xc, &yf, vm)?))
    }

    /// `np.kron(a, b)` — Kronecker product (a 2-D classic; we generalize via
    /// einsum-style outer product + reshape).
    #[pyfunction]
    fn kron(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        // Pad x and y to the same ndim by prepending 1-axes.
        let nd = x.ndim().max(y.ndim());
        let pad = |a: &ArraysD| -> ArraysD {
            let mut s = a.shape().to_vec();
            while s.len() < nd {
                s.insert(0, 1);
            }
            crate::linalg::reshape(a, &s).unwrap_or_else(|| a.clone())
        };
        let xp = pad(&x);
        let yp = pad(&y);
        // Out shape: x.shape[i] * y.shape[i] per axis.
        let out_shape: Vec<usize> = xp
            .shape()
            .iter()
            .zip(yp.shape())
            .map(|(a, b)| a * b)
            .collect();
        // For each output index, value is x[i//bs] * y[i%bs] (componentwise).
        use crate::dtype::CoerceArray;
        let xf = xp.coerce::<f64>();
        let yf = yp.coerce::<f64>();
        let total: usize = out_shape.iter().product();
        let mut data = Vec::with_capacity(total);
        // walk in C order.
        let mut idx = vec![0usize; nd];
        for _ in 0..total {
            let mut xi = vec![0usize; nd];
            let mut yi = vec![0usize; nd];
            for d in 0..nd {
                let bs = yp.shape()[d];
                xi[d] = idx[d] / bs;
                yi[d] = idx[d] % bs;
            }
            data.push(xf[ndarray::IxDyn(&xi)] * yf[ndarray::IxDyn(&yi)]);
            // increment
            for d in (0..nd).rev() {
                idx[d] += 1;
                if idx[d] < out_shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&out_shape), data)
            .map_err(|e| crate::internal::internal(vm, format!("kron: {e}")))?;
        Ok(PyNdArray::from_arrays(ArraysD::F64(arr)))
    }

    #[derive(FromArgs)]
    pub(crate) struct TensordotArgs {
        #[pyarg(any, optional)]
        axes: OptionalArg<PyObjectRef>,
    }

    /// `np.tensordot(a, b, axes)` — limited to axes=int form for now.
    #[pyfunction]
    fn tensordot(
        a: PyObjectRef,
        b: PyObjectRef,
        td: TensordotArgs,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        let n: isize = match &td.axes {
            OptionalArg::Missing => 2,
            OptionalArg::Present(o) => o.try_int(vm)?.try_to_primitive::<isize>(vm)?,
        };
        let n = n as usize;
        let xs = x.shape().to_vec();
        let ys = y.shape().to_vec();
        if n > xs.len() || n > ys.len() {
            return Err(vm.new_value_error("tensordot: axes too large".to_string()));
        }
        let x_outer: Vec<usize> = xs[..xs.len() - n].to_vec();
        let x_inner: usize = xs[xs.len() - n..].iter().product();
        let y_inner: usize = ys[..n].iter().product();
        let y_outer: Vec<usize> = ys[n..].to_vec();
        if x_inner != y_inner {
            return Err(vm.new_value_error(format!(
                "tensordot: contracted dimensions mismatch ({x_inner} vs {y_inner})"
            )));
        }
        let mx: usize = x_outer.iter().product::<usize>().max(1);
        let my: usize = y_outer.iter().product::<usize>().max(1);
        let xr = crate::linalg::reshape(&x, &[mx, x_inner.max(1)])
            .ok_or_else(|| crate::internal::internal(vm, "tensordot: reshape x"))?;
        let yr = crate::linalg::reshape(&y, &[y_inner.max(1), my])
            .ok_or_else(|| crate::internal::internal(vm, "tensordot: reshape y"))?;
        let prod = linalg::dot(&xr, &yr, vm)?;
        let mut shape = x_outer;
        shape.extend(y_outer);
        if shape.is_empty() {
            return Ok(PyNdArray::from_arrays(prod));
        }
        let final_arr = crate::linalg::reshape(&prod, &shape)
            .ok_or_else(|| crate::internal::internal(vm, "tensordot: reshape final"))?;
        Ok(PyNdArray::from_arrays(final_arr))
    }

    #[derive(FromArgs)]
    pub(crate) struct ConvolveArgs {
        #[pyarg(any, optional)]
        mode: OptionalArg<PyObjectRef>,
    }

    /// `np.convolve(a, v, mode='full')`.
    #[pyfunction]
    fn convolve(
        a: PyObjectRef,
        v: PyObjectRef,
        args: ConvolveArgs,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        convolve_or_correlate(a, v, args.mode, /*reverse_v=*/ false, "full", vm)
    }

    /// `np.correlate(a, v, mode='valid')`.
    #[pyfunction]
    fn correlate(
        a: PyObjectRef,
        v: PyObjectRef,
        args: ConvolveArgs,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        convolve_or_correlate(a, v, args.mode, /*reverse_v=*/ true, "valid", vm)
    }

    fn convolve_or_correlate(
        a: PyObjectRef,
        v: PyObjectRef,
        mode: OptionalArg<PyObjectRef>,
        reverse_v: bool,
        default_mode: &str,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        use crate::dtype::CoerceArray;
        let xa = obj_to_array(&a, None, vm)?;
        let xv = obj_to_array(&v, None, vm)?;
        let xf = xa.coerce::<f64>();
        let mut vf = xv.coerce::<f64>();
        if xf.ndim() != 1 || vf.ndim() != 1 {
            return Err(vm.new_value_error("convolve/correlate inputs must be 1-D".to_string()));
        }
        if reverse_v {
            let mut v: Vec<f64> = vf.iter().copied().collect();
            v.reverse();
            vf = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[v.len()]), v)
                .map_err(|e| crate::internal::internal(vm, format!("convolve: {e}")))?;
        }
        let n = xf.len();
        let m = vf.len();
        // Mode determines output size: 'full' (n+m-1), 'same' (n), 'valid' (n-m+1)
        let mode_s = match &mode {
            OptionalArg::Missing => default_mode.to_string(),
            OptionalArg::Present(o) => o
                .downcast_ref::<rustpython_vm::builtins::PyStr>()
                .map(|s| s.as_wtf8().to_string_lossy().into_owned())
                .ok_or_else(|| vm.new_type_error("convolve mode must be a string".to_string()))?,
        };
        // Compute 'full' first.
        let full_len = n + m - 1;
        let mut full = vec![0.0f64; full_len];
        let x_slice = xf
            .as_slice()
            .ok_or_else(|| crate::internal::internal(vm, "convolve: input not contiguous"))?;
        let v_slice = vf
            .as_slice()
            .ok_or_else(|| crate::internal::internal(vm, "convolve: kernel not contiguous"))?;
        for i in 0..n {
            for j in 0..m {
                full[i + j] += x_slice[i] * v_slice[j];
            }
        }
        let (start, end) = match mode_s.as_str() {
            "full" => (0, full_len),
            "same" => {
                let pad = (m - 1) / 2;
                (pad, pad + n)
            }
            "valid" => {
                if m > n {
                    return Err(vm.new_value_error(
                        "convolve/correlate 'valid' mode: kernel longer than input".to_string(),
                    ));
                }
                (m - 1, n)
            }
            other => {
                return Err(vm.new_value_error(format!("invalid mode: {other:?}")));
            }
        };
        let data: Vec<f64> = full[start..end].to_vec();
        let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[data.len()]), data)
            .map_err(|e| crate::internal::internal(vm, format!("convolve: {e}")))?;
        Ok(PyNdArray::from_arrays(ArraysD::F64(arr)))
    }

    // ---- split family ----

    #[pyfunction]
    fn split(
        ary: PyObjectRef,
        indices_or_sections: PyObjectRef,
        axis: OptionalArg<isize>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        do_split(
            ary,
            indices_or_sections,
            axis,
            /*allow_uneven=*/ false,
            vm,
        )
    }

    #[pyfunction]
    fn array_split(
        ary: PyObjectRef,
        indices_or_sections: PyObjectRef,
        axis: OptionalArg<isize>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        do_split(
            ary,
            indices_or_sections,
            axis,
            /*allow_uneven=*/ true,
            vm,
        )
    }

    #[pyfunction]
    fn hsplit(
        ary: PyObjectRef,
        sections: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let arr = obj_to_array(&ary, None, vm)?;
        let axis = if arr.ndim() == 1 { 0 } else { 1 };
        do_split(ary, sections, OptionalArg::Present(axis), false, vm)
    }

    #[pyfunction]
    fn vsplit(
        ary: PyObjectRef,
        sections: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        do_split(ary, sections, OptionalArg::Present(0), false, vm)
    }

    #[pyfunction]
    fn dsplit(
        ary: PyObjectRef,
        sections: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        do_split(ary, sections, OptionalArg::Present(2), false, vm)
    }

    fn do_split(
        ary: PyObjectRef,
        sections: PyObjectRef,
        axis: OptionalArg<isize>,
        allow_uneven: bool,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let arr = obj_to_array(&ary, None, vm)?;
        let nd = arr.ndim() as isize;
        let ax = match axis {
            OptionalArg::Missing => 0,
            OptionalArg::Present(v) => v,
        };
        let ax = if ax < 0 { ax + nd } else { ax };
        if ax < 0 || ax >= nd {
            return Err(vm.new_value_error(format!("axis {ax} out of bounds for {nd}-D array")));
        }
        let ax = ax as usize;
        let dim = arr.shape()[ax];
        let cut_points: Vec<usize> = if let Ok(n) = sections
            .try_int(vm)
            .and_then(|i| i.try_to_primitive::<isize>(vm))
        {
            // Integer N: split into N equal-ish parts.
            let n = n as usize;
            if n == 0 {
                return Err(vm.new_value_error("number of sections must be > 0".to_string()));
            }
            if !allow_uneven && dim % n != 0 {
                return Err(vm.new_value_error(format!(
                    "array split does not result in an equal division ({dim} / {n})"
                )));
            }
            let mut pts = Vec::with_capacity(n - 1);
            let base = dim / n;
            let extra = dim % n;
            let mut cum = 0;
            for k in 0..n - 1 {
                cum += base + if k < extra { 1 } else { 0 };
                pts.push(cum);
            }
            pts
        } else {
            // Sequence of indices.
            let items: Vec<PyObjectRef> = if let Some(t) = sections.downcast_ref::<PyTuple>() {
                t.as_slice().to_vec()
            } else if let Some(l) = sections.downcast_ref::<rustpython_vm::builtins::PyList>() {
                l.borrow_vec().to_vec()
            } else {
                return Err(
                    vm.new_type_error("indices_or_sections must be int or sequence".to_string())
                );
            };
            let mut pts = Vec::with_capacity(items.len());
            for it in items {
                let v = it.try_int(vm)?.try_to_primitive::<isize>(vm)?;
                pts.push(v.max(0) as usize);
            }
            pts
        };
        let mut parts: Vec<PyObjectRef> = Vec::with_capacity(cut_points.len() + 1);
        let mut prev = 0usize;
        for &c in &cut_points {
            let end = c.min(dim);
            parts.push(
                PyNdArray::from_arrays(slice_along_axis(&arr, ax, prev, end)).into_pyobject(vm),
            );
            prev = end;
        }
        parts.push(PyNdArray::from_arrays(slice_along_axis(&arr, ax, prev, dim)).into_pyobject(vm));
        Ok(vm.ctx.new_list(parts).into())
    }

    fn slice_along_axis(a: &ArraysD, axis: usize, start: usize, end: usize) -> ArraysD {
        macro_rules! per {
            ($var:ident, $arr:ident) => {{
                let s = $arr.slice_axis(ndarray::Axis(axis), ndarray::Slice::from(start..end));
                ArraysD::$var(s.to_owned())
            }};
        }
        match a {
            ArraysD::Bool(arr) => per!(Bool, arr),
            ArraysD::I8(arr) => per!(I8, arr),
            ArraysD::I16(arr) => per!(I16, arr),
            ArraysD::I32(arr) => per!(I32, arr),
            ArraysD::I64(arr) => per!(I64, arr),
            ArraysD::U8(arr) => per!(U8, arr),
            ArraysD::U16(arr) => per!(U16, arr),
            ArraysD::U32(arr) => per!(U32, arr),
            ArraysD::U64(arr) => per!(U64, arr),
            ArraysD::F16(arr) => per!(F16, arr),
            ArraysD::F32(arr) => per!(F32, arr),
            ArraysD::F64(arr) => per!(F64, arr),
            ArraysD::C64(arr) => per!(C64, arr),
            ArraysD::C128(arr) => per!(C128, arr),
            _ => a.clone(),
        }
    }

    // ---- axis manipulation ----

    #[pyfunction]
    fn swapaxes(
        a: PyObjectRef,
        axis1: isize,
        axis2: isize,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        let nd = arr.ndim();
        let n1 = normalize_axis_arg(axis1, nd, vm)?;
        let n2 = normalize_axis_arg(axis2, nd, vm)?;
        let mut perm: Vec<usize> = (0..nd).collect();
        perm.swap(n1, n2);
        Ok(PyNdArray::from_arrays(transpose_with_perm(&arr, &perm)))
    }

    #[pyfunction]
    fn moveaxis(
        a: PyObjectRef,
        source: PyObjectRef,
        destination: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        let nd = arr.ndim();
        let parse_one_or_seq = |o: &PyObjectRef| -> PyResult<Vec<usize>> {
            if let Some(t) = o.downcast_ref::<PyTuple>() {
                t.as_slice()
                    .iter()
                    .map(|x| {
                        let v = x.try_int(vm)?.try_to_primitive::<isize>(vm)?;
                        normalize_axis_arg(v, nd, vm)
                    })
                    .collect()
            } else if let Some(l) = o.downcast_ref::<rustpython_vm::builtins::PyList>() {
                l.borrow_vec()
                    .iter()
                    .map(|x| {
                        let v = x.try_int(vm)?.try_to_primitive::<isize>(vm)?;
                        normalize_axis_arg(v, nd, vm)
                    })
                    .collect()
            } else {
                let v = o.try_int(vm)?.try_to_primitive::<isize>(vm)?;
                Ok(vec![normalize_axis_arg(v, nd, vm)?])
            }
        };
        let src = parse_one_or_seq(&source)?;
        let dst = parse_one_or_seq(&destination)?;
        if src.len() != dst.len() {
            return Err(vm.new_value_error(
                "moveaxis: source and destination must match in length".to_string(),
            ));
        }
        // Build the perm: start with [0..nd], remove src in order, then insert at dst.
        let mut order: Vec<usize> = (0..nd).filter(|i| !src.contains(i)).collect();
        let mut pairs: Vec<(usize, usize)> = src.iter().copied().zip(dst.iter().copied()).collect();
        pairs.sort_by_key(|&(_, d)| d);
        for (s, d) in pairs {
            order.insert(d.min(order.len()), s);
        }
        Ok(PyNdArray::from_arrays(transpose_with_perm(&arr, &order)))
    }

    /// `np.insert(arr, obj, values, axis=None)` — insert values along an axis.
    /// Only int-position `obj` is supported (not slices or sequence-of-ints).
    #[pyfunction]
    fn insert(
        arr: PyObjectRef,
        obj: isize,
        values: PyObjectRef,
        axis: OptionalArg<isize>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let a = obj_to_array(&arr, None, vm)?;
        let v = obj_to_array(&values, None, vm)?;
        let (a, ax) = match axis {
            OptionalArg::Missing => (crate::linalg::flatten(&a), 0),
            OptionalArg::Present(k) => {
                let ax = normalize_axis_arg(k, a.ndim(), vm)?;
                (a, ax)
            }
        };
        let dim = a.shape()[ax];
        let pos = if obj < 0 {
            (obj + dim as isize).max(0) as usize
        } else {
            (obj as usize).min(dim)
        };
        // Build new array via three concatenations: a[:pos], values, a[pos:].
        let before = slice_along_axis(&a, ax, 0, pos);
        let after = slice_along_axis(&a, ax, pos, dim);
        // values needs to be broadcast to the right shape.
        let mut target_shape = a.shape().to_vec();
        target_shape[ax] = 1;
        // If values is 1-D and not the right shape, treat as a scalar broadcast.
        let target_product: usize = target_shape.iter().product();
        let v_shaped = if v.ndim() == 0 {
            crate::extras::broadcast_to(&v, &target_shape, vm)?
        } else if v.shape() == &target_shape[..] {
            v
        } else if v.ndim() == 1 && v.len() == target_product {
            crate::linalg::reshape(&v, &target_shape).unwrap_or(v)
        } else {
            // assume the user knows what they're doing; let concatenate fail later
            v
        };
        let cat = linalg::concatenate(&[before, v_shaped, after], ax, vm)?;
        Ok(PyNdArray::from_arrays(cat))
    }

    /// `np.pad(arr, pad_width, mode='constant', constant_values=0)` — limited to
    /// the `constant` mode (the overwhelmingly common case).
    #[pyfunction]
    fn pad(
        arr: PyObjectRef,
        pad_width: PyObjectRef,
        args: PadArgs,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let a = obj_to_array(&arr, None, vm)?;
        let nd = a.ndim();
        let widths = parse_pad_width(&pad_width, nd, vm)?;
        let mode = match &args.mode {
            OptionalArg::Missing => "constant".to_string(),
            OptionalArg::Present(o) => o
                .downcast_ref::<rustpython_vm::builtins::PyStr>()
                .map(|s| s.as_wtf8().to_string_lossy().into_owned())
                .ok_or_else(|| vm.new_type_error("pad mode must be str".to_string()))?,
        };
        if mode != "constant" {
            return Err(vm.new_not_implemented_error(format!(
                "pad mode {mode:?} not implemented (only 'constant')"
            )));
        }
        let fill = match &args.constant_values {
            OptionalArg::Missing => 0.0f64,
            OptionalArg::Present(o) => o.try_float(vm)?.to_f64(),
        };
        // Build output array filled with `fill`, then copy the original in.
        let new_shape: Vec<usize> = (0..nd)
            .map(|i| widths[i].0 + a.shape()[i] + widths[i].1)
            .collect();
        let dt = a.dtype();
        let filled = crate::create::full_f64(&new_shape, fill, dt);
        // Set the central slice = a.
        let mut out = filled;
        macro_rules! per {
            ($variant:ident, $arr:ident, $ty:ty) => {{
                let src = a.coerce::<$ty>();
                if let Some(dst) = $arr.as_array_mut::<$ty>() {
                    let mut slice = dst.view_mut();
                    let info: Vec<ndarray::SliceInfoElem> = (0..nd)
                        .map(|i| ndarray::SliceInfoElem::Slice {
                            start: widths[i].0 as isize,
                            end: Some((widths[i].0 + a.shape()[i]) as isize),
                            step: 1,
                        })
                        .collect();
                    let si =
                        ndarray::SliceInfo::<_, ndarray::IxDyn, ndarray::IxDyn>::try_from(info)
                            .map_err(|e| vm.new_value_error(e.to_string()))?;
                    let mut sub = slice.slice_mut(si.as_ref());
                    sub.assign(&src);
                }
            }};
        }
        use crate::dtype::CoerceArray;
        match &mut out {
            ArraysD::Bool(_) => per!(Bool, out, bool),
            ArraysD::I8(_) => per!(I8, out, i8),
            ArraysD::I16(_) => per!(I16, out, i16),
            ArraysD::I32(_) => per!(I32, out, i32),
            ArraysD::I64(_) => per!(I64, out, i64),
            ArraysD::U8(_) => per!(U8, out, u8),
            ArraysD::U16(_) => per!(U16, out, u16),
            ArraysD::U32(_) => per!(U32, out, u32),
            ArraysD::U64(_) => per!(U64, out, u64),
            ArraysD::F16(_) => per!(F16, out, half::f16),
            ArraysD::F32(_) => per!(F32, out, f32),
            ArraysD::F64(_) => per!(F64, out, f64),
            ArraysD::C64(_) => per!(C64, out, crate::dtype::C32),
            ArraysD::C128(_) => per!(C128, out, crate::dtype::C64),
            _ => return Err(crate::internal::unsupported_dtype(vm, "pad", out.dtype())),
        }
        Ok(PyNdArray::from_arrays(out))
    }

    #[derive(FromArgs)]
    pub(crate) struct PadArgs {
        #[pyarg(any, optional)]
        mode: OptionalArg<PyObjectRef>,
        #[pyarg(any, optional)]
        constant_values: OptionalArg<PyObjectRef>,
    }

    fn parse_pad_width(
        obj: &PyObjectRef,
        nd: usize,
        vm: &VirtualMachine,
    ) -> PyResult<Vec<(usize, usize)>> {
        // Accept: int → (n,n) for every axis;
        //         (b, a) → (b, a) for every axis;
        //         sequence of (b, a) per axis.
        if let Ok(n) = obj
            .try_int(vm)
            .and_then(|i| i.try_to_primitive::<isize>(vm))
        {
            let n = n.max(0) as usize;
            return Ok(vec![(n, n); nd]);
        }
        let items: Vec<PyObjectRef> = if let Some(t) = obj.downcast_ref::<PyTuple>() {
            t.as_slice().to_vec()
        } else if let Some(l) = obj.downcast_ref::<rustpython_vm::builtins::PyList>() {
            l.borrow_vec().to_vec()
        } else {
            return Err(vm.new_type_error("pad_width must be int or sequence".to_string()));
        };
        // (b, a) shorthand?
        if items.len() == 2 && items.iter().all(|x| x.try_int(vm).is_ok()) {
            let b = items[0].try_int(vm)?.try_to_primitive::<isize>(vm)?.max(0) as usize;
            let a = items[1].try_int(vm)?.try_to_primitive::<isize>(vm)?.max(0) as usize;
            return Ok(vec![(b, a); nd]);
        }
        if items.len() != nd {
            return Err(vm.new_value_error(format!(
                "pad_width has {} entries but array is {}-D",
                items.len(),
                nd
            )));
        }
        items
            .iter()
            .map(|p| {
                let pair: Vec<PyObjectRef> = if let Some(t) = p.downcast_ref::<PyTuple>() {
                    t.as_slice().to_vec()
                } else if let Some(l) = p.downcast_ref::<rustpython_vm::builtins::PyList>() {
                    l.borrow_vec().to_vec()
                } else {
                    return Err(
                        vm.new_type_error("pad_width entries must be sequences".to_string())
                    );
                };
                if pair.len() != 2 {
                    return Err(
                        vm.new_value_error("pad_width entries must have length 2".to_string())
                    );
                }
                let b = pair[0].try_int(vm)?.try_to_primitive::<isize>(vm)?.max(0) as usize;
                let a = pair[1].try_int(vm)?.try_to_primitive::<isize>(vm)?.max(0) as usize;
                Ok((b, a))
            })
            .collect()
    }

    /// `np.block(arrays)` — construct a block array from nested lists.
    /// Limited support: 2-D blocks (list of list of 2-D arrays).
    #[pyfunction]
    fn block(arrays: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        // Determine if outer is a list-of-lists (2-D block) or flat (1-D block).
        if let Some(outer) = arrays.downcast_ref::<rustpython_vm::builtins::PyList>() {
            let outer_vec = outer.borrow_vec().to_vec();
            // Check if any element is itself a list → 2-D block.
            let any_list = outer_vec.iter().any(|o| {
                o.downcast_ref::<rustpython_vm::builtins::PyList>()
                    .is_some()
            });
            if any_list {
                // Each inner list becomes one row.
                let mut row_arrays: Vec<ArraysD> = Vec::with_capacity(outer_vec.len());
                for inner in outer_vec {
                    let inner_list = inner
                        .downcast_ref::<rustpython_vm::builtins::PyList>()
                        .ok_or_else(|| {
                            vm.new_type_error(
                                "block: mixed nested/flat structure not supported".to_string(),
                            )
                        })?;
                    let row_items: Vec<ArraysD> = inner_list
                        .borrow_vec()
                        .iter()
                        .map(|o| obj_to_array(o, None, vm))
                        .collect::<PyResult<_>>()?;
                    let row = linalg::concatenate(&row_items, 1, vm)?;
                    row_arrays.push(row);
                }
                return Ok(PyNdArray::from_arrays(linalg::concatenate(
                    &row_arrays,
                    0,
                    vm,
                )?));
            }
            // Flat list — concatenate along last axis.
            let arrs: Vec<ArraysD> = outer_vec
                .iter()
                .map(|o| obj_to_array(o, None, vm))
                .collect::<PyResult<_>>()?;
            let nd = arrs.first().map(|a| a.ndim()).unwrap_or(0);
            let axis = if nd == 0 { 0 } else { nd - 1 };
            return Ok(PyNdArray::from_arrays(linalg::concatenate(
                &arrs, axis, vm,
            )?));
        }
        Err(vm.new_type_error("block: argument must be a list".to_string()))
    }

    #[pyfunction]
    fn rollaxis(
        a: PyObjectRef,
        axis: isize,
        start: OptionalArg<isize>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        let nd = arr.ndim();
        let from = normalize_axis_arg(axis, nd, vm)?;
        let start_i = match start {
            OptionalArg::Missing => 0,
            OptionalArg::Present(v) => v,
        };
        // numpy: rollaxis lets start range 0..=nd.
        let to = if start_i < 0 {
            start_i + nd as isize
        } else {
            start_i
        };
        if to < 0 || to > nd as isize {
            return Err(vm.new_value_error(format!("rollaxis start {start_i} out of range")));
        }
        let mut order: Vec<usize> = (0..nd).filter(|&i| i != from).collect();
        let to = (to as usize).min(order.len());
        order.insert(to, from);
        Ok(PyNdArray::from_arrays(transpose_with_perm(&arr, &order)))
    }

    fn normalize_axis_arg(ax: isize, nd: usize, vm: &VirtualMachine) -> PyResult<usize> {
        let nd_i = nd as isize;
        let real = if ax < 0 { ax + nd_i } else { ax };
        if real < 0 || real >= nd_i {
            return Err(vm.new_value_error(format!("axis {ax} out of bounds for {nd}-D array")));
        }
        Ok(real as usize)
    }

    fn transpose_with_perm(a: &ArraysD, perm: &[usize]) -> ArraysD {
        macro_rules! per {
            ($var:ident, $arr:ident, $ty:ty) => {{
                let v = $arr.view().permuted_axes(ndarray::IxDyn(perm));
                let shape: Vec<usize> = v.shape().to_vec();
                let data: Vec<$ty> = v.iter().copied().collect();
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
            _ => a.clone(),
        }
    }

    #[pyfunction(name = "absolute")]
    fn absolute_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::absolute(&arr)))
    }

    #[pyfunction(name = "abs")]
    fn abs_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        absolute_fn(a, vm)
    }

    #[pyfunction(name = "negative")]
    fn negative_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::negate(&arr, vm)?))
    }

    #[pyfunction(name = "sign")]
    fn sign_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::unary_real_only(
            &arr,
            "sign",
            |x: f64| {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            },
            vm,
        )?))
    }

    #[pyfunction(name = "real")]
    fn real_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::real_part(&arr)))
    }

    #[pyfunction(name = "imag")]
    fn imag_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::imag_part(&arr)))
    }

    #[pyfunction(name = "conjugate")]
    fn conjugate_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::conj(&arr)))
    }

    #[pyfunction(name = "conj")]
    fn conj_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        conjugate_fn(a, vm)
    }

    // ---------------- binary ufuncs ----------------

    #[pyfunction]
    fn add(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::binary_op(
            &x,
            &y,
            vm,
            ops::Add,
        )?))
    }
    #[pyfunction]
    fn subtract(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::binary_op(
            &x,
            &y,
            vm,
            ops::Sub,
        )?))
    }
    #[pyfunction]
    fn multiply(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::binary_op(
            &x,
            &y,
            vm,
            ops::Mul,
        )?))
    }
    #[pyfunction]
    fn divide(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::true_divide(&x, &y, vm)?))
    }
    #[pyfunction]
    fn power(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::power(&x, &y, vm)?))
    }
    #[pyfunction(name = "true_divide")]
    fn true_divide_fn(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        divide(a, b, vm)
    }
    #[pyfunction(name = "floor_divide")]
    fn floor_divide_fn(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::floor_divide(&x, &y, vm)?))
    }
    #[pyfunction(name = "mod")]
    fn mod_fn(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::remainder(&x, &y, vm)?))
    }
    #[pyfunction(name = "remainder")]
    fn remainder_fn(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        mod_fn(a, b, vm)
    }
    #[pyfunction]
    fn maximum(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::elem_max(&x, &y, vm)?))
    }
    #[pyfunction]
    fn minimum(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::elem_min(&x, &y, vm)?))
    }

    // ---------------- comparison ufuncs ----------------

    #[pyfunction(name = "equal")]
    fn eq_fn(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::compare(&x, &y, CmpOp::Eq, vm)?))
    }
    #[pyfunction(name = "not_equal")]
    fn ne_fn(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::compare(&x, &y, CmpOp::Ne, vm)?))
    }
    #[pyfunction(name = "less")]
    fn lt_fn(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::compare(&x, &y, CmpOp::Lt, vm)?))
    }
    #[pyfunction(name = "less_equal")]
    fn le_fn(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::compare(&x, &y, CmpOp::Le, vm)?))
    }
    #[pyfunction(name = "greater")]
    fn gt_fn(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::compare(&x, &y, CmpOp::Gt, vm)?))
    }
    #[pyfunction(name = "greater_equal")]
    fn ge_fn(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        Ok(PyNdArray::from_arrays(ops::compare(&x, &y, CmpOp::Ge, vm)?))
    }

    // ---------------- reductions (free) ----------------

    #[pyfunction(name = "sum")]
    fn sum_fn(a: PyObjectRef, args: ReduceArgs, vm: &VirtualMachine) -> PyResult {
        let arr = obj_to_array(&a, None, vm)?;
        let r = do_reduce(&arr, args, Reduce::Sum, vm)?;
        Ok(scalar_or_array(r, vm))
    }
    #[pyfunction(name = "prod")]
    fn prod_fn(a: PyObjectRef, args: ReduceArgs, vm: &VirtualMachine) -> PyResult {
        let arr = obj_to_array(&a, None, vm)?;
        let r = do_reduce(&arr, args, Reduce::Prod, vm)?;
        Ok(scalar_or_array(r, vm))
    }
    #[pyfunction(name = "mean")]
    fn mean_fn(a: PyObjectRef, args: ReduceArgs, vm: &VirtualMachine) -> PyResult {
        let arr = obj_to_array(&a, None, vm)?;
        let r = do_reduce(&arr, args, Reduce::Mean, vm)?;
        Ok(scalar_or_array(r, vm))
    }
    #[pyfunction(name = "amin")]
    fn amin_fn(a: PyObjectRef, args: ReduceArgs, vm: &VirtualMachine) -> PyResult {
        let arr = obj_to_array(&a, None, vm)?;
        let r = do_reduce(&arr, args, Reduce::Min, vm)?;
        Ok(scalar_or_array(r, vm))
    }
    #[pyfunction(name = "amax")]
    fn amax_fn(a: PyObjectRef, args: ReduceArgs, vm: &VirtualMachine) -> PyResult {
        let arr = obj_to_array(&a, None, vm)?;
        let r = do_reduce(&arr, args, Reduce::Max, vm)?;
        Ok(scalar_or_array(r, vm))
    }

    // Aliases for the numpy 2.x function names.
    #[pyfunction(name = "max")]
    fn max_fn(a: PyObjectRef, args: ReduceArgs, vm: &VirtualMachine) -> PyResult {
        amax_fn(a, args, vm)
    }
    #[pyfunction(name = "min")]
    fn min_fn(a: PyObjectRef, args: ReduceArgs, vm: &VirtualMachine) -> PyResult {
        amin_fn(a, args, vm)
    }

    /// `np.argmin(a)` / `np.argmax(a)` — flat-index extremum.
    #[pyfunction]
    fn argmin(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<usize> {
        let arr = obj_to_array(&a, None, vm)?;
        reduce::arg_extremum(&arr, false, vm)
    }
    #[pyfunction]
    fn argmax(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<usize> {
        let arr = obj_to_array(&a, None, vm)?;
        reduce::arg_extremum(&arr, true, vm)
    }

    // ---------------- extras: logical / bitwise / predicates ----------------

    #[pyfunction]
    fn logical_and(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::extras::logical_and(
            &x, &y, vm,
        )?))
    }
    #[pyfunction]
    fn logical_or(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::extras::logical_or(
            &x, &y, vm,
        )?))
    }
    #[pyfunction]
    fn logical_xor(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::extras::logical_xor(
            &x, &y, vm,
        )?))
    }
    #[pyfunction]
    fn logical_not(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::extras::logical_not(&x)))
    }
    #[pyfunction]
    fn bitwise_and(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::extras::bitwise_and(
            &x, &y, vm,
        )?))
    }
    #[pyfunction]
    fn bitwise_or(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::extras::bitwise_or(
            &x, &y, vm,
        )?))
    }
    #[pyfunction]
    fn bitwise_xor(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::extras::bitwise_xor(
            &x, &y, vm,
        )?))
    }
    #[pyfunction]
    fn invert(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::extras::invert(&x, vm)?))
    }
    #[pyfunction(name = "bitwise_not")]
    fn bitwise_not(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        invert(a, vm)
    }
    #[pyfunction]
    fn isnan(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::extras::isnan(&x)))
    }
    #[pyfunction]
    fn isinf(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::extras::isinf(&x)))
    }
    #[pyfunction]
    fn isfinite(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::extras::isfinite(&x)))
    }

    #[derive(FromArgs)]
    pub(crate) struct CloseArgs {
        #[pyarg(positional)]
        a: PyObjectRef,
        #[pyarg(positional)]
        b: PyObjectRef,
        #[pyarg(any, optional)]
        rtol: OptionalArg<f64>,
        #[pyarg(any, optional)]
        atol: OptionalArg<f64>,
        #[pyarg(any, optional)]
        equal_nan: OptionalArg<bool>,
    }

    #[pyfunction]
    fn isclose(args: CloseArgs, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&args.a, None, vm)?;
        let y = obj_to_array(&args.b, None, vm)?;
        let rtol = args.rtol.unwrap_or(1e-5);
        let atol = args.atol.unwrap_or(1e-8);
        let eq_nan = args.equal_nan.unwrap_or(false);
        Ok(PyNdArray::from_arrays(crate::extras::isclose(
            &x, &y, rtol, atol, eq_nan, vm,
        )?))
    }

    #[pyfunction]
    fn allclose(args: CloseArgs, vm: &VirtualMachine) -> PyResult<bool> {
        let x = obj_to_array(&args.a, None, vm)?;
        let y = obj_to_array(&args.b, None, vm)?;
        let rtol = args.rtol.unwrap_or(1e-5);
        let atol = args.atol.unwrap_or(1e-8);
        let eq_nan = args.equal_nan.unwrap_or(false);
        crate::extras::allclose(&x, &y, rtol, atol, eq_nan, vm)
    }

    #[pyfunction]
    fn array_equal(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<bool> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        Ok(crate::extras::array_equal(&x, &y))
    }

    #[pyfunction]
    fn any(a: PyObjectRef, args: ReduceArgs, vm: &VirtualMachine) -> PyResult {
        let arr = obj_to_array(&a, None, vm)?;
        let r = any_all_kw(&arr, args, /*want_all=*/ false, vm)?;
        Ok(scalar_or_array(r, vm))
    }
    #[pyfunction]
    fn all(a: PyObjectRef, args: ReduceArgs, vm: &VirtualMachine) -> PyResult {
        let arr = obj_to_array(&a, None, vm)?;
        let r = any_all_kw(&arr, args, /*want_all=*/ true, vm)?;
        Ok(scalar_or_array(r, vm))
    }

    fn any_all_kw(
        arr: &ArraysD,
        args: ReduceArgs,
        want_all: bool,
        vm: &VirtualMachine,
    ) -> PyResult<ArraysD> {
        let axes = parse_axes(&args.axis, vm)?;
        let keepdims = args.keepdims.unwrap_or(false);
        // None → reduce over all axes; otherwise iterate descending.
        let mut sorted_axes: Vec<usize> = match &axes {
            None => (0..arr.ndim()).collect(),
            Some(list) => {
                let nd = arr.ndim() as isize;
                let mut v: Vec<usize> = Vec::with_capacity(list.len());
                for &ax in list {
                    let na = if ax < 0 { ax + nd } else { ax };
                    if na < 0 || na >= nd {
                        return Err(vm.new_value_error(format!(
                            "axis {ax} out of bounds for {}-D array",
                            arr.ndim()
                        )));
                    }
                    v.push(na as usize);
                }
                v
            }
        };
        sorted_axes.sort_by(|x, y| y.cmp(x));
        let mut current = arr.clone();
        if axes.is_none() {
            let result = if want_all {
                crate::extras::all(&current, None, vm)?
            } else {
                crate::extras::any(&current, None, vm)?
            };
            return apply_keepdims_local(arr, &sorted_axes, result, keepdims, vm);
        }
        for &ax in &sorted_axes {
            current = if want_all {
                crate::extras::all(&current, Some(ax as isize), vm)?
            } else {
                crate::extras::any(&current, Some(ax as isize), vm)?
            };
        }
        apply_keepdims_local(arr, &sorted_axes, current, keepdims, vm)
    }

    fn apply_keepdims_local(
        original: &ArraysD,
        reduced_axes_desc: &[usize],
        reduced: ArraysD,
        keepdims: bool,
        vm: &VirtualMachine,
    ) -> PyResult<ArraysD> {
        if !keepdims {
            return Ok(reduced);
        }
        let mut full_shape: Vec<usize> = original.shape().to_vec();
        for ax in 0..original.ndim() {
            if reduced_axes_desc.contains(&ax) {
                full_shape[ax] = 1;
            }
        }
        crate::linalg::reshape(&reduced, &full_shape)
            .ok_or_else(|| crate::internal::internal(vm, "any/all keepdims reshape failed"))
    }

    #[pyfunction]
    fn cumsum(a: PyObjectRef, args: AxisArg, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::extras::cumsum_axis(
            &arr,
            args.axis.flatten(),
            vm,
        )?))
    }
    #[pyfunction]
    fn cumprod(a: PyObjectRef, args: AxisArg, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::extras::cumprod_axis(
            &arr,
            args.axis.flatten(),
            vm,
        )?))
    }
    #[pyfunction]
    fn diff(a: PyObjectRef, args: AxisArg, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        // numpy default axis is -1; use that when caller doesn't specify.
        let axis = match args.axis {
            OptionalArg::Present(v) => v,
            OptionalArg::Missing => Some(-1),
        };
        Ok(PyNdArray::from_arrays(crate::extras::diff_axis(
            &arr, axis, vm,
        )?))
    }

    #[derive(FromArgs)]
    pub(crate) struct ClipArgs {
        #[pyarg(positional)]
        a: PyObjectRef,
        #[pyarg(positional)]
        a_min: PyObjectRef,
        #[pyarg(positional)]
        a_max: PyObjectRef,
    }
    #[pyfunction]
    fn clip(args: ClipArgs, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&args.a, None, vm)?;
        let lo = if args.a_min.is(&vm.ctx.none) {
            None
        } else {
            Some(args.a_min.try_float(vm)?.to_f64())
        };
        let hi = if args.a_max.is(&vm.ctx.none) {
            None
        } else {
            Some(args.a_max.try_float(vm)?.to_f64())
        };
        Ok(PyNdArray::from_arrays(crate::extras::clip(&arr, lo, hi)))
    }

    #[pyfunction(name = "round")]
    fn round_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::extras::round_half_even(&arr)))
    }
    #[pyfunction]
    fn trunc(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::extras::trunc(&arr)))
    }
    #[pyfunction(name = "fix")]
    fn fix_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        trunc(a, vm)
    }

    #[pyfunction(name = "where")]
    fn where_fn(args: FuncArgs, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        // Pull arguments through `Option::ok_or` instead of `.unwrap()` so a
        // future change to `FuncArgs` that violates the length invariant
        // can't panic.
        let need = |it: &mut std::vec::IntoIter<PyObjectRef>| -> PyResult<PyObjectRef> {
            it.next()
                .ok_or_else(|| crate::internal::internal(vm, "where(): missing argument"))
        };
        match args.args.len() {
            1 => {
                let mut it = args.args.into_iter();
                nonzero(need(&mut it)?, vm)
            }
            3 => {
                let mut it = args.args.into_iter();
                let c = obj_to_array(&need(&mut it)?, None, vm)?;
                let xa = obj_to_array(&need(&mut it)?, None, vm)?;
                let ya = obj_to_array(&need(&mut it)?, None, vm)?;
                Ok(
                    PyNdArray::from_arrays(crate::extras::where_op(&c, &xa, &ya, vm)?)
                        .into_pyobject(vm),
                )
            }
            n => Err(vm.new_type_error(format!(
                "where() takes 1 or 3 positional arguments; got {n}"
            ))),
        }
    }
    #[pyfunction]
    fn nonzero(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        // numpy.nonzero returns a *tuple* of N arrays (one per axis); for our
        // 1-D-style implementation we return a length-1 tuple so callers can
        // do `np.nonzero(a)[0]`.
        let arr = obj_to_array(&a, None, vm)?;
        let idx_arr = PyNdArray::from_arrays(crate::extras::nonzero(&arr));
        let tup = PyTuple::new_ref(vec![idx_arr.into_pyobject(vm)], &vm.ctx);
        Ok(tup.into())
    }

    #[derive(FromArgs)]
    pub(crate) struct SortArgs {
        #[pyarg(positional)]
        a: PyObjectRef,
        #[pyarg(any, optional)]
        axis: OptionalArg<Option<isize>>,
    }
    #[pyfunction]
    fn sort(args: SortArgs, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&args.a, None, vm)?;
        // numpy default is axis=-1; OptionalArg::Missing keeps that.
        let axis = match args.axis {
            OptionalArg::Present(v) => v,
            OptionalArg::Missing => Some(-1),
        };
        Ok(PyNdArray::from_arrays(crate::extras::sort(&arr, axis, vm)?))
    }
    #[pyfunction]
    fn argsort(args: SortArgs, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&args.a, None, vm)?;
        let axis = match args.axis {
            OptionalArg::Present(v) => v,
            OptionalArg::Missing => Some(-1),
        };
        Ok(PyNdArray::from_arrays(crate::extras::argsort(
            &arr, axis, vm,
        )?))
    }

    /// `np.partition(a, kth)` — partition such that element kth ends up in its
    /// sorted position, with smaller before and larger after (in any order).
    /// Our implementation simplifies: fully sorts along the last axis (correct
    /// but not the optimal partitioning algorithm).
    #[pyfunction]
    fn partition(a: PyObjectRef, _kth: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::extras::sort(
            &arr,
            Some(-1),
            vm,
        )?))
    }

    /// `np.argpartition(a, kth)` — argsort along the last axis (similar
    /// simplification as `partition`).
    #[pyfunction]
    fn argpartition(a: PyObjectRef, _kth: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::extras::argsort(
            &arr,
            Some(-1),
            vm,
        )?))
    }

    /// `np.lexsort(keys)` — stable indirect sort using a sequence of keys, last
    /// key as primary.
    #[pyfunction]
    fn lexsort(keys: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let key_arrs = seq_to_arrays(&keys, vm)?;
        if key_arrs.is_empty() {
            return Err(vm.new_value_error("lexsort needs at least one key".to_string()));
        }
        use crate::dtype::CoerceArray;
        // All keys must have the same length; flatten each.
        let flat_keys: Vec<Vec<f64>> = key_arrs
            .iter()
            .map(|a| {
                crate::linalg::flatten(a)
                    .coerce::<f64>()
                    .iter()
                    .copied()
                    .collect()
            })
            .collect();
        let n = flat_keys[0].len();
        if !flat_keys.iter().all(|k| k.len() == n) {
            return Err(
                vm.new_value_error("lexsort: keys must all have the same length".to_string())
            );
        }
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| {
            // Last key is primary; compare in reverse order.
            for key in flat_keys.iter().rev() {
                match key[i]
                    .partial_cmp(&key[j])
                    .unwrap_or(std::cmp::Ordering::Equal)
                {
                    std::cmp::Ordering::Equal => continue,
                    other => return other,
                }
            }
            std::cmp::Ordering::Equal
        });
        let data: Vec<i64> = indices.iter().map(|&i| i as i64).collect();
        let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n]), data)
            .map_err(|e| crate::internal::internal(vm, format!("lexsort: {e}")))?;
        Ok(PyNdArray::from_arrays(ArraysD::I64(arr)))
    }

    #[derive(FromArgs)]
    pub(crate) struct UniqueArgs {
        #[pyarg(any, optional)]
        return_index: OptionalArg<bool>,
        #[pyarg(any, optional)]
        return_inverse: OptionalArg<bool>,
        #[pyarg(any, optional)]
        return_counts: OptionalArg<bool>,
    }

    #[pyfunction]
    fn unique(a: PyObjectRef, args: UniqueArgs, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let arr = obj_to_array(&a, None, vm)?;
        let ret_idx = args.return_index.unwrap_or(false);
        let ret_inv = args.return_inverse.unwrap_or(false);
        let ret_cnt = args.return_counts.unwrap_or(false);
        let flat = crate::linalg::flatten(&arr);
        let uniq = crate::extras::unique(&arr, vm)?;
        if !ret_idx && !ret_inv && !ret_cnt {
            return Ok(PyNdArray::from_arrays(uniq).into_pyobject(vm));
        }
        // For the auxiliary returns, build them by scanning flat against uniq.
        use crate::dtype::CoerceArray;
        let uniq_f = uniq.coerce::<f64>();
        let flat_f = flat.coerce::<f64>();
        let mut out_items: Vec<PyObjectRef> =
            vec![PyNdArray::from_arrays(uniq.clone()).into_pyobject(vm)];
        if ret_idx {
            // index of *first* occurrence in flat for each unique value.
            let mut idx: Vec<i64> = Vec::with_capacity(uniq_f.len());
            for u in uniq_f.iter() {
                let mut found = -1i64;
                for (i, v) in flat_f.iter().enumerate() {
                    if v == u || (v.is_nan() && u.is_nan()) {
                        found = i as i64;
                        break;
                    }
                }
                idx.push(found);
            }
            let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[idx.len()]), idx)
                .map_err(|e| crate::internal::internal(vm, format!("unique idx: {e}")))?;
            out_items.push(PyNdArray::from_arrays(ArraysD::I64(arr)).into_pyobject(vm));
        }
        if ret_inv {
            // For each element of flat, the index into uniq.
            let mut inv: Vec<i64> = Vec::with_capacity(flat_f.len());
            for v in flat_f.iter() {
                let mut found = -1i64;
                for (i, u) in uniq_f.iter().enumerate() {
                    if v == u || (v.is_nan() && u.is_nan()) {
                        found = i as i64;
                        break;
                    }
                }
                inv.push(found);
            }
            let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[inv.len()]), inv)
                .map_err(|e| crate::internal::internal(vm, format!("unique inv: {e}")))?;
            out_items.push(PyNdArray::from_arrays(ArraysD::I64(arr)).into_pyobject(vm));
        }
        if ret_cnt {
            // Number of times each unique value appears in flat.
            let mut counts: Vec<i64> = Vec::with_capacity(uniq_f.len());
            for u in uniq_f.iter() {
                let mut c = 0i64;
                for v in flat_f.iter() {
                    if v == u || (v.is_nan() && u.is_nan()) {
                        c += 1;
                    }
                }
                counts.push(c);
            }
            let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[counts.len()]), counts)
                .map_err(|e| crate::internal::internal(vm, format!("unique counts: {e}")))?;
            out_items.push(PyNdArray::from_arrays(ArraysD::I64(arr)).into_pyobject(vm));
        }
        Ok(PyTuple::new_ref(out_items, &vm.ctx).into())
    }

    fn seq_to_arrays(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<Vec<ArraysD>> {
        let list = obj
            .downcast_ref::<rustpython_vm::builtins::PyList>()
            .map(|l| l.borrow_vec().to_vec())
            .or_else(|| obj.downcast_ref::<PyTuple>().map(|t| t.as_slice().to_vec()))
            .ok_or_else(|| vm.new_type_error("expected sequence of arrays".to_string()))?;
        list.iter().map(|o| obj_to_array(o, None, vm)).collect()
    }

    #[pyfunction]
    fn stack(
        arrays: PyObjectRef,
        axis: OptionalArg<usize>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let arrs = seq_to_arrays(&arrays, vm)?;
        Ok(PyNdArray::from_arrays(crate::extras::stack(
            &arrs,
            axis.unwrap_or(0),
            vm,
        )?))
    }
    #[pyfunction]
    fn hstack(arrays: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arrs = seq_to_arrays(&arrays, vm)?;
        Ok(PyNdArray::from_arrays(crate::extras::hstack(&arrs, vm)?))
    }
    #[pyfunction]
    fn vstack(arrays: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arrs = seq_to_arrays(&arrays, vm)?;
        Ok(PyNdArray::from_arrays(crate::extras::vstack(&arrs, vm)?))
    }

    #[pyfunction]
    fn squeeze(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::extras::squeeze(&arr, vm)?))
    }
    #[derive(FromArgs)]
    pub(crate) struct ExpandDimsArgs {
        #[pyarg(any)]
        axis: PyObjectRef,
    }

    #[pyfunction]
    fn expand_dims(
        a: PyObjectRef,
        args: ExpandDimsArgs,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        // numpy: axis can be int or tuple of ints; we expand them in order.
        let axes: Vec<isize> = if let Some(t) = args.axis.downcast_ref::<PyTuple>() {
            t.as_slice()
                .iter()
                .map(|o| o.try_int(vm)?.try_to_primitive::<isize>(vm))
                .collect::<PyResult<_>>()?
        } else if let Some(l) = args.axis.downcast_ref::<rustpython_vm::builtins::PyList>() {
            l.borrow_vec()
                .iter()
                .map(|o| o.try_int(vm)?.try_to_primitive::<isize>(vm))
                .collect::<PyResult<_>>()?
        } else {
            vec![args.axis.try_int(vm)?.try_to_primitive::<isize>(vm)?]
        };
        let mut sorted = axes.clone();
        sorted.sort();
        let target_nd = arr.ndim() + sorted.len();
        let mut out = arr;
        // For each requested axis (in ascending order), insert a length-1 dim.
        for ax in sorted {
            let nd_after = out.ndim() + 1;
            let pos = if ax < 0 {
                (ax + target_nd as isize).max(0) as usize
            } else {
                (ax as usize).min(nd_after - 1)
            };
            out = crate::extras::expand_dims(&out, pos, vm)?;
        }
        Ok(PyNdArray::from_arrays(out))
    }
    #[pyfunction]
    fn broadcast_to(
        a: PyObjectRef,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        let s = parse_shape(&shape, vm)?;
        Ok(PyNdArray::from_arrays(crate::extras::broadcast_to(
            &arr, &s, vm,
        )?))
    }
    #[derive(FromArgs)]
    pub(crate) struct RepeatArgs {
        #[pyarg(any)]
        repeats: PyObjectRef,
        #[pyarg(any, optional)]
        axis: OptionalArg<Option<isize>>,
    }

    #[pyfunction(name = "repeat")]
    fn repeat_fn(a: PyObjectRef, args: RepeatArgs, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        // `repeats` can be an int or an array. If int → repeat each element n times.
        // Per-element repeats (array of repeats) not yet supported.
        let n = if let Some(arr_obj) = args.repeats.downcast_ref::<PyNdArray>() {
            if arr_obj.view().len() == 1 {
                use crate::dtype::CoerceArray;
                arr_obj
                    .view()
                    .coerce::<i64>()
                    .iter()
                    .next()
                    .copied()
                    .unwrap_or(0) as usize
            } else {
                return Err(vm.new_not_implemented_error(
                    "repeat: per-element repeat counts not yet implemented".to_string(),
                ));
            }
        } else {
            args.repeats.try_int(vm)?.try_to_primitive::<usize>(vm)?
        };
        let axis = match args.axis {
            OptionalArg::Missing => None,
            OptionalArg::Present(v) => v,
        };
        let target = match axis {
            None => crate::linalg::flatten(&arr),
            Some(_) => arr, // numpy.repeat with axis: keep shape but repeat along axis
        };
        // For axis=None: simply repeat each element n times → 1-D result.
        // For axis=Some(k): repeat each slice along axis k (matches our flat repeat
        // when k is 0 on a 1-D array).
        Ok(PyNdArray::from_arrays(crate::extras::repeat(&target, n)))
    }
    #[pyfunction(name = "tile")]
    fn tile_fn(a: PyObjectRef, reps: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        // reps can be an int or a tuple of ints. For tuple form, we tile along
        // each axis in turn; pad ndim with leading 1s if reps has more entries.
        let reps_vec: Vec<usize> = if let Some(t) = reps.downcast_ref::<PyTuple>() {
            t.as_slice()
                .iter()
                .map(|o| {
                    o.try_int(vm)?
                        .try_to_primitive::<isize>(vm)
                        .map(|v| v.max(0) as usize)
                })
                .collect::<PyResult<_>>()?
        } else if let Some(l) = reps.downcast_ref::<rustpython_vm::builtins::PyList>() {
            l.borrow_vec()
                .iter()
                .map(|o| {
                    o.try_int(vm)?
                        .try_to_primitive::<isize>(vm)
                        .map(|v| v.max(0) as usize)
                })
                .collect::<PyResult<_>>()?
        } else {
            vec![reps.try_int(vm)?.try_to_primitive::<usize>(vm)?]
        };
        if reps_vec.len() == 1 {
            return Ok(PyNdArray::from_arrays(crate::extras::tile(
                &arr,
                reps_vec[0],
            )));
        }
        // Multi-axis tile: pad shape with leading 1s if needed, then tile along each.
        let mut current = arr;
        while current.ndim() < reps_vec.len() {
            let mut new_shape = vec![1usize];
            new_shape.extend(current.shape());
            current = crate::linalg::reshape(&current, &new_shape)
                .ok_or_else(|| crate::internal::internal(vm, "tile: shape pad failed"))?;
        }
        let offset = current.ndim() - reps_vec.len();
        for (i, &r) in reps_vec.iter().enumerate() {
            let axis = offset + i;
            current = tile_along_axis(&current, axis, r, vm)?;
        }
        Ok(PyNdArray::from_arrays(current))
    }

    /// Repeat the array `r` times along `axis`, concatenating the copies.
    fn tile_along_axis(
        a: &ArraysD,
        axis: usize,
        r: usize,
        vm: &VirtualMachine,
    ) -> PyResult<ArraysD> {
        if r == 0 {
            // Zero-length along axis.
            let mut s = a.shape().to_vec();
            s[axis] = 0;
            return crate::linalg::reshape(a, &s)
                .or_else(|| {
                    // If reshape from non-empty to empty isn't possible, build
                    // a fresh zero-array.
                    Some(empty_with_shape(a.dtype(), &s))
                })
                .ok_or_else(|| crate::internal::internal(vm, "tile r=0"));
        }
        if r == 1 {
            return Ok(a.clone());
        }
        let parts: Vec<ArraysD> = (0..r).map(|_| a.clone()).collect();
        crate::linalg::concatenate(&parts, axis, vm)
    }

    fn empty_with_shape(dt: DType, shape: &[usize]) -> ArraysD {
        crate::create::zeros(shape, dt)
    }

    #[pyfunction]
    fn ptp(a: PyObjectRef, vm: &VirtualMachine) -> PyResult {
        let arr = obj_to_array(&a, None, vm)?;
        let r = crate::extras::ptp(&arr, vm)?;
        Ok(scalar_or_array(r, vm))
    }
    #[pyfunction]
    fn median(a: PyObjectRef, args: ReduceArgs, vm: &VirtualMachine) -> PyResult {
        let arr = obj_to_array(&a, None, vm)?;
        let axes = parse_axes(&args.axis, vm)?;
        let keepdims = args.keepdims.unwrap_or(false);
        let r = median_axis(&arr, axes.as_deref(), keepdims, vm)?;
        Ok(scalar_or_array(r, vm))
    }

    /// Median over the given axes (or full flat for `None`). For tuple axes,
    /// we permute & flatten the target axes into one, then take the median
    /// along that.
    fn median_axis(
        arr: &ArraysD,
        axes: Option<&[isize]>,
        keepdims: bool,
        vm: &VirtualMachine,
    ) -> PyResult<ArraysD> {
        use crate::dtype::CoerceArray;
        let nd = arr.ndim();
        // Normalize axes.
        let norm: Vec<usize> = match axes {
            None => (0..nd).collect(),
            Some(list) => {
                let mut v: Vec<usize> = Vec::with_capacity(list.len());
                for &ax in list {
                    let na = if ax < 0 { ax + nd as isize } else { ax };
                    if na < 0 || na >= nd as isize {
                        return Err(vm.new_value_error(format!(
                            "median: axis {ax} out of bounds for {nd}-D"
                        )));
                    }
                    v.push(na as usize);
                }
                v
            }
        };
        // Full flatten case.
        if norm.len() == nd {
            let res = crate::extras::median(arr, vm)?;
            return if keepdims {
                let new_shape = vec![1usize; nd];
                crate::linalg::reshape(&res, &new_shape)
                    .ok_or_else(|| crate::internal::internal(vm, "median keepdims reshape"))
            } else {
                Ok(res)
            };
        }
        // Permute target axes to the front, then merge.
        let mut perm: Vec<usize> = norm.iter().copied().collect();
        for ax in 0..nd {
            if !perm.contains(&ax) {
                perm.push(ax);
            }
        }
        // Materialize the transposed array contiguously.
        let f = arr.coerce::<f64>();
        let permuted = f.view().permuted_axes(ndarray::IxDyn(&perm));
        let permuted_shape: Vec<usize> = permuted.shape().to_vec();
        let permuted_data: Vec<f64> = permuted.iter().copied().collect();
        let target_axes_size: usize = norm.iter().map(|&i| arr.shape()[i]).product();
        let outer_shape: Vec<usize> = permuted_shape[norm.len()..].to_vec();
        // Re-shape into (target_axes_size, outer...) then take median along axis 0.
        let mut merged_shape = vec![target_axes_size];
        merged_shape.extend(&outer_shape);
        let merged = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&merged_shape), permuted_data)
            .map_err(|e| crate::internal::internal(vm, format!("median merge: {e}")))?;
        // Now compute median along axis 0 of merged.
        let out_data: Vec<f64> = if outer_shape.is_empty() {
            // Whole flat-flat case (shouldn't happen given the early return),
            // but fall back to sorting all values.
            let mut v: Vec<f64> = merged.iter().copied().collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            vec![if v.is_empty() {
                f64::NAN
            } else if v.len() % 2 == 1 {
                v[v.len() / 2]
            } else {
                0.5 * (v[v.len() / 2 - 1] + v[v.len() / 2])
            }]
        } else {
            let mut out = Vec::with_capacity(outer_shape.iter().product::<usize>());
            // Walk the outer index in C-order.
            let outer_size: usize = outer_shape.iter().product::<usize>().max(1);
            for outer_i in 0..outer_size {
                // Compute multi-index for outer
                let mut idx = vec![0usize; outer_shape.len()];
                let mut rem = outer_i;
                for d in (0..outer_shape.len()).rev() {
                    idx[d] = rem % outer_shape[d];
                    rem /= outer_shape[d];
                }
                let mut col: Vec<f64> = Vec::with_capacity(target_axes_size);
                for k in 0..target_axes_size {
                    let mut full_idx = vec![k];
                    full_idx.extend(&idx);
                    col.push(merged[ndarray::IxDyn(&full_idx)]);
                }
                col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let m = if col.is_empty() {
                    f64::NAN
                } else if col.len() % 2 == 1 {
                    col[col.len() / 2]
                } else {
                    0.5 * (col[col.len() / 2 - 1] + col[col.len() / 2])
                };
                out.push(m);
            }
            out
        };
        // Final shape: outer_shape, possibly with the reduced axes restored as 1s.
        let mut final_shape = outer_shape.clone();
        if keepdims {
            final_shape = arr.shape().to_vec();
            for &ax in &norm {
                final_shape[ax] = 1;
            }
        }
        let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&final_shape), out_data)
            .map_err(|e| crate::internal::internal(vm, format!("median out: {e}")))?;
        Ok(ArraysD::F64(arr))
    }

    #[pyfunction]
    fn trace(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::linalg_extra::trace(
            &arr, vm,
        )?))
    }
    #[pyfunction]
    fn cross(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::linalg_extra::cross(
            &x, &y, vm,
        )?))
    }

    // ---------------- more_ops: flip / roll / column_stack / diag / atleast / etc. ----------------

    #[pyfunction]
    fn flip(
        a: PyObjectRef,
        axis: OptionalArg<Option<isize>>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::more_ops::flip(
            &arr,
            axis.flatten(),
        )))
    }
    #[pyfunction]
    fn flipud(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::more_ops::flipud(&arr)))
    }
    #[pyfunction]
    fn fliplr(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::more_ops::fliplr(&arr, vm)?))
    }
    #[pyfunction]
    fn roll(a: PyObjectRef, shift: isize, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::more_ops::roll(
            &arr, shift, vm,
        )?))
    }
    #[pyfunction]
    fn rot90(a: PyObjectRef, args: KArg, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::more_ops::rot90(
            &arr,
            args.k.unwrap_or(1),
            vm,
        )?))
    }

    #[pyfunction]
    fn column_stack(arrays: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arrs = seq_to_arrays(&arrays, vm)?;
        Ok(PyNdArray::from_arrays(crate::more_ops::column_stack(
            &arrs, vm,
        )?))
    }
    #[pyfunction]
    fn dstack(arrays: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arrs = seq_to_arrays(&arrays, vm)?;
        Ok(PyNdArray::from_arrays(crate::more_ops::dstack(&arrs, vm)?))
    }

    #[derive(FromArgs)]
    pub(crate) struct KArg {
        #[pyarg(any, optional)]
        k: OptionalArg<isize>,
    }

    #[pyfunction]
    fn diag(a: PyObjectRef, args: KArg, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::more_ops::diag(
            &arr,
            args.k.unwrap_or(0),
            vm,
        )?))
    }
    #[pyfunction]
    fn diagflat(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::more_ops::diagflat(&arr)))
    }
    #[pyfunction]
    fn triu(a: PyObjectRef, args: KArg, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::more_ops::triu(
            &arr,
            args.k.unwrap_or(0),
            vm,
        )?))
    }
    #[pyfunction]
    fn tril(a: PyObjectRef, args: KArg, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::more_ops::tril(
            &arr,
            args.k.unwrap_or(0),
            vm,
        )?))
    }

    #[derive(FromArgs)]
    pub(crate) struct TriArgs {
        #[pyarg(positional)]
        n: usize,
        #[pyarg(any, optional)]
        m: OptionalArg<usize>,
        #[pyarg(any, optional)]
        k: OptionalArg<isize>,
        #[pyarg(any, optional)]
        dtype: OptionalArg<PyObjectRef>,
    }
    #[pyfunction(name = "tri")]
    fn tri_fn(args: TriArgs, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let m = args.m.unwrap_or(args.n);
        let k = args.k.unwrap_or(0);
        let dt =
            crate::convert::parse_dtype_arg(&args.dtype.into_option(), vm)?.unwrap_or(DType::F64);
        Ok(PyNdArray::from_arrays(crate::more_ops::tri(
            args.n, m, k, dt,
        )))
    }

    #[pyfunction]
    fn atleast_1d(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::more_ops::atleast_1d(&arr)))
    }
    #[pyfunction]
    fn atleast_2d(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::more_ops::atleast_2d(&arr)))
    }
    #[pyfunction]
    fn atleast_3d(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::more_ops::atleast_3d(&arr)))
    }

    #[pyfunction]
    fn count_nonzero(a: PyObjectRef, vm: &VirtualMachine) -> PyResult {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(scalar_or_array(crate::more_ops::count_nonzero(&arr), vm))
    }
    #[derive(FromArgs)]
    pub(crate) struct BincountArgs {
        #[pyarg(any, optional)]
        weights: OptionalArg<PyObjectRef>,
        #[pyarg(any, optional)]
        minlength: OptionalArg<usize>,
    }

    #[pyfunction]
    fn bincount(a: PyObjectRef, args: BincountArgs, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        use crate::dtype::CoerceArray;
        let flat = arr.coerce::<i64>();
        let max_val = flat.iter().copied().max().unwrap_or(0).max(0) as usize;
        let minlen = args.minlength.unwrap_or(0);
        let n = (max_val + 1).max(minlen);
        // weights
        let weights = match &args.weights {
            OptionalArg::Missing => None,
            OptionalArg::Present(o) if o.is(&vm.ctx.none) => None,
            OptionalArg::Present(o) => {
                let w_arr = obj_to_array(o, None, vm)?;
                let w_f = w_arr.coerce::<f64>();
                if w_f.len() != flat.len() {
                    return Err(vm.new_value_error(format!(
                        "bincount: weights length {} != input length {}",
                        w_f.len(),
                        flat.len()
                    )));
                }
                Some(w_f)
            }
        };
        if let Some(w) = weights {
            let mut counts = vec![0.0f64; n];
            for (i, &v) in flat.iter().enumerate() {
                if v < 0 {
                    return Err(
                        vm.new_value_error("bincount: input must be non-negative".to_string())
                    );
                }
                let idx = v as usize;
                if idx < n {
                    counts[idx] += w[ndarray::IxDyn(&[i])];
                }
            }
            let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n]), counts)
                .map_err(|e| crate::internal::internal(vm, format!("bincount w: {e}")))?;
            Ok(PyNdArray::from_arrays(ArraysD::F64(arr)))
        } else {
            let mut counts = vec![0i64; n];
            for &v in flat.iter() {
                if v < 0 {
                    return Err(
                        vm.new_value_error("bincount: input must be non-negative".to_string())
                    );
                }
                let idx = v as usize;
                if idx < n {
                    counts[idx] += 1;
                }
            }
            let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n]), counts)
                .map_err(|e| crate::internal::internal(vm, format!("bincount: {e}")))?;
            Ok(PyNdArray::from_arrays(ArraysD::I64(arr)))
        }
    }

    #[derive(FromArgs)]
    pub(crate) struct HistArgs {
        #[pyarg(positional)]
        a: PyObjectRef,
        #[pyarg(any, optional)]
        bins: OptionalArg<usize>,
        #[pyarg(any, optional)]
        range: OptionalArg<PyObjectRef>,
    }
    #[pyfunction]
    fn histogram(args: HistArgs, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let arr = obj_to_array(&args.a, None, vm)?;
        let bins = args.bins.unwrap_or(10);
        let range = match args.range.into_option() {
            None => None,
            Some(o) if o.is(&vm.ctx.none) => None,
            Some(o) => {
                let s = crate::convert::parse_shape_signed(&o, vm)?;
                if s.len() != 2 {
                    return Err(vm.new_value_error("range must be a (lo, hi) pair".to_string()));
                }
                Some((s[0] as f64, s[1] as f64))
            }
        };
        let (counts, edges) = crate::more_ops::histogram(&arr, bins, range, vm)?;
        let tup = PyTuple::new_ref(
            vec![
                PyNdArray::from_arrays(counts).into_pyobject(vm),
                PyNdArray::from_arrays(edges).into_pyobject(vm),
            ],
            &vm.ctx,
        );
        Ok(tup.into())
    }

    #[pyfunction]
    fn nansum(a: PyObjectRef, vm: &VirtualMachine) -> PyResult {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(scalar_or_array(crate::more_ops::nansum(&arr), vm))
    }
    #[pyfunction]
    fn nanmean(a: PyObjectRef, vm: &VirtualMachine) -> PyResult {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(scalar_or_array(crate::more_ops::nanmean(&arr), vm))
    }
    #[pyfunction]
    fn nanmin(a: PyObjectRef, vm: &VirtualMachine) -> PyResult {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(scalar_or_array(crate::more_ops::nanmin(&arr), vm))
    }
    #[pyfunction]
    fn nanmax(a: PyObjectRef, vm: &VirtualMachine) -> PyResult {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(scalar_or_array(crate::more_ops::nanmax(&arr), vm))
    }
    #[pyfunction]
    fn nanstd(a: PyObjectRef, ddof: OptionalArg<usize>, vm: &VirtualMachine) -> PyResult {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(scalar_or_array(
            crate::more_ops::nanstd(&arr, ddof.unwrap_or(0)),
            vm,
        ))
    }
    #[pyfunction]
    fn nanvar(a: PyObjectRef, ddof: OptionalArg<usize>, vm: &VirtualMachine) -> PyResult {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(scalar_or_array(
            crate::more_ops::nanvar(&arr, ddof.unwrap_or(0)),
            vm,
        ))
    }
    #[pyfunction]
    fn nanmedian(a: PyObjectRef, vm: &VirtualMachine) -> PyResult {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(scalar_or_array(crate::more_ops::nanmedian(&arr), vm))
    }

    #[derive(FromArgs)]
    pub(crate) struct SearchSortedArgs {
        #[pyarg(any, optional)]
        side: OptionalArg<PyObjectRef>,
        #[pyarg(any, optional)]
        sorter: OptionalArg<PyObjectRef>,
    }

    #[pyfunction]
    fn searchsorted(
        a: PyObjectRef,
        v: PyObjectRef,
        args: SearchSortedArgs,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let aa = obj_to_array(&a, None, vm)?;
        let va = obj_to_array(&v, None, vm)?;
        // Apply sorter if provided.
        let aa = match &args.sorter {
            OptionalArg::Missing => aa,
            OptionalArg::Present(o) if o.is(&vm.ctx.none) => aa,
            OptionalArg::Present(o) => {
                let sorter = obj_to_array(o, None, vm)?;
                use crate::dtype::CoerceArray;
                let s_i: Vec<usize> = sorter.coerce::<i64>().iter().map(|&i| i as usize).collect();
                let flat = crate::linalg::flatten(&aa);
                let flat_f = flat.coerce::<f64>();
                let reordered: Vec<f64> =
                    s_i.iter().map(|&i| flat_f[ndarray::IxDyn(&[i])]).collect();
                let arr =
                    ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[reordered.len()]), reordered)
                        .map_err(|e| crate::internal::internal(vm, format!("sorter: {e}")))?;
                ArraysD::F64(arr).cast(aa.dtype())
            }
        };
        let side = match &args.side {
            OptionalArg::Missing => "left".to_string(),
            OptionalArg::Present(o) if o.is(&vm.ctx.none) => "left".to_string(),
            OptionalArg::Present(o) => o
                .downcast_ref::<rustpython_vm::builtins::PyStr>()
                .map(|s| s.as_wtf8().to_string_lossy().into_owned())
                .ok_or_else(|| vm.new_type_error("searchsorted side= must be str".to_string()))?,
        };
        if side != "left" && side != "right" {
            return Err(vm.new_value_error(format!("invalid side: {side:?}")));
        }
        // The current more_ops::searchsorted uses left-bisect; for "right" we
        // emulate by shifting equal-key matches up by one.
        let base = crate::more_ops::searchsorted(&aa, &va);
        if side == "left" {
            return Ok(PyNdArray::from_arrays(base));
        }
        use crate::dtype::CoerceArray;
        let aa_f = aa.coerce::<f64>();
        let va_f = va.coerce::<f64>();
        let base_i = base.coerce::<i64>();
        let mut out = Vec::with_capacity(base_i.len());
        for (i, idx_i) in base_i.iter().copied().enumerate() {
            // For "right": advance while a[idx] == v.
            let needle = va_f[ndarray::IxDyn(&[i])];
            let mut k = idx_i as usize;
            while k < aa_f.len() && aa_f[ndarray::IxDyn(&[k])] == needle {
                k += 1;
            }
            out.push(k as i64);
        }
        let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[out.len()]), out)
            .map_err(|e| crate::internal::internal(vm, format!("searchsorted right: {e}")))?;
        Ok(PyNdArray::from_arrays(ArraysD::I64(arr)))
    }

    #[pyfunction]
    fn meshgrid(x: PyObjectRef, y: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let xa = obj_to_array(&x, None, vm)?;
        let ya = obj_to_array(&y, None, vm)?;
        let (xx, yy) = crate::more_ops::meshgrid(&xa, &ya);
        let tup = PyTuple::new_ref(
            vec![
                PyNdArray::from_arrays(xx).into_pyobject(vm),
                PyNdArray::from_arrays(yy).into_pyobject(vm),
            ],
            &vm.ctx,
        );
        Ok(tup.into())
    }

    #[pyfunction]
    fn interp(
        x: PyObjectRef,
        xp: PyObjectRef,
        fp: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let xa = obj_to_array(&x, None, vm)?;
        let xpa = obj_to_array(&xp, None, vm)?;
        let fpa = obj_to_array(&fp, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::more_ops::interp(
            &xa, &xpa, &fpa,
        )))
    }

    #[pyfunction]
    fn trapezoid(y: PyObjectRef, dx: OptionalArg<f64>, vm: &VirtualMachine) -> PyResult {
        let arr = obj_to_array(&y, None, vm)?;
        Ok(scalar_or_array(
            crate::more_ops::trapz(&arr, dx.unwrap_or(1.0)),
            vm,
        ))
    }

    #[pyfunction]
    fn gradient(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::more_ops::gradient(&arr)))
    }

    #[pyfunction(name = "delete")]
    fn delete_fn(a: PyObjectRef, idx: usize, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::more_ops::delete(
            &arr, idx, vm,
        )?))
    }

    #[pyfunction]
    fn append(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let aa = obj_to_array(&a, None, vm)?;
        let bb = obj_to_array(&b, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::more_ops::append(
            &aa, &bb, vm,
        )?))
    }

    // ---------------- trig & stats fillers ----------------

    #[pyfunction]
    fn arctan2(y: PyObjectRef, x: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let ya = obj_to_array(&y, None, vm)?;
        let xa = obj_to_array(&x, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::extras2::arctan2(
            &ya, &xa, vm,
        )?))
    }
    #[pyfunction(name = "atan2")]
    fn atan2_fn(y: PyObjectRef, x: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        arctan2(y, x, vm)
    }
    #[pyfunction]
    fn hypot(x: PyObjectRef, y: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let xa = obj_to_array(&x, None, vm)?;
        let ya = obj_to_array(&y, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::extras2::hypot(&xa, &ya, vm)?))
    }
    #[pyfunction]
    fn deg2rad(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::extras2::deg2rad(&arr)))
    }
    #[pyfunction(name = "radians")]
    fn radians_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        deg2rad(a, vm)
    }
    #[pyfunction]
    fn rad2deg(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::extras2::rad2deg(&arr)))
    }
    #[pyfunction(name = "degrees")]
    fn degrees_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        rad2deg(a, vm)
    }
    #[pyfunction]
    fn unwrap(
        a: PyObjectRef,
        discont: OptionalArg<f64>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        let d = discont.unwrap_or(std::f64::consts::PI);
        Ok(PyNdArray::from_arrays(crate::extras2::unwrap(&arr, d)))
    }

    #[derive(FromArgs)]
    pub(crate) struct AverageArgs {
        #[pyarg(positional)]
        a: PyObjectRef,
        #[pyarg(any, optional)]
        weights: OptionalArg<PyObjectRef>,
    }
    #[pyfunction]
    fn average(args: AverageArgs, vm: &VirtualMachine) -> PyResult {
        let a = obj_to_array(&args.a, None, vm)?;
        let w = match args.weights {
            OptionalArg::Present(w) if !w.is(&vm.ctx.none) => Some(obj_to_array(&w, None, vm)?),
            _ => None,
        };
        let r = crate::extras2::average(&a, w.as_ref(), vm)?;
        Ok(scalar_or_array(r, vm))
    }

    #[pyfunction]
    fn percentile(a: PyObjectRef, q: PyObjectRef, vm: &VirtualMachine) -> PyResult {
        percentile_or_quantile(a, q, /*scale_to_1=*/ true, vm)
    }
    #[pyfunction]
    fn quantile(a: PyObjectRef, q: PyObjectRef, vm: &VirtualMachine) -> PyResult {
        percentile_or_quantile(a, q, /*scale_to_1=*/ false, vm)
    }

    /// Shared body: percentile uses `q in [0,100]`, quantile uses `q in [0,1]`.
    fn percentile_or_quantile(
        a: PyObjectRef,
        q: PyObjectRef,
        is_percentile: bool,
        vm: &VirtualMachine,
    ) -> PyResult {
        let arr = obj_to_array(&a, None, vm)?;
        // Accept scalar q or array q.
        if let Some(q_arr_ref) = q.downcast_ref::<PyNdArray>() {
            use crate::dtype::CoerceArray;
            let q_f = q_arr_ref.view().coerce::<f64>();
            let mut out = Vec::with_capacity(q_f.len());
            for &qv in q_f.iter() {
                let scaled = if is_percentile { qv } else { qv * 100.0 };
                out.push(crate::extras2::percentile_scalar(&arr, scaled, vm)?);
            }
            let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(q_f.shape()), out)
                .map_err(|e| crate::internal::internal(vm, format!("perc: {e}")))?;
            return Ok(PyNdArray::from_arrays(ArraysD::F64(arr)).into_pyobject(vm));
        }
        if let Some(lst) = q.downcast_ref::<rustpython_vm::builtins::PyList>() {
            let mut out = Vec::with_capacity(lst.borrow_vec().len());
            for item in lst.borrow_vec().iter() {
                let qv = item.try_float(vm)?.to_f64();
                let scaled = if is_percentile { qv } else { qv * 100.0 };
                out.push(crate::extras2::percentile_scalar(&arr, scaled, vm)?);
            }
            let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[out.len()]), out)
                .map_err(|e| crate::internal::internal(vm, format!("perc: {e}")))?;
            return Ok(PyNdArray::from_arrays(ArraysD::F64(arr)).into_pyobject(vm));
        }
        let qf = q.try_float(vm)?.to_f64();
        let r = if is_percentile {
            crate::extras2::percentile(&arr, qf, vm)?
        } else {
            crate::extras2::quantile(&arr, qf, vm)?
        };
        Ok(scalar_or_array(r, vm))
    }
    #[pyfunction]
    fn cov(m: PyObjectRef, ddof: OptionalArg<usize>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&m, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::extras2::cov(
            &arr,
            ddof.unwrap_or(1),
            vm,
        )?))
    }
    #[pyfunction]
    fn corrcoef(m: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&m, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::extras2::corrcoef(&arr, vm)?))
    }

    // ---------------- text I/O (savetxt/loadtxt) ----------------

    #[derive(FromArgs)]
    pub(crate) struct SavetxtArgs {
        #[pyarg(positional)]
        fname: PyObjectRef,
        #[pyarg(positional)]
        a: PyObjectRef,
        #[pyarg(any, optional)]
        delimiter: OptionalArg<PyObjectRef>,
        #[pyarg(any, optional)]
        header: OptionalArg<PyObjectRef>,
        #[pyarg(any, optional)]
        comments: OptionalArg<PyObjectRef>,
        #[pyarg(any, optional)]
        fmt: OptionalArg<PyObjectRef>,
    }
    #[pyfunction]
    fn savetxt(args: SavetxtArgs, vm: &VirtualMachine) -> PyResult<()> {
        let path = path_arg(&args.fname, vm)?;
        let arr = obj_to_array(&args.a, None, vm)?;
        let delim = str_arg_default(&args.delimiter.into_option(), " ", vm)?;
        let header = match args.header.into_option() {
            Some(o) if !o.is(&vm.ctx.none) => Some(str_arg(&o, vm)?),
            _ => None,
        };
        let comments = str_arg_default(&args.comments.into_option(), "# ", vm)?;
        let fmt = str_arg_default(&args.fmt.into_option(), "%.18e", vm)?;
        crate::textio::savetxt(
            std::path::Path::new(&path),
            &arr,
            &delim,
            header.as_deref(),
            &comments,
            &fmt,
        )
        .map_err(|e| vm.new_os_error(format!("savetxt: {e}")))
    }

    #[derive(FromArgs)]
    pub(crate) struct LoadtxtArgs {
        #[pyarg(positional)]
        fname: PyObjectRef,
        #[pyarg(any, optional)]
        delimiter: OptionalArg<PyObjectRef>,
        #[pyarg(any, optional)]
        comments: OptionalArg<PyObjectRef>,
        #[pyarg(any, optional)]
        skiprows: OptionalArg<usize>,
    }
    #[pyfunction]
    fn loadtxt(args: LoadtxtArgs, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let path = path_arg(&args.fname, vm)?;
        let delim_opt = args.delimiter.into_option();
        let delim: Option<String> = match delim_opt {
            Some(o) if !o.is(&vm.ctx.none) => Some(str_arg(&o, vm)?),
            _ => None,
        };
        let comments = str_arg_default(&args.comments.into_option(), "#", vm)?;
        let skip = args.skiprows.unwrap_or(0);
        let arr = crate::textio::loadtxt(
            std::path::Path::new(&path),
            delim.as_deref(),
            &comments,
            skip,
        )
        .map_err(|e| vm.new_os_error(format!("loadtxt: {e}")))?;
        Ok(PyNdArray::from_arrays(arr))
    }

    #[pyfunction]
    fn tofile(file: PyObjectRef, a: PyObjectRef, vm: &VirtualMachine) -> PyResult<()> {
        let path = path_arg(&file, vm)?;
        let arr = obj_to_array(&a, None, vm)?;
        crate::textio::tofile(std::path::Path::new(&path), &arr)
            .map_err(|e| vm.new_os_error(format!("tofile: {e}")))
    }

    #[derive(FromArgs)]
    pub(crate) struct FromfileArgs {
        #[pyarg(positional)]
        file: PyObjectRef,
        #[pyarg(any, optional)]
        dtype: OptionalArg<PyObjectRef>,
        #[pyarg(any, optional)]
        count: OptionalArg<isize>,
    }
    #[pyfunction]
    fn fromfile(args: FromfileArgs, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let path = path_arg(&args.file, vm)?;
        let dt = parse_dtype_arg(&args.dtype.into_option(), vm)?.unwrap_or(DType::F64);
        let count = args.count.unwrap_or(-1);
        let arr = crate::textio::fromfile(std::path::Path::new(&path), dt, count)
            .map_err(|e| vm.new_os_error(format!("fromfile: {e}")))?;
        Ok(PyNdArray::from_arrays(arr))
    }

    fn path_arg(o: &PyObjectRef, vm: &VirtualMachine) -> PyResult<String> {
        Ok(o.downcast_ref::<rustpython_vm::builtins::PyStr>()
            .ok_or_else(|| vm.new_type_error("file argument must be a str path".to_string()))?
            .as_wtf8()
            .to_string_lossy()
            .into_owned())
    }
    fn str_arg(o: &PyObjectRef, vm: &VirtualMachine) -> PyResult<String> {
        Ok(o.downcast_ref::<rustpython_vm::builtins::PyStr>()
            .ok_or_else(|| vm.new_type_error("expected a string".to_string()))?
            .as_wtf8()
            .to_string_lossy()
            .into_owned())
    }
    fn str_arg_default(
        o: &Option<PyObjectRef>,
        default: &str,
        vm: &VirtualMachine,
    ) -> PyResult<String> {
        match o {
            Some(s) if !s.is(&vm.ctx.none) => str_arg(s, vm),
            _ => Ok(default.to_string()),
        }
    }

    // ---------------- polynomial (top-level) ----------------

    #[pyfunction]
    fn polyval(p: PyObjectRef, x: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let pa = obj_to_array(&p, None, vm)?;
        let xa = obj_to_array(&x, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::poly::polyval(&pa, &xa, vm)?))
    }
    #[pyfunction]
    fn roots(p: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let pa = obj_to_array(&p, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::poly::roots(&pa, vm)?))
    }
    #[pyfunction]
    fn polyfit(
        x: PyObjectRef,
        y: PyObjectRef,
        deg: usize,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let xa = obj_to_array(&x, None, vm)?;
        let ya = obj_to_array(&y, None, vm)?;
        Ok(PyNdArray::from_arrays(crate::poly::polyfit(
            &xa, &ya, deg, vm,
        )?))
    }

    /// `np.polyder(p, m=1)` — derivative of a polynomial with descending-power
    /// coefficients.
    #[pyfunction]
    fn polyder(p: PyObjectRef, m: OptionalArg<usize>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        use crate::dtype::CoerceArray;
        let pa = obj_to_array(&p, None, vm)?;
        let mut coeffs: Vec<f64> = pa.coerce::<f64>().iter().copied().collect();
        let order = m.unwrap_or(1);
        for _ in 0..order {
            if coeffs.len() <= 1 {
                coeffs = vec![0.0];
                break;
            }
            let n = coeffs.len();
            let new_n = n - 1;
            let mut next = Vec::with_capacity(new_n);
            for (i, &c) in coeffs.iter().take(new_n).enumerate() {
                let power = (n - 1 - i) as f64;
                next.push(c * power);
            }
            coeffs = next;
        }
        let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[coeffs.len()]), coeffs)
            .map_err(|e| crate::internal::internal(vm, format!("polyder: {e}")))?;
        Ok(PyNdArray::from_arrays(ArraysD::F64(arr)))
    }

    /// `np.polyint(p, m=1, k=0)` — antiderivative.
    #[pyfunction]
    fn polyint(p: PyObjectRef, m: OptionalArg<usize>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        use crate::dtype::CoerceArray;
        let pa = obj_to_array(&p, None, vm)?;
        let mut coeffs: Vec<f64> = pa.coerce::<f64>().iter().copied().collect();
        let order = m.unwrap_or(1);
        for _ in 0..order {
            let n = coeffs.len();
            let mut next = Vec::with_capacity(n + 1);
            for (i, &c) in coeffs.iter().enumerate() {
                let power = (n - i) as f64;
                next.push(c / power);
            }
            next.push(0.0); // integration constant
            coeffs = next;
        }
        let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[coeffs.len()]), coeffs)
            .map_err(|e| crate::internal::internal(vm, format!("polyint: {e}")))?;
        Ok(PyNdArray::from_arrays(ArraysD::F64(arr)))
    }

    // ---------------- np.fromiter / fromstring / logspace / geomspace ----------------

    #[derive(FromArgs)]
    pub(crate) struct FromIterArgs {
        #[pyarg(any)]
        dtype: PyObjectRef,
        #[pyarg(any, optional)]
        count: OptionalArg<isize>,
    }

    /// `np.fromiter(iterable, dtype, count=-1)` — build an ndarray from an iterable.
    #[pyfunction]
    fn fromiter(
        iterable: PyObjectRef,
        args: FromIterArgs,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let dt = parse_dtype_arg(&Some(args.dtype), vm)?.unwrap_or(DType::F64);
        let max = args.count.unwrap_or(-1);
        // Convert to an iterator object (call iter() on non-iterators).
        let iter_obj = iterable.get_iter(vm)?;
        let mut data: Vec<f64> = Vec::new();
        let mut n_collected = 0i64;
        loop {
            match iter_obj.next(vm)? {
                rustpython_vm::protocol::PyIterReturn::Return(it_res) => {
                    let f = it_res.try_float(vm)?.to_f64();
                    data.push(f);
                    n_collected += 1;
                    if max >= 0 && n_collected as isize >= max {
                        break;
                    }
                }
                rustpython_vm::protocol::PyIterReturn::StopIteration(_) => break,
            }
        }
        let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[data.len()]), data)
            .map_err(|e| crate::internal::internal(vm, format!("fromiter: {e}")))?;
        Ok(PyNdArray::from_arrays(ArraysD::F64(arr).cast(dt)))
    }

    #[derive(FromArgs)]
    pub(crate) struct FromStringArgs {
        #[pyarg(any, optional)]
        dtype: OptionalArg<PyObjectRef>,
        #[pyarg(any, optional)]
        sep: OptionalArg<PyObjectRef>,
    }

    /// `np.fromstring(s, dtype=float, sep=' ')` — parse whitespace/comma-separated
    /// numbers from a string. (Deprecated in numpy but still common.)
    #[pyfunction]
    fn fromstring(
        s: PyObjectRef,
        args: FromStringArgs,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let dt = parse_dtype_arg(&args.dtype.into_option(), vm)?.unwrap_or(DType::F64);
        let sep_str = match args.sep {
            OptionalArg::Missing => " ".to_string(),
            OptionalArg::Present(o) => o
                .downcast_ref::<rustpython_vm::builtins::PyStr>()
                .map(|s| s.as_wtf8().to_string_lossy().into_owned())
                .unwrap_or(" ".to_string()),
        };
        let s_str = s
            .downcast_ref::<rustpython_vm::builtins::PyStr>()
            .ok_or_else(|| vm.new_type_error("fromstring: first arg must be str".to_string()))?
            .as_wtf8()
            .to_string_lossy()
            .into_owned();
        // Tokenize by sep or by whitespace if sep is empty.
        let tokens: Vec<&str> = if sep_str.is_empty() || sep_str.trim().is_empty() {
            s_str.split_whitespace().collect()
        } else {
            s_str
                .split(sep_str.as_str())
                .map(|t| t.trim())
                .filter(|t| !t.is_empty())
                .collect()
        };
        let mut data = Vec::with_capacity(tokens.len());
        for t in tokens {
            let v: f64 = t
                .parse()
                .map_err(|_| vm.new_value_error(format!("fromstring: cannot parse '{t}'")))?;
            data.push(v);
        }
        let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[data.len()]), data)
            .map_err(|e| crate::internal::internal(vm, format!("fromstring: {e}")))?;
        Ok(PyNdArray::from_arrays(ArraysD::F64(arr).cast(dt)))
    }

    /// `np.array_equiv(a, b)` — equal as arrays modulo broadcasting.
    #[pyfunction]
    fn array_equiv(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<bool> {
        let xa = obj_to_array(&a, None, vm)?;
        let xb = obj_to_array(&b, None, vm)?;
        let common = match crate::extras::broadcast_shape(xa.shape(), xb.shape()) {
            Some(s) => s,
            None => return Ok(false),
        };
        let ba = crate::extras::broadcast_to(&xa, &common, vm)?;
        let bb = crate::extras::broadcast_to(&xb, &common, vm)?;
        Ok(crate::extras::array_equal(&ba, &bb))
    }

    /// `np.isneginf` / `np.isposinf`.
    #[pyfunction]
    fn isneginf(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        use crate::dtype::CoerceArray;
        let arr = obj_to_array(&a, None, vm)?;
        let f = arr.coerce::<f64>();
        let out: ndarray::ArrayD<bool> = f.mapv(|x| x.is_infinite() && x < 0.0);
        Ok(PyNdArray::from_arrays(ArraysD::Bool(out)))
    }
    #[pyfunction]
    fn isposinf(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        use crate::dtype::CoerceArray;
        let arr = obj_to_array(&a, None, vm)?;
        let f = arr.coerce::<f64>();
        let out: ndarray::ArrayD<bool> = f.mapv(|x| x.is_infinite() && x > 0.0);
        Ok(PyNdArray::from_arrays(ArraysD::Bool(out)))
    }

    /// `np.iscomplexobj(x)` / `np.isrealobj(x)` — whether the array dtype is complex.
    #[pyfunction]
    fn iscomplexobj(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<bool> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(arr.dtype().is_complex())
    }
    #[pyfunction]
    fn isrealobj(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<bool> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(!arr.dtype().is_complex())
    }

    // ---------------- np.vectorize / apply_along_axis / frompyfunc ----------------

    /// `np.vectorize(pyfunc, otypes=None)` — wrap a Python function so it
    /// applies element-wise to an ndarray.
    ///
    /// Returns a callable wrapper. The wrapper, when called with an ndarray,
    /// calls `pyfunc` on each element and collects the results.
    #[pyfunction]
    fn vectorize(pyfunc: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        // Build a Python wrapper: lambda *args: np.array([pyfunc(*xs) for xs in zip(*args)])
        // Easier: use a closure-like helper that the Rust side dispatches.
        let captured = pyfunc;
        let wrapper = vm.new_function(
            "vectorized",
            move |args: FuncArgs, vm: &VirtualMachine| -> PyResult<PyObjectRef> {
                // Convert each positional arg to an ndarray.
                let arrs: Vec<ArraysD> = args
                    .args
                    .iter()
                    .map(|o| obj_to_array(o, None, vm))
                    .collect::<PyResult<_>>()?;
                if arrs.is_empty() {
                    return Err(
                        vm.new_type_error("vectorize: needs at least one argument".to_string())
                    );
                }
                // Broadcast all to common shape.
                let mut common: Vec<usize> = arrs[0].shape().to_vec();
                for a in arrs.iter().skip(1) {
                    common =
                        crate::extras::broadcast_shape(&common, a.shape()).ok_or_else(|| {
                            vm.new_value_error(format!(
                                "vectorize: shapes not broadcastable: {:?}",
                                arrs.iter().map(|a| a.shape().to_vec()).collect::<Vec<_>>()
                            ))
                        })?;
                }
                let broadcast: Vec<ArraysD> = arrs
                    .iter()
                    .map(|a| crate::extras::broadcast_to(a, &common, vm))
                    .collect::<PyResult<_>>()?;
                use crate::dtype::CoerceArray;
                let broadcast_f: Vec<ndarray::ArrayD<f64>> =
                    broadcast.iter().map(|a| a.coerce::<f64>()).collect();
                let total = broadcast_f.first().map(|a| a.len()).unwrap_or(0);
                let mut out = Vec::with_capacity(total);
                for k in 0..total {
                    let call_args: Vec<PyObjectRef> = broadcast_f
                        .iter()
                        .map(|a| {
                            vm.ctx
                                .new_float(a.iter().nth(k).copied().unwrap_or(0.0))
                                .into()
                        })
                        .collect();
                    let r = captured.call(call_args, vm)?;
                    let f = r.try_float(vm)?.to_f64();
                    out.push(f);
                }
                let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&common), out)
                    .map_err(|e| crate::internal::internal(vm, format!("vectorize: {e}")))?;
                Ok(PyNdArray::from_arrays(ArraysD::F64(arr)).into_pyobject(vm))
            },
        );
        Ok(wrapper.into())
    }

    /// `np.frompyfunc(func, nin, nout)` — like vectorize but always returns
    /// object-dtype-ish output. We map to vectorize since we don't have an
    /// object dtype.
    #[pyfunction]
    fn frompyfunc(
        func: PyObjectRef,
        _nin: usize,
        _nout: usize,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        vectorize(func, vm)
    }

    /// `np.apply_along_axis(func1d, axis, arr)` — apply a 1-D function along
    /// an axis. Returns an array with the same shape as `arr` (or different
    /// last-axis depending on what `func1d` returns; we require it to return
    /// a 1-D array of the same length).
    #[pyfunction]
    fn apply_along_axis(
        func: PyObjectRef,
        axis: isize,
        arr: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        use crate::dtype::CoerceArray;
        let a = obj_to_array(&arr, None, vm)?;
        let nd = a.ndim() as isize;
        let ax = if axis < 0 { axis + nd } else { axis };
        if ax < 0 || ax >= nd {
            return Err(vm.new_value_error(format!("axis {axis} out of bounds")));
        }
        let ax = ax as usize;
        let f = a.coerce::<f64>();
        // For each lane along axis ax, call func1d on it.
        let mut results: Vec<Vec<f64>> = Vec::new();
        let mut result_len: Option<usize> = None;
        for lane in f.lanes(ndarray::Axis(ax)).into_iter() {
            let lane_vec: Vec<f64> = lane.iter().copied().collect();
            let n = lane_vec.len();
            let lane_arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n]), lane_vec)
                .map_err(|e| crate::internal::internal(vm, format!("apply_along: {e}")))?;
            let lane_py = PyNdArray::from_arrays(ArraysD::F64(lane_arr)).into_pyobject(vm);
            let r = func.call((lane_py,), vm)?;
            // Convert result to f64 array.
            let r_arr = obj_to_array(&r, None, vm)?;
            let r_f: Vec<f64> = r_arr.coerce::<f64>().iter().copied().collect();
            match result_len {
                None => result_len = Some(r_f.len()),
                Some(len) if len == r_f.len() => {}
                Some(_) => {
                    return Err(vm.new_value_error(
                        "apply_along_axis: func1d must return same-length 1-D arrays".to_string(),
                    ));
                }
            }
            results.push(r_f);
        }
        let new_axis_len = result_len.unwrap_or(0);
        // Output shape: same as input but axis `ax` replaced by new_axis_len.
        let mut out_shape: Vec<usize> = f.shape().to_vec();
        out_shape[ax] = new_axis_len;
        // Build output by walking outer iterator order (same as lanes().into_iter()).
        // Each lane contributes `new_axis_len` values that should go into position
        // along axis `ax`.
        let mut out_data = vec![0.0f64; out_shape.iter().product()];
        let mut lane_idx = 0usize;
        // Walk all multi-coords of `out_shape` excluding axis ax in lane-iteration order.
        let outer_shape: Vec<usize> = out_shape
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if i == ax { None } else { Some(d) })
            .collect();
        let outer_total: usize = outer_shape.iter().product::<usize>().max(1);
        for outer_flat in 0..outer_total {
            // Walk outer coord in C-order over outer_shape.
            let mut outer_coord = vec![0usize; outer_shape.len()];
            let mut rem = outer_flat;
            for d in (0..outer_shape.len()).rev() {
                outer_coord[d] = rem % outer_shape[d];
                rem /= outer_shape[d];
            }
            // Build the full multi-coord for each new-axis position.
            for k in 0..new_axis_len {
                let mut full = vec![0usize; out_shape.len()];
                let mut oc = outer_coord.iter().copied();
                for (i, slot) in full.iter_mut().enumerate() {
                    if i == ax {
                        *slot = k;
                    } else {
                        *slot = oc.next().unwrap_or(0);
                    }
                }
                // Linearize using out_shape strides.
                let strides: Vec<usize> = (0..out_shape.len())
                    .map(|d| out_shape[d + 1..].iter().product::<usize>().max(1))
                    .collect();
                let flat: usize = full.iter().zip(&strides).map(|(c, s)| c * s).sum();
                out_data[flat] = results[lane_idx][k];
            }
            lane_idx += 1;
        }
        let out_arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&out_shape), out_data)
            .map_err(|e| crate::internal::internal(vm, format!("apply_along: {e}")))?;
        Ok(PyNdArray::from_arrays(
            ArraysD::F64(out_arr).cast(a.dtype()),
        ))
    }

    // ---------------- np.broadcast_arrays / copyto / asarray_family ----------------

    /// `np.broadcast_arrays(*arrays)` — broadcast all to a common shape.
    #[pyfunction]
    fn broadcast_arrays(args: FuncArgs, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let arrs: Vec<ArraysD> = args
            .args
            .iter()
            .map(|o| obj_to_array(o, None, vm))
            .collect::<PyResult<_>>()?;
        if arrs.is_empty() {
            return Ok(vm.ctx.new_list(vec![]).into());
        }
        let mut common: Vec<usize> = arrs[0].shape().to_vec();
        for a in arrs.iter().skip(1) {
            common = crate::extras::broadcast_shape(&common, a.shape()).ok_or_else(|| {
                vm.new_value_error("broadcast: shapes not compatible".to_string())
            })?;
        }
        let out: Vec<PyObjectRef> = arrs
            .iter()
            .map(|a| {
                let b = crate::extras::broadcast_to(a, &common, vm)?;
                Ok(PyNdArray::from_arrays(b).into_pyobject(vm))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(vm.ctx.new_list(out).into())
    }

    /// `np.copyto(dst, src, casting='same_kind', where=True)` — copy src into dst.
    #[pyfunction]
    fn copyto(dst: PyObjectRef, src: PyObjectRef, vm: &VirtualMachine) -> PyResult<()> {
        let dst_arr = dst
            .downcast::<PyNdArray>()
            .map_err(|_| vm.new_type_error("copyto: dst must be ndarray".to_string()))?;
        let s = obj_to_array(&src, None, vm)?;
        let target_shape = dst_arr.view().shape().to_vec();
        let broadcast = crate::extras::broadcast_to(&s, &target_shape, vm)?;
        let cast = broadcast.cast(dst_arr.view().dtype());
        *dst_arr.view_mut() = cast;
        Ok(())
    }

    /// `np.asanyarray(a, dtype=None)` — same as `asarray` since we don't have
    /// ndarray subclasses.
    #[pyfunction]
    fn asanyarray(obj: PyObjectRef, dtype: DTypeArg, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        asarray(obj, dtype, vm)
    }

    /// `np.ascontiguousarray(a, dtype=None)` — same as `asarray` (everything is
    /// row-major contiguous in our representation).
    #[pyfunction]
    fn ascontiguousarray(
        obj: PyObjectRef,
        dtype: DTypeArg,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        asarray(obj, dtype, vm)
    }

    /// `np.asfortranarray(a, dtype=None)` — we don't support Fortran order;
    /// fall back to `asarray`.
    #[pyfunction]
    fn asfortranarray(
        obj: PyObjectRef,
        dtype: DTypeArg,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        asarray(obj, dtype, vm)
    }

    /// `np.require(a, dtype=None, requirements=None)` — coerce to array.
    #[pyfunction]
    fn require(
        obj: PyObjectRef,
        dtype: DTypeArg,
        _requirements: OptionalArg<PyObjectRef>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        asarray(obj, dtype, vm)
    }

    /// `np.resize(a, new_shape)` — return a new array with the requested
    /// shape, repeating data if needed.
    #[pyfunction]
    fn resize(a: PyObjectRef, new_shape: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        let target = parse_shape(&new_shape, vm)?;
        let total: usize = target.iter().product();
        let flat = crate::linalg::flatten(&arr);
        use crate::dtype::CoerceArray;
        macro_rules! per {
            ($var:ident, $ty:ty) => {{
                let src = flat.coerce::<$ty>();
                let mut data = Vec::with_capacity(total);
                if src.is_empty() {
                    return Err(
                        vm.new_value_error("resize: cannot resize an empty array".to_string())
                    );
                }
                for i in 0..total {
                    data.push(src[ndarray::IxDyn(&[i % src.len()])]);
                }
                let out = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&target), data)
                    .map_err(|e| crate::internal::internal(vm, format!("resize: {e}")))?;
                ArraysD::$var(out)
            }};
        }
        let out = match arr.dtype() {
            DType::Bool => per!(Bool, bool),
            DType::I8 => per!(I8, i8),
            DType::I16 => per!(I16, i16),
            DType::I32 => per!(I32, i32),
            DType::I64 => per!(I64, i64),
            DType::U8 => per!(U8, u8),
            DType::U16 => per!(U16, u16),
            DType::U32 => per!(U32, u32),
            DType::U64 => per!(U64, u64),
            DType::F16 => per!(F16, half::f16),
            DType::F32 => per!(F32, f32),
            DType::F64 => per!(F64, f64),
            DType::C64 => per!(C64, crate::dtype::C32),
            DType::C128 => per!(C128, crate::dtype::C64),
            other => {
                return Err(crate::internal::unsupported_dtype(vm, "resize", other));
            }
        };
        Ok(PyNdArray::from_arrays(out))
    }

    // ---------------- np.compress / extract / place ----------------

    /// `np.compress(condition, a, axis=None)` — select elements where condition is True.
    #[pyfunction]
    fn compress(
        condition: PyObjectRef,
        a: PyObjectRef,
        axis: OptionalArg<isize>,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        use crate::dtype::CoerceArray;
        let arr = obj_to_array(&a, None, vm)?;
        let cond = obj_to_array(&condition, None, vm)?;
        let mask: Vec<bool> = cond.coerce::<bool>().iter().copied().collect();
        match axis {
            OptionalArg::Missing | OptionalArg::Present(_) if axis.into_option().is_none() => {
                // Flat: like boolean indexing on flat.
                let flat = crate::linalg::flatten(&arr);
                use crate::index::IdxItem;
                // Pad mask with falses if too short, or truncate if too long.
                let mut padded = mask.clone();
                padded.resize(flat.len(), false);
                crate::index::apply_index(&flat, &[IdxItem::BoolMask(padded)], vm)
                    .map(PyNdArray::from_arrays)
            }
            _ => Err(vm.new_not_implemented_error("compress: axis= not yet supported".to_string())),
        }
    }

    /// `np.extract(condition, a)` — flat boolean selection.
    #[pyfunction]
    fn extract(condition: PyObjectRef, a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        compress(condition, a, OptionalArg::Missing, vm)
    }

    /// `np.place(arr, mask, vals)` — set masked positions in `arr` to the
    /// repeating sequence `vals`.
    #[pyfunction]
    fn place(
        arr: PyObjectRef,
        mask: PyObjectRef,
        vals: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<()> {
        let arr = arr
            .downcast::<PyNdArray>()
            .map_err(|_| vm.new_type_error("place: first arg must be ndarray".to_string()))?;
        let m = obj_to_array(&mask, None, vm)?;
        let v = obj_to_array(&vals, None, vm)?;
        use crate::dtype::CoerceArray;
        let mask_b: Vec<bool> = m.coerce::<bool>().iter().copied().collect();
        let n_true = mask_b.iter().filter(|&&b| b).count();
        if n_true == 0 {
            return Ok(());
        }
        let v_flat = crate::linalg::flatten(&v);
        let v_len = v_flat.len();
        if v_len == 0 {
            return Err(vm.new_value_error("place: values must not be empty".to_string()));
        }
        // Build a "filled out" value array of length n_true by repeating v.
        let dt = arr.view().dtype();
        macro_rules! per {
            ($var:ident, $ty:ty) => {{
                let src: Vec<$ty> = v_flat.coerce::<$ty>().iter().copied().collect();
                let cast_arr: ArraysD = {
                    let mut data = Vec::with_capacity(n_true);
                    for i in 0..n_true {
                        data.push(src[i % v_len]);
                    }
                    ArraysD::$var(
                        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n_true]), data)
                            .map_err(|e| crate::internal::internal(vm, format!("place: {e}")))?,
                    )
                };
                let mut inner = arr.view_mut();
                crate::index::set_via_index(
                    &mut inner,
                    &[crate::index::IdxItem::BoolMask(mask_b.clone())],
                    &cast_arr.cast(dt),
                    vm,
                )?;
            }};
        }
        match dt {
            DType::Bool => per!(Bool, bool),
            DType::I8 => per!(I8, i8),
            DType::I16 => per!(I16, i16),
            DType::I32 => per!(I32, i32),
            DType::I64 => per!(I64, i64),
            DType::U8 => per!(U8, u8),
            DType::U16 => per!(U16, u16),
            DType::U32 => per!(U32, u32),
            DType::U64 => per!(U64, u64),
            DType::F16 => per!(F16, half::f16),
            DType::F32 => per!(F32, f32),
            DType::F64 => per!(F64, f64),
            DType::C64 => per!(C64, crate::dtype::C32),
            DType::C128 => per!(C128, crate::dtype::C64),
            _ => return Err(crate::internal::unsupported_dtype(vm, "place", dt)),
        }
        Ok(())
    }

    // ---------------- np.put_along_axis ----------------

    #[pyfunction]
    fn put_along_axis(
        a: PyObjectRef,
        indices: PyObjectRef,
        values: PyObjectRef,
        axis: isize,
        vm: &VirtualMachine,
    ) -> PyResult<()> {
        use crate::dtype::CoerceArray;
        let arr = a
            .downcast::<PyNdArray>()
            .map_err(|_| vm.new_type_error("put_along_axis: dst must be ndarray".to_string()))?;
        let idx = obj_to_array(&indices, None, vm)?;
        let vals = obj_to_array(&values, None, vm)?;
        let nd = arr.view().ndim() as isize;
        let ax = if axis < 0 { axis + nd } else { axis };
        if ax < 0 || ax >= nd {
            return Err(vm.new_value_error(format!("axis {axis} out of bounds")));
        }
        let ax = ax as usize;
        let idx_i = idx.coerce::<i64>();
        let dt = arr.view().dtype();
        let vals_cast = vals.cast(dt);
        macro_rules! per {
            ($var:ident, $ty:ty) => {{
                let mut inner = arr.view_mut();
                if let ArraysD::$var(dst) = &mut *inner {
                    let vals_v: Vec<$ty> = match &vals_cast {
                        ArraysD::$var(x) => x.iter().copied().collect(),
                        _ => return Err(crate::internal::internal(vm, "put_along_axis cast")),
                    };
                    let idx_shape = idx_i.shape().to_vec();
                    for flat in 0..idx_i.len() {
                        let mut coord = vec![0usize; idx_shape.len()];
                        let mut rem = flat;
                        for d in (0..idx_shape.len()).rev() {
                            coord[d] = rem % idx_shape[d];
                            rem /= idx_shape[d];
                        }
                        let mut dst_coord = coord.clone();
                        dst_coord[ax] = idx_i[ndarray::IxDyn(&coord)] as usize;
                        let v = vals_v[flat % vals_v.len()];
                        dst[ndarray::IxDyn(&dst_coord)] = v;
                    }
                }
                Ok(())
            }};
        }
        match dt {
            DType::Bool => per!(Bool, bool),
            DType::I8 => per!(I8, i8),
            DType::I16 => per!(I16, i16),
            DType::I32 => per!(I32, i32),
            DType::I64 => per!(I64, i64),
            DType::U8 => per!(U8, u8),
            DType::U16 => per!(U16, u16),
            DType::U32 => per!(U32, u32),
            DType::U64 => per!(U64, u64),
            DType::F16 => per!(F16, half::f16),
            DType::F32 => per!(F32, f32),
            DType::F64 => per!(F64, f64),
            DType::C64 => per!(C64, crate::dtype::C32),
            DType::C128 => per!(C128, crate::dtype::C64),
            _ => return Err(crate::internal::unsupported_dtype(vm, "put_along_axis", dt)),
        }
    }

    // ---------------- np.fmin / fmax / fmod / divmod / modf / frexp / ldexp ----------------

    #[pyfunction]
    fn fmin(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        use crate::dtype::CoerceArray;
        let xf = x.coerce::<f64>();
        let yf = y.coerce::<f64>();
        let out = broadcast_binary_f64(&xf, &yf, vm, |a, b| {
            if a.is_nan() {
                b
            } else if b.is_nan() {
                a
            } else if a < b {
                a
            } else {
                b
            }
        })?;
        Ok(PyNdArray::from_arrays(ArraysD::F64(out)))
    }

    #[pyfunction]
    fn fmax(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        use crate::dtype::CoerceArray;
        let xf = x.coerce::<f64>();
        let yf = y.coerce::<f64>();
        let out = broadcast_binary_f64(&xf, &yf, vm, |a, b| {
            if a.is_nan() {
                b
            } else if b.is_nan() {
                a
            } else if a > b {
                a
            } else {
                b
            }
        })?;
        Ok(PyNdArray::from_arrays(ArraysD::F64(out)))
    }

    #[pyfunction]
    fn fmod(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        use crate::dtype::CoerceArray;
        let xf = x.coerce::<f64>();
        let yf = y.coerce::<f64>();
        let out = broadcast_binary_f64(&xf, &yf, vm, |a, b| a - (a / b).trunc() * b)?;
        Ok(PyNdArray::from_arrays(ArraysD::F64(out)))
    }

    /// `np.divmod(a, b)` — return (a // b, a % b) elementwise.
    #[pyfunction(name = "divmod")]
    fn divmod_fn(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        let q = ops::floor_divide(&x, &y, vm)?;
        let r = ops::remainder(&x, &y, vm)?;
        let tup = PyTuple::new_ref(
            vec![
                PyNdArray::from_arrays(q).into_pyobject(vm),
                PyNdArray::from_arrays(r).into_pyobject(vm),
            ],
            &vm.ctx,
        );
        Ok(tup.into())
    }

    /// `np.modf(a)` — return (fractional part, integer part).
    #[pyfunction]
    fn modf(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        use crate::dtype::CoerceArray;
        let arr = obj_to_array(&a, None, vm)?;
        let f = arr.coerce::<f64>();
        let shape = f.shape().to_vec();
        let frac: ndarray::ArrayD<f64> = f.mapv(|x| x.fract());
        let int_part: ndarray::ArrayD<f64> = f.mapv(|x| x.trunc());
        let _ = shape;
        let tup = PyTuple::new_ref(
            vec![
                PyNdArray::from_arrays(ArraysD::F64(frac)).into_pyobject(vm),
                PyNdArray::from_arrays(ArraysD::F64(int_part)).into_pyobject(vm),
            ],
            &vm.ctx,
        );
        Ok(tup.into())
    }

    /// `np.frexp(a)` — return (mantissa, exponent) with x = mantissa * 2**exponent.
    #[pyfunction]
    fn frexp(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        use crate::dtype::CoerceArray;
        let arr = obj_to_array(&a, None, vm)?;
        let f = arr.coerce::<f64>();
        let shape = f.shape().to_vec();
        let n = f.len();
        let mut mantissa = Vec::with_capacity(n);
        let mut exponent = Vec::with_capacity(n);
        for &v in f.iter() {
            if v == 0.0 {
                mantissa.push(0.0);
                exponent.push(0i64);
            } else {
                // m * 2^e == v, with 0.5 <= |m| < 1.0
                let bits = v.to_bits();
                let exp = (((bits >> 52) & 0x7ff) as i64) - 1022;
                let m = v * 2f64.powi(-exp as i32);
                mantissa.push(m);
                exponent.push(exp);
            }
        }
        let m_arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), mantissa)
            .map_err(|e| crate::internal::internal(vm, format!("frexp m: {e}")))?;
        let e_arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), exponent)
            .map_err(|e| crate::internal::internal(vm, format!("frexp e: {e}")))?;
        let tup = PyTuple::new_ref(
            vec![
                PyNdArray::from_arrays(ArraysD::F64(m_arr)).into_pyobject(vm),
                PyNdArray::from_arrays(ArraysD::I64(e_arr)).into_pyobject(vm),
            ],
            &vm.ctx,
        );
        Ok(tup.into())
    }

    /// `np.ldexp(x1, x2)` — x1 * 2**x2 elementwise.
    #[pyfunction]
    fn ldexp(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        use crate::dtype::CoerceArray;
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        let xf = x.coerce::<f64>();
        let yf = y.coerce::<f64>();
        let out = broadcast_binary_f64(&xf, &yf, vm, |a, b| a * 2f64.powi(b as i32))?;
        Ok(PyNdArray::from_arrays(ArraysD::F64(out)))
    }

    /// `np.positive(a)` — unary plus (identity).
    #[pyfunction]
    fn positive(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        Ok(PyNdArray::from_arrays(arr))
    }

    /// `np.spacing(x)` — distance to the next floating-point value after x.
    #[pyfunction]
    fn spacing(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&a, None, vm)?;
        use crate::dtype::CoerceArray;
        let f = arr.coerce::<f64>();
        let out: ndarray::ArrayD<f64> = f.mapv(|x| {
            // ulp(x) — distance to next representable double.
            if x.is_nan() {
                f64::NAN
            } else if x.is_infinite() {
                f64::INFINITY
            } else {
                let bits = x.to_bits();
                let next = f64::from_bits(bits + 1);
                next - x
            }
        });
        Ok(PyNdArray::from_arrays(ArraysD::F64(out)))
    }

    /// `np.nextafter(x1, x2)` — next-representable value after x1 toward x2.
    #[pyfunction]
    fn nextafter(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        use crate::dtype::CoerceArray;
        let x = obj_to_array(&a, None, vm)?;
        let y = obj_to_array(&b, None, vm)?;
        let xf = x.coerce::<f64>();
        let yf = y.coerce::<f64>();
        let out = broadcast_binary_f64(&xf, &yf, vm, |a, b| {
            if a == b {
                a
            } else if a < b {
                f64::from_bits(a.to_bits() + 1)
            } else {
                f64::from_bits(a.to_bits() - 1)
            }
        })?;
        Ok(PyNdArray::from_arrays(ArraysD::F64(out)))
    }

    // ---------------- np.seterr / geterr / errstate (no-op stubs) ----------------

    /// `np.geterr()` — always returns a dict with everything 'warn'.
    #[pyfunction]
    fn geterr(vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let d = vm.ctx.new_dict();
        for k in ["divide", "over", "under", "invalid"] {
            d.set_item(k, vm.ctx.new_str("warn").into(), vm)?;
        }
        Ok(d.into())
    }

    /// `np.seterr(...)` — accept and ignore; returns the previous (default) settings.
    #[pyfunction]
    fn seterr(_args: FuncArgs, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        geterr(vm)
    }

    /// `np.geterrcall()` — always None.
    #[pyfunction]
    fn geterrcall(vm: &VirtualMachine) -> PyObjectRef {
        vm.ctx.none()
    }

    /// `np.errstate(...)` — return a context manager that's a no-op.
    #[pyfunction]
    fn errstate(_args: FuncArgs, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        // Build a no-op context manager via a tiny Python class on the fly.
        let src = "class _ErrState:\n    def __enter__(self): return self\n    def __exit__(self, *args): return False\n_es = _ErrState()\n";
        let g = vm.ctx.new_dict();
        let code = vm
            .compile(
                src,
                rustpython_vm::compiler::Mode::Exec,
                "<errstate>".into(),
            )
            .map_err(|e| vm.new_syntax_error(&e, Some(src)))?;
        let scope = rustpython_vm::scope::Scope::with_builtins(None, g.clone(), vm);
        vm.run_code_obj(code, scope)?;
        Ok(g.get_item("_es", vm)?)
    }

    // ---------------- index helpers ----------------

    /// `np.indices(shape)` — return a tuple of index arrays (or just the
    /// stacked array). We return a stacked array of shape (ndim, *shape).
    #[pyfunction]
    fn indices(shape: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let dims = parse_shape(&shape, vm)?;
        let total: usize = dims.iter().product::<usize>().max(1);
        let nd = dims.len();
        let mut out = Vec::with_capacity(nd * total);
        // For each axis k, emit the k-th coordinate of every element in
        // C-order over the product shape.
        for k in 0..nd {
            for flat in 0..total {
                let mut rem = flat;
                let mut coord = 0usize;
                for (d, &_dim) in dims.iter().enumerate() {
                    let stride: usize = dims[d + 1..].iter().product();
                    let coord_d = rem / stride.max(1);
                    rem %= stride.max(1);
                    if d == k {
                        coord = coord_d;
                    }
                }
                out.push(coord as i64);
            }
        }
        let mut out_shape = vec![nd];
        out_shape.extend(&dims);
        let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&out_shape), out)
            .map_err(|e| crate::internal::internal(vm, format!("indices: {e}")))?;
        Ok(PyNdArray::from_arrays(ArraysD::I64(arr)))
    }

    /// `np.unravel_index(indices, shape)` — convert flat indices to coordinate
    /// tuples. Returns a tuple of N arrays, one per axis.
    #[pyfunction]
    fn unravel_index(
        idx: PyObjectRef,
        shape: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        use crate::dtype::CoerceArray;
        let dims = parse_shape(&shape, vm)?;
        let i_arr = obj_to_array(&idx, None, vm)?;
        let i_flat: Vec<i64> = i_arr.coerce::<i64>().iter().copied().collect();
        // For each axis, gather coordinates.
        let mut per_axis: Vec<Vec<i64>> = vec![Vec::with_capacity(i_flat.len()); dims.len()];
        let strides: Vec<usize> = (0..dims.len())
            .map(|d| dims[d + 1..].iter().product::<usize>().max(1))
            .collect();
        for &flat in &i_flat {
            let mut rem = flat as usize;
            for (d, &s) in strides.iter().enumerate() {
                per_axis[d].push((rem / s) as i64);
                rem %= s;
            }
        }
        let n = i_flat.len();
        let mut tup_items: Vec<PyObjectRef> = Vec::with_capacity(dims.len());
        for col in per_axis {
            let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n]), col)
                .map_err(|e| crate::internal::internal(vm, format!("unravel: {e}")))?;
            tup_items.push(PyNdArray::from_arrays(ArraysD::I64(arr)).into_pyobject(vm));
        }
        Ok(PyTuple::new_ref(tup_items, &vm.ctx).into())
    }

    /// `np.ravel_multi_index(multi_index, dims)` — inverse of unravel_index.
    #[pyfunction]
    fn ravel_multi_index(
        multi_index: PyObjectRef,
        dims: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        use crate::dtype::CoerceArray;
        let dim_vec = parse_shape(&dims, vm)?;
        // multi_index is a sequence of N arrays (one per axis). Accept tuple or list.
        let arrays: Vec<ArraysD> = seq_to_arrays(&multi_index, vm)?;
        if arrays.len() != dim_vec.len() {
            return Err(vm.new_value_error(format!(
                "ravel_multi_index: expected {} index arrays for dims {:?}, got {}",
                dim_vec.len(),
                dim_vec,
                arrays.len()
            )));
        }
        let n = arrays[0].len();
        let strides: Vec<usize> = (0..dim_vec.len())
            .map(|d| dim_vec[d + 1..].iter().product::<usize>().max(1))
            .collect();
        let mut out = vec![0i64; n];
        for (axis, arr) in arrays.iter().enumerate() {
            let f: Vec<i64> = arr.coerce::<i64>().iter().copied().collect();
            for (i, &v) in f.iter().enumerate() {
                out[i] += v * strides[axis] as i64;
            }
        }
        let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n]), out)
            .map_err(|e| crate::internal::internal(vm, format!("ravel_multi: {e}")))?;
        Ok(PyNdArray::from_arrays(ArraysD::I64(arr)))
    }

    /// `np.diag_indices(n, ndim=2)` — indices that select the diagonal of an
    /// n×n×…×n array.
    #[pyfunction]
    fn diag_indices(
        n: usize,
        ndim: OptionalArg<usize>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let nd = ndim.unwrap_or(2);
        let idx: Vec<i64> = (0..n as i64).collect();
        let mut items = Vec::with_capacity(nd);
        for _ in 0..nd {
            let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n]), idx.clone())
                .map_err(|e| crate::internal::internal(vm, format!("diag_indices: {e}")))?;
            items.push(PyNdArray::from_arrays(ArraysD::I64(arr)).into_pyobject(vm));
        }
        Ok(PyTuple::new_ref(items, &vm.ctx).into())
    }

    /// `np.tril_indices(n, k=0, m=None)` / `np.triu_indices(n, k=0, m=None)`.
    #[pyfunction]
    fn tril_indices(
        n: usize,
        k: OptionalArg<isize>,
        m: OptionalArg<usize>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let cols = m.unwrap_or(n);
        let kk = k.unwrap_or(0);
        let mut rows = Vec::new();
        let mut col_idx = Vec::new();
        for r in 0..n {
            for c in 0..cols {
                if (c as isize) <= r as isize + kk {
                    rows.push(r as i64);
                    col_idx.push(c as i64);
                }
            }
        }
        let nn = rows.len();
        let arr_r = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[nn]), rows)
            .map_err(|e| crate::internal::internal(vm, format!("tril_indices: {e}")))?;
        let arr_c = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[nn]), col_idx)
            .map_err(|e| crate::internal::internal(vm, format!("tril_indices: {e}")))?;
        Ok(PyTuple::new_ref(
            vec![
                PyNdArray::from_arrays(ArraysD::I64(arr_r)).into_pyobject(vm),
                PyNdArray::from_arrays(ArraysD::I64(arr_c)).into_pyobject(vm),
            ],
            &vm.ctx,
        )
        .into())
    }

    #[pyfunction]
    fn triu_indices(
        n: usize,
        k: OptionalArg<isize>,
        m: OptionalArg<usize>,
        vm: &VirtualMachine,
    ) -> PyResult<PyObjectRef> {
        let cols = m.unwrap_or(n);
        let kk = k.unwrap_or(0);
        let mut rows = Vec::new();
        let mut col_idx = Vec::new();
        for r in 0..n {
            for c in 0..cols {
                if (c as isize) >= r as isize + kk {
                    rows.push(r as i64);
                    col_idx.push(c as i64);
                }
            }
        }
        let nn = rows.len();
        let arr_r = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[nn]), rows)
            .map_err(|e| crate::internal::internal(vm, format!("triu_indices: {e}")))?;
        let arr_c = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[nn]), col_idx)
            .map_err(|e| crate::internal::internal(vm, format!("triu_indices: {e}")))?;
        Ok(PyTuple::new_ref(
            vec![
                PyNdArray::from_arrays(ArraysD::I64(arr_r)).into_pyobject(vm),
                PyNdArray::from_arrays(ArraysD::I64(arr_c)).into_pyobject(vm),
            ],
            &vm.ctx,
        )
        .into())
    }

    // ---------------- nan-aware reductions ----------------

    #[pyfunction]
    fn nanargmin(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<usize> {
        use crate::dtype::CoerceArray;
        let arr = obj_to_array(&a, None, vm)?;
        let f = arr.coerce::<f64>();
        let mut best_idx = 0usize;
        let mut best_val = f64::INFINITY;
        let mut found = false;
        for (i, &v) in f.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            if !found || v < best_val {
                best_val = v;
                best_idx = i;
                found = true;
            }
        }
        if !found {
            return Err(vm.new_value_error("all-NaN slice encountered".to_string()));
        }
        Ok(best_idx)
    }

    #[pyfunction]
    fn nanargmax(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<usize> {
        use crate::dtype::CoerceArray;
        let arr = obj_to_array(&a, None, vm)?;
        let f = arr.coerce::<f64>();
        let mut best_idx = 0usize;
        let mut best_val = f64::NEG_INFINITY;
        let mut found = false;
        for (i, &v) in f.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            if !found || v > best_val {
                best_val = v;
                best_idx = i;
                found = true;
            }
        }
        if !found {
            return Err(vm.new_value_error("all-NaN slice encountered".to_string()));
        }
        Ok(best_idx)
    }

    #[pyfunction]
    fn nanprod(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        use crate::dtype::CoerceArray;
        let arr = obj_to_array(&a, None, vm)?;
        let f = arr.coerce::<f64>();
        let prod: f64 = f.iter().copied().filter(|x| !x.is_nan()).product();
        Ok(PyNdArray::from_arrays(ArraysD::F64(
            ndarray::ArrayD::from_elem(ndarray::IxDyn(&[]), prod),
        )))
    }

    #[pyfunction]
    fn nancumsum(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        use crate::dtype::CoerceArray;
        let arr = obj_to_array(&a, None, vm)?;
        let f = arr.coerce::<f64>();
        let shape = f.shape().to_vec();
        let mut acc = 0.0;
        let out: Vec<f64> = f
            .iter()
            .copied()
            .map(|x| {
                if !x.is_nan() {
                    acc += x;
                }
                acc
            })
            .collect();
        let out_arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), out)
            .map_err(|e| crate::internal::internal(vm, format!("nancumsum: {e}")))?;
        Ok(PyNdArray::from_arrays(ArraysD::F64(out_arr)))
    }

    #[pyfunction]
    fn nancumprod(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        use crate::dtype::CoerceArray;
        let arr = obj_to_array(&a, None, vm)?;
        let f = arr.coerce::<f64>();
        let shape = f.shape().to_vec();
        let mut acc = 1.0;
        let out: Vec<f64> = f
            .iter()
            .copied()
            .map(|x| {
                if !x.is_nan() {
                    acc *= x;
                }
                acc
            })
            .collect();
        let out_arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), out)
            .map_err(|e| crate::internal::internal(vm, format!("nancumprod: {e}")))?;
        Ok(PyNdArray::from_arrays(ArraysD::F64(out_arr)))
    }

    #[pyfunction]
    fn nanpercentile(a: PyObjectRef, q: ArgIntoFloat, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let q: f64 = q.into();
        use crate::dtype::CoerceArray;
        let arr = obj_to_array(&a, None, vm)?;
        let f = arr.coerce::<f64>();
        let mut v: Vec<f64> = f.iter().copied().filter(|x| !x.is_nan()).collect();
        if v.is_empty() {
            return Err(vm.new_value_error("all-NaN slice".to_string()));
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // Linear interpolation between adjacent sorted points.
        let p = q.clamp(0.0, 100.0) / 100.0;
        let pos = p * (v.len() as f64 - 1.0);
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        let frac = pos - lo as f64;
        let val = v[lo] + frac * (v[hi.min(v.len() - 1)] - v[lo]);
        Ok(PyNdArray::from_arrays(ArraysD::F64(
            ndarray::ArrayD::from_elem(ndarray::IxDyn(&[]), val),
        )))
    }

    #[pyfunction]
    fn nanquantile(a: PyObjectRef, q: ArgIntoFloat, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let q: f64 = q.into();
        // Convert to 0-100 percentile by multiplying by 100.
        // Use the same impl via a fresh ArgIntoFloat wrapper is awkward —
        // inline:
        use crate::dtype::CoerceArray;
        let arr = obj_to_array(&a, None, vm)?;
        let f = arr.coerce::<f64>();
        let mut v: Vec<f64> = f.iter().copied().filter(|x| !x.is_nan()).collect();
        if v.is_empty() {
            return Err(vm.new_value_error("all-NaN slice".to_string()));
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p = q.clamp(0.0, 1.0);
        let pos = p * (v.len() as f64 - 1.0);
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        let frac = pos - lo as f64;
        let val = v[lo] + frac * (v[hi.min(v.len() - 1)] - v[lo]);
        Ok(PyNdArray::from_arrays(ArraysD::F64(
            ndarray::ArrayD::from_elem(ndarray::IxDyn(&[]), val),
        )))
    }

    // ---------------- np.put / np.take_along_axis / np.choose / np.select ----------------

    /// `np.put(a, indices, values)` — set elements at flat-indices.
    #[pyfunction]
    fn put(
        a: PyObjectRef,
        indices: PyObjectRef,
        values: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<()> {
        use crate::dtype::CoerceArray;
        let arr_obj = a
            .downcast::<PyNdArray>()
            .map_err(|_| vm.new_type_error("put: first arg must be ndarray".to_string()))?;
        let i_arr = obj_to_array(&indices, None, vm)?;
        let v_arr = obj_to_array(&values, None, vm)?;
        let i_flat: Vec<i64> = i_arr.coerce::<i64>().iter().copied().collect();
        let dt = arr_obj.view().dtype();
        let v_cast = v_arr.cast(dt);
        macro_rules! per {
            ($var:ident, $ty:ty) => {{
                let mut inner = arr_obj.view_mut();
                let flat_len = inner.len();
                if let ArraysD::$var(arr) = &mut *inner {
                    let mut flat = arr
                        .as_slice_mut()
                        .map(|s| s.to_vec())
                        .unwrap_or_else(|| arr.iter().copied().collect());
                    let vs: Vec<$ty> = match &v_cast {
                        ArraysD::$var(x) => x.iter().copied().collect(),
                        _ => return Err(crate::internal::internal(vm, "put: cast")),
                    };
                    for (k, &i) in i_flat.iter().enumerate() {
                        let pos = if i < 0 {
                            (i + flat_len as i64) as usize
                        } else {
                            i as usize
                        };
                        if pos >= flat_len {
                            return Err(vm.new_index_error(format!(
                                "put: index {i} out of bounds for size {flat_len}"
                            )));
                        }
                        let val = if vs.is_empty() {
                            <$ty as Default>::default()
                        } else {
                            vs[k % vs.len()]
                        };
                        flat[pos] = val;
                    }
                    let new = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(arr.shape()), flat)
                        .map_err(|e| crate::internal::internal(vm, format!("put: {e}")))?;
                    *arr = new;
                }
                Ok(())
            }};
        }
        match arr_obj.view().dtype() {
            DType::Bool => per!(Bool, bool),
            DType::I8 => per!(I8, i8),
            DType::I16 => per!(I16, i16),
            DType::I32 => per!(I32, i32),
            DType::I64 => per!(I64, i64),
            DType::U8 => per!(U8, u8),
            DType::U16 => per!(U16, u16),
            DType::U32 => per!(U32, u32),
            DType::U64 => per!(U64, u64),
            DType::F16 => per!(F16, half::f16),
            DType::F32 => per!(F32, f32),
            DType::F64 => per!(F64, f64),
            DType::C64 => per!(C64, crate::dtype::C32),
            DType::C128 => per!(C128, crate::dtype::C64),
            _ => return Err(crate::internal::unsupported_dtype(vm, "put", dt)),
        }
    }

    #[derive(FromArgs)]
    pub(crate) struct TakeAlongAxisArgs {
        #[pyarg(any)]
        axis: isize,
    }

    /// `np.take_along_axis(a, indices, axis)` — gather elements along an axis.
    #[pyfunction]
    fn take_along_axis(
        a: PyObjectRef,
        indices: PyObjectRef,
        args: TakeAlongAxisArgs,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        let axis = args.axis;
        use crate::dtype::CoerceArray;
        let arr = obj_to_array(&a, None, vm)?;
        let idx = obj_to_array(&indices, None, vm)?;
        let nd = arr.ndim() as isize;
        let ax = if axis < 0 { axis + nd } else { axis };
        if ax < 0 || ax >= nd {
            return Err(vm.new_value_error(format!("axis {axis} out of bounds")));
        }
        let ax = ax as usize;
        // Use np.take semantics along single axis: result has the shape of `idx`.
        let arr_f = arr.coerce::<f64>();
        let idx_i = idx.coerce::<i64>();
        if arr_f.shape().len() != idx_i.shape().len() {
            return Err(
                vm.new_value_error("take_along_axis: indices must match input ndim".to_string())
            );
        }
        let out_shape: Vec<usize> = idx_i.shape().to_vec();
        let mut out = Vec::with_capacity(idx_i.len());
        let total = idx_i.len();
        // Walk every flat element of idx_i; for each, build the source coords
        // by reusing the idx multi-coord but substituting axis `ax` with idx_i's value.
        for flat in 0..total {
            let mut idx_coord = vec![0usize; out_shape.len()];
            let mut rem = flat;
            for d in (0..out_shape.len()).rev() {
                idx_coord[d] = rem % out_shape[d];
                rem /= out_shape[d];
            }
            let mut src_coord = idx_coord.clone();
            let take = idx_i[ndarray::IxDyn(&idx_coord)] as usize;
            src_coord[ax] = take;
            out.push(arr_f[ndarray::IxDyn(&src_coord)]);
        }
        let out_arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&out_shape), out)
            .map_err(|e| crate::internal::internal(vm, format!("take_along_axis: {e}")))?;
        Ok(PyNdArray::from_arrays(
            ArraysD::F64(out_arr).cast(arr.dtype()),
        ))
    }

    /// `np.choose(a, choices)` — pick from a list of arrays based on indices.
    #[pyfunction]
    fn choose(a: PyObjectRef, choices: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        use crate::dtype::CoerceArray;
        let idx = obj_to_array(&a, None, vm)?;
        let arrs = seq_to_arrays(&choices, vm)?;
        if arrs.is_empty() {
            return Err(vm.new_value_error("choose: choices must be non-empty".to_string()));
        }
        let n = idx.len();
        let idx_flat = crate::linalg::flatten(&idx).coerce::<i64>();
        let shape = idx.shape().to_vec();
        // All choice arrays should broadcast to idx's shape; we simplify by
        // requiring same total length and same shape as idx.
        let mut out = Vec::with_capacity(n);
        let choice_flats: Vec<Vec<f64>> = arrs
            .iter()
            .map(|c| {
                let cf = c.coerce::<f64>();
                cf.iter().copied().collect()
            })
            .collect();
        for i in 0..n {
            let choice = idx_flat[ndarray::IxDyn(&[i])] as usize;
            if choice >= choice_flats.len() {
                return Err(vm.new_value_error(format!(
                    "choose: index {choice} out of range (have {} choices)",
                    choice_flats.len()
                )));
            }
            let cf = &choice_flats[choice];
            out.push(cf[i % cf.len()]);
        }
        let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), out)
            .map_err(|e| crate::internal::internal(vm, format!("choose: {e}")))?;
        Ok(PyNdArray::from_arrays(ArraysD::F64(arr)))
    }

    // ---------------- string / number formatting ----------------

    #[derive(FromArgs)]
    pub(crate) struct BinaryReprArgs {
        #[pyarg(any, optional)]
        width: OptionalArg<usize>,
    }

    /// `np.binary_repr(n, width=None)` — binary string of an integer.
    #[pyfunction]
    fn binary_repr(n: i64, args: BinaryReprArgs, _vm: &VirtualMachine) -> String {
        let width = args.width;
        let s = if n >= 0 {
            format!("{n:b}")
        } else if let OptionalArg::Present(w) = &width {
            let mask = if *w >= 64 { !0u64 } else { (1u64 << *w) - 1 };
            format!("{:b}", (n as i128 & mask as i128) as u64)
        } else {
            format!("-{:b}", -n)
        };
        match width {
            OptionalArg::Present(w) => format!("{:0>width$}", s, width = w),
            OptionalArg::Missing => s,
        }
    }

    #[derive(FromArgs)]
    pub(crate) struct BaseReprArgs {
        #[pyarg(any, optional)]
        base: OptionalArg<u32>,
        #[pyarg(any, optional)]
        padding: OptionalArg<usize>,
    }

    /// `np.base_repr(n, base=2, padding=0)` — string in a given base.
    #[pyfunction]
    fn base_repr(n: i64, args: BaseReprArgs, vm: &VirtualMachine) -> PyResult<String> {
        let base = args.base;
        let padding = args.padding;
        let b = base.unwrap_or(2);
        if !(2..=36).contains(&b) {
            return Err(vm.new_value_error("base must be in [2, 36]".to_string()));
        }
        let pad = padding.unwrap_or(0);
        let mut abs = n.unsigned_abs();
        let mut digits: Vec<char> = Vec::new();
        if abs == 0 {
            digits.push('0');
        }
        while abs > 0 {
            let r = (abs % b as u64) as u32;
            let c = if r < 10 {
                std::char::from_digit(r, 36).unwrap_or('0')
            } else {
                ((r - 10) as u8 + b'A') as char
            };
            digits.push(c);
            abs /= b as u64;
        }
        for _ in 0..pad {
            digits.push('0');
        }
        let mut s: String = digits.into_iter().rev().collect();
        if n < 0 {
            s.insert(0, '-');
        }
        Ok(s)
    }

    /// `np.angle(z, deg=False)` — phase angle of a complex (or real) array.
    #[pyfunction]
    fn angle(z: PyObjectRef, deg: OptionalArg<bool>, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let arr = obj_to_array(&z, None, vm)?;
        let degrees = deg.unwrap_or(false);
        let out: ndarray::ArrayD<f64> = match &arr {
            ArraysD::C64(x) => x.mapv(|c| {
                let a = c.im.atan2(c.re) as f64;
                if degrees { a.to_degrees() } else { a }
            }),
            ArraysD::C128(x) => x.mapv(|c| {
                let a = c.im.atan2(c.re);
                if degrees { a.to_degrees() } else { a }
            }),
            other => {
                use crate::dtype::CoerceArray;
                let f = other.coerce::<f64>();
                f.mapv(|v| {
                    let a = if v < 0.0 { std::f64::consts::PI } else { 0.0 };
                    if degrees { a.to_degrees() } else { a }
                })
            }
        };
        Ok(PyNdArray::from_arrays(ArraysD::F64(out)))
    }

    // ---------------- einsum ----------------

    #[pyfunction]
    fn einsum(args: FuncArgs, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        let mut it = args.args.into_iter();
        let spec_obj = it
            .next()
            .ok_or_else(|| vm.new_type_error("einsum: missing subscripts string".to_string()))?;
        let spec = str_arg(&spec_obj, vm)?;
        let operands: Vec<ArraysD> = it
            .map(|o| obj_to_array(&o, None, vm))
            .collect::<PyResult<_>>()?;
        Ok(PyNdArray::from_arrays(crate::einsum::einsum(
            &spec, &operands, vm,
        )?))
    }

    // ---------------- save / load (.npy) ----------------

    #[pyfunction]
    fn save(file: PyObjectRef, arr: PyObjectRef, vm: &VirtualMachine) -> PyResult<()> {
        let path_str = file
            .downcast_ref::<rustpython_vm::builtins::PyStr>()
            .ok_or_else(|| vm.new_type_error("save: file argument must be a str path".to_string()))?
            .as_wtf8()
            .to_string_lossy()
            .into_owned();
        // numpy appends `.npy` if missing.
        let final_path = if std::path::Path::new(&path_str)
            .extension()
            .and_then(|e| e.to_str())
            == Some("npy")
        {
            path_str
        } else {
            format!("{path_str}.npy")
        };
        let array = obj_to_array(&arr, None, vm)?;
        crate::npy::save(std::path::Path::new(&final_path), &array)
            .map_err(|e| vm.new_os_error(format!("save failed: {e}")))
    }

    /// `np.load(file)` — returns either an `ndarray` (for `.npy`) or a
    /// `dict` mapping member-name → `ndarray` (for `.npz`).
    #[pyfunction]
    fn load(file: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        let path_str = file
            .downcast_ref::<rustpython_vm::builtins::PyStr>()
            .ok_or_else(|| vm.new_type_error("load: file argument must be a str path".to_string()))?
            .as_wtf8()
            .to_string_lossy()
            .into_owned();
        let path = std::path::Path::new(&path_str);
        // Discriminate by the file's leading bytes. .npy starts with
        // `\x93NUMPY`, .npz starts with `PK\x03\x04`.
        let mut head = [0u8; 4];
        {
            use std::io::Read;
            let mut f = std::fs::File::open(path)
                .map_err(|e| vm.new_os_error(format!("open {path_str}: {e}")))?;
            let _ = f.read(&mut head);
        }
        if head.starts_with(b"PK\x03\x04") {
            let entries = crate::npz::load(path).map_err(|e| match e {
                crate::npz::NpzError::Io(io) => vm.new_os_error(format!("load: {io}")),
                crate::npz::NpzError::Format(s) => vm.new_value_error(format!("bad .npz: {s}")),
                crate::npz::NpzError::Compression => {
                    vm.new_value_error("compressed .npz not supported".to_string())
                }
            })?;
            let dict = vm.ctx.new_dict();
            for (name, arr) in entries {
                let key: PyObjectRef = vm.ctx.new_str(name).into();
                let val = PyNdArray::from_arrays(arr).into_pyobject(vm);
                dict.set_item(&*key, val, vm)?;
            }
            return Ok(dict.into());
        }
        let arr = crate::npy::load(path).map_err(|e| match e {
            crate::npy::LoadError::Io(io) => vm.new_os_error(format!("load failed: {io}")),
            crate::npy::LoadError::Format(s) => vm.new_value_error(format!("bad .npy: {s}")),
        })?;
        Ok(PyNdArray::from_arrays(arr).into_pyobject(vm))
    }

    /// `np.savez(file, **kwargs)` — write each kwarg as a member of a `.npz`
    /// archive. Positional arrays would be saved as `arr_0`, `arr_1`, … in
    /// numpy; we accept them here too.
    #[pyfunction]
    fn savez(args: FuncArgs, vm: &VirtualMachine) -> PyResult<()> {
        savez_impl(args, false, vm)
    }

    #[pyfunction]
    fn savez_compressed(args: FuncArgs, vm: &VirtualMachine) -> PyResult<()> {
        // No deflate implementation; fall back to stored. numpy.load reads
        // both transparently, so the on-disk format is still a valid .npz.
        savez_impl(args, true, vm)
    }

    fn savez_impl(args: FuncArgs, _compressed: bool, vm: &VirtualMachine) -> PyResult<()> {
        let mut it = args.args.into_iter();
        let file = it.next().ok_or_else(|| {
            vm.new_type_error("savez() missing positional argument: 'file'".to_string())
        })?;
        let path_str = file
            .downcast_ref::<rustpython_vm::builtins::PyStr>()
            .ok_or_else(|| vm.new_type_error("savez: file must be a str path".to_string()))?
            .as_wtf8()
            .to_string_lossy()
            .into_owned();
        let final_path = if std::path::Path::new(&path_str)
            .extension()
            .and_then(|e| e.to_str())
            == Some("npz")
        {
            path_str
        } else {
            format!("{path_str}.npz")
        };

        let mut named: Vec<(String, ArraysD)> = Vec::new();
        for (i, a) in it.enumerate() {
            let arr = obj_to_array(&a, None, vm)?;
            named.push((format!("arr_{i}"), arr));
        }
        for (k, v) in args.kwargs.into_iter() {
            let arr = obj_to_array(&v, None, vm)?;
            named.push((k, arr));
        }
        crate::npz::save(std::path::Path::new(&final_path), &named)
            .map_err(|e| vm.new_os_error(format!("savez failed: {e}")))?;
        Ok(())
    }

    // ---------------- numpy.linalg submodule ----------------

    #[pymodule(name = "linalg")]
    pub(crate) mod linalg_sub {
        use crate::{convert::obj_to_array, linalg_extra};
        use rustpython_vm::{
            AsObject, PyObjectRef, PyPayload, PyResult, VirtualMachine, function::OptionalArg,
        };

        #[pyfunction]
        fn norm(a: PyObjectRef, args: NormArgs, vm: &VirtualMachine) -> PyResult<super::PyNdArray> {
            let arr = obj_to_array(&a, None, vm)?;
            let ord = parse_norm_ord(&args.ord, vm)?;
            let axis = parse_norm_axis(&args.axis, vm)?;
            let keepdims = args.keepdims.unwrap_or(false);
            Ok(super::PyNdArray::from_arrays(linalg_extra::norm(
                &arr, ord, axis, keepdims, vm,
            )?))
        }

        #[derive(rustpython_vm::FromArgs)]
        struct NormArgs {
            #[pyarg(any, optional)]
            ord: OptionalArg<PyObjectRef>,
            #[pyarg(any, optional)]
            axis: OptionalArg<PyObjectRef>,
            #[pyarg(any, optional)]
            keepdims: OptionalArg<bool>,
        }

        fn parse_norm_ord(
            arg: &OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<Option<linalg_extra::NormOrd>> {
            let obj = match arg {
                OptionalArg::Missing => return Ok(None),
                OptionalArg::Present(o) if o.is(&vm.ctx.none) => return Ok(None),
                OptionalArg::Present(o) => o,
            };
            // String first: "fro", "nuc".
            if let Some(s) = obj.downcast_ref::<rustpython_vm::builtins::PyStr>() {
                let bytes = s.as_wtf8().to_string_lossy();
                return match bytes.as_ref() {
                    "fro" => Ok(Some(linalg_extra::NormOrd::Fro)),
                    "nuc" => Ok(Some(linalg_extra::NormOrd::Nuc)),
                    other => Err(vm.new_value_error(format!("invalid ord: '{other}'"))),
                };
            }
            // Numeric — distinguish ±inf.
            let f = obj.try_float(vm)?.to_f64();
            if f.is_infinite() {
                Ok(Some(if f > 0.0 {
                    linalg_extra::NormOrd::PosInf
                } else {
                    linalg_extra::NormOrd::NegInf
                }))
            } else {
                Ok(Some(linalg_extra::NormOrd::Num(f)))
            }
        }

        fn parse_norm_axis(
            arg: &OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<Option<linalg_extra::NormAxis>> {
            let obj = match arg {
                OptionalArg::Missing => return Ok(None),
                OptionalArg::Present(o) if o.is(&vm.ctx.none) => return Ok(None),
                OptionalArg::Present(o) => o,
            };
            if let Some(t) = obj.downcast_ref::<rustpython_vm::builtins::PyTuple>() {
                let items = t.as_slice();
                if items.len() == 1 {
                    let v = items[0].try_int(vm)?.try_to_primitive::<isize>(vm)?;
                    return Ok(Some(linalg_extra::NormAxis::Single(v)));
                }
                if items.len() == 2 {
                    let i = items[0].try_int(vm)?.try_to_primitive::<isize>(vm)?;
                    let j = items[1].try_int(vm)?.try_to_primitive::<isize>(vm)?;
                    return Ok(Some(linalg_extra::NormAxis::Pair(i, j)));
                }
                return Err(vm.new_value_error(format!(
                    "axis tuple must have 1 or 2 elements; got {}",
                    items.len()
                )));
            }
            let v = obj.try_int(vm)?.try_to_primitive::<isize>(vm)?;
            Ok(Some(linalg_extra::NormAxis::Single(v)))
        }
        #[pyfunction]
        fn det(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<super::PyNdArray> {
            let arr = obj_to_array(&a, None, vm)?;
            Ok(super::PyNdArray::from_arrays(linalg_extra::det(&arr, vm)?))
        }
        #[pyfunction]
        fn inv(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<super::PyNdArray> {
            let arr = obj_to_array(&a, None, vm)?;
            Ok(super::PyNdArray::from_arrays(linalg_extra::inv(&arr, vm)?))
        }
        #[pyfunction]
        fn solve(
            a: PyObjectRef,
            b: PyObjectRef,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let x = obj_to_array(&a, None, vm)?;
            let y = obj_to_array(&b, None, vm)?;
            Ok(super::PyNdArray::from_arrays(linalg_extra::solve(
                &x, &y, vm,
            )?))
        }
        #[pyfunction]
        fn matrix_rank(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<super::PyNdArray> {
            let arr = obj_to_array(&a, None, vm)?;
            Ok(super::PyNdArray::from_arrays(linalg_extra::matrix_rank(
                &arr, vm,
            )?))
        }
        #[pyfunction]
        fn cholesky(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<super::PyNdArray> {
            let arr = obj_to_array(&a, None, vm)?;
            Ok(super::PyNdArray::from_arrays(linalg_extra::cholesky(
                &arr, vm,
            )?))
        }
        #[derive(rustpython_vm::FromArgs)]
        pub(crate) struct QrArgs {
            #[pyarg(positional)]
            a: PyObjectRef,
            #[pyarg(any, optional)]
            mode: OptionalArg<PyObjectRef>,
        }
        #[pyfunction]
        fn qr(args: QrArgs, vm: &VirtualMachine) -> PyResult<rustpython_vm::PyObjectRef> {
            use rustpython_vm::{PyPayload, builtins::PyTuple};
            let arr = obj_to_array(&args.a, None, vm)?;
            let mode_str = match args.mode {
                OptionalArg::Present(m) if !m.is(&vm.ctx.none) => m
                    .downcast_ref::<rustpython_vm::builtins::PyStr>()
                    .ok_or_else(|| vm.new_type_error("qr: mode must be a string".to_string()))?
                    .as_wtf8()
                    .to_string_lossy()
                    .into_owned(),
                _ => "reduced".to_string(),
            };
            let mode = linalg_extra::QrMode::parse(&mode_str).ok_or_else(|| {
                vm.new_value_error(format!(
                    "qr: invalid mode {mode_str:?}; expected 'reduced', 'complete', or 'r'"
                ))
            })?;
            let (q, r) = linalg_extra::qr(&arr, mode, vm)?;
            if matches!(mode, linalg_extra::QrMode::R) {
                return Ok(super::PyNdArray::from_arrays(r).into_pyobject(vm));
            }
            let tup = PyTuple::new_ref(
                vec![
                    super::PyNdArray::from_arrays(q).into_pyobject(vm),
                    super::PyNdArray::from_arrays(r).into_pyobject(vm),
                ],
                &vm.ctx,
            );
            Ok(tup.into())
        }

        #[pyfunction]
        fn lstsq(
            a: PyObjectRef,
            b: PyObjectRef,
            vm: &VirtualMachine,
        ) -> PyResult<rustpython_vm::PyObjectRef> {
            use rustpython_vm::{PyPayload, builtins::PyTuple};
            let ax = obj_to_array(&a, None, vm)?;
            let bx = obj_to_array(&b, None, vm)?;
            let r = linalg_extra::lstsq_full(&ax, &bx, vm)?;
            let tup = PyTuple::new_ref(
                vec![
                    super::PyNdArray::from_arrays(r.solution).into_pyobject(vm),
                    super::PyNdArray::from_arrays(r.residuals).into_pyobject(vm),
                    vm.ctx.new_int(r.rank).into(),
                    super::PyNdArray::from_arrays(r.singular).into_pyobject(vm),
                ],
                &vm.ctx,
            );
            Ok(tup.into())
        }
        #[pyfunction]
        fn pinv(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<super::PyNdArray> {
            let ax = obj_to_array(&a, None, vm)?;
            Ok(super::PyNdArray::from_arrays(linalg_extra::pinv(&ax, vm)?))
        }
        #[pyfunction]
        fn eigvalsh(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<super::PyNdArray> {
            let ax = obj_to_array(&a, None, vm)?;
            Ok(super::PyNdArray::from_arrays(linalg_extra::eigvalsh(
                &ax, vm,
            )?))
        }

        #[pyfunction]
        fn eigh(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
            let ax = obj_to_array(&a, None, vm)?;
            let (vals, vecs) = linalg_extra::eigh(&ax, vm)?;
            let v = super::PyNdArray::from_arrays(vals).into_pyobject(vm);
            let m = super::PyNdArray::from_arrays(vecs).into_pyobject(vm);
            Ok(rustpython_vm::builtins::PyTuple::new_ref(vec![v, m], &vm.ctx).into())
        }

        #[pyfunction]
        fn eig(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
            let ax = obj_to_array(&a, None, vm)?;
            let (vals, vecs) = linalg_extra::eig(&ax, vm)?;
            let v = super::PyNdArray::from_arrays(vals).into_pyobject(vm);
            let m = super::PyNdArray::from_arrays(vecs).into_pyobject(vm);
            Ok(rustpython_vm::builtins::PyTuple::new_ref(vec![v, m], &vm.ctx).into())
        }

        #[pyfunction]
        fn eigvals(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<super::PyNdArray> {
            let ax = obj_to_array(&a, None, vm)?;
            Ok(super::PyNdArray::from_arrays(linalg_extra::eigvals(
                &ax, vm,
            )?))
        }

        #[pyfunction]
        fn slogdet(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
            let ax = obj_to_array(&a, None, vm)?;
            let (sign, log_abs) = linalg_extra::slogdet(&ax, vm)?;
            let s = super::PyNdArray::from_arrays(sign).into_pyobject(vm);
            let l = super::PyNdArray::from_arrays(log_abs).into_pyobject(vm);
            Ok(rustpython_vm::builtins::PyTuple::new_ref(vec![s, l], &vm.ctx).into())
        }

        #[pyfunction]
        fn matrix_power(
            a: PyObjectRef,
            n: isize,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let ax = obj_to_array(&a, None, vm)?;
            Ok(super::PyNdArray::from_arrays(linalg_extra::matrix_power(
                &ax, n, vm,
            )?))
        }

        #[derive(rustpython_vm::FromArgs)]
        struct SvdArgs {
            #[pyarg(any, optional)]
            full_matrices: OptionalArg<bool>,
        }

        #[pyfunction]
        fn svd(a: PyObjectRef, args: SvdArgs, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
            let ax = obj_to_array(&a, None, vm)?;
            let full = args.full_matrices.unwrap_or(true);
            let (u, s, vh) = linalg_extra::svd(&ax, full, vm)?;
            let up = super::PyNdArray::from_arrays(u).into_pyobject(vm);
            let sp = super::PyNdArray::from_arrays(s).into_pyobject(vm);
            let vp = super::PyNdArray::from_arrays(vh).into_pyobject(vm);
            Ok(rustpython_vm::builtins::PyTuple::new_ref(vec![up, sp, vp], &vm.ctx).into())
        }

        /// `np.linalg.vector_norm(a)` — 2-norm of a flattened vector
        /// (numpy 2.0+ alias for `norm` with vector semantics).
        #[pyfunction]
        fn vector_norm(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<super::PyNdArray> {
            let ax = obj_to_array(&a, None, vm)?;
            Ok(super::PyNdArray::from_arrays(linalg_extra::norm(
                &ax,
                None,
                Some(linalg_extra::NormAxis::Single(0)),
                false,
                vm,
            )?))
        }

        /// `np.linalg.matrix_norm(a)` — Frobenius norm of a 2-D matrix.
        #[pyfunction]
        fn matrix_norm(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<super::PyNdArray> {
            let ax = obj_to_array(&a, None, vm)?;
            Ok(super::PyNdArray::from_arrays(linalg_extra::norm(
                &ax,
                Some(linalg_extra::NormOrd::Fro),
                None,
                false,
                vm,
            )?))
        }

        /// `np.linalg.vecdot(a, b)` — dot product of two vectors.
        #[pyfunction]
        fn vecdot(
            a: PyObjectRef,
            b: PyObjectRef,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let ax = obj_to_array(&a, None, vm)?;
            let bx = obj_to_array(&b, None, vm)?;
            Ok(super::PyNdArray::from_arrays(crate::linalg::dot(
                &ax, &bx, vm,
            )?))
        }

        /// `np.linalg.matmul(a, b)` — same as `np.matmul`.
        #[pyfunction]
        fn matmul(
            a: PyObjectRef,
            b: PyObjectRef,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let ax = obj_to_array(&a, None, vm)?;
            let bx = obj_to_array(&b, None, vm)?;
            Ok(super::PyNdArray::from_arrays(crate::linalg::dot(
                &ax, &bx, vm,
            )?))
        }

        /// `np.linalg.cross(a, b)` — cross product of two 3-vectors.
        #[pyfunction]
        fn cross(
            a: PyObjectRef,
            b: PyObjectRef,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let ax = obj_to_array(&a, None, vm)?;
            let bx = obj_to_array(&b, None, vm)?;
            Ok(super::PyNdArray::from_arrays(linalg_extra::cross(
                &ax, &bx, vm,
            )?))
        }

        /// `np.linalg.outer(a, b)` — outer product of two 1-D arrays.
        #[pyfunction]
        fn outer(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
            let numpy_mod = vm.import("numpy", 0)?;
            let f = numpy_mod.get_attr("outer", vm)?;
            f.call((a, b), vm)
        }

        /// `np.linalg.tensordot(a, b, axes=2)` — re-export of the top-level
        /// tensordot for the array-API namespace.
        #[pyfunction]
        fn tensordot(
            a: PyObjectRef,
            b: PyObjectRef,
            axes: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<PyObjectRef> {
            let numpy_mod = vm.import("numpy", 0)?;
            let f = numpy_mod.get_attr("tensordot", vm)?;
            match axes.into_option() {
                Some(ax) => f.call((a, b, ax), vm),
                None => f.call((a, b), vm),
            }
        }

        /// `np.linalg.trace(a)` — sum along the main diagonal.
        #[pyfunction]
        fn trace(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
            let numpy_mod = vm.import("numpy", 0)?;
            let f = numpy_mod.get_attr("trace", vm)?;
            f.call((a,), vm)
        }

        /// `np.linalg.diagonal(a)` — main diagonal.
        #[pyfunction]
        fn diagonal(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
            let numpy_mod = vm.import("numpy", 0)?;
            let f = numpy_mod.get_attr("diagonal", vm)?;
            f.call((a,), vm)
        }

        /// `np.linalg.matrix_transpose(a)` — swap last two axes.
        #[pyfunction]
        fn matrix_transpose(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
            let numpy_mod = vm.import("numpy", 0)?;
            let f = numpy_mod.get_attr("matrix_transpose", vm)?;
            f.call((a,), vm)
        }

        /// `np.linalg.svdvals(a)` — singular values of `a`.
        #[pyfunction]
        fn svdvals(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<super::PyNdArray> {
            let arr = obj_to_array(&a, None, vm)?;
            let (_u, s, _v) = crate::linalg_extra::svd(&arr, false, vm)?;
            Ok(super::PyNdArray::from_arrays(s))
        }

        /// `np.linalg.multi_dot([a, b, c, ...])` — chained matmul.
        #[pyfunction]
        fn multi_dot(arrays: PyObjectRef, vm: &VirtualMachine) -> PyResult<super::PyNdArray> {
            let list = arrays
                .downcast_ref::<rustpython_vm::builtins::PyList>()
                .map(|l| l.borrow_vec().to_vec())
                .or_else(|| {
                    arrays
                        .downcast_ref::<rustpython_vm::builtins::PyTuple>()
                        .map(|t| t.as_slice().to_vec())
                })
                .ok_or_else(|| {
                    vm.new_type_error("multi_dot expects a sequence of arrays".to_string())
                })?;
            if list.is_empty() {
                return Err(vm.new_value_error("multi_dot requires at least one array".to_string()));
            }
            let mut acc = obj_to_array(&list[0], None, vm)?;
            for o in &list[1..] {
                let next = obj_to_array(o, None, vm)?;
                acc = crate::linalg::dot(&acc, &next, vm)?;
            }
            Ok(super::PyNdArray::from_arrays(acc))
        }

        /// `np.linalg.cond(a)` — condition number wrt. the 2-norm.
        #[pyfunction]
        fn cond(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<super::PyNdArray> {
            let arr = obj_to_array(&a, None, vm)?;
            let (_u, s, _v) = crate::linalg_extra::svd(&arr, false, vm)?;
            let f = s.cast(crate::DType::F64);
            let vals_arr_f64 = f.cast(crate::DType::F64);
            let crate::ArraysD::F64(vals) = vals_arr_f64 else {
                return Err(crate::internal::internal(vm, "expected F64 after cast"));
            };
            let mut hi = f64::MIN;
            let mut lo = f64::MAX;
            for &v in vals.iter() {
                let av = v.abs();
                if av > hi {
                    hi = av;
                }
                if av < lo {
                    lo = av;
                }
            }
            let c = if lo == 0.0 || lo == f64::MAX {
                f64::INFINITY
            } else {
                hi / lo
            };
            Ok(super::PyNdArray::from_arrays(crate::ArraysD::F64(
                ndarray::Array::from_elem(ndarray::IxDyn(&[]), c),
            )))
        }

        /// `np.linalg.tensorinv(a, ind=2)` — generalised tensor inverse.
        #[pyfunction]
        fn tensorinv(
            a: PyObjectRef,
            ind: OptionalArg<usize>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let arr = obj_to_array(&a, None, vm)?;
            let ind = ind.unwrap_or(2);
            let shape = arr.shape().to_vec();
            if ind == 0 || ind >= shape.len() {
                return Err(
                    vm.new_value_error("tensorinv: `ind` must be > 0 and < ndim".to_string())
                );
            }
            let lead: usize = shape[..ind].iter().product();
            let trail: usize = shape[ind..].iter().product();
            if lead != trail {
                return Err(vm.new_value_error(format!(
                    "tensorinv: implied matrix shape {lead}x{trail} is not square"
                )));
            }
            let f = arr.cast(crate::DType::F64);
            let vals_arr_f64 = f.cast(crate::DType::F64);
            let crate::ArraysD::F64(flat) = vals_arr_f64 else {
                return Err(crate::internal::internal(vm, "expected F64 after cast"));
            };
            let mat = flat
                .into_shape_with_order(ndarray::IxDyn(&[lead, trail]))
                .map_err(|e| vm.new_value_error(format!("tensorinv reshape: {e}")))?;
            let inv = crate::linalg_extra::inv(&crate::ArraysD::F64(mat), vm)?;
            let vals_arr_f64 = inv.cast(crate::DType::F64);
            let crate::ArraysD::F64(inv_mat) = vals_arr_f64 else {
                return Err(crate::internal::internal(vm, "expected F64 after cast"));
            };
            let mut out_shape: Vec<usize> = shape[ind..].to_vec();
            out_shape.extend_from_slice(&shape[..ind]);
            let reshaped = inv_mat
                .into_shape_with_order(ndarray::IxDyn(&out_shape))
                .map_err(|e| vm.new_value_error(format!("tensorinv shape-back: {e}")))?;
            Ok(super::PyNdArray::from_arrays(crate::ArraysD::F64(reshaped)))
        }

        /// `np.linalg.tensorsolve(a, b, axes=None)` — solve a tensor system.
        #[pyfunction]
        fn tensorsolve(
            a: PyObjectRef,
            b: PyObjectRef,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let arr = obj_to_array(&a, None, vm)?;
            let rhs = obj_to_array(&b, None, vm)?;
            let n: usize = rhs.shape().iter().product();
            let total: usize = arr.shape().iter().product();
            if total % n != 0 || total / n != n {
                return Err(vm.new_value_error(
                    "tensorsolve: a.size/b.size must be a square integer".to_string(),
                ));
            }
            let fa = arr.cast(crate::DType::F64);
            let fb = rhs.cast(crate::DType::F64);
            let vals_arr_f64 = fa.cast(crate::DType::F64);
            let crate::ArraysD::F64(da) = vals_arr_f64 else {
                return Err(crate::internal::internal(vm, "expected F64 after cast"));
            };
            let vals_arr_f64 = fb.cast(crate::DType::F64);
            let crate::ArraysD::F64(db) = vals_arr_f64 else {
                return Err(crate::internal::internal(vm, "expected F64 after cast"));
            };
            let mat = da
                .into_shape_with_order(ndarray::IxDyn(&[n, n]))
                .map_err(|e| vm.new_value_error(format!("tensorsolve reshape A: {e}")))?;
            let vec = db
                .into_shape_with_order(ndarray::IxDyn(&[n]))
                .map_err(|e| vm.new_value_error(format!("tensorsolve reshape b: {e}")))?;
            let sol = crate::linalg_extra::solve(
                &crate::ArraysD::F64(mat),
                &crate::ArraysD::F64(vec),
                vm,
            )?;
            let vals_arr_f64 = sol.cast(crate::DType::F64);
            let crate::ArraysD::F64(out) = vals_arr_f64 else {
                return Err(crate::internal::internal(vm, "expected F64 after cast"));
            };
            let lead = arr.shape().len() - rhs.shape().len();
            let out_shape: Vec<usize> = arr.shape()[lead..].to_vec();
            let reshaped = out
                .into_shape_with_order(ndarray::IxDyn(&out_shape))
                .map_err(|e| vm.new_value_error(format!("tensorsolve shape-back: {e}")))?;
            Ok(super::PyNdArray::from_arrays(crate::ArraysD::F64(reshaped)))
        }

        // `test` — rumpy doesn't ship a test runner. The pyfunction below
        // ignores all arguments and returns True so test-runner harness code
        // (`np.linalg.test()`) succeeds without running anything.
        #[pyfunction]
        fn test(_args: rustpython_vm::function::FuncArgs) -> bool {
            true
        }
    }

    // ---------------- numpy.fft submodule ----------------

    #[pymodule(name = "fft")]
    pub(crate) mod fft_sub {
        use crate::{convert::obj_to_array, fft as ff};
        use rustpython_vm::{PyObjectRef, PyResult, VirtualMachine, function::OptionalArg};

        #[pyfunction]
        fn fft(
            a: PyObjectRef,
            n: OptionalArg<usize>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let arr = obj_to_array(&a, None, vm)?;
            Ok(super::PyNdArray::from_arrays(ff::fft(
                &arr,
                n.into_option(),
                vm,
            )?))
        }
        #[pyfunction]
        fn ifft(
            a: PyObjectRef,
            n: OptionalArg<usize>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let arr = obj_to_array(&a, None, vm)?;
            Ok(super::PyNdArray::from_arrays(ff::ifft(
                &arr,
                n.into_option(),
                vm,
            )?))
        }
        #[pyfunction]
        fn rfft(
            a: PyObjectRef,
            n: OptionalArg<usize>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let arr = obj_to_array(&a, None, vm)?;
            Ok(super::PyNdArray::from_arrays(ff::rfft(
                &arr,
                n.into_option(),
                vm,
            )?))
        }
        #[pyfunction]
        fn irfft(
            a: PyObjectRef,
            n: OptionalArg<usize>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let arr = obj_to_array(&a, None, vm)?;
            Ok(super::PyNdArray::from_arrays(ff::irfft(
                &arr,
                n.into_option(),
                vm,
            )?))
        }
        #[pyfunction]
        fn fft2(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<super::PyNdArray> {
            let arr = obj_to_array(&a, None, vm)?;
            Ok(super::PyNdArray::from_arrays(ff::fft2(&arr, vm)?))
        }
        #[pyfunction]
        fn ifft2(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<super::PyNdArray> {
            let arr = obj_to_array(&a, None, vm)?;
            Ok(super::PyNdArray::from_arrays(ff::ifft2(&arr, vm)?))
        }

        /// `np.fft.rfft2(a)` — 2-D real FFT. Output equals `fft2(a)` when `a`
        /// is purely real; we delegate to fft2 since the rustfft-backed
        /// implementation already returns complex.
        #[pyfunction]
        fn rfft2(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<super::PyNdArray> {
            let arr = obj_to_array(&a, None, vm)?;
            Ok(super::PyNdArray::from_arrays(ff::fft2(&arr, vm)?))
        }

        /// `np.fft.irfft2(a)` — inverse of `rfft2`.
        #[pyfunction]
        fn irfft2(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<super::PyNdArray> {
            let arr = obj_to_array(&a, None, vm)?;
            Ok(super::PyNdArray::from_arrays(ff::ifft2(&arr, vm)?))
        }

        /// `np.fft.rfftn(a)` — n-D real FFT.
        #[pyfunction]
        fn rfftn(
            a: PyObjectRef,
            args: FftnArgs,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let arr = obj_to_array(&a, None, vm)?;
            let _ = args.s;
            let axes = parse_axes(args.axes, vm)?;
            Ok(super::PyNdArray::from_arrays(ff::fftn(&arr, axes, vm)?))
        }

        /// `np.fft.irfftn(a)` — inverse of `rfftn`.
        #[pyfunction]
        fn irfftn(
            a: PyObjectRef,
            args: FftnArgs,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let arr = obj_to_array(&a, None, vm)?;
            let _ = args.s;
            let axes = parse_axes(args.axes, vm)?;
            Ok(super::PyNdArray::from_arrays(ff::ifftn(&arr, axes, vm)?))
        }

        /// `np.fft.hfft(a, n=None)` — Hermitian-symmetric FFT.
        ///
        /// For Hermitian-symmetric input ``a`` (length ``n//2 + 1``) this
        /// returns the real-valued FFT of length ``n``. rumpy uses
        /// ``irfft(a)`` and accepts the numpy-equivalent normalisation
        /// difference (callers who need the exact ``hfft == n * irfft``
        /// scaling can multiply by ``n`` after the fact).
        #[pyfunction]
        fn hfft(
            a: PyObjectRef,
            n: OptionalArg<usize>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let arr = obj_to_array(&a, None, vm)?;
            Ok(super::PyNdArray::from_arrays(ff::irfft(
                &arr,
                n.into_option(),
                vm,
            )?))
        }

        /// `np.fft.ihfft(a, n=None)` — inverse Hermitian FFT.
        ///
        /// Equivalent in rumpy to ``rfft(a)`` modulo the conjugate /
        /// normalisation difference noted on ``hfft``.
        #[pyfunction]
        fn ihfft(
            a: PyObjectRef,
            n: OptionalArg<usize>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let arr = obj_to_array(&a, None, vm)?;
            Ok(super::PyNdArray::from_arrays(ff::rfft(
                &arr,
                n.into_option(),
                vm,
            )?))
        }

        /// `np.fft.test()` — placeholder test runner (rumpy ships none).
        #[pyfunction]
        fn test(_args: rustpython_vm::function::FuncArgs) -> bool {
            true
        }

        #[derive(rustpython_vm::FromArgs)]
        pub(crate) struct FftnArgs {
            #[pyarg(any, optional)]
            s: OptionalArg<PyObjectRef>,
            #[pyarg(any, optional)]
            axes: OptionalArg<PyObjectRef>,
        }

        #[pyfunction]
        fn fftn(a: PyObjectRef, args: FftnArgs, vm: &VirtualMachine) -> PyResult<super::PyNdArray> {
            let arr = obj_to_array(&a, None, vm)?;
            // For now we ignore `s` (shape) — only `axes=` is honored.
            let _ = args.s;
            let axes = parse_axes(args.axes, vm)?;
            Ok(super::PyNdArray::from_arrays(ff::fftn(&arr, axes, vm)?))
        }

        #[pyfunction]
        fn ifftn(
            a: PyObjectRef,
            args: FftnArgs,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let arr = obj_to_array(&a, None, vm)?;
            let _ = args.s;
            let axes = parse_axes(args.axes, vm)?;
            Ok(super::PyNdArray::from_arrays(ff::ifftn(&arr, axes, vm)?))
        }
        #[derive(rustpython_vm::FromArgs)]
        pub(crate) struct FreqArgs {
            #[pyarg(positional)]
            n: usize,
            #[pyarg(any, optional)]
            d: OptionalArg<f64>,
        }
        #[pyfunction]
        fn fftfreq(args: FreqArgs, _vm: &VirtualMachine) -> super::PyNdArray {
            super::PyNdArray::from_arrays(ff::fftfreq(args.n, args.d.unwrap_or(1.0)))
        }
        #[pyfunction]
        fn rfftfreq(args: FreqArgs, _vm: &VirtualMachine) -> super::PyNdArray {
            super::PyNdArray::from_arrays(ff::rfftfreq(args.n, args.d.unwrap_or(1.0)))
        }
        fn parse_axes(
            obj: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<Option<Vec<isize>>> {
            use rustpython_vm::AsObject;
            match obj {
                OptionalArg::Missing => Ok(None),
                OptionalArg::Present(o) if o.is(&vm.ctx.none) => Ok(None),
                OptionalArg::Present(o) => {
                    if let Some(i) = o.downcast_ref::<rustpython_vm::builtins::PyInt>() {
                        Ok(Some(vec![i.try_to_primitive::<isize>(vm)?]))
                    } else {
                        let raw = crate::convert::parse_shape_signed(&o, vm)?;
                        Ok(Some(raw.into_iter().map(|v| v as isize).collect()))
                    }
                }
            }
        }

        #[derive(rustpython_vm::FromArgs)]
        pub(crate) struct ShiftArgs {
            #[pyarg(positional)]
            a: PyObjectRef,
            #[pyarg(any, optional)]
            axes: OptionalArg<PyObjectRef>,
        }
        #[pyfunction]
        fn fftshift(args: ShiftArgs, vm: &VirtualMachine) -> PyResult<super::PyNdArray> {
            let arr = obj_to_array(&args.a, None, vm)?;
            let ax = parse_axes(args.axes, vm)?;
            Ok(super::PyNdArray::from_arrays(ff::fftshift(&arr, ax, vm)?))
        }
        #[pyfunction]
        fn ifftshift(args: ShiftArgs, vm: &VirtualMachine) -> PyResult<super::PyNdArray> {
            let arr = obj_to_array(&args.a, None, vm)?;
            let ax = parse_axes(args.axes, vm)?;
            Ok(super::PyNdArray::from_arrays(ff::ifftshift(&arr, ax, vm)?))
        }
    }

    // ---------------- numpy.random submodule ----------------

    #[pymodule(name = "random")]
    pub(crate) mod random_sub {
        use crate::{convert::parse_shape, random as rnd};
        use rustpython_vm::{
            AsObject, PyObjectRef, PyResult, VirtualMachine, function::OptionalArg,
        };

        #[pyfunction]
        fn seed(s: u64) {
            rnd::seed(s);
        }

        fn shape_from_args(
            args: rustpython_vm::function::FuncArgs,
            vm: &VirtualMachine,
        ) -> PyResult<Vec<usize>> {
            // Accept rand(), rand(n), rand(*shape), rand((n, m))
            let mut out = Vec::with_capacity(args.args.len());
            if args.args.len() == 1
                && let Ok(s) = parse_shape(&args.args[0], vm)
            {
                return Ok(s);
            }
            for a in &args.args {
                out.push(a.try_int(vm)?.try_to_primitive::<usize>(vm)?);
            }
            Ok(out)
        }

        #[pyfunction]
        fn rand(
            args: rustpython_vm::function::FuncArgs,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = shape_from_args(args, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::rand(&s)))
        }
        #[pyfunction]
        fn randn(
            args: rustpython_vm::function::FuncArgs,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = shape_from_args(args, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::randn(&s)))
        }
        #[pyfunction]
        fn randint(
            low: i64,
            high: OptionalArg<i64>,
            shape: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            // numpy.random.randint(low[, high, size])
            let (lo, hi) = match high {
                OptionalArg::Present(h) => (low, h),
                OptionalArg::Missing => (0, low),
            };
            let s = match shape {
                OptionalArg::Present(o) => parse_shape(&o, vm)?,
                OptionalArg::Missing => vec![],
            };
            Ok(super::PyNdArray::from_arrays(rnd::randint(lo, hi, &s)))
        }
        #[pyfunction]
        fn uniform(
            low: OptionalArg<f64>,
            high: OptionalArg<f64>,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let lo = low.unwrap_or(0.0);
            let hi = high.unwrap_or(1.0);
            let s = match size {
                OptionalArg::Present(o) => parse_shape(&o, vm)?,
                OptionalArg::Missing => vec![],
            };
            Ok(super::PyNdArray::from_arrays(rnd::uniform(lo, hi, &s)))
        }
        #[pyfunction]
        fn normal(
            loc: OptionalArg<f64>,
            scale: OptionalArg<f64>,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let m = loc.unwrap_or(0.0);
            let s = scale.unwrap_or(1.0);
            let sh = match size {
                OptionalArg::Present(o) => parse_shape(&o, vm)?,
                OptionalArg::Missing => vec![],
            };
            Ok(super::PyNdArray::from_arrays(rnd::normal(m, s, &sh)))
        }

        fn size_to_shape(
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<Vec<usize>> {
            match size {
                OptionalArg::Present(o) if !o.is(&vm.ctx.none) => parse_shape(&o, vm),
                _ => Ok(vec![]),
            }
        }

        // ---- continuous distributions ----

        #[pyfunction]
        fn exponential(
            scale: OptionalArg<f64>,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::exponential(
                scale.unwrap_or(1.0),
                &s,
            )))
        }
        #[pyfunction]
        fn standard_exponential(
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::standard_exponential(&s)))
        }
        #[pyfunction]
        fn gamma(
            shape_k: f64,
            scale: OptionalArg<f64>,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::gamma(
                shape_k,
                scale.unwrap_or(1.0),
                &s,
            )))
        }
        #[pyfunction]
        fn standard_gamma(
            shape_k: f64,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::standard_gamma(
                shape_k, &s,
            )))
        }
        #[pyfunction]
        fn beta(
            a: f64,
            b: f64,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::beta(a, b, &s)))
        }
        #[pyfunction]
        fn chisquare(
            df: f64,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::chisquare(df, &s)))
        }
        #[pyfunction]
        fn standard_t(
            df: f64,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::standard_t(df, &s)))
        }
        #[pyfunction]
        fn f(
            dfnum: f64,
            dfden: f64,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::f_dist(dfnum, dfden, &s)))
        }
        #[pyfunction]
        fn noncentral_chisquare(
            df: f64,
            nonc: f64,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::noncentral_chisquare(
                df, nonc, &s,
            )))
        }
        #[pyfunction]
        fn noncentral_f(
            dfnum: f64,
            dfden: f64,
            nonc: f64,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::noncentral_f(
                dfnum, dfden, nonc, &s,
            )))
        }
        #[pyfunction]
        fn lognormal(
            mean: OptionalArg<f64>,
            sigma: OptionalArg<f64>,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::lognormal(
                mean.unwrap_or(0.0),
                sigma.unwrap_or(1.0),
                &s,
            )))
        }
        #[pyfunction]
        fn standard_cauchy(
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::standard_cauchy(&s)))
        }
        #[pyfunction]
        fn laplace(
            loc: OptionalArg<f64>,
            scale: OptionalArg<f64>,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::laplace(
                loc.unwrap_or(0.0),
                scale.unwrap_or(1.0),
                &s,
            )))
        }
        #[pyfunction]
        fn logistic(
            loc: OptionalArg<f64>,
            scale: OptionalArg<f64>,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::logistic(
                loc.unwrap_or(0.0),
                scale.unwrap_or(1.0),
                &s,
            )))
        }
        #[pyfunction]
        fn gumbel(
            loc: OptionalArg<f64>,
            scale: OptionalArg<f64>,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::gumbel(
                loc.unwrap_or(0.0),
                scale.unwrap_or(1.0),
                &s,
            )))
        }
        #[pyfunction]
        fn pareto(
            a: f64,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::pareto(a, &s)))
        }
        #[pyfunction]
        fn power(
            a: f64,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::power_dist(a, &s)))
        }
        #[pyfunction]
        fn rayleigh(
            scale: OptionalArg<f64>,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::rayleigh(
                scale.unwrap_or(1.0),
                &s,
            )))
        }
        #[pyfunction]
        fn triangular(
            left: f64,
            mode: f64,
            right: f64,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::triangular(
                left, mode, right, &s,
            )))
        }
        #[pyfunction]
        fn vonmises(
            mu: f64,
            kappa: f64,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::vonmises(mu, kappa, &s)))
        }
        #[pyfunction]
        fn wald(
            mean: f64,
            scale: f64,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::wald(mean, scale, &s)))
        }
        #[pyfunction]
        fn weibull(
            a: f64,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::weibull(a, &s)))
        }

        // ---- discrete distributions ----

        #[pyfunction]
        fn poisson(
            lam: OptionalArg<f64>,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::poisson(
                lam.unwrap_or(1.0),
                &s,
            )))
        }
        #[pyfunction]
        fn binomial(
            n: i64,
            p: f64,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::binomial(n, p, &s)))
        }
        #[pyfunction]
        fn geometric(
            p: f64,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::geometric(p, &s)))
        }
        #[pyfunction]
        fn negative_binomial(
            n: f64,
            p: f64,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::negative_binomial(
                n, p, &s,
            )))
        }
        #[pyfunction]
        fn hypergeometric(
            ngood: i64,
            nbad: i64,
            nsample: i64,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::hypergeometric(
                ngood, nbad, nsample, &s,
            )))
        }
        #[pyfunction]
        fn logseries(
            p: f64,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::logseries(p, &s)))
        }
        #[pyfunction]
        fn zipf(
            a: f64,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::zipf(a, &s)))
        }

        // ---- aliases for uniform [0,1) ----

        #[pyfunction]
        fn random(
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::rand(&s)))
        }
        #[pyfunction]
        fn random_sample(
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            random(size, vm)
        }
        #[pyfunction]
        fn ranf(size: OptionalArg<PyObjectRef>, vm: &VirtualMachine) -> PyResult<super::PyNdArray> {
            random(size, vm)
        }
        #[pyfunction]
        fn sample(
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            random(size, vm)
        }
        #[pyfunction]
        fn random_integers(
            low: i64,
            high: OptionalArg<i64>,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let (lo, hi) = match high {
                OptionalArg::Present(h) => (low, h + 1),
                OptionalArg::Missing => (1, low + 1),
            };
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::randint(lo, hi, &s)))
        }
        #[pyfunction]
        fn standard_normal(
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let s = size_to_shape(size, vm)?;
            Ok(super::PyNdArray::from_arrays(rnd::randn(&s)))
        }

        // ---- permutation / shuffle / choice ----

        #[pyfunction]
        fn permutation(n: i64) -> super::PyNdArray {
            super::PyNdArray::from_arrays(rnd::permutation(n))
        }
        #[pyfunction]
        fn shuffle(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<()> {
            // In-place shuffle by mutating the underlying ndarray. Caller
            // must pass a rumpy ndarray; passing anything else is a no-op.
            if let Some(p) = a.downcast_ref::<super::PyNdArray>() {
                let mut g = p.view_mut();
                rnd::shuffle(&mut g);
            } else {
                return Err(
                    vm.new_type_error("shuffle: argument must be a numpy ndarray".to_string())
                );
            }
            Ok(())
        }
        #[pyfunction]
        fn choice(
            a: PyObjectRef,
            size: OptionalArg<usize>,
            replace: OptionalArg<bool>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            // numpy.random.choice(a, size=None, replace=True): if a is int n,
            // sample from arange(n). Otherwise sample from arr.
            let pool: Vec<i64> = if let Some(i) = a.downcast_ref::<rustpython_vm::builtins::PyInt>()
            {
                let n = i.try_to_primitive::<i64>(vm)?.max(0);
                (0..n).collect()
            } else {
                let arr = crate::convert::obj_to_array(&a, None, vm)?;
                let cast = arr.cast(crate::DType::I64);
                let crate::ArraysD::I64(x) = cast else {
                    return Err(crate::internal::internal(
                        vm,
                        "choice: cast(I64) did not yield I64 variant",
                    ));
                };
                x.iter().copied().collect()
            };
            let n = size.unwrap_or(1);
            Ok(super::PyNdArray::from_arrays(rnd::choice(
                &pool,
                n,
                replace.unwrap_or(true),
            )))
        }
        #[pyfunction]
        fn bytes(length: usize) -> Vec<u8> {
            rnd::random_bytes(length)
        }

        // ---- state introspection ----

        #[pyfunction]
        fn get_state(_vm: &VirtualMachine) -> &'static str {
            "rumpy/std_rng"
        }
        #[pyfunction]
        fn set_state(_state: PyObjectRef) {
            // Stateless: setting state is a no-op (use `seed` to reseed).
        }
        #[pyfunction]
        fn get_bit_generator(_vm: &VirtualMachine) -> &'static str {
            "StdRng"
        }
        #[pyfunction]
        fn set_bit_generator(_bg: PyObjectRef) {}

        // ---- placeholders for Generator / BitGenerator family ----
        //
        // rumpy's RNG is a single global StdRng; the per-Generator class
        // hierarchy that real numpy uses (BitGenerator -> PCG64 / MT19937,
        // wrapped by Generator) is replaced by these inert factory
        // functions that simply return the module so users can still do
        // ``rng = np.random.default_rng()`` and then ``rng.normal(...)``.

        #[pyfunction]
        fn default_rng(
            _seed: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<PyObjectRef> {
            vm.import("numpy", 0)?.get_attr("random", vm)
        }

        // ---- additional distribution functions ----

        /// Dirichlet distribution: sample `(k,)` proportions whose components
        /// sum to 1 according to concentration parameters ``alpha``.
        #[pyfunction]
        fn dirichlet(
            alpha: PyObjectRef,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let alpha_arr = crate::convert::obj_to_array(&alpha, None, vm)?;
            let cast = alpha_arr.cast(crate::DType::F64);
            let vals_arr_f64 = cast.cast(crate::DType::F64);
            let crate::ArraysD::F64(av) = vals_arr_f64 else {
                return Err(crate::internal::internal(vm, "expected F64 after cast"));
            };
            let alphas: Vec<f64> = av.iter().copied().collect();
            let k = alphas.len();
            if k == 0 {
                return Err(vm.new_value_error("dirichlet: alpha must be non-empty".to_string()));
            }
            let sh = size_to_shape(size, vm)?;
            let n: usize = sh.iter().product::<usize>().max(1);

            // Sample each row as gamma(alpha_i, 1.0) and normalise.
            let mut out: Vec<f64> = Vec::with_capacity(n * k);
            for _ in 0..n {
                let mut row: Vec<f64> = alphas
                    .iter()
                    .map(|a| {
                        let gam = rnd::gamma(*a, 1.0, &[1]);
                        if let crate::ArraysD::F64(g) = gam {
                            g[0]
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let s: f64 = row.iter().sum::<f64>().max(f64::MIN_POSITIVE);
                for v in row.iter_mut() {
                    *v /= s;
                }
                out.extend_from_slice(&row);
            }

            let mut full_shape = sh;
            full_shape.push(k);
            let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&full_shape), out)
                .map_err(|e| vm.new_value_error(format!("dirichlet output: {e}")))?;
            Ok(super::PyNdArray::from_arrays(crate::ArraysD::F64(arr)))
        }

        /// Multinomial: throw ``n`` balls into ``len(pvals)`` bins, each ball
        /// landing in bin ``i`` with probability ``pvals[i]``. Returns the
        /// per-bin counts.
        #[pyfunction]
        fn multinomial(
            n: i64,
            pvals: PyObjectRef,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let p_arr = crate::convert::obj_to_array(&pvals, None, vm)?;
            let cast = p_arr.cast(crate::DType::F64);
            let vals_arr_f64 = cast.cast(crate::DType::F64);
            let crate::ArraysD::F64(pv) = vals_arr_f64 else {
                return Err(crate::internal::internal(vm, "expected F64 after cast"));
            };
            let ps: Vec<f64> = pv.iter().copied().collect();
            let k = ps.len();
            if k == 0 {
                return Err(vm.new_value_error("multinomial: pvals must be non-empty".to_string()));
            }
            let sh = size_to_shape(size, vm)?;
            let n_draws: usize = sh.iter().product::<usize>().max(1);
            let cum: Vec<f64> = {
                let mut acc = 0.0;
                ps.iter()
                    .map(|p| {
                        acc += p;
                        acc
                    })
                    .collect()
            };
            let mut out: Vec<i64> = Vec::with_capacity(n_draws * k);
            for _ in 0..n_draws {
                let mut row = vec![0i64; k];
                let uniforms = rnd::rand(&[n.max(0) as usize]);
                if let crate::ArraysD::F64(u) = uniforms {
                    for x in u.iter() {
                        // Pick the first bin whose cumulative probability >= u.
                        let bin = cum.iter().position(|c| *c >= *x).unwrap_or(k - 1);
                        row[bin] += 1;
                    }
                }
                out.extend_from_slice(&row);
            }
            let mut full_shape = sh;
            full_shape.push(k);
            let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&full_shape), out)
                .map_err(|e| vm.new_value_error(format!("multinomial output: {e}")))?;
            Ok(super::PyNdArray::from_arrays(crate::ArraysD::I64(arr)))
        }

        /// Multivariate normal: sample `(d,)` vectors from N(mean, cov) using
        /// the Cholesky factorisation of ``cov``.
        #[pyfunction]
        fn multivariate_normal(
            mean: PyObjectRef,
            cov: PyObjectRef,
            size: OptionalArg<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult<super::PyNdArray> {
            let m_arr = crate::convert::obj_to_array(&mean, None, vm)?;
            let c_arr = crate::convert::obj_to_array(&cov, None, vm)?;
            let mf = m_arr.cast(crate::DType::F64);
            let vals_arr_f64 = mf.cast(crate::DType::F64);
            let crate::ArraysD::F64(mv) = vals_arr_f64 else {
                return Err(crate::internal::internal(vm, "expected F64 after cast"));
            };
            let d = mv.len();
            // Cholesky of cov.
            let chol = crate::linalg_extra::cholesky(&c_arr, vm)?;
            let vals_arr_f64 = chol.cast(crate::DType::F64);
            let crate::ArraysD::F64(ch) = vals_arr_f64 else {
                return Err(crate::internal::internal(vm, "expected F64 after cast"));
            };
            let ch_mat: Vec<Vec<f64>> = {
                let mut out = vec![vec![0.0; d]; d];
                // Lower-triangular Cholesky factor — index it by hand.
                let shape = ch.shape().to_vec();
                if shape.len() == 2 && shape[0] == d && shape[1] == d {
                    for i in 0..d {
                        for j in 0..=i {
                            out[i][j] = ch[ndarray::IxDyn(&[i, j])];
                        }
                    }
                }
                out
            };
            let mean_vec: Vec<f64> = mv.iter().copied().collect();
            let sh = size_to_shape(size, vm)?;
            let n: usize = sh.iter().product::<usize>().max(1);
            let mut out: Vec<f64> = Vec::with_capacity(n * d);
            for _ in 0..n {
                let z = rnd::randn(&[d]);
                let vals_arr_f64 = z.cast(crate::DType::F64);
                let crate::ArraysD::F64(zv) = vals_arr_f64 else {
                    return Err(crate::internal::internal(vm, "expected F64 after cast"));
                };
                let z_vec: Vec<f64> = zv.iter().copied().collect();
                for i in 0..d {
                    let mut s = mean_vec[i];
                    for j in 0..=i.min(d - 1) {
                        s += ch_mat[i][j] * z_vec[j];
                    }
                    out.push(s);
                }
            }
            let mut full_shape = sh;
            full_shape.push(d);
            let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&full_shape), out)
                .map_err(|e| vm.new_value_error(format!("multivariate_normal output: {e}")))?;
            Ok(super::PyNdArray::from_arrays(crate::ArraysD::F64(arr)))
        }

        // ---- BitGenerator / Generator family ----
        //
        // rumpy backs all RNG operations with one global StdRng; the
        // BitGenerator-flavoured class hierarchy that numpy 2.x exposes is
        // mirrored by these tiny wrapper classes that all delegate back to
        // the module-level samplers. They satisfy `isinstance(x, np.random.Generator)`
        // checks and `np.random.PCG64(seed=42)`-style construction without
        // pretending to be different RNGs under the hood.

        #[pyfunction]
        fn _make_generator_classes(vm: &VirtualMachine) -> PyResult<PyObjectRef> {
            let src = r#"
class BitGenerator:
    """Thin proxy for numpy's BitGenerator base. rumpy stores no per-instance
    state — calls fall through to the shared global RNG."""

    def __init__(self, seed=None):
        if seed is not None:
            import numpy
            numpy.random.seed(int(seed))

    def random_raw(self, size=None):
        import numpy
        return numpy.random.randint(0, 1 << 63, size=size)


class Generator(BitGenerator):
    """Numpy 2.x Generator analog backed by the shared RNG."""

    def __init__(self, bit_generator=None):
        if bit_generator is None or isinstance(bit_generator, int):
            super().__init__(bit_generator)
        else:
            super().__init__()

    def __getattr__(self, name):
        import numpy
        if hasattr(numpy.random, name):
            return getattr(numpy.random, name)
        raise AttributeError(name)


class MT19937(BitGenerator): pass
class PCG64(BitGenerator): pass
class PCG64DXSM(BitGenerator): pass
class Philox(BitGenerator): pass
class SFC64(BitGenerator): pass


class RandomState:
    """Legacy numpy 1.x RandomState — rumpy proxies everything through the
    module-level samplers."""

    def __init__(self, seed=None):
        if seed is not None:
            import numpy
            numpy.random.seed(int(seed))

    def __getattr__(self, name):
        import numpy
        if hasattr(numpy.random, name):
            return getattr(numpy.random, name)
        raise AttributeError(name)


class SeedSequence:
    def __init__(self, entropy=None, *, spawn_key=(), pool_size=4, n_children_spawned=0):
        self.entropy = entropy
        self.spawn_key = tuple(spawn_key)
        self.pool_size = pool_size

    def spawn(self, n):
        return [SeedSequence(self.entropy, spawn_key=self.spawn_key + (i,)) for i in range(n)]

    def generate_state(self, n_words, dtype="uint32"):
        import numpy
        return numpy.random.randint(0, 1 << 31, size=n_words).astype(dtype)
"#;
            let dict = vm.ctx.new_dict();
            let scope = rustpython_vm::scope::Scope::with_builtins(None, dict.clone(), vm);
            let code = vm
                .compile(
                    src,
                    rustpython_vm::compiler::Mode::Exec,
                    "random_classes.py".into(),
                )
                .map_err(|e| vm.new_syntax_error(&e, Some(src)))?;
            vm.run_code_obj(code, scope)?;
            Ok(dict.into())
        }

        // Helper that lazily constructs and caches the generator-class dict
        // for the current VM, then returns the named class.
        fn fetch_random_class(vm: &VirtualMachine, name: &'static str) -> PyResult<PyObjectRef> {
            thread_local! {
                static CACHE: std::cell::RefCell<Option<(usize, PyObjectRef)>> =
                    const { std::cell::RefCell::new(None) };
            }
            let key = vm as *const VirtualMachine as usize;
            let cached = CACHE.with(|c| {
                c.borrow()
                    .as_ref()
                    .and_then(|(k, m)| if *k == key { Some(m.clone()) } else { None })
            });
            let dict = match cached {
                Some(d) => d,
                None => {
                    let d = _make_generator_classes(vm)?;
                    CACHE.with(|c| {
                        *c.borrow_mut() = Some((key, d.clone()));
                    });
                    d
                }
            };
            dict.get_item(name, vm)
        }

        #[pyfunction(name = "BitGenerator")]
        fn _bg(vm: &VirtualMachine) -> PyResult<PyObjectRef> {
            fetch_random_class(vm, "BitGenerator")
        }
        #[pyfunction(name = "Generator")]
        fn _gen(vm: &VirtualMachine) -> PyResult<PyObjectRef> {
            fetch_random_class(vm, "Generator")
        }
        #[pyfunction(name = "MT19937")]
        fn _mt(vm: &VirtualMachine) -> PyResult<PyObjectRef> {
            fetch_random_class(vm, "MT19937")
        }
        #[pyfunction(name = "PCG64")]
        fn _pcg64(vm: &VirtualMachine) -> PyResult<PyObjectRef> {
            fetch_random_class(vm, "PCG64")
        }
        #[pyfunction(name = "PCG64DXSM")]
        fn _pcg64d(vm: &VirtualMachine) -> PyResult<PyObjectRef> {
            fetch_random_class(vm, "PCG64DXSM")
        }
        #[pyfunction(name = "Philox")]
        fn _philox(vm: &VirtualMachine) -> PyResult<PyObjectRef> {
            fetch_random_class(vm, "Philox")
        }
        #[pyfunction(name = "SFC64")]
        fn _sfc(vm: &VirtualMachine) -> PyResult<PyObjectRef> {
            fetch_random_class(vm, "SFC64")
        }
        #[pyfunction(name = "RandomState")]
        fn _rs(vm: &VirtualMachine) -> PyResult<PyObjectRef> {
            fetch_random_class(vm, "RandomState")
        }
        #[pyfunction(name = "SeedSequence")]
        fn _ss(vm: &VirtualMachine) -> PyResult<PyObjectRef> {
            fetch_random_class(vm, "SeedSequence")
        }

        // ---- legacy submodule names ----
        //
        // numpy exposes `numpy.random.bit_generator` and `numpy.random.mtrand`
        // as importable submodules. We expose them as the same shim dict
        // that holds the generator classes.

        #[pyfunction(name = "bit_generator")]
        fn _bit_generator_mod(vm: &VirtualMachine) -> PyResult<PyObjectRef> {
            fetch_random_class(vm, "BitGenerator")
        }
        #[pyfunction(name = "mtrand")]
        fn _mtrand_mod(vm: &VirtualMachine) -> PyResult<PyObjectRef> {
            fetch_random_class(vm, "RandomState")
        }

        #[pyfunction]
        fn test(_args: rustpython_vm::function::FuncArgs) -> bool {
            true
        }
    }

    // ---------------- module attributes ----------------

    #[pyattr]
    fn pi(_vm: &VirtualMachine) -> f64 {
        core::f64::consts::PI
    }
    #[pyattr]
    fn e(_vm: &VirtualMachine) -> f64 {
        core::f64::consts::E
    }
    #[pyattr]
    fn inf(_vm: &VirtualMachine) -> f64 {
        f64::INFINITY
    }
    #[pyattr]
    fn nan(_vm: &VirtualMachine) -> f64 {
        f64::NAN
    }
    // Euler–Mascheroni constant γ (matches numpy's `np.euler_gamma`).
    #[pyattr]
    fn euler_gamma(_vm: &VirtualMachine) -> f64 {
        0.577_215_664_901_532_9_f64
    }
    // `np.newaxis is None` in numpy — used as a slice marker.
    #[pyattr]
    fn newaxis(vm: &VirtualMachine) -> PyObjectRef {
        vm.ctx.none()
    }
    // True for little-endian targets, false for big-endian. We assume the
    // host endianness mirrors what numpy would report.
    #[pyattr]
    fn little_endian(_vm: &VirtualMachine) -> bool {
        cfg!(target_endian = "little")
    }
    #[pyattr(once, name = "False_")]
    fn false_attr(vm: &VirtualMachine) -> PyObjectRef {
        let bool_cls = fetch_scalar_class(vm, "bool_");
        // Call bool_(False) to get the 0-D False scalar.
        match bool_cls.call((vm.ctx.false_value.clone(),), vm) {
            Ok(v) => v,
            Err(_) => vm.ctx.false_value.clone().into(),
        }
    }
    #[pyattr(once, name = "True_")]
    fn true_attr(vm: &VirtualMachine) -> PyObjectRef {
        let bool_cls = fetch_scalar_class(vm, "bool_");
        match bool_cls.call((vm.ctx.true_value.clone(),), vm) {
            Ok(v) => v,
            Err(_) => vm.ctx.true_value.clone().into(),
        }
    }

    // ---------------- alias-only top-level functions ----------------
    //
    // Names that numpy 2.x exposes as synonyms of an already-implemented
    // function. Each one re-points the user-facing name to the canonical
    // implementation; behaviour is identical.

    #[pyfunction(name = "around")]
    fn around_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        round_fn(a, vm)
    }
    #[pyfunction(name = "permute_dims")]
    fn permute_dims_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        transpose_fn(a, vm)
    }
    #[pyfunction(name = "pow")]
    fn pow_alias(a: PyObjectRef, b: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        power(a, b, vm)
    }
    #[pyfunction(name = "row_stack")]
    fn row_stack_fn(arrays: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        vstack(arrays, vm)
    }
    #[pyfunction(name = "concat")]
    fn concat_fn(
        arrays: PyObjectRef,
        args: ConcatenateArgs,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        concatenate(arrays, args, vm)
    }
    #[pyfunction(name = "cumulative_sum")]
    fn cumulative_sum_fn(
        a: PyObjectRef,
        args: AxisArg,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        cumsum(a, args, vm)
    }
    #[pyfunction(name = "cumulative_prod")]
    fn cumulative_prod_fn(
        a: PyObjectRef,
        args: AxisArg,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        cumprod(a, args, vm)
    }
    #[pyfunction(name = "bitwise_invert")]
    fn bitwise_invert_fn(a: PyObjectRef, vm: &VirtualMachine) -> PyResult<PyNdArray> {
        invert(a, vm)
    }
    #[pyfunction(name = "bitwise_left_shift")]
    fn bitwise_left_shift_fn(
        a: PyObjectRef,
        b: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        left_shift(a, b, vm)
    }
    #[pyfunction(name = "bitwise_right_shift")]
    fn bitwise_right_shift_fn(
        a: PyObjectRef,
        b: PyObjectRef,
        vm: &VirtualMachine,
    ) -> PyResult<PyNdArray> {
        right_shift(a, b, vm)
    }

    // ---------------- pure-Python submodules ----------------
    //
    // These mirror the user-facing slice of numpy's pure-Python submodules
    // (numpy.typing, numpy.exceptions, numpy.version). Each one lives as a
    // separate `.py` file under `py-src/`, embedded at build time via
    // `include_str!`. On first access we compile the source and execute it
    // in a fresh module namespace, injecting any Rust-side names the source
    // needs (e.g. the `ndarray` class for `typing.NDArray`).

    #[pyattr(once)]
    fn typing(vm: &VirtualMachine) -> PyObjectRef {
        build_py_submodule(
            vm,
            "numpy.typing",
            include_str!("../py-src/typing.py"),
            &[("_ndarray", {
                let cls = <PyNdArray as rustpython_vm::class::StaticType>::static_type();
                cls.to_owned().into()
            })],
        )
        .unwrap_or_else(|err| typing_panic(vm, "numpy.typing", &err))
    }

    #[pyattr(once)]
    fn exceptions(vm: &VirtualMachine) -> PyObjectRef {
        build_py_submodule(
            vm,
            "numpy.exceptions",
            include_str!("../py-src/exceptions.py"),
            &[],
        )
        .unwrap_or_else(|err| typing_panic(vm, "numpy.exceptions", &err))
    }

    // `LinAlgError` lives in numpy.exceptions (which can use Python class
    // syntax) but real numpy also exposes it at numpy.linalg.LinAlgError.
    // The `#[pymodule]` macro forbids nested pyattrs, so we resolve the
    // class lazily at numpy-module-level and ensure it's available via
    // both `np.LinAlgError` and (post-patch) `np.linalg.LinAlgError`.

    fn fetch_lin_alg_error(vm: &VirtualMachine) -> PyObjectRef {
        let numpy_mod = match vm.import("numpy", 0) {
            Ok(m) => m,
            Err(_) => return vm.ctx.exceptions.exception_type.to_owned().into(),
        };
        let exceptions_mod = match numpy_mod.get_attr("exceptions", vm) {
            Ok(m) => m,
            Err(_) => return vm.ctx.exceptions.exception_type.to_owned().into(),
        };
        let cls = match exceptions_mod.get_attr("LinAlgError", vm) {
            Ok(c) => c,
            Err(_) => return vm.ctx.exceptions.exception_type.to_owned().into(),
        };
        // Side-effect: also patch the class onto numpy.linalg so existing
        // `except np.linalg.LinAlgError:` clauses pick it up. Currently
        // this fails: `#[pyattr(once)]` items in this build of rustpython
        // are evaluated *during* `extend_module`, so `numpy.linalg` (added
        // by the macro's submodule_inits) isn't yet visible from the
        // child pyattr body. Embedders who want `np.linalg.LinAlgError`
        // attached should run, after `import numpy`,
        // ``numpy.linalg.LinAlgError = numpy.exceptions.LinAlgError``.
        if let Ok(linalg_mod) = numpy_mod.get_attr("linalg", vm) {
            let _ = linalg_mod.set_attr("LinAlgError", cls.clone(), vm);
        }
        cls
    }

    #[pyattr(once, name = "LinAlgError")]
    fn top_lin_alg_error(vm: &VirtualMachine) -> PyObjectRef {
        fetch_lin_alg_error(vm)
    }

    #[pyattr(once)]
    fn version(vm: &VirtualMachine) -> PyObjectRef {
        let ver = env!("CARGO_PKG_VERSION");
        build_py_submodule(
            vm,
            "numpy.version",
            include_str!("../py-src/version.py"),
            &[
                ("version", vm.ctx.new_str(ver).into()),
                ("full_version", vm.ctx.new_str(ver).into()),
                ("short_version", vm.ctx.new_str(ver).into()),
                ("git_revision", vm.ctx.new_str("unknown").into()),
                ("release", vm.ctx.true_value.clone().into()),
            ],
        )
        .unwrap_or_else(|err| typing_panic(vm, "numpy.version", &err))
    }

    #[pyattr(once)]
    fn compat(vm: &VirtualMachine) -> PyObjectRef {
        build_py_submodule(vm, "numpy.compat", include_str!("../py-src/compat.py"), &[])
            .unwrap_or_else(|err| typing_panic(vm, "numpy.compat", &err))
    }

    #[pyattr(once)]
    fn doc(vm: &VirtualMachine) -> PyObjectRef {
        build_py_submodule(vm, "numpy.doc", include_str!("../py-src/doc.py"), &[])
            .unwrap_or_else(|err| typing_panic(vm, "numpy.doc", &err))
    }

    #[pyattr(once, name = "core")]
    fn core_module(vm: &VirtualMachine) -> PyObjectRef {
        build_py_submodule(vm, "numpy.core", include_str!("../py-src/core.py"), &[])
            .unwrap_or_else(|err| typing_panic(vm, "numpy.core", &err))
    }

    #[pyattr(once)]
    fn ctypeslib(vm: &VirtualMachine) -> PyObjectRef {
        build_py_submodule(
            vm,
            "numpy.ctypeslib",
            include_str!("../py-src/ctypeslib.py"),
            &[],
        )
        .unwrap_or_else(|err| typing_panic(vm, "numpy.ctypeslib", &err))
    }

    #[pyattr(once)]
    fn char(vm: &VirtualMachine) -> PyObjectRef {
        build_py_submodule(vm, "numpy.char", include_str!("../py-src/char.py"), &[])
            .unwrap_or_else(|err| typing_panic(vm, "numpy.char", &err))
    }

    #[pyattr(once)]
    fn rec(vm: &VirtualMachine) -> PyObjectRef {
        build_py_submodule(vm, "numpy.rec", include_str!("../py-src/rec.py"), &[])
            .unwrap_or_else(|err| typing_panic(vm, "numpy.rec", &err))
    }

    #[pyattr(once)]
    fn dtypes(vm: &VirtualMachine) -> PyObjectRef {
        build_py_submodule(vm, "numpy.dtypes", include_str!("../py-src/dtypes.py"), &[])
            .unwrap_or_else(|err| typing_panic(vm, "numpy.dtypes", &err))
    }

    #[pyattr(once)]
    fn testing(vm: &VirtualMachine) -> PyObjectRef {
        build_py_submodule(
            vm,
            "numpy.testing",
            include_str!("../py-src/testing.py"),
            &[],
        )
        .unwrap_or_else(|err| typing_panic(vm, "numpy.testing", &err))
    }

    #[pyattr(once)]
    fn emath(vm: &VirtualMachine) -> PyObjectRef {
        let math = math_injections(vm);
        build_py_submodule(vm, "numpy.emath", include_str!("../py-src/emath.py"), &math)
            .unwrap_or_else(|err| typing_panic(vm, "numpy.emath", &err))
    }

    #[pyattr(once)]
    fn polynomial(vm: &VirtualMachine) -> PyObjectRef {
        let math = math_injections(vm);
        build_py_submodule(
            vm,
            "numpy.polynomial",
            include_str!("../py-src/polynomial.py"),
            &math,
        )
        .unwrap_or_else(|err| typing_panic(vm, "numpy.polynomial", &err))
    }

    #[pyattr(once)]
    fn strings(vm: &VirtualMachine) -> PyObjectRef {
        build_py_submodule(
            vm,
            "numpy.strings",
            include_str!("../py-src/strings.py"),
            &[],
        )
        .unwrap_or_else(|err| typing_panic(vm, "numpy.strings", &err))
    }

    #[pyattr(once)]
    fn f2py(vm: &VirtualMachine) -> PyObjectRef {
        build_py_submodule(vm, "numpy.f2py", include_str!("../py-src/f2py.py"), &[])
            .unwrap_or_else(|err| typing_panic(vm, "numpy.f2py", &err))
    }

    #[pyattr(once)]
    fn ma(vm: &VirtualMachine) -> PyObjectRef {
        let numpy_mod = vm
            .import("numpy", 0)
            .unwrap_or_else(|err| typing_panic(vm, "numpy.ma", &err));
        build_py_submodule(
            vm,
            "numpy.ma",
            include_str!("../py-src/ma.py"),
            &[("_np", numpy_mod)],
        )
        .unwrap_or_else(|err| typing_panic(vm, "numpy.ma", &err))
    }

    #[pyattr(once)]
    fn matrixlib(vm: &VirtualMachine) -> PyObjectRef {
        let numpy_mod = vm
            .import("numpy", 0)
            .unwrap_or_else(|err| typing_panic(vm, "numpy.matrixlib", &err));
        build_py_submodule(
            vm,
            "numpy.matrixlib",
            include_str!("../py-src/matrixlib.py"),
            &[("_np", numpy_mod)],
        )
        .unwrap_or_else(|err| typing_panic(vm, "numpy.matrixlib", &err))
    }

    fn matrixlib_export(vm: &VirtualMachine, name: &'static str) -> PyObjectRef {
        let numpy_mod = vm
            .import("numpy", 0)
            .unwrap_or_else(|err| typing_panic(vm, "numpy.matrixlib", &err));
        let matrixlib_mod = numpy_mod
            .get_attr("matrixlib", vm)
            .unwrap_or_else(|err| typing_panic(vm, "numpy.matrixlib", &err));
        matrixlib_mod
            .get_attr(name, vm)
            .unwrap_or_else(|err| typing_panic(vm, name, &err))
    }

    #[pyattr(once, name = "matrix")]
    fn top_matrix(vm: &VirtualMachine) -> PyObjectRef {
        matrixlib_export(vm, "matrix")
    }
    #[pyattr(once, name = "asmatrix")]
    fn top_asmatrix(vm: &VirtualMachine) -> PyObjectRef {
        matrixlib_export(vm, "asmatrix")
    }
    #[pyattr(once, name = "bmat")]
    fn top_bmat(vm: &VirtualMachine) -> PyObjectRef {
        matrixlib_export(vm, "bmat")
    }

    #[pyattr(once)]
    fn matlib(vm: &VirtualMachine) -> PyObjectRef {
        let numpy_mod = vm
            .import("numpy", 0)
            .unwrap_or_else(|err| typing_panic(vm, "numpy.matlib", &err));
        // Pull `matrix` / `asmatrix` / `mat` / `bmat` out of the
        // sibling `numpy.matrixlib` module so callers don't need to
        // `from numpy.matrixlib import ...` (the submodule isn't
        // registered in sys.modules).
        let matrixlib_mod = numpy_mod
            .get_attr("matrixlib", vm)
            .unwrap_or_else(|err| typing_panic(vm, "numpy.matlib", &err));
        let fetch = |name: &'static str| -> PyObjectRef {
            matrixlib_mod
                .get_attr(name, vm)
                .unwrap_or_else(|err| typing_panic(vm, "numpy.matlib", &err))
        };
        build_py_submodule(
            vm,
            "numpy.matlib",
            include_str!("../py-src/matlib.py"),
            &[
                ("_np", numpy_mod),
                ("matrix", fetch("matrix")),
                ("asmatrix", fetch("asmatrix")),
                ("mat", fetch("mat")),
                ("bmat", fetch("bmat")),
            ],
        )
        .unwrap_or_else(|err| typing_panic(vm, "numpy.matlib", &err))
    }

    // ---------------- top-level pure-Python extras ----------------
    //
    // Functions that real numpy exposes at `np.<name>` but were missing
    // from rumpy. They live in `py-src/_top_extras.py` and lean on the
    // already-implemented core via the injected `_np` handle.

    fn top_extras_module(vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        thread_local! {
            static CACHE: std::cell::RefCell<Option<(usize, PyObjectRef)>> =
                const { std::cell::RefCell::new(None) };
        }
        let key = vm as *const VirtualMachine as usize;
        let hit = CACHE.with(|c| {
            c.borrow()
                .as_ref()
                .and_then(|(k, m)| if *k == key { Some(m.clone()) } else { None })
        });
        if let Some(m) = hit {
            return Ok(m);
        }
        let numpy_mod = vm.import("numpy", 0).map_err(|e| {
            let mut s = String::new();
            let _ = vm.write_exception(&mut s, &e);
            vm.new_runtime_error(format!("could not import numpy for _top_extras: {s}"))
        })?;
        let m = build_py_submodule(
            vm,
            "numpy._top_extras",
            include_str!("../py-src/_top_extras.py"),
            &[("_np", numpy_mod)],
        )?;
        CACHE.with(|c| {
            *c.borrow_mut() = Some((key, m.clone()));
        });
        Ok(m)
    }

    fn fetch_top_extra(vm: &VirtualMachine, name: &'static str) -> PyObjectRef {
        match top_extras_module(vm) {
            Ok(m) => m.get_attr(name, vm).unwrap_or_else(|_| vm.ctx.none()),
            Err(_) => vm.ctx.none(),
        }
    }

    #[pyattr(once)]
    fn sinc(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "sinc")
    }
    #[pyattr(once)]
    fn float_power(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "float_power")
    }
    #[pyattr(once)]
    fn logaddexp(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "logaddexp")
    }
    #[pyattr(once)]
    fn logaddexp2(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "logaddexp2")
    }
    #[pyattr(once)]
    fn nan_to_num(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "nan_to_num")
    }
    #[pyattr(once)]
    fn real_if_close(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "real_if_close")
    }
    #[pyattr(once)]
    fn trim_zeros(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "trim_zeros")
    }
    #[pyattr(once)]
    fn bartlett(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "bartlett")
    }
    #[pyattr(once)]
    fn hamming(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "hamming")
    }
    #[pyattr(once)]
    fn hanning(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "hanning")
    }
    #[pyattr(once)]
    fn blackman(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "blackman")
    }
    #[pyattr(once)]
    fn kaiser(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "kaiser")
    }
    #[pyattr(once)]
    fn i0(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "i0")
    }
    #[pyattr(once)]
    fn broadcast_shapes(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "broadcast_shapes")
    }
    #[pyattr(once)]
    fn vander(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "vander")
    }
    #[pyattr(once)]
    fn diag_indices_from(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "diag_indices_from")
    }
    #[pyattr(once)]
    fn tril_indices_from(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "tril_indices_from")
    }
    #[pyattr(once)]
    fn triu_indices_from(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "triu_indices_from")
    }
    #[pyattr(once)]
    fn mask_indices(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "mask_indices")
    }
    #[pyattr(once)]
    fn fill_diagonal(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "fill_diagonal")
    }
    #[pyattr(once)]
    fn ediff1d(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "ediff1d")
    }
    #[pyattr(once)]
    fn intersect1d(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "intersect1d")
    }
    #[pyattr(once)]
    fn union1d(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "union1d")
    }
    #[pyattr(once)]
    fn setdiff1d(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "setdiff1d")
    }
    #[pyattr(once)]
    fn setxor1d(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "setxor1d")
    }
    #[pyattr(once)]
    fn isin(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "isin")
    }
    #[pyattr(once)]
    fn sort_complex(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "sort_complex")
    }
    #[pyattr(once)]
    fn unique_values(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "unique_values")
    }
    #[pyattr(once)]
    fn unique_counts(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "unique_counts")
    }
    #[pyattr(once)]
    fn unique_inverse(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "unique_inverse")
    }
    #[pyattr(once)]
    fn unique_all(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "unique_all")
    }
    #[pyattr(once)]
    fn digitize(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "digitize")
    }
    #[pyattr(once)]
    fn histogram_bin_edges(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "histogram_bin_edges")
    }
    #[pyattr(once)]
    fn histogram2d(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "histogram2d")
    }
    #[pyattr(once)]
    fn histogramdd(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "histogramdd")
    }
    #[pyattr(once)]
    fn ravel(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "ravel")
    }
    #[pyattr(once)]
    fn astype(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "astype")
    }
    #[pyattr(once)]
    fn take(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "take")
    }
    #[pyattr(once)]
    fn matrix_transpose(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "matrix_transpose")
    }
    #[pyattr(once)]
    fn vecdot(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "vecdot")
    }
    #[pyattr(once)]
    fn matvec(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "matvec")
    }
    #[pyattr(once)]
    fn vecmat(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "vecmat")
    }
    #[pyattr(once)]
    fn unstack(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "unstack")
    }
    #[pyattr(once)]
    fn isfortran(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "isfortran")
    }
    #[pyattr(once)]
    fn issubdtype(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "issubdtype")
    }
    #[pyattr(once)]
    fn isdtype(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "isdtype")
    }
    #[pyattr(once)]
    fn isnat(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "isnat")
    }
    #[pyattr(once)]
    fn iterable(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "iterable")
    }
    #[pyattr(once)]
    fn bitwise_count(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "bitwise_count")
    }
    #[pyattr(once)]
    fn array_repr(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "array_repr")
    }
    #[pyattr(once)]
    fn array_str(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "array_str")
    }
    #[pyattr(once)]
    fn array2string(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "array2string")
    }
    #[pyattr(once)]
    fn format_float_positional(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "format_float_positional")
    }
    #[pyattr(once)]
    fn format_float_scientific(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "format_float_scientific")
    }
    #[pyattr(once)]
    fn set_printoptions(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "set_printoptions")
    }
    #[pyattr(once)]
    fn get_printoptions(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "get_printoptions")
    }
    #[pyattr(once)]
    fn printoptions(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "printoptions")
    }
    #[pyattr(once)]
    fn getbufsize(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "getbufsize")
    }
    #[pyattr(once)]
    fn setbufsize(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "setbufsize")
    }
    #[pyattr(once)]
    fn seterrcall(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "seterrcall")
    }
    #[pyattr(once)]
    fn frombuffer(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "frombuffer")
    }
    #[pyattr(once)]
    fn from_dlpack(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "from_dlpack")
    }
    #[pyattr(once)]
    fn fromfunction(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "fromfunction")
    }
    #[pyattr(once)]
    fn fromregex(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "fromregex")
    }
    #[pyattr(once)]
    fn genfromtxt(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "genfromtxt")
    }
    #[pyattr(once)]
    fn asarray_chkfinite(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "asarray_chkfinite")
    }
    #[pyattr(once)]
    fn packbits(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "packbits")
    }
    #[pyattr(once)]
    fn unpackbits(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "unpackbits")
    }
    #[pyattr(once)]
    fn putmask(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "putmask")
    }
    #[pyattr(once)]
    fn shares_memory(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "shares_memory")
    }
    #[pyattr(once)]
    fn may_share_memory(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "may_share_memory")
    }
    #[pyattr(once)]
    fn info(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "info")
    }
    #[pyattr(once)]
    fn show_config(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "show_config")
    }
    #[pyattr(once)]
    fn show_runtime(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "show_runtime")
    }
    #[pyattr(once)]
    fn get_include(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "get_include")
    }
    #[pyattr(once)]
    fn common_type(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "common_type")
    }
    #[pyattr(once)]
    fn mintypecode(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "mintypecode")
    }
    #[pyattr(once)]
    fn typename(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "typename")
    }
    #[pyattr(once)]
    fn typecodes(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "typecodes")
    }
    #[pyattr(once)]
    fn select(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "select")
    }
    #[pyattr(once)]
    fn piecewise(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "piecewise")
    }
    #[pyattr(once)]
    fn apply_over_axes(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "apply_over_axes")
    }
    #[pyattr(once)]
    fn einsum_path(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "einsum_path")
    }
    #[pyattr(once)]
    fn index_exp(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "index_exp")
    }

    // ---- datetime / timedelta family ----
    //
    // Lives in py-src/_datetime.py and is wired via the same
    // build_py_submodule/cache mechanism as _top_extras. The classes
    // wrap Python's `datetime` for scalar arithmetic; rumpy currently
    // doesn't expose datetime64 as a real ndarray dtype.

    fn datetime_module(vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        thread_local! {
            static CACHE: std::cell::RefCell<Option<(usize, PyObjectRef)>> =
                const { std::cell::RefCell::new(None) };
        }
        let key = vm as *const VirtualMachine as usize;
        let hit = CACHE.with(|c| {
            c.borrow()
                .as_ref()
                .and_then(|(k, m)| if *k == key { Some(m.clone()) } else { None })
        });
        if let Some(m) = hit {
            return Ok(m);
        }
        let m = build_py_submodule(
            vm,
            "numpy._datetime",
            include_str!("../py-src/_datetime.py"),
            &[],
        )?;
        CACHE.with(|c| {
            *c.borrow_mut() = Some((key, m.clone()));
        });
        Ok(m)
    }

    fn fetch_datetime(vm: &VirtualMachine, name: &'static str) -> PyObjectRef {
        match datetime_module(vm) {
            Ok(m) => m.get_attr(name, vm).unwrap_or_else(|_| vm.ctx.none()),
            Err(_) => vm.ctx.none(),
        }
    }

    fn iter_module(vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        thread_local! {
            static CACHE: std::cell::RefCell<Option<(usize, PyObjectRef)>> =
                const { std::cell::RefCell::new(None) };
        }
        let key = vm as *const VirtualMachine as usize;
        let hit = CACHE.with(|c| {
            c.borrow()
                .as_ref()
                .and_then(|(k, m)| if *k == key { Some(m.clone()) } else { None })
        });
        if let Some(m) = hit {
            return Ok(m);
        }
        let numpy_mod = vm.import("numpy", 0)?;
        let m = build_py_submodule(
            vm,
            "numpy._iter",
            include_str!("../py-src/_iter.py"),
            &[("_np", numpy_mod)],
        )?;
        CACHE.with(|c| {
            *c.borrow_mut() = Some((key, m.clone()));
        });
        Ok(m)
    }

    fn fetch_iter(vm: &VirtualMachine, name: &'static str) -> PyObjectRef {
        match iter_module(vm) {
            Ok(m) => m.get_attr(name, vm).unwrap_or_else(|_| vm.ctx.none()),
            Err(_) => vm.ctx.none(),
        }
    }

    // Top-level wrappers from _top_extras.
    #[pyattr(once)]
    fn poly1d(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "poly1d")
    }
    #[pyattr(once)]
    fn poly(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "poly")
    }
    #[pyattr(once)]
    fn polyadd(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "polyadd")
    }
    #[pyattr(once)]
    fn polysub(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "polysub")
    }
    #[pyattr(once)]
    fn polymul(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "polymul")
    }
    #[pyattr(once)]
    fn polydiv(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "polydiv")
    }
    #[pyattr(once)]
    fn ufunc(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "ufunc")
    }

    // recarray / record — re-export from numpy.rec.
    #[pyattr(once)]
    fn recarray(vm: &VirtualMachine) -> PyObjectRef {
        let numpy_mod = vm
            .import("numpy", 0)
            .unwrap_or_else(|err| typing_panic(vm, "recarray", &err));
        let rec_mod = numpy_mod
            .get_attr("rec", vm)
            .unwrap_or_else(|err| typing_panic(vm, "recarray", &err));
        rec_mod
            .get_attr("recarray", vm)
            .unwrap_or_else(|err| typing_panic(vm, "recarray", &err))
    }
    #[pyattr(once)]
    fn record(vm: &VirtualMachine) -> PyObjectRef {
        let numpy_mod = vm
            .import("numpy", 0)
            .unwrap_or_else(|err| typing_panic(vm, "record", &err));
        let rec_mod = numpy_mod
            .get_attr("rec", vm)
            .unwrap_or_else(|err| typing_panic(vm, "record", &err));
        rec_mod
            .get_attr("record", vm)
            .unwrap_or_else(|err| typing_panic(vm, "record", &err))
    }

    #[pyattr(once)]
    fn ndindex(vm: &VirtualMachine) -> PyObjectRef {
        fetch_iter(vm, "ndindex")
    }
    #[pyattr(once)]
    fn ndenumerate(vm: &VirtualMachine) -> PyObjectRef {
        fetch_iter(vm, "ndenumerate")
    }
    #[pyattr(once)]
    fn broadcast(vm: &VirtualMachine) -> PyObjectRef {
        fetch_iter(vm, "broadcast")
    }
    #[pyattr(once)]
    fn nditer(vm: &VirtualMachine) -> PyObjectRef {
        fetch_iter(vm, "nditer")
    }
    #[pyattr(once)]
    fn flatiter(vm: &VirtualMachine) -> PyObjectRef {
        fetch_iter(vm, "flatiter")
    }
    #[pyattr(once)]
    fn nested_iters(vm: &VirtualMachine) -> PyObjectRef {
        fetch_iter(vm, "nested_iters")
    }
    #[pyattr(once)]
    fn memmap(vm: &VirtualMachine) -> PyObjectRef {
        fetch_iter(vm, "memmap")
    }

    #[pyattr(once)]
    fn datetime64(vm: &VirtualMachine) -> PyObjectRef {
        fetch_datetime(vm, "datetime64")
    }
    #[pyattr(once)]
    fn timedelta64(vm: &VirtualMachine) -> PyObjectRef {
        fetch_datetime(vm, "timedelta64")
    }
    #[pyattr(once)]
    fn datetime_as_string(vm: &VirtualMachine) -> PyObjectRef {
        fetch_datetime(vm, "datetime_as_string")
    }
    #[pyattr(once)]
    fn datetime_data(vm: &VirtualMachine) -> PyObjectRef {
        fetch_datetime(vm, "datetime_data")
    }
    #[pyattr(once)]
    fn busdaycalendar(vm: &VirtualMachine) -> PyObjectRef {
        fetch_datetime(vm, "busdaycalendar")
    }
    #[pyattr(once)]
    fn is_busday(vm: &VirtualMachine) -> PyObjectRef {
        fetch_datetime(vm, "is_busday")
    }
    #[pyattr(once)]
    fn busday_count(vm: &VirtualMachine) -> PyObjectRef {
        fetch_datetime(vm, "busday_count")
    }
    #[pyattr(once)]
    fn busday_offset(vm: &VirtualMachine) -> PyObjectRef {
        fetch_datetime(vm, "busday_offset")
    }

    // These need custom names because they clash with Rust keywords or
    // are intentionally exposed differently.
    #[pyattr(once, name = "copy")]
    fn np_copy(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "copy")
    }
    #[pyattr(once, name = "shape")]
    fn np_shape(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "shape")
    }
    #[pyattr(once, name = "size")]
    fn np_size(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "size")
    }
    #[pyattr(once, name = "ndim")]
    fn np_ndim(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "ndim")
    }
    #[pyattr(once, name = "diagonal")]
    fn np_diagonal(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "diagonal")
    }
    #[pyattr(once, name = "std")]
    fn np_std(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "std")
    }
    #[pyattr(once, name = "var")]
    fn np_var(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "var")
    }
    #[pyattr(once, name = "test")]
    fn np_test(vm: &VirtualMachine) -> PyObjectRef {
        fetch_top_extra(vm, "test")
    }

    // ---------------- mgrid / ogrid / r_ / c_ / s_ / ix_ ----------------

    fn index_helpers_module(vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        thread_local! {
            static CACHE: std::cell::RefCell<Option<(usize, PyObjectRef)>> =
                const { std::cell::RefCell::new(None) };
        }
        let key = vm as *const VirtualMachine as usize;
        let hit = CACHE.with(|c| {
            c.borrow()
                .as_ref()
                .and_then(|(k, m)| if *k == key { Some(m.clone()) } else { None })
        });
        if let Some(m) = hit {
            return Ok(m);
        }
        // Build with `_np` set to the current `numpy` module.
        let numpy_mod = vm.import("numpy", 0).map_err(|e| {
            let mut s = String::new();
            let _ = vm.write_exception(&mut s, &e);
            vm.new_runtime_error(format!("could not import numpy: {s}"))
        })?;
        let m = build_py_submodule(
            vm,
            "numpy._index_helpers",
            include_str!("../py-src/index_helpers.py"),
            &[("_np", numpy_mod)],
        )?;
        CACHE.with(|c| {
            *c.borrow_mut() = Some((key, m.clone()));
        });
        Ok(m)
    }

    fn fetch_index_helper(vm: &VirtualMachine, name: &'static str) -> PyObjectRef {
        match index_helpers_module(vm) {
            Ok(m) => m.get_attr(name, vm).unwrap_or_else(|_| vm.ctx.none()),
            Err(_) => vm.ctx.none(),
        }
    }

    #[pyattr(once)]
    fn mgrid(vm: &VirtualMachine) -> PyObjectRef {
        fetch_index_helper(vm, "mgrid")
    }
    #[pyattr(once)]
    fn ogrid(vm: &VirtualMachine) -> PyObjectRef {
        fetch_index_helper(vm, "ogrid")
    }
    #[pyattr(once, name = "r_")]
    fn r_helper(vm: &VirtualMachine) -> PyObjectRef {
        fetch_index_helper(vm, "r_")
    }
    #[pyattr(once, name = "c_")]
    fn c_helper(vm: &VirtualMachine) -> PyObjectRef {
        fetch_index_helper(vm, "c_")
    }
    #[pyattr(once, name = "s_")]
    fn s_helper(vm: &VirtualMachine) -> PyObjectRef {
        fetch_index_helper(vm, "s_")
    }
    #[pyattr(once)]
    fn ix_(vm: &VirtualMachine) -> PyObjectRef {
        fetch_index_helper(vm, "ix_")
    }

    #[pyattr(once)]
    fn lib(vm: &VirtualMachine) -> PyObjectRef {
        // Build numpy.lib with a `stride_tricks` attribute that is itself a
        // submodule. We build both, then patch.
        let stride = build_py_submodule(
            vm,
            "numpy.lib.stride_tricks",
            include_str!("../py-src/lib_stride_tricks.py"),
            &[],
        )
        .unwrap_or_else(|err| typing_panic(vm, "numpy.lib.stride_tricks", &err));
        let lib_mod = build_py_submodule(vm, "numpy.lib", include_str!("../py-src/lib.py"), &[])
            .unwrap_or_else(|err| typing_panic(vm, "numpy.lib", &err));
        let _ = lib_mod.set_attr("stride_tricks", stride, vm);
        lib_mod
    }

    // ---------------- numpy scalar type hierarchy ----------------
    //
    // np.int32, np.float64, np.integer, np.floating, etc. are exposed as
    // Python classes that build 0-D ndarrays when called. The hierarchy
    // matches numpy so `isinstance(x, np.integer)` works for x = int32(...).

    fn build_scalars_module(vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        // Inject _np_array so the embedded source can build 0-D ndarrays
        // without needing to `import numpy` (which would recurse).
        let array_fn = vm.new_function("array", |args: FuncArgs, vm: &VirtualMachine| {
            // Match `np.array(value, dtype=...)`.
            let mut it = args.args.into_iter();
            let obj = it
                .next()
                .ok_or_else(|| vm.new_type_error("array() needs a value".to_string()))?;
            let dtype_obj = args.kwargs.get("dtype").cloned().or_else(|| it.next());
            let dt = parse_dtype_arg(&dtype_obj, vm)?;
            let arr = obj_to_array(&obj, dt, vm)?;
            Ok::<_, rustpython_vm::PyRef<rustpython_vm::builtins::PyBaseException>>(
                PyNdArray::from_arrays(arr).into_pyobject(vm),
            )
        });
        build_py_submodule(
            vm,
            "numpy._scalars",
            include_str!("../py-src/scalars.py"),
            &[("_np_array", array_fn.into())],
        )
    }

    /// Build the scalar-types module *once* per `VirtualMachine` and cache it.
    /// All scalar pyattrs read from the same instance so the class hierarchy
    /// shares MRO links (`issubclass(int32, integer)` works).
    fn scalars_module(vm: &VirtualMachine) -> PyResult<PyObjectRef> {
        thread_local! {
            static CACHE: std::cell::RefCell<Option<(usize, PyObjectRef)>> =
                const { std::cell::RefCell::new(None) };
        }
        let key = vm as *const VirtualMachine as usize;
        let hit = CACHE.with(|c| {
            c.borrow()
                .as_ref()
                .and_then(|(k, m)| if *k == key { Some(m.clone()) } else { None })
        });
        if let Some(m) = hit {
            return Ok(m);
        }
        let m = build_scalars_module(vm)?;
        CACHE.with(|c| {
            *c.borrow_mut() = Some((key, m.clone()));
        });
        Ok(m)
    }

    fn fetch_scalar_class(vm: &VirtualMachine, name: &'static str) -> PyObjectRef {
        match scalars_module(vm) {
            Ok(m) => match m.get_attr(name, vm) {
                Ok(cls) => cls,
                Err(_) => vm.ctx.none(),
            },
            Err(_) => vm.ctx.none(),
        }
    }

    // Abstract base classes.
    #[pyattr(once)]
    fn generic(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "generic")
    }
    #[pyattr(once)]
    fn number(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "number")
    }
    #[pyattr(once)]
    fn integer(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "integer")
    }
    #[pyattr(once)]
    fn signedinteger(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "signedinteger")
    }
    #[pyattr(once)]
    fn unsignedinteger(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "unsignedinteger")
    }
    #[pyattr(once)]
    fn inexact(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "inexact")
    }
    #[pyattr(once)]
    fn floating(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "floating")
    }
    #[pyattr(once)]
    fn complexfloating(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "complexfloating")
    }

    // Concrete leaf classes (numeric dtypes).
    #[pyattr(once)]
    fn bool_(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "bool_")
    }
    #[pyattr(once)]
    fn int8(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "int8")
    }
    #[pyattr(once)]
    fn int16(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "int16")
    }
    #[pyattr(once)]
    fn int32(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "int32")
    }
    #[pyattr(once)]
    fn int64(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "int64")
    }
    #[pyattr(once)]
    fn uint8(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "uint8")
    }
    #[pyattr(once)]
    fn uint16(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "uint16")
    }
    #[pyattr(once)]
    fn uint32(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "uint32")
    }
    #[pyattr(once)]
    fn uint64(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "uint64")
    }
    #[pyattr(once)]
    fn float16(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "float16")
    }
    #[pyattr(once)]
    fn float32(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "float32")
    }
    #[pyattr(once)]
    fn float64(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "float64")
    }
    #[pyattr(once)]
    fn complex64(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "complex64")
    }
    #[pyattr(once)]
    fn complex128(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "complex128")
    }

    // numpy aliases.
    #[pyattr(once)]
    fn intp(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "intp")
    }
    #[pyattr(once)]
    fn uintp(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "uintp")
    }
    #[pyattr(once)]
    fn intc(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "intc")
    }
    #[pyattr(once)]
    fn uintc(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "uintc")
    }
    #[pyattr(once)]
    fn short(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "short")
    }
    #[pyattr(once)]
    fn ushort(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "ushort")
    }
    #[pyattr(once)]
    fn byte(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "byte")
    }
    #[pyattr(once)]
    fn ubyte(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "ubyte")
    }
    #[pyattr(once)]
    fn longlong(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "longlong")
    }
    #[pyattr(once)]
    fn ulonglong(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "ulonglong")
    }
    #[pyattr(once)]
    fn single(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "single")
    }
    #[pyattr(once)]
    fn double(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "double")
    }
    #[pyattr(once, name = "half")]
    fn half_scalar(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "half")
    }
    #[pyattr(once)]
    fn csingle(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "csingle")
    }
    #[pyattr(once)]
    fn cdouble(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "cdouble")
    }

    #[pyattr(once, name = "ScalarType")]
    fn scalar_type(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "ScalarType")
    }
    #[pyattr(once, name = "sctypeDict")]
    fn sctype_dict(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "sctypeDict")
    }

    // Python-builtin-shadowing aliases that numpy 2.x re-added.
    #[pyattr(once, name = "bool")]
    fn bool_alias(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "bool")
    }
    #[pyattr(once, name = "int_")]
    fn int_alias(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "int_")
    }
    #[pyattr(once, name = "uint")]
    fn uint_alias(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "uint")
    }
    #[pyattr(once, name = "long")]
    fn long_alias(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "long")
    }
    #[pyattr(once, name = "ulong")]
    fn ulong_alias(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "ulong")
    }

    // Extended precision (rumpy lacks 80-bit floats; these are aliased to f64).
    #[pyattr(once)]
    fn longdouble(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "longdouble")
    }
    #[pyattr(once)]
    fn clongdouble(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "clongdouble")
    }
    #[pyattr(once)]
    fn float128(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "float128")
    }
    #[pyattr(once)]
    fn complex256(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "complex256")
    }

    // String / object / void scalar types.
    #[pyattr(once)]
    fn flexible(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "flexible")
    }
    #[pyattr(once)]
    fn character(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "character")
    }
    #[pyattr(once)]
    fn str_(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "str_")
    }
    #[pyattr(once)]
    fn bytes_(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "bytes_")
    }
    #[pyattr(once)]
    fn object_(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "object_")
    }
    #[pyattr(once)]
    fn void(vm: &VirtualMachine) -> PyObjectRef {
        fetch_scalar_class(vm, "void")
    }

    /// Build a list of math primitives (functions + constants) implemented
    /// in Rust, suitable for injecting into a submodule's namespace under
    /// `_pi`, `_sqrt`, `_log`, … This lets pure-Python submodules avoid
    /// `import math`, which our minimal rustpython build doesn't ship.
    fn math_injections(vm: &VirtualMachine) -> Vec<(&'static str, PyObjectRef)> {
        macro_rules! f1 {
            ($name:literal, $f:expr) => {{
                let func = vm.new_function($name, $f as fn(f64) -> f64);
                ($name, func.into())
            }};
        }
        macro_rules! f2 {
            ($name:literal, $f:expr) => {{
                let func = vm.new_function($name, $f as fn(f64, f64) -> f64);
                ($name, func.into())
            }};
        }
        vec![
            ("_pi", vm.ctx.new_float(std::f64::consts::PI).into()),
            ("_e", vm.ctx.new_float(std::f64::consts::E).into()),
            f1!("_sqrt", |x| x.sqrt()),
            f1!("_log", |x| x.ln()),
            f1!("_log10", |x| x.log10()),
            f1!("_log2", |x| x.log2()),
            f1!("_exp", |x| x.exp()),
            f1!("_cos", |x| x.cos()),
            f1!("_sin", |x| x.sin()),
            f1!("_tan", |x| x.tan()),
            f1!("_acos", |x| x.acos()),
            f1!("_asin", |x| x.asin()),
            f1!("_atan", |x| x.atan()),
            f1!("_atanh", |x| x.atanh()),
            f1!("_cosh", |x| x.cosh()),
            f1!("_sinh", |x| x.sinh()),
            f1!("_tanh", |x| x.tanh()),
            f2!("_atan2", |y, x| y.atan2(x)),
            f2!("_hypot", |x, y| x.hypot(y)),
        ]
    }

    /// Compile + execute an embedded Python source as a child module of
    /// `numpy`. `injections` lets the host pre-bind names (e.g. the ndarray
    /// class) into the module dict before the source runs.
    fn build_py_submodule(
        vm: &VirtualMachine,
        name: &str,
        source: &str,
        injections: &[(&str, PyObjectRef)],
    ) -> PyResult<PyObjectRef> {
        let dict = vm.ctx.new_dict();
        for (k, v) in injections {
            dict.set_item(*k, v.clone(), vm)?;
        }
        let code = vm
            .compile(
                source,
                rustpython_vm::compiler::Mode::Exec,
                format!("{}.py", name.replace('.', "/")),
            )
            .map_err(|e| vm.new_syntax_error(&e, Some(source)))?;
        let module = vm.new_module(name, dict.clone(), None);
        let scope = rustpython_vm::scope::Scope::with_builtins(None, dict, vm);
        vm.run_code_obj(code, scope)?;
        Ok(module.into())
    }

    /// Build a stand-in module when one of the Python-shim submodules fails
    /// to load at init time. This previously `panic!()`'d, which kills the
    /// embedding host; instead we return an empty module whose docstring
    /// records the build error, so the embedder stays alive and any user
    /// inspecting the failed submodule can see why it's empty.
    fn typing_panic(
        vm: &VirtualMachine,
        what: &str,
        err: &rustpython_vm::PyRef<rustpython_vm::builtins::PyBaseException>,
    ) -> PyObjectRef {
        let mut msg = format!("rumpy: failed to build {what} — ");
        let _ = vm.write_exception(&mut msg, err);
        let dict = vm.ctx.new_dict();
        let _ = dict.set_item("__build_error__", vm.ctx.new_str(msg.clone()).into(), vm);
        let _ = dict.set_item("__doc__", vm.ctx.new_str(msg).into(), vm);
        vm.new_module(what, dict, None).into()
    }

    // -----------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------

    fn parse_shape_from_args(args: &FuncArgs, vm: &VirtualMachine) -> PyResult<Vec<i64>> {
        if args.args.len() == 1 {
            let first = &args.args[0];
            if first
                .downcast_ref::<rustpython_vm::builtins::PyList>()
                .is_some()
                || first.downcast_ref::<PyTuple>().is_some()
            {
                return parse_shape_signed(first, vm);
            }
        }
        let mut out = Vec::with_capacity(args.args.len());
        for a in &args.args {
            out.push(a.try_int(vm)?.try_to_primitive::<i64>(vm)?);
        }
        Ok(out)
    }
}
