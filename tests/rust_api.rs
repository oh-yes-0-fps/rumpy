//! Rust-side ergonomic API tests — no Python involved. Exercises
//! `ArrayElement`, the typed accessors on `ArraysD`, and the `AsRef`/
//! `AsMut`/`Deref`/`From`/`TryFrom` impls on `PyNdArray`.

use ndarray::{ArrayD, IxDyn};
use rumpy::{ArrayElement, ArraysD, DType, PyNdArray};

// --- ArrayElement on each numpy element type -----------------------------

#[test]
fn array_element_dtype_tags() {
    assert_eq!(<bool as ArrayElement>::DTYPE, DType::Bool);
    assert_eq!(<i8 as ArrayElement>::DTYPE, DType::I8);
    assert_eq!(<i16 as ArrayElement>::DTYPE, DType::I16);
    assert_eq!(<i32 as ArrayElement>::DTYPE, DType::I32);
    assert_eq!(<i64 as ArrayElement>::DTYPE, DType::I64);
    assert_eq!(<u8 as ArrayElement>::DTYPE, DType::U8);
    assert_eq!(<u16 as ArrayElement>::DTYPE, DType::U16);
    assert_eq!(<u32 as ArrayElement>::DTYPE, DType::U32);
    assert_eq!(<u64 as ArrayElement>::DTYPE, DType::U64);
    assert_eq!(<half::f16 as ArrayElement>::DTYPE, DType::F16);
    assert_eq!(<f32 as ArrayElement>::DTYPE, DType::F32);
    assert_eq!(<f64 as ArrayElement>::DTYPE, DType::F64);
    assert_eq!(<num_complex::Complex<f32> as ArrayElement>::DTYPE, DType::C64);
    assert_eq!(<num_complex::Complex<f64> as ArrayElement>::DTYPE, DType::C128);
}

// --- as_array<T>() borrow ------------------------------------------------

#[test]
fn as_array_matches_variant() {
    let arrs: ArraysD = ArrayD::<f64>::zeros(IxDyn(&[2, 3])).into();
    assert_eq!(arrs.dtype(), DType::F64);

    let a: &ArrayD<f64> = arrs.as_array::<f64>().expect("variant matches");
    assert_eq!(a.shape(), &[2, 3]);

    // Wrong dtype turbofish → None.
    assert!(arrs.as_array::<i32>().is_none());
    assert!(arrs.as_array::<bool>().is_none());

    // Per-dtype shortcut accessors mirror as_array<T>().
    assert!(arrs.as_f64().is_some());
    assert!(arrs.as_i32().is_none());
}

#[test]
fn as_array_mut_lets_us_mutate() {
    let mut arrs: ArraysD =
        ArrayD::<i32>::from_shape_vec(IxDyn(&[3]), vec![1, 2, 3]).unwrap().into();

    let view: &mut ArrayD<i32> = arrs.as_i32_mut().expect("is i32");
    view[IxDyn(&[1])] = 42;

    let view: &ArrayD<i32> = arrs.as_i32().unwrap();
    assert_eq!(view[IxDyn(&[1])], 42);
}

// --- into_array consumes by value ---------------------------------------

#[test]
fn into_array_recovers_underlying() {
    let owned = ArrayD::<u8>::from_shape_vec(IxDyn(&[4]), vec![10, 20, 30, 40]).unwrap();
    let arrs: ArraysD = owned.clone().into();

    // Right type: get the ArrayD back.
    let recovered: ArrayD<u8> = arrs.into_array::<u8>().expect("u8 variant");
    assert_eq!(recovered, owned);

    // Wrong type: the ArraysD comes back unchanged.
    let arrs2: ArraysD = owned.into();
    let returned = arrs2.into_array::<f32>().unwrap_err();
    assert_eq!(returned.dtype(), DType::U8);
}

// --- From<ArrayD<T>> / TryFrom<ArraysD> ---------------------------------

#[test]
fn from_arrayd_for_arraysd() {
    let a: ArraysD = ArrayD::<f32>::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0])
        .unwrap()
        .into();
    assert_eq!(a.dtype(), DType::F32);
}

#[test]
fn try_from_arraysd_for_arrayd() {
    let a: ArraysD = ArrayD::<i64>::from_shape_vec(IxDyn(&[2, 2]), vec![1, 2, 3, 4])
        .unwrap()
        .into();

    let i: ArrayD<i64> = a.clone().try_into().expect("i64 variant");
    assert_eq!(i.shape(), &[2, 2]);
    assert_eq!(i[IxDyn(&[1, 1])], 4);

    // Mismatched: we get the ArraysD back as the error.
    let err: Result<ArrayD<f64>, ArraysD> = a.try_into();
    assert!(err.is_err());
    assert_eq!(err.unwrap_err().dtype(), DType::I64);
}

// --- PyNdArray ergonomics -----------------------------------------------

#[test]
fn pyndarray_from_arrayd_and_arraysd() {
    let inner: ArrayD<f64> = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
    let py: PyNdArray = inner.into(); // From<ArrayD<T>> for PyNdArray
    assert_eq!(py.dtype(), DType::F64);
    assert_eq!(py.shape(), &[3]);

    let a: ArraysD = ArrayD::<bool>::from_elem(IxDyn(&[2]), true).into();
    let py: PyNdArray = a.into(); // From<ArraysD> for PyNdArray
    assert_eq!(py.dtype(), DType::Bool);
}

#[test]
fn pyndarray_as_ref_and_deref() {
    let py: PyNdArray = ArrayD::<i32>::zeros(IxDyn(&[4])).into();

    // AsRef<ArraysD>
    let r: &ArraysD = py.as_ref();
    assert_eq!(r.dtype(), DType::I32);

    // Deref: methods on ArraysD reachable directly on PyNdArray.
    assert_eq!(py.dtype(), DType::I32);
    assert_eq!(py.shape(), &[4]);
    assert_eq!(py.ndim(), 1);
    assert_eq!(py.len(), 4);

    // Inner typed view via the shortcut accessors.
    let view: &ArrayD<i32> = py.as_i32().expect("i32");
    assert_eq!(view.len(), 4);
}

#[test]
fn pyndarray_as_mut_and_deref_mut() {
    let mut py: PyNdArray =
        ArrayD::<f64>::from_shape_vec(IxDyn(&[3]), vec![0.0, 0.0, 0.0])
            .unwrap()
            .into();

    // AsMut<ArraysD>
    {
        let r: &mut ArraysD = py.as_mut();
        let view = r.as_f64_mut().unwrap();
        view[IxDyn(&[1])] = 9.0;
    }
    // DerefMut: same thing, terser.
    {
        let view = py.as_f64_mut().unwrap();
        view[IxDyn(&[2])] = 7.0;
    }
    let view = py.as_f64().unwrap();
    assert_eq!(view[IxDyn(&[0])], 0.0);
    assert_eq!(view[IxDyn(&[1])], 9.0);
    assert_eq!(view[IxDyn(&[2])], 7.0);
}

// --- Generic functions over ArrayElement --------------------------------

fn sum_of_squares<T>(arrs: &ArraysD) -> Option<f64>
where
    T: ArrayElement + Into<f64> + Copy,
{
    let view = arrs.as_array::<T>()?;
    let mut acc = 0.0f64;
    for &v in view.iter() {
        let f: f64 = v.into();
        acc += f * f;
    }
    Some(acc)
}

#[test]
fn generic_over_array_element() {
    let a: ArraysD = ArrayD::<f32>::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0])
        .unwrap()
        .into();

    let s = sum_of_squares::<f32>(&a).unwrap();
    assert!((s - 14.0).abs() < 1e-6);

    // Wrong T -> None
    assert!(sum_of_squares::<f64>(&a).is_none());
}

// --- AsRef<PyNdArray> for PyNdArray (lets callers take generic input) ---

fn array_size<A: AsRef<PyNdArray>>(x: A) -> usize {
    x.as_ref().len()
}

#[test]
fn asref_pyndarray_generic_caller() {
    let py: PyNdArray = ArrayD::<u8>::zeros(IxDyn(&[5])).into();
    assert_eq!(array_size(&py), 5);
    assert_eq!(array_size(py), 5); // also works by value
}

// --- Generic coercion API ----------------------------------------------

#[test]
fn coerce_array_same_dtype() {
    use rumpy::CoerceArray;
    let arrs: ArraysD = ArrayD::<f64>::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0])
        .unwrap()
        .into();
    let out: ArrayD<f64> = arrs.coerce::<f64>();
    assert_eq!(out.shape(), &[3]);
    assert_eq!(out[IxDyn(&[0])], 1.0);
}

#[test]
fn coerce_array_casts_when_needed() {
    use rumpy::CoerceArray;
    let arrs: ArraysD = ArrayD::<i32>::from_shape_vec(IxDyn(&[3]), vec![1, 2, 3])
        .unwrap()
        .into();
    let out: ArrayD<f64> = arrs.coerce::<f64>();
    assert_eq!(out[IxDyn(&[1])], 2.0);
    // Original ArraysD is untouched.
    assert_eq!(arrs.dtype(), DType::I32);

    // try_borrow_as: succeeds when dtypes match, None otherwise.
    assert!(arrs.try_borrow_as::<i32>().is_some());
    assert!(arrs.try_borrow_as::<f64>().is_none());
}

#[test]
fn coerce_pyndarray() {
    use rumpy::CoerceArray;
    let py: PyNdArray = ArrayD::<u16>::from_shape_vec(IxDyn(&[2]), vec![10, 20])
        .unwrap()
        .into();
    let owned: ArrayD<f32> = py.coerce::<f32>();
    assert_eq!(owned[IxDyn(&[0])], 10.0);
    assert_eq!(owned[IxDyn(&[1])], 20.0);
}

#[test]
fn coerce_into_consumes() {
    use rumpy::CoerceArray;
    let arrs: ArraysD = ArrayD::<i64>::from_shape_vec(IxDyn(&[4]), vec![1, 2, 3, 4])
        .unwrap()
        .into();
    let consumed: ArrayD<i64> = arrs.into_coerced::<i64>();
    assert_eq!(consumed.shape(), &[4]);
}

// --- DType introspection ------------------------------------------------

#[test]
fn dtype_name_and_kind() {
    assert_eq!(DType::Bool.name(), "bool");
    assert_eq!(DType::I32.name(), "int32");
    assert_eq!(DType::U64.name(), "uint64");
    assert_eq!(DType::F16.name(), "float16");
    assert_eq!(DType::C128.name(), "complex128");

    assert_eq!(DType::Bool.kind(), 'b');
    assert_eq!(DType::I8.kind(), 'i');
    assert_eq!(DType::U16.kind(), 'u');
    assert_eq!(DType::F32.kind(), 'f');
    assert_eq!(DType::C64.kind(), 'c');
}

#[test]
fn dtype_itemsize_matches_layout() {
    assert_eq!(DType::Bool.itemsize(), 1);
    assert_eq!(DType::I8.itemsize(), 1);
    assert_eq!(DType::U16.itemsize(), 2);
    assert_eq!(DType::F16.itemsize(), 2);
    assert_eq!(DType::I32.itemsize(), 4);
    assert_eq!(DType::F32.itemsize(), 4);
    assert_eq!(DType::I64.itemsize(), 8);
    assert_eq!(DType::F64.itemsize(), 8);
    assert_eq!(DType::C64.itemsize(), 8);
    assert_eq!(DType::C128.itemsize(), 16);
}

#[test]
fn dtype_classifiers() {
    assert!(DType::Bool.is_integer());
    assert!(DType::I32.is_integer());
    assert!(DType::U32.is_integer());
    assert!(!DType::F32.is_integer());
    assert!(!DType::C64.is_integer());

    assert!(DType::F16.is_float());
    assert!(DType::F32.is_float());
    assert!(DType::F64.is_float());
    assert!(!DType::I32.is_float());
    assert!(!DType::C128.is_float());

    assert!(DType::C64.is_complex());
    assert!(DType::C128.is_complex());
    assert!(!DType::F64.is_complex());

    assert!(DType::I8.is_signed());
    assert!(DType::I64.is_signed());
    assert!(DType::F32.is_signed());
    assert!(DType::C128.is_signed());
    assert!(!DType::U8.is_signed());
    assert!(!DType::U64.is_signed());
    assert!(!DType::Bool.is_signed());
}

#[test]
fn dtype_parse_canonical_and_short_forms() {
    assert_eq!(DType::parse("bool"), Some(DType::Bool));
    assert_eq!(DType::parse("?"), Some(DType::Bool));
    assert_eq!(DType::parse("int32"), Some(DType::I32));
    assert_eq!(DType::parse("i4"), Some(DType::I32));
    assert_eq!(DType::parse("uint8"), Some(DType::U8));
    assert_eq!(DType::parse("u1"), Some(DType::U8));
    assert_eq!(DType::parse("float64"), Some(DType::F64));
    assert_eq!(DType::parse("f8"), Some(DType::F64));
    assert_eq!(DType::parse("float"), Some(DType::F64));
    assert_eq!(DType::parse("complex64"), Some(DType::C64));
    assert_eq!(DType::parse("c16"), Some(DType::C128));
    assert_eq!(DType::parse("complex"), Some(DType::C128));
    // Byte-order prefixes are stripped.
    assert_eq!(DType::parse("<f4"), Some(DType::F32));
    assert_eq!(DType::parse(">i8"), Some(DType::I64));
    assert_eq!(DType::parse("|u1"), Some(DType::U8));
}

#[test]
fn dtype_parse_rejects_unknown_and_legacy() {
    assert_eq!(DType::parse("nope"), None);
    assert_eq!(DType::parse(""), None);
    // 1.x aliases are intentionally not accepted.
    assert_eq!(DType::parse("float_"), None);
    assert_eq!(DType::parse("int_"), None);
    assert_eq!(DType::parse("complex_"), None);
}

#[test]
fn dtype_round_trips_through_name_and_parse() {
    for d in [
        DType::Bool, DType::I8, DType::I16, DType::I32, DType::I64,
        DType::U8, DType::U16, DType::U32, DType::U64,
        DType::F16, DType::F32, DType::F64,
        DType::C64, DType::C128,
    ] {
        assert_eq!(DType::parse(d.name()), Some(d), "round-trip for {d:?}");
    }
}

// --- ArraysD geometry / nbytes ------------------------------------------

#[test]
fn shape_and_ndim_and_raw_dim() {
    let arrs: ArraysD = ArrayD::<f32>::zeros(IxDyn(&[2, 3, 4])).into();
    assert_eq!(arrs.shape(), &[2, 3, 4]);
    assert_eq!(arrs.ndim(), 3);
    assert_eq!(arrs.len(), 24);
    assert!(!arrs.is_empty());
    assert_eq!(arrs.raw_dim(), IxDyn(&[2, 3, 4]));
}

#[test]
fn is_empty_for_zero_size() {
    let arrs: ArraysD = ArrayD::<i32>::zeros(IxDyn(&[0])).into();
    assert!(arrs.is_empty());
    assert_eq!(arrs.len(), 0);
    // Empty along one axis still has zero elements.
    let arrs: ArraysD = ArrayD::<i32>::zeros(IxDyn(&[3, 0, 5])).into();
    assert!(arrs.is_empty());
    assert_eq!(arrs.nbytes(), 0);
}

#[test]
fn nbytes_matches_len_times_itemsize() {
    let arrs: ArraysD = ArrayD::<f64>::zeros(IxDyn(&[7])).into();
    assert_eq!(arrs.nbytes(), 7 * 8);

    let arrs: ArraysD = ArrayD::<u16>::zeros(IxDyn(&[2, 5])).into();
    assert_eq!(arrs.nbytes(), 10 * 2);

    let arrs: ArraysD = ArrayD::<num_complex::Complex<f64>>::zeros(IxDyn(&[4])).into();
    assert_eq!(arrs.nbytes(), 4 * 16);
}

// --- ArraysD::cast / cast_cow -------------------------------------------

#[test]
fn cast_changes_dtype_and_values() {
    let arrs: ArraysD = ArrayD::<i32>::from_shape_vec(IxDyn(&[3]), vec![1, 2, 3])
        .unwrap()
        .into();
    let cast = arrs.cast(DType::F64);
    assert_eq!(cast.dtype(), DType::F64);
    let view = cast.as_f64().unwrap();
    assert_eq!(view[IxDyn(&[0])], 1.0);
    assert_eq!(view[IxDyn(&[2])], 3.0);
}

#[test]
fn cast_cow_borrows_when_same_dtype() {
    use std::borrow::Cow;
    let arrs: ArraysD = ArrayD::<f32>::from_shape_vec(IxDyn(&[2]), vec![1.5, 2.5])
        .unwrap()
        .into();
    match arrs.cast_cow(DType::F32) {
        Cow::Borrowed(b) => assert_eq!(b.dtype(), DType::F32),
        Cow::Owned(_) => panic!("expected borrow for same-dtype cast"),
    }
    match arrs.cast_cow(DType::F64) {
        Cow::Owned(o) => {
            assert_eq!(o.dtype(), DType::F64);
            assert_eq!(o.as_f64().unwrap()[IxDyn(&[1])], 2.5);
        }
        Cow::Borrowed(_) => panic!("expected owned for cross-dtype cast"),
    }
}

#[test]
fn cast_float_to_int_truncates() {
    let arrs: ArraysD = ArrayD::<f64>::from_shape_vec(IxDyn(&[4]), vec![1.9, -1.9, 2.5, -2.5])
        .unwrap()
        .into();
    let cast = arrs.cast(DType::I32);
    let v = cast.as_i32().unwrap();
    // numpy.astype on float->int truncates toward zero.
    assert_eq!(v[IxDyn(&[0])], 1);
    assert_eq!(v[IxDyn(&[1])], -1);
    assert_eq!(v[IxDyn(&[2])], 2);
    assert_eq!(v[IxDyn(&[3])], -2);
}

#[test]
fn cast_bool_to_int_and_back() {
    let arrs: ArraysD = ArrayD::<bool>::from_shape_vec(
        IxDyn(&[4]),
        vec![true, false, true, true],
    )
    .unwrap()
    .into();
    let as_i32 = arrs.cast(DType::I32);
    let v = as_i32.as_i32().unwrap();
    assert_eq!(v[IxDyn(&[0])], 1);
    assert_eq!(v[IxDyn(&[1])], 0);
    assert_eq!(v[IxDyn(&[2])], 1);

    let back = as_i32.cast(DType::Bool);
    let b = back.as_bool().unwrap();
    assert_eq!(b[IxDyn(&[0])], true);
    assert_eq!(b[IxDyn(&[1])], false);
}

// --- Less-covered dtype round trips -------------------------------------

#[test]
fn half_precision_round_trip() {
    use half::f16;
    let owned = ArrayD::<f16>::from_shape_vec(
        IxDyn(&[3]),
        vec![f16::from_f32(0.5), f16::from_f32(1.5), f16::from_f32(-2.0)],
    )
    .unwrap();
    let arrs: ArraysD = owned.clone().into();
    assert_eq!(arrs.dtype(), DType::F16);
    assert_eq!(arrs.nbytes(), 6);

    let view = arrs.as_f16().expect("f16 variant");
    assert_eq!(view[IxDyn(&[0])].to_f32(), 0.5);
    assert_eq!(view[IxDyn(&[2])].to_f32(), -2.0);

    let recovered: ArrayD<f16> = arrs.into_array::<f16>().unwrap();
    assert_eq!(recovered, owned);
}

#[test]
fn complex64_round_trip() {
    use num_complex::Complex;
    let owned = ArrayD::<Complex<f32>>::from_shape_vec(
        IxDyn(&[2]),
        vec![Complex::new(1.0, -2.0), Complex::new(3.0, 4.0)],
    )
    .unwrap();
    let arrs: ArraysD = owned.clone().into();
    assert_eq!(arrs.dtype(), DType::C64);

    let view = arrs.as_c64().expect("c64 variant");
    assert_eq!(view[IxDyn(&[0])], Complex::new(1.0, -2.0));
    assert_eq!(view[IxDyn(&[1])].im, 4.0);
}

#[test]
fn complex128_mutate_via_typed_view() {
    use num_complex::Complex;
    let mut arrs: ArraysD =
        ArrayD::<Complex<f64>>::zeros(IxDyn(&[2])).into();
    {
        let v = arrs.as_c128_mut().unwrap();
        v[IxDyn(&[0])] = Complex::new(0.0, 1.0);
        v[IxDyn(&[1])] = Complex::new(-1.0, 0.0);
    }
    let v = arrs.as_c128().unwrap();
    assert_eq!(v[IxDyn(&[0])].im, 1.0);
    assert_eq!(v[IxDyn(&[1])].re, -1.0);
}

// --- PyNdArray Deref to ArraysD methods --------------------------------

#[test]
fn pyndarray_deref_exposes_nbytes_and_raw_dim() {
    let py: PyNdArray = ArrayD::<u32>::zeros(IxDyn(&[3, 4])).into();
    assert_eq!(py.nbytes(), 12 * 4);
    assert_eq!(py.raw_dim(), IxDyn(&[3, 4]));
    assert!(!py.is_empty());
}

#[test]
fn pyndarray_cast_through_deref() {
    let py: PyNdArray = ArrayD::<i16>::from_shape_vec(IxDyn(&[3]), vec![10, 20, 30])
        .unwrap()
        .into();
    let cast = py.cast(DType::F32);
    assert_eq!(cast.dtype(), DType::F32);
    let v = cast.as_f32().unwrap();
    assert_eq!(v[IxDyn(&[1])], 20.0);
}

// --- CoerceArray to integer / boolean targets ---------------------------

#[test]
fn coerce_to_smaller_int_wraps() {
    use rumpy::CoerceArray;
    let arrs: ArraysD = ArrayD::<i32>::from_shape_vec(IxDyn(&[3]), vec![1, 256, -1])
        .unwrap()
        .into();
    let out: ArrayD<u8> = arrs.coerce::<u8>();
    // Cast semantics follow `as` (truncation/wraparound).
    assert_eq!(out[IxDyn(&[0])], 1);
    assert_eq!(out[IxDyn(&[1])], 0);
    assert_eq!(out[IxDyn(&[2])], 255);
}

#[test]
fn coerce_to_bool() {
    use rumpy::CoerceArray;
    let arrs: ArraysD = ArrayD::<i32>::from_shape_vec(IxDyn(&[3]), vec![0, 1, -3])
        .unwrap()
        .into();
    let out: ArrayD<bool> = arrs.coerce::<bool>();
    assert_eq!(out[IxDyn(&[0])], false);
    assert_eq!(out[IxDyn(&[1])], true);
    assert_eq!(out[IxDyn(&[2])], true);
}

// --- create:: constructors -------------------------------------------------

#[test]
fn create_zeros_each_dtype() {
    use rumpy::create;
    for d in [
        DType::Bool, DType::I8, DType::I16, DType::I32, DType::I64,
        DType::U8, DType::U16, DType::U32, DType::U64,
        DType::F16, DType::F32, DType::F64,
        DType::C64, DType::C128,
    ] {
        let a = create::zeros(&[3], d);
        assert_eq!(a.dtype(), d, "zeros dtype for {d:?}");
        assert_eq!(a.shape(), &[3]);
        assert_eq!(a.len(), 3);

        // Casting zeros -> f64 should yield all zeros.
        let f = a.cast(DType::F64);
        let v = f.as_f64().unwrap();
        for x in v.iter() {
            assert_eq!(*x, 0.0);
        }
    }
}

#[test]
fn create_ones_each_dtype() {
    use rumpy::create;
    for d in [
        DType::Bool, DType::I8, DType::I16, DType::I32, DType::I64,
        DType::U8, DType::U16, DType::U32, DType::U64,
        DType::F16, DType::F32, DType::F64,
        DType::C64, DType::C128,
    ] {
        let a = create::ones(&[4], d);
        assert_eq!(a.dtype(), d);
        let f = a.cast(DType::F64);
        let v = f.as_f64().unwrap();
        for x in v.iter() {
            assert_eq!(*x, 1.0);
        }
    }
}

#[test]
fn create_full_f64_cast_to_target() {
    use rumpy::create;
    let a = create::full_f64(&[2, 2], 3.7, DType::F64);
    assert_eq!(a.shape(), &[2, 2]);
    let v = a.as_f64().unwrap();
    for x in v.iter() {
        assert_eq!(*x, 3.7);
    }
    // Truncation to int when the target is integer.
    let a = create::full_f64(&[3], 3.7, DType::I32);
    let v = a.as_i32().unwrap();
    for x in v.iter() {
        assert_eq!(*x, 3);
    }
    // Non-zero floats coerce to true for bool.
    let a = create::full_f64(&[3], 0.0, DType::Bool);
    assert_eq!(a.as_bool().unwrap()[IxDyn(&[1])], false);
    let a = create::full_f64(&[3], 0.5, DType::Bool);
    assert_eq!(a.as_bool().unwrap()[IxDyn(&[1])], true);
}

#[test]
fn create_eye_diagonal_only() {
    use rumpy::create;
    let a = create::eye(3, 4, DType::F64);
    assert_eq!(a.shape(), &[3, 4]);
    let v = a.as_f64().unwrap();
    for i in 0..3 {
        for j in 0..4 {
            let want = if i == j { 1.0 } else { 0.0 };
            assert_eq!(v[IxDyn(&[i, j])], want, "eye[{i}, {j}]");
        }
    }
}

#[test]
fn create_arange_int_inference() {
    use rumpy::create;
    // All integer arguments → i64.
    let a = create::arange(0.0, 5.0, 1.0, None);
    assert_eq!(a.dtype(), DType::I64);
    assert_eq!(a.shape(), &[5]);
    let v = a.as_i64().unwrap();
    assert_eq!(v[IxDyn(&[0])], 0);
    assert_eq!(v[IxDyn(&[4])], 4);

    // Fractional step → stays f64.
    let a = create::arange(0.0, 1.0, 0.25, None);
    assert_eq!(a.dtype(), DType::F64);
    assert_eq!(a.len(), 4);

    // Explicit dtype override casts.
    let a = create::arange(0.0, 4.0, 1.0, Some(DType::F32));
    assert_eq!(a.dtype(), DType::F32);
}

#[test]
fn create_linspace_endpoints_and_edge_cases() {
    use rumpy::create;
    let a = create::linspace(0.0, 1.0, 5, None);
    assert_eq!(a.shape(), &[5]);
    let v = a.as_f64().unwrap();
    assert_eq!(v[IxDyn(&[0])], 0.0);
    assert_eq!(v[IxDyn(&[4])], 1.0);
    // Middle element should be exactly 0.5.
    assert!((v[IxDyn(&[2])] - 0.5).abs() < 1e-12);

    // num=0 → empty array.
    let a = create::linspace(0.0, 1.0, 0, None);
    assert_eq!(a.len(), 0);
    assert!(a.is_empty());

    // num=1 → single start.
    let a = create::linspace(2.5, 7.5, 1, None);
    assert_eq!(a.shape(), &[1]);
    assert_eq!(a.as_f64().unwrap()[IxDyn(&[0])], 2.5);
}

// --- Cast cross-product (sanity check every source/target pair) -----------

#[test]
fn cast_cross_product_preserves_shape() {
    let dtypes = [
        DType::Bool, DType::I8, DType::I16, DType::I32, DType::I64,
        DType::U8, DType::U16, DType::U32, DType::U64,
        DType::F16, DType::F32, DType::F64,
        DType::C64, DType::C128,
    ];
    for &src in &dtypes {
        let a = rumpy::create::ones(&[3, 2], src);
        for &tgt in &dtypes {
            let b = a.cast(tgt);
            assert_eq!(b.dtype(), tgt, "{src:?} → {tgt:?} dtype");
            assert_eq!(b.shape(), &[3, 2], "{src:?} → {tgt:?} shape");
            assert_eq!(b.len(), 6);
        }
    }
}

// --- PyNdArray view / view_mut --------------------------------------------

#[test]
fn pyndarray_view_reads_inner() {
    let py: PyNdArray = ArrayD::<f64>::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0])
        .unwrap()
        .into();
    let v = py.view();
    assert_eq!(v.shape(), &[3]);
    assert_eq!(v.dtype(), DType::F64);
    let inner = v.as_f64().expect("f64 variant");
    assert_eq!(inner[IxDyn(&[1])], 2.0);
}

#[test]
fn pyndarray_view_mut_writes_through() {
    let py: PyNdArray = ArrayD::<i32>::from_shape_vec(IxDyn(&[4]), vec![0, 0, 0, 0])
        .unwrap()
        .into();
    {
        let mut vm = py.view_mut();
        let inner = vm.as_i32_mut().expect("i32 variant");
        inner[IxDyn(&[2])] = 99;
    }
    let inner = py.view().as_i32().unwrap().clone();
    assert_eq!(inner[IxDyn(&[2])], 99);
    assert_eq!(inner[IxDyn(&[0])], 0);
}

// --- TryFrom round-trip for every dtype -----------------------------------

#[test]
fn try_from_for_each_dtype() {
    macro_rules! roundtrip {
        ($t:ty, $variant:path) => {{
            let owned = ArrayD::<$t>::zeros(IxDyn(&[2]));
            let arrs: ArraysD = owned.clone().into();
            assert!(matches!(arrs, $variant(_)));
            let back: ArrayD<$t> = arrs.try_into().expect("dtype match");
            assert_eq!(back.shape(), &[2]);
        }};
    }
    // bool doesn't implement Zero; use from_elem instead.
    let owned = ArrayD::<bool>::from_elem(IxDyn(&[2]), false);
    let arrs: ArraysD = owned.into();
    assert!(matches!(arrs, ArraysD::Bool(_)));
    let _: ArrayD<bool> = arrs.try_into().expect("bool");
    roundtrip!(i8,   ArraysD::I8);
    roundtrip!(i16,  ArraysD::I16);
    roundtrip!(i32,  ArraysD::I32);
    roundtrip!(i64,  ArraysD::I64);
    roundtrip!(u8,   ArraysD::U8);
    roundtrip!(u16,  ArraysD::U16);
    roundtrip!(u32,  ArraysD::U32);
    roundtrip!(u64,  ArraysD::U64);
    roundtrip!(f32,  ArraysD::F32);
    roundtrip!(f64,  ArraysD::F64);
    // f16 / complex zero-init isn't via ArrayD::zeros — exercise via from_elem.
    use half::f16;
    use num_complex::Complex;
    let a = ArrayD::<f16>::from_elem(IxDyn(&[2]), f16::ZERO);
    let arrs: ArraysD = a.into();
    assert!(matches!(arrs, ArraysD::F16(_)));
    let _: ArrayD<f16> = arrs.try_into().expect("f16");

    let a = ArrayD::<Complex<f32>>::from_elem(IxDyn(&[2]), Complex::new(0.0, 0.0));
    let arrs: ArraysD = a.into();
    assert!(matches!(arrs, ArraysD::C64(_)));

    let a = ArrayD::<Complex<f64>>::from_elem(IxDyn(&[2]), Complex::new(0.0, 0.0));
    let arrs: ArraysD = a.into();
    assert!(matches!(arrs, ArraysD::C128(_)));
}

// --- as_array on every dtype ----------------------------------------------

#[test]
fn as_array_returns_some_only_for_matching_variant() {
    // Build one ArraysD for each dtype and verify the matching accessor
    // returns Some and exactly one of the typed shortcuts is Some.
    let pairs: &[(DType, fn(&ArraysD) -> bool)] = &[
        (DType::Bool, |a| a.as_bool().is_some()),
        (DType::I8,   |a| a.as_i8().is_some()),
        (DType::I16,  |a| a.as_i16().is_some()),
        (DType::I32,  |a| a.as_i32().is_some()),
        (DType::I64,  |a| a.as_i64().is_some()),
        (DType::U8,   |a| a.as_u8().is_some()),
        (DType::U16,  |a| a.as_u16().is_some()),
        (DType::U32,  |a| a.as_u32().is_some()),
        (DType::U64,  |a| a.as_u64().is_some()),
        (DType::F16,  |a| a.as_f16().is_some()),
        (DType::F32,  |a| a.as_f32().is_some()),
        (DType::F64,  |a| a.as_f64().is_some()),
        (DType::C64,  |a| a.as_c64().is_some()),
        (DType::C128, |a| a.as_c128().is_some()),
    ];
    for (i, &(d, check)) in pairs.iter().enumerate() {
        let arr = rumpy::create::zeros(&[2], d);
        for (j, &(_, other)) in pairs.iter().enumerate() {
            let got = other(&arr);
            assert_eq!(got, i == j, "dtype {d:?} as_* accessor row {j} mismatch");
        }
        assert!(check(&arr));
    }
}

// --- ArraysD::Clone preserves dtype and data ------------------------------

#[test]
fn clone_preserves_contents_and_is_independent() {
    let mut a: ArraysD = ArrayD::<f64>::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0])
        .unwrap()
        .into();
    let b = a.clone();
    // Mutate the original.
    a.as_f64_mut().unwrap()[IxDyn(&[0])] = 99.0;
    // Clone is unchanged.
    assert_eq!(b.as_f64().unwrap()[IxDyn(&[0])], 1.0);
    assert_eq!(a.as_f64().unwrap()[IxDyn(&[0])], 99.0);
}

// --- DType equality / Copy ------------------------------------------------

#[test]
fn dtype_is_copy_and_eq() {
    let a = DType::F32;
    let b = a; // Copy
    assert_eq!(a, b);
    assert_eq!(a, DType::F32);
    assert_ne!(a, DType::F64);
}

// --- PyNdArray from_arrays constructor ------------------------------------

#[test]
fn pyndarray_from_arrays_constructor() {
    let inner: ArraysD = ArrayD::<u8>::from_shape_vec(IxDyn(&[3]), vec![10, 20, 30])
        .unwrap()
        .into();
    let py = PyNdArray::from_arrays(inner);
    assert_eq!(py.dtype(), DType::U8);
    assert_eq!(py.shape(), &[3]);
    let v = py.view();
    assert_eq!(v.as_u8().unwrap()[IxDyn(&[2])], 30);
}