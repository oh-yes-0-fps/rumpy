//! Dtype machinery: the `DType` enum, the `ArraysD` storage enum that holds
//! one `ndarray::ArrayD<T>` per numpy element type, and the dispatch macros
//! and helpers used by every operation.
//!
//! Numpy supports a fixed set of numeric dtypes. We mirror those exactly:
//!
//! ```text
//!   bool                bool
//!   int8 / uint8        i8  / u8
//!   int16 / uint16      i16 / u16
//!   int32 / uint32      i32 / u32
//!   int64 / uint64      i64 / u64
//!   float16             half::f16
//!   float32 / float64   f32 / f64
//!   complex64           num_complex::Complex<f32>
//!   complex128          num_complex::Complex<f64>
//! ```
//!
//! Operations dispatch through the `dispatch_*` macros below. Type promotion
//! lives in `promote.rs`.

use half::f16;
use ndarray::{ArrayD, IxDyn};
use num_complex::Complex;

pub type C32 = Complex<f32>;
pub type C64 = Complex<f64>;

/// "View-style" dispatch — runs `$body` once per arm with `$a` bound to a
/// reference to the inner array. Useful for read-only inspection where the
/// element type does not appear in the result.
#[macro_export]
macro_rules! dispatch_view {
    ($arr:expr, |$a:ident| $body:expr) => {
        match $arr {
            $crate::dtype::ArraysD::Bool($a) => $body,
            $crate::dtype::ArraysD::I8($a) => $body,
            $crate::dtype::ArraysD::I16($a) => $body,
            $crate::dtype::ArraysD::I32($a) => $body,
            $crate::dtype::ArraysD::I64($a) => $body,
            $crate::dtype::ArraysD::U8($a) => $body,
            $crate::dtype::ArraysD::U16($a) => $body,
            $crate::dtype::ArraysD::U32($a) => $body,
            $crate::dtype::ArraysD::U64($a) => $body,
            $crate::dtype::ArraysD::F16($a) => $body,
            $crate::dtype::ArraysD::F32($a) => $body,
            $crate::dtype::ArraysD::F64($a) => $body,
            $crate::dtype::ArraysD::C64($a) => $body,
            $crate::dtype::ArraysD::C128($a) => $body,
        }
    };
}

/// All numpy dtypes that `rumpy` understands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    F32,
    F64,
    C64,  // complex64  (2× f32)
    C128, // complex128 (2× f64)
}

impl DType {
    /// Numpy's `dtype.name`.
    #[no_panic::no_panic]
    #[inline]
    pub fn name(self) -> &'static str {
        match self {
            DType::Bool => "bool",
            DType::I8 => "int8",
            DType::I16 => "int16",
            DType::I32 => "int32",
            DType::I64 => "int64",
            DType::U8 => "uint8",
            DType::U16 => "uint16",
            DType::U32 => "uint32",
            DType::U64 => "uint64",
            DType::F16 => "float16",
            DType::F32 => "float32",
            DType::F64 => "float64",
            DType::C64 => "complex64",
            DType::C128 => "complex128",
        }
    }

    /// Numpy's dtype kind code: b/i/u/f/c.
    #[no_panic::no_panic]
    #[inline]
    pub fn kind(self) -> char {
        match self {
            DType::Bool => 'b',
            DType::I8 | DType::I16 | DType::I32 | DType::I64 => 'i',
            DType::U8 | DType::U16 | DType::U32 | DType::U64 => 'u',
            DType::F16 | DType::F32 | DType::F64 => 'f',
            DType::C64 | DType::C128 => 'c',
        }
    }

    /// Bytes per element.
    #[no_panic::no_panic]
    #[inline]
    pub fn itemsize(self) -> usize {
        match self {
            DType::Bool => 1,
            DType::I8 | DType::U8 => 1,
            DType::I16 | DType::U16 | DType::F16 => 2,
            DType::I32 | DType::U32 | DType::F32 => 4,
            DType::I64 | DType::U64 | DType::F64 | DType::C64 => 8,
            DType::C128 => 16,
        }
    }

    /// Parse a numpy-style dtype string ("float64", "f8", "<f4", "int", "?", …).
    /// Parse a numpy 2.x dtype string.  Deprecated 1.x aliases like
    /// `float_`, `int_`, `complex_` are intentionally not accepted.
    #[no_panic::no_panic]
    #[inline]
    pub fn parse(s: &str) -> Option<DType> {
        let bare = s.trim_start_matches(['<', '>', '=', '|']);
        Some(match bare {
            "bool" | "b1" | "?" => DType::Bool,
            "int8" | "i1" | "byte" => DType::I8,
            "int16" | "i2" | "short" => DType::I16,
            "int32" | "i4" | "intc" => DType::I32,
            "int64" | "i8" | "int" | "long" | "intp" => DType::I64,
            "uint8" | "u1" | "ubyte" => DType::U8,
            "uint16" | "u2" | "ushort" => DType::U16,
            "uint32" | "u4" | "uintc" => DType::U32,
            "uint64" | "u8" | "uintp" | "uint" | "ulong" => DType::U64,
            "float16" | "f2" | "half" => DType::F16,
            "float32" | "f4" | "single" => DType::F32,
            "float64" | "f8" | "float" | "double" => DType::F64,
            "complex64" | "c8" | "csingle" => DType::C64,
            "complex128" | "c16" | "complex" | "cdouble" => DType::C128,
            _ => return None,
        })
    }

    #[no_panic::no_panic]
    #[inline]
    pub fn is_integer(self) -> bool {
        matches!(self.kind(), 'i' | 'u') || matches!(self, DType::Bool)
    }
    #[no_panic::no_panic]
    #[inline]
    pub fn is_float(self) -> bool {
        matches!(self.kind(), 'f')
    }
    #[no_panic::no_panic]
    #[inline]
    pub fn is_complex(self) -> bool {
        matches!(self.kind(), 'c')
    }
    #[no_panic::no_panic]
    #[inline]
    pub fn is_signed(self) -> bool {
        matches!(self, DType::I8 | DType::I16 | DType::I32 | DType::I64)
            || self.is_float()
            || self.is_complex()
    }
}

/// A dynamic-shape array tagged with its numpy dtype.
#[derive(Debug, Clone)]
pub enum ArraysD {
    Bool(ArrayD<bool>),
    I8(ArrayD<i8>),
    I16(ArrayD<i16>),
    I32(ArrayD<i32>),
    I64(ArrayD<i64>),
    U8(ArrayD<u8>),
    U16(ArrayD<u16>),
    U32(ArrayD<u32>),
    U64(ArrayD<u64>),
    F16(ArrayD<f16>),
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
    C64(ArrayD<C32>),
    C128(ArrayD<C64>),
}

impl ArraysD {
    #[no_panic::no_panic]
    #[inline]
    pub fn dtype(&self) -> DType {
        match self {
            ArraysD::Bool(_) => DType::Bool,
            ArraysD::I8(_) => DType::I8,
            ArraysD::I16(_) => DType::I16,
            ArraysD::I32(_) => DType::I32,
            ArraysD::I64(_) => DType::I64,
            ArraysD::U8(_) => DType::U8,
            ArraysD::U16(_) => DType::U16,
            ArraysD::U32(_) => DType::U32,
            ArraysD::U64(_) => DType::U64,
            ArraysD::F16(_) => DType::F16,
            ArraysD::F32(_) => DType::F32,
            ArraysD::F64(_) => DType::F64,
            ArraysD::C64(_) => DType::C64,
            ArraysD::C128(_) => DType::C128,
        }
    }

    #[no_panic::no_panic]
    #[inline]
    pub fn shape(&self) -> &[usize] {
        dispatch_view!(self, |a| a.shape())
    }
    #[no_panic::no_panic]
    #[inline]
    pub fn ndim(&self) -> usize {
        dispatch_view!(self, |a| a.ndim())
    }
    #[no_panic::no_panic]
    #[inline]
    pub fn len(&self) -> usize {
        dispatch_view!(self, |a| a.len())
    }
    #[no_panic::no_panic]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    #[inline]
    pub fn raw_dim(&self) -> IxDyn {
        dispatch_view!(self, |a| a.raw_dim())
    }

    /// Number-of-bytes the data takes (excluding headers).
    #[no_panic::no_panic]
    #[inline]
    pub fn nbytes(&self) -> usize {
        self.len().saturating_mul(self.dtype().itemsize())
    }

    /// Return a new array of the requested dtype, with values cast.
    /// This is `ndarray.astype(new_dtype)`.
    pub fn cast(&self, target: DType) -> ArraysD {
        if self.dtype() == target {
            return self.clone();
        }
        // Two-step: source → f64/C128 staging → target. Slower but small.
        // For exactness we instead expand explicitly for each (src, tgt) pair
        // — the macros below take care of the source side, the target side
        // is matched here.
        cast_impl(self, target)
    }

    /// Like [`cast`] but borrows the original storage when no conversion is
    /// needed. Use this from the hot binary-op paths so we don't clone the
    /// underlying buffer just to discover the dtypes already match.
    pub fn cast_cow(&self, target: DType) -> std::borrow::Cow<'_, ArraysD> {
        if self.dtype() == target {
            std::borrow::Cow::Borrowed(self)
        } else {
            std::borrow::Cow::Owned(cast_impl(self, target))
        }
    }

    // ----- Typed accessors (ergonomic Rust-side API) -----

    /// Return `&ArrayD<T>` if the stored variant matches `T`, otherwise
    /// `None`. The `ArrayElement` trait below is impl'd for every numpy
    /// element type, so this works generically:
    ///
    /// ```ignore
    /// if let Some(a) = arrs.as_array::<f64>() { /* use &ArrayD<f64> */ }
    /// ```
    #[inline]
    pub fn as_array<T: ArrayElement>(&self) -> Option<&ArrayD<T>> {
        T::array_ref(self)
    }

    /// Mutable variant of `as_array`.
    #[inline]
    pub fn as_array_mut<T: ArrayElement>(&mut self) -> Option<&mut ArrayD<T>> {
        T::array_mut(self)
    }

    /// Consume `self` and return the inner `ArrayD<T>` if the variant
    /// matches, otherwise hand back the original `ArraysD` unchanged.
    #[inline]
    pub fn into_array<T: ArrayElement>(self) -> Result<ArrayD<T>, ArraysD> {
        T::into_array(self)
    }

    /// Per-dtype shortcut accessors — equivalent to `as_array::<T>()` but
    /// don't require a turbofish.
    #[inline] pub fn as_bool(&self)  -> Option<&ArrayD<bool>> { self.as_array() }
    #[inline] pub fn as_i8(&self)    -> Option<&ArrayD<i8>>   { self.as_array() }
    #[inline] pub fn as_i16(&self)   -> Option<&ArrayD<i16>>  { self.as_array() }
    #[inline] pub fn as_i32(&self)   -> Option<&ArrayD<i32>>  { self.as_array() }
    #[inline] pub fn as_i64(&self)   -> Option<&ArrayD<i64>>  { self.as_array() }
    #[inline] pub fn as_u8(&self)    -> Option<&ArrayD<u8>>   { self.as_array() }
    #[inline] pub fn as_u16(&self)   -> Option<&ArrayD<u16>>  { self.as_array() }
    #[inline] pub fn as_u32(&self)   -> Option<&ArrayD<u32>>  { self.as_array() }
    #[inline] pub fn as_u64(&self)   -> Option<&ArrayD<u64>>  { self.as_array() }
    #[inline] pub fn as_f16(&self)   -> Option<&ArrayD<f16>>  { self.as_array() }
    #[inline] pub fn as_f32(&self)   -> Option<&ArrayD<f32>>  { self.as_array() }
    #[inline] pub fn as_f64(&self)   -> Option<&ArrayD<f64>>  { self.as_array() }
    #[inline] pub fn as_c64(&self)   -> Option<&ArrayD<C32>>  { self.as_array() }
    #[inline] pub fn as_c128(&self)  -> Option<&ArrayD<C64>>  { self.as_array() }

    #[inline] pub fn as_bool_mut(&mut self) -> Option<&mut ArrayD<bool>> { self.as_array_mut() }
    #[inline] pub fn as_i8_mut(&mut self)   -> Option<&mut ArrayD<i8>>   { self.as_array_mut() }
    #[inline] pub fn as_i16_mut(&mut self)  -> Option<&mut ArrayD<i16>>  { self.as_array_mut() }
    #[inline] pub fn as_i32_mut(&mut self)  -> Option<&mut ArrayD<i32>>  { self.as_array_mut() }
    #[inline] pub fn as_i64_mut(&mut self)  -> Option<&mut ArrayD<i64>>  { self.as_array_mut() }
    #[inline] pub fn as_u8_mut(&mut self)   -> Option<&mut ArrayD<u8>>   { self.as_array_mut() }
    #[inline] pub fn as_u16_mut(&mut self)  -> Option<&mut ArrayD<u16>>  { self.as_array_mut() }
    #[inline] pub fn as_u32_mut(&mut self)  -> Option<&mut ArrayD<u32>>  { self.as_array_mut() }
    #[inline] pub fn as_u64_mut(&mut self)  -> Option<&mut ArrayD<u64>>  { self.as_array_mut() }
    #[inline] pub fn as_f16_mut(&mut self)  -> Option<&mut ArrayD<f16>>  { self.as_array_mut() }
    #[inline] pub fn as_f32_mut(&mut self)  -> Option<&mut ArrayD<f32>>  { self.as_array_mut() }
    #[inline] pub fn as_f64_mut(&mut self)  -> Option<&mut ArrayD<f64>>  { self.as_array_mut() }
    #[inline] pub fn as_c64_mut(&mut self)  -> Option<&mut ArrayD<C32>>  { self.as_array_mut() }
    #[inline] pub fn as_c128_mut(&mut self) -> Option<&mut ArrayD<C64>>  { self.as_array_mut() }
}

// =====================================================================
// ArrayElement — one impl per numpy element type
// =====================================================================

mod sealed {
    pub trait Sealed {}
}

/// A type that can be the element type of an [`ArraysD`] variant.
///
/// Sealed: only implemented for the 14 numpy element types
/// (`bool`, the 8 fixed-width integers, `f16`/`f32`/`f64`,
/// `Complex<f32>`/`Complex<f64>`).
pub trait ArrayElement: sealed::Sealed + Copy + Default + 'static {
    /// Numpy dtype tag for this element type.
    const DTYPE: DType;

    /// Borrow a typed array out of an `ArraysD`, or `None` if the variant
    /// doesn't match.
    fn array_ref(a: &ArraysD) -> Option<&ArrayD<Self>>
    where
        Self: Sized;

    /// Mutable variant.
    fn array_mut(a: &mut ArraysD) -> Option<&mut ArrayD<Self>>
    where
        Self: Sized;

    /// Consume `ArraysD` and recover the typed array, returning the
    /// original `ArraysD` on dtype mismatch.
    fn into_array(a: ArraysD) -> Result<ArrayD<Self>, ArraysD>
    where
        Self: Sized;

    /// Wrap a typed array back into an `ArraysD`.
    fn from_array(a: ArrayD<Self>) -> ArraysD
    where
        Self: Sized;
}

macro_rules! impl_array_element {
    ($t:ty, $variant:ident, $dt:expr) => {
        impl sealed::Sealed for $t {}
        impl ArrayElement for $t {
            const DTYPE: DType = $dt;
            #[no_panic::no_panic]
            #[inline]
            fn array_ref(a: &ArraysD) -> Option<&ArrayD<Self>> {
                match a {
                    ArraysD::$variant(x) => Some(x),
                    _ => None,
                }
            }
            #[no_panic::no_panic]
            #[inline]
            fn array_mut(a: &mut ArraysD) -> Option<&mut ArrayD<Self>> {
                match a {
                    ArraysD::$variant(x) => Some(x),
                    _ => None,
                }
            }
            #[no_panic::no_panic]
            #[inline]
            fn into_array(a: ArraysD) -> Result<ArrayD<Self>, ArraysD> {
                match a {
                    ArraysD::$variant(x) => Ok(x),
                    other => Err(other),
                }
            }
            #[no_panic::no_panic]
            #[inline]
            fn from_array(a: ArrayD<Self>) -> ArraysD {
                ArraysD::$variant(a)
            }
        }

        impl From<ArrayD<$t>> for ArraysD {
            #[inline]
            fn from(a: ArrayD<$t>) -> ArraysD {
                ArraysD::$variant(a)
            }
        }

        impl TryFrom<ArraysD> for ArrayD<$t> {
            type Error = ArraysD;
            #[inline]
            fn try_from(a: ArraysD) -> Result<Self, ArraysD> {
                <$t as ArrayElement>::into_array(a)
            }
        }
    };
}

impl_array_element!(bool, Bool, DType::Bool);
impl_array_element!(i8, I8, DType::I8);
impl_array_element!(i16, I16, DType::I16);
impl_array_element!(i32, I32, DType::I32);
impl_array_element!(i64, I64, DType::I64);
impl_array_element!(u8, U8, DType::U8);
impl_array_element!(u16, U16, DType::U16);
impl_array_element!(u32, U32, DType::U32);
impl_array_element!(u64, U64, DType::U64);
impl_array_element!(f16, F16, DType::F16);
impl_array_element!(f32, F32, DType::F32);
impl_array_element!(f64, F64, DType::F64);
impl_array_element!(C32, C64, DType::C64);
impl_array_element!(C64, C128, DType::C128);

// =====================================================================
// Generic-driven coercion API
// =====================================================================

/// Generic array coercion: ask for an `ArrayD<T>` and get one back,
/// performing a dtype cast if the underlying storage doesn't already match.
///
/// ```ignore
/// use rumpy::{ArraysD, CoerceArray};
/// use ndarray::{ArrayD, IxDyn};
///
/// let mixed: ArraysD = ArrayD::<i32>::from_shape_vec(IxDyn(&[3]), vec![1, 2, 3])
///     .unwrap()
///     .into();
/// let floats: ArrayD<f64> = mixed.coerce::<f64>();   // cast on demand
/// let same: ArrayD<i32> = mixed.coerce::<i32>();     // no-op cast (clone)
/// ```
///
/// The trait has methods for both borrowing and consuming variants:
///
///   * `coerce::<T>()` — always returns owned `ArrayD<T>`, casts when needed.
///   * `try_borrow_as::<T>()` — `Option<&ArrayD<T>>`, never casts.
///   * `into_coerced::<T>()` — consume `self` and return `ArrayD<T>`.
pub trait CoerceArray {
    /// Return an `ArrayD<T>` containing the same elements as `self`, casting
    /// to `T` if the underlying dtype differs.
    fn coerce<T: ArrayElement>(&self) -> ArrayD<T>;

    /// Borrow `&ArrayD<T>` only if the underlying dtype already matches.
    fn try_borrow_as<T: ArrayElement>(&self) -> Option<&ArrayD<T>>;

    /// Consume `self` and return `ArrayD<T>`, casting if necessary.
    fn into_coerced<T: ArrayElement>(self) -> ArrayD<T>
    where
        Self: Sized;
}

impl CoerceArray for ArraysD {
    #[inline]
    fn coerce<T: ArrayElement>(&self) -> ArrayD<T> {
        if let Some(a) = T::array_ref(self) {
            return a.clone();
        }
        // dtype differs: cast then extract. The extraction can only fail
        // if the cast somehow yielded a different variant, which is a bug.
        // Fall back to an empty array rather than panic.
        T::into_array(self.cast(T::DTYPE)).unwrap_or_else(|_| empty_array_d::<T>())
    }

    #[no_panic::no_panic]
    #[inline]
    fn try_borrow_as<T: ArrayElement>(&self) -> Option<&ArrayD<T>> {
        T::array_ref(self)
    }

    #[inline]
    fn into_coerced<T: ArrayElement>(self) -> ArrayD<T> {
        let cast = if self.dtype() == T::DTYPE { self } else { self.cast(T::DTYPE) };
        T::into_array(cast).unwrap_or_else(|_| empty_array_d::<T>())
    }
}

/// Empty `ArrayD<T>` used as a panic-free fallback.
#[inline]
fn empty_array_d<T: ArrayElement>() -> ArrayD<T> {
    ArrayD::<T>::default(IxDyn(&[0]))
}

// =====================================================================
// Casting (every source dtype → every target dtype).
// =====================================================================

fn cast_impl(src: &ArraysD, tgt: DType) -> ArraysD {
    match tgt {
        DType::Bool => ArraysD::Bool(dispatch_view!(src, |a| a.mapv(to_bool))),
        DType::I8 => ArraysD::I8(dispatch_view!(src, |a| a.mapv(cast_to_i8))),
        DType::I16 => ArraysD::I16(dispatch_view!(src, |a| a.mapv(cast_to_i16))),
        DType::I32 => ArraysD::I32(dispatch_view!(src, |a| a.mapv(cast_to_i32))),
        DType::I64 => ArraysD::I64(dispatch_view!(src, |a| a.mapv(cast_to_i64))),
        DType::U8 => ArraysD::U8(dispatch_view!(src, |a| a.mapv(cast_to_u8))),
        DType::U16 => ArraysD::U16(dispatch_view!(src, |a| a.mapv(cast_to_u16))),
        DType::U32 => ArraysD::U32(dispatch_view!(src, |a| a.mapv(cast_to_u32))),
        DType::U64 => ArraysD::U64(dispatch_view!(src, |a| a.mapv(cast_to_u64))),
        DType::F16 => ArraysD::F16(dispatch_view!(src, |a| a.mapv(cast_to_f16))),
        DType::F32 => ArraysD::F32(dispatch_view!(src, |a| a.mapv(cast_to_f32))),
        DType::F64 => ArraysD::F64(dispatch_view!(src, |a| a.mapv(cast_to_f64))),
        DType::C64 => ArraysD::C64(dispatch_view!(src, |a| a.mapv(cast_to_c32))),
        DType::C128 => ArraysD::C128(dispatch_view!(src, |a| a.mapv(cast_to_c64))),
    }
}

/// Trait of element types we can cast *from*. Each numpy element type
/// implements all the `cast_to_*` conversions below.
pub(crate) trait CastFrom: Copy {
    fn to_bool_(self) -> bool;
    fn to_i64_(self) -> i64;
    fn to_u64_(self) -> u64;
    fn to_f64_(self) -> f64;
    fn to_c64_(self) -> C64;
}

impl CastFrom for bool {
    fn to_bool_(self) -> bool {
        self
    }
    fn to_i64_(self) -> i64 {
        self as i64
    }
    fn to_u64_(self) -> u64 {
        self as u64
    }
    fn to_f64_(self) -> f64 {
        if self { 1.0 } else { 0.0 }
    }
    fn to_c64_(self) -> C64 {
        C64::new(self.to_f64_(), 0.0)
    }
}

macro_rules! int_cast_from {
    ($t:ty) => {
        impl CastFrom for $t {
            fn to_bool_(self) -> bool {
                self != 0
            }
            fn to_i64_(self) -> i64 {
                self as i64
            }
            fn to_u64_(self) -> u64 {
                self as u64
            }
            fn to_f64_(self) -> f64 {
                self as f64
            }
            fn to_c64_(self) -> C64 {
                C64::new(self as f64, 0.0)
            }
        }
    };
}
int_cast_from!(i8);
int_cast_from!(i16);
int_cast_from!(i32);
int_cast_from!(i64);
int_cast_from!(u8);
int_cast_from!(u16);
int_cast_from!(u32);
int_cast_from!(u64);

impl CastFrom for f32 {
    fn to_bool_(self) -> bool {
        self != 0.0 && !self.is_nan()
    }
    fn to_i64_(self) -> i64 {
        self as i64
    }
    fn to_u64_(self) -> u64 {
        self as u64
    }
    fn to_f64_(self) -> f64 {
        self as f64
    }
    fn to_c64_(self) -> C64 {
        C64::new(self as f64, 0.0)
    }
}
impl CastFrom for f64 {
    fn to_bool_(self) -> bool {
        self != 0.0 && !self.is_nan()
    }
    fn to_i64_(self) -> i64 {
        self as i64
    }
    fn to_u64_(self) -> u64 {
        self as u64
    }
    fn to_f64_(self) -> f64 {
        self
    }
    fn to_c64_(self) -> C64 {
        C64::new(self, 0.0)
    }
}
impl CastFrom for f16 {
    fn to_bool_(self) -> bool {
        f32::from(self) != 0.0 && !f32::from(self).is_nan()
    }
    fn to_i64_(self) -> i64 {
        f32::from(self) as i64
    }
    fn to_u64_(self) -> u64 {
        f32::from(self) as u64
    }
    fn to_f64_(self) -> f64 {
        f32::from(self) as f64
    }
    fn to_c64_(self) -> C64 {
        C64::new(self.to_f64_(), 0.0)
    }
}
impl CastFrom for C32 {
    fn to_bool_(self) -> bool {
        self.re != 0.0 || self.im != 0.0
    }
    fn to_i64_(self) -> i64 {
        self.re as i64
    }
    fn to_u64_(self) -> u64 {
        self.re as u64
    }
    fn to_f64_(self) -> f64 {
        self.re as f64
    }
    fn to_c64_(self) -> C64 {
        C64::new(self.re as f64, self.im as f64)
    }
}
impl CastFrom for C64 {
    fn to_bool_(self) -> bool {
        self.re != 0.0 || self.im != 0.0
    }
    fn to_i64_(self) -> i64 {
        self.re as i64
    }
    fn to_u64_(self) -> u64 {
        self.re as u64
    }
    fn to_f64_(self) -> f64 {
        self.re
    }
    fn to_c64_(self) -> C64 {
        self
    }
}

fn to_bool<T: CastFrom>(v: T) -> bool {
    v.to_bool_()
}
fn cast_to_i64<T: CastFrom>(v: T) -> i64 {
    v.to_i64_()
}
fn cast_to_i8<T: CastFrom>(v: T) -> i8 {
    v.to_i64_() as i8
}
fn cast_to_i16<T: CastFrom>(v: T) -> i16 {
    v.to_i64_() as i16
}
fn cast_to_i32<T: CastFrom>(v: T) -> i32 {
    v.to_i64_() as i32
}
fn cast_to_u64<T: CastFrom>(v: T) -> u64 {
    v.to_u64_()
}
fn cast_to_u8<T: CastFrom>(v: T) -> u8 {
    v.to_u64_() as u8
}
fn cast_to_u16<T: CastFrom>(v: T) -> u16 {
    v.to_u64_() as u16
}
fn cast_to_u32<T: CastFrom>(v: T) -> u32 {
    v.to_u64_() as u32
}
fn cast_to_f64<T: CastFrom>(v: T) -> f64 {
    v.to_f64_()
}
fn cast_to_f32<T: CastFrom>(v: T) -> f32 {
    v.to_f64_() as f32
}
fn cast_to_f16<T: CastFrom>(v: T) -> f16 {
    f16::from_f64(v.to_f64_())
}
fn cast_to_c64<T: CastFrom>(v: T) -> C64 {
    v.to_c64_()
}
fn cast_to_c32<T: CastFrom>(v: T) -> C32 {
    let c = v.to_c64_();
    C32::new(c.re as f32, c.im as f32)
}
