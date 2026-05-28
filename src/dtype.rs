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
use rustpython_vm::PyObjectRef;
use std::sync::Arc;

pub type C32 = Complex<f32>;
pub type C64 = Complex<f64>;

/// Datetime / timedelta unit code (numpy's [unit] suffix on `M8`/`m8`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeUnit {
    Y, M, W, D, H, Min, S, Ms, Us, Ns, Ps, Fs, As,
}

impl TimeUnit {
    /// Short code as used in numpy dtype strings: `Y`, `M`, `W`, `D`, `h`,
    /// `m`, `s`, `ms`, `us`, `ns`, `ps`, `fs`, `as`.
    #[inline]
    pub fn code(self) -> &'static str {
        match self {
            TimeUnit::Y => "Y",
            TimeUnit::M => "M",
            TimeUnit::W => "W",
            TimeUnit::D => "D",
            TimeUnit::H => "h",
            TimeUnit::Min => "m",
            TimeUnit::S => "s",
            TimeUnit::Ms => "ms",
            TimeUnit::Us => "us",
            TimeUnit::Ns => "ns",
            TimeUnit::Ps => "ps",
            TimeUnit::Fs => "fs",
            TimeUnit::As => "as",
        }
    }

    /// Parse a unit code; returns `None` for unrecognized inputs.
    #[inline]
    pub fn parse(s: &str) -> Option<TimeUnit> {
        Some(match s {
            "Y" => TimeUnit::Y,
            "M" => TimeUnit::M,
            "W" => TimeUnit::W,
            "D" => TimeUnit::D,
            "h" => TimeUnit::H,
            "m" => TimeUnit::Min,
            "s" => TimeUnit::S,
            "ms" => TimeUnit::Ms,
            "us" | "μs" => TimeUnit::Us,
            "ns" => TimeUnit::Ns,
            "ps" => TimeUnit::Ps,
            "fs" => TimeUnit::Fs,
            "as" => TimeUnit::As,
            _ => return None,
        })
    }
}

/// Field descriptor for structured / record dtypes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructField {
    pub name: String,
    pub dtype: DType,
    pub offset: usize,
}

/// Layout for `Void`/structured dtypes. Cheaply clone via `Arc`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructLayout {
    pub fields: Vec<StructField>,
    pub itemsize: usize,
}

impl StructLayout {
    pub fn new(fields: Vec<StructField>, itemsize: usize) -> Self {
        Self { fields, itemsize }
    }

    pub fn field(&self, name: &str) -> Option<&StructField> {
        self.fields.iter().find(|f| f.name == name)
    }
}

/// "View-style" dispatch — runs `$body` once per arm with `$a` bound to a
/// reference to the inner array. Useful for read-only inspection where the
/// element type does not appear in the result.
///
/// Covers all variants of [`ArraysD`], including non-numeric ones. The body
/// must be generic enough to apply to `ArrayD<bool>`, `ArrayD<i32>` …
/// `ArrayD<PyObjectRef>`, `ArrayD<String>`, `ArrayD<Vec<u8>>`, `ArrayD<i64>`.
/// In practice this works for the universal ndarray methods (`.shape()`,
/// `.ndim()`, `.len()`, `.raw_dim()`).
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
            $crate::dtype::ArraysD::Object($a) => $body,
            $crate::dtype::ArraysD::Str { data: $a, .. } => $body,
            $crate::dtype::ArraysD::Bytes { data: $a, .. } => $body,
            $crate::dtype::ArraysD::Datetime64 { data: $a, .. } => $body,
            $crate::dtype::ArraysD::Timedelta64 { data: $a, .. } => $body,
            $crate::dtype::ArraysD::Void { data: $a, .. } => $body,
        }
    };
}

/// Dispatch that ONLY covers the 14 numeric variants. Non-numeric variants
/// (`Object`, `Str`, …) take the wildcard arm `$other`, useful for ops that
/// only make sense on numeric data.
#[macro_export]
macro_rules! dispatch_numeric {
    ($arr:expr, |$a:ident| $body:expr, _ => $other:expr) => {
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
            _ => $other,
        }
    };
}

/// All numpy dtypes that `rumpy` understands.
///
/// Numeric variants (`Bool` … `C128`) are the "fast path" used by every
/// vectorized op. Non-numeric variants (`Object`, `Str`, `Bytes`,
/// `Datetime64`, `Timedelta64`, `Void`) carry their own metadata (itemsize
/// or unit) and are only meaningful for the corresponding storage path.
///
/// `DType: Copy`. The full struct layout for record/structured arrays lives
/// on the [`ArraysD::Void`] variant itself (as an `Arc<StructLayout>`); this
/// keeps `DType` cheap to pass around while still letting record arrays
/// carry their field schema.
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
    /// Object dtype — element is an arbitrary Python object reference.
    Object,
    /// Fixed-width unicode string. Inner is the number of code points
    /// (each stored as 4 bytes UCS-4 internally).
    Str(u32),
    /// Fixed-width bytes string. Inner is the number of bytes per element.
    Bytes(u32),
    /// `datetime64[unit]`.
    Datetime64(TimeUnit),
    /// `timedelta64[unit]`.
    Timedelta64(TimeUnit),
    /// Void itemsize in bytes. Structured layouts (field descriptors) live
    /// on the [`ArraysD::Void`] variant rather than in this dtype tag.
    Void(u32),
}

impl DType {
    /// Numpy's `dtype.name`. Returns a `&'static str` only for the numeric
    /// (always-the-same-name) variants; sized / parameterized variants like
    /// `Str(8)`, `Bytes(4)`, `Datetime64(s)` build their name dynamically —
    /// use [`name_owned`](DType::name_owned) for those.
    #[cfg_attr(feature = "no-panic", no_panic::no_panic)]
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
            DType::Object => "object",
            DType::Str(_) => "str",
            DType::Bytes(_) => "bytes",
            DType::Datetime64(_) => "datetime64",
            DType::Timedelta64(_) => "timedelta64",
            DType::Void(_) => "void",
        }
    }

    /// Numpy's `dtype.name` for any variant, including the parameterised
    /// non-numeric ones. e.g. `str32`, `bytes8`, `datetime64[ns]`.
    pub fn name_owned(self) -> String {
        match self {
            DType::Str(n) => format!("str{n}"),
            DType::Bytes(n) => format!("bytes{n}"),
            DType::Void(n) => format!("void{n}"),
            DType::Datetime64(u) => format!("datetime64[{}]", u.code()),
            DType::Timedelta64(u) => format!("timedelta64[{}]", u.code()),
            _ => self.name().to_string(),
        }
    }

    /// Numpy's dtype kind code: b/i/u/f/c/O/U/S/M/m/V.
    #[cfg_attr(feature = "no-panic", no_panic::no_panic)]
    #[inline]
    pub fn kind(self) -> char {
        match self {
            DType::Bool => 'b',
            DType::I8 | DType::I16 | DType::I32 | DType::I64 => 'i',
            DType::U8 | DType::U16 | DType::U32 | DType::U64 => 'u',
            DType::F16 | DType::F32 | DType::F64 => 'f',
            DType::C64 | DType::C128 => 'c',
            DType::Object => 'O',
            DType::Str(_) => 'U',
            DType::Bytes(_) => 'S',
            DType::Datetime64(_) => 'M',
            DType::Timedelta64(_) => 'm',
            DType::Void(_) => 'V',
        }
    }

    /// Bytes per element. For `Object` returns the size of a `PyObjectRef`
    /// (matches numpy's `np.dtype('O').itemsize` on 64-bit: 8).
    #[cfg_attr(feature = "no-panic", no_panic::no_panic)]
    #[inline]
    pub fn itemsize(self) -> usize {
        match self {
            DType::Bool => 1,
            DType::I8 | DType::U8 => 1,
            DType::I16 | DType::U16 | DType::F16 => 2,
            DType::I32 | DType::U32 | DType::F32 => 4,
            DType::I64 | DType::U64 | DType::F64 | DType::C64 => 8,
            DType::C128 => 16,
            DType::Object => std::mem::size_of::<usize>(),
            // Unicode is stored as UCS-4 internally — 4 bytes per code point.
            DType::Str(n) => (n as usize) * 4,
            DType::Bytes(n) => n as usize,
            DType::Datetime64(_) | DType::Timedelta64(_) => 8,
            DType::Void(n) => n as usize,
        }
    }

    /// Parse a numpy-style dtype string ("float64", "f8", "<f4", "int", "?", …).
    /// Parse a numpy 2.x dtype string.  Deprecated 1.x aliases like
    /// `float_`, `int_`, `complex_` are intentionally not accepted.
    ///
    /// Also handles non-numeric forms:
    /// - `"O"`, `"object"` → `Object`
    /// - `"U10"`, `"<U10"` → `Str(10)`
    /// - `"S5"`, `"|S5"` → `Bytes(5)`
    /// - `"V8"` → `Void(8)`
    /// - `"M8[ns]"`, `"datetime64[ns]"` → `Datetime64(Ns)`
    /// - `"m8[us]"`, `"timedelta64[us]"` → `Timedelta64(Us)`
    #[inline]
    pub fn parse(s: &str) -> Option<DType> {
        let bare = s.trim_start_matches(['<', '>', '=', '|']);
        // Parameterised non-numeric forms first.
        if let Some(rest) = bare.strip_prefix(['U']) {
            let n: u32 = rest.parse().ok()?;
            return Some(DType::Str(n));
        }
        if let Some(rest) = bare.strip_prefix(['S']) {
            // Avoid clobbering "Sxx" with single-letter form
            if rest.is_empty() {
                return Some(DType::Bytes(0));
            }
            if let Ok(n) = rest.parse::<u32>() {
                return Some(DType::Bytes(n));
            }
        }
        if let Some(rest) = bare.strip_prefix(['V']) {
            if rest.is_empty() {
                return Some(DType::Void(0));
            }
            if let Ok(n) = rest.parse::<u32>() {
                return Some(DType::Void(n));
            }
        }
        // datetime64[unit] / m8[unit]
        for (prefix, build) in [
            ("datetime64[", DType::Datetime64 as fn(TimeUnit) -> DType),
            ("M8[", DType::Datetime64),
            ("M[", DType::Datetime64),
            ("timedelta64[", DType::Timedelta64),
            ("m8[", DType::Timedelta64),
            ("m[", DType::Timedelta64),
        ] {
            if let Some(rest) = bare.strip_prefix(prefix) {
                let inner = rest.strip_suffix(']')?;
                let unit = TimeUnit::parse(inner)?;
                return Some(build(unit));
            }
        }
        if bare == "datetime64" || bare == "M8" || bare == "M" {
            return Some(DType::Datetime64(TimeUnit::Us));
        }
        if bare == "timedelta64" || bare == "m8" || bare == "m" {
            return Some(DType::Timedelta64(TimeUnit::Us));
        }
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
            "O" | "object" | "object_" => DType::Object,
            _ => return None,
        })
    }

    #[cfg_attr(feature = "no-panic", no_panic::no_panic)]
    #[inline]
    pub fn is_integer(self) -> bool {
        matches!(self.kind(), 'i' | 'u') || matches!(self, DType::Bool)
    }
    #[cfg_attr(feature = "no-panic", no_panic::no_panic)]
    #[inline]
    pub fn is_float(self) -> bool {
        matches!(self.kind(), 'f')
    }
    #[cfg_attr(feature = "no-panic", no_panic::no_panic)]
    #[inline]
    pub fn is_complex(self) -> bool {
        matches!(self.kind(), 'c')
    }
    #[cfg_attr(feature = "no-panic", no_panic::no_panic)]
    #[inline]
    pub fn is_signed(self) -> bool {
        matches!(self, DType::I8 | DType::I16 | DType::I32 | DType::I64)
            || self.is_float()
            || self.is_complex()
    }
    /// True for the 14 plain numeric/bool dtypes — the ones that flow through
    /// the vectorized op paths. Non-numeric dtypes (`Object`, `Str`, …) are
    /// handled by dedicated codepaths.
    #[inline]
    pub fn is_numeric(self) -> bool {
        matches!(
            self,
            DType::Bool
                | DType::I8
                | DType::I16
                | DType::I32
                | DType::I64
                | DType::U8
                | DType::U16
                | DType::U32
                | DType::U64
                | DType::F16
                | DType::F32
                | DType::F64
                | DType::C64
                | DType::C128
        )
    }
}

/// A dynamic-shape array tagged with its numpy dtype.
///
/// Numeric variants hold an `ArrayD<T>` of fixed-width copy element types.
/// Non-numeric variants carry their own metadata (`itemsize`, unit, layout):
///
/// - `Object`: `ArrayD<PyObjectRef>` — refcounted Python references per cell.
/// - `Str { itemsize_chars, data }`: ASCII / UCS-4 stored as a single
///   `String` per element (padded to `itemsize_chars` code points logically;
///   we just truncate-on-store and never read past).
/// - `Bytes { itemsize, data }`: `ArrayD<Vec<u8>>`, each Vec of length
///   `itemsize`.
/// - `Datetime64`/`Timedelta64`: `ArrayD<i64>` (the raw unit-count) plus the
///   resolution unit.
/// - `Void { layout, data }`: `ArrayD<Vec<u8>>` (each element of size
///   `layout.itemsize`), plus an `Arc<StructLayout>` describing fields.
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
    Object(ArrayD<PyObjectRef>),
    Str { itemsize_chars: u32, data: ArrayD<String> },
    Bytes { itemsize: u32, data: ArrayD<Vec<u8>> },
    Datetime64 { unit: TimeUnit, data: ArrayD<i64> },
    Timedelta64 { unit: TimeUnit, data: ArrayD<i64> },
    Void { layout: Arc<StructLayout>, data: ArrayD<Vec<u8>> },
}

impl ArraysD {
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
            ArraysD::Object(_) => DType::Object,
            ArraysD::Str { itemsize_chars, .. } => DType::Str(*itemsize_chars),
            ArraysD::Bytes { itemsize, .. } => DType::Bytes(*itemsize),
            ArraysD::Datetime64 { unit, .. } => DType::Datetime64(*unit),
            ArraysD::Timedelta64 { unit, .. } => DType::Timedelta64(*unit),
            ArraysD::Void { layout, .. } => DType::Void(layout.itemsize as u32),
        }
    }

    #[cfg_attr(feature = "no-panic", no_panic::no_panic)]
    #[inline]
    pub fn shape(&self) -> &[usize] {
        dispatch_view!(self, |a| a.shape())
    }
    #[cfg_attr(feature = "no-panic", no_panic::no_panic)]
    #[inline]
    pub fn ndim(&self) -> usize {
        dispatch_view!(self, |a| a.ndim())
    }
    #[cfg_attr(feature = "no-panic", no_panic::no_panic)]
    #[inline]
    pub fn len(&self) -> usize {
        dispatch_view!(self, |a| a.len())
    }
    #[cfg_attr(feature = "no-panic", no_panic::no_panic)]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    #[inline]
    pub fn raw_dim(&self) -> IxDyn {
        dispatch_view!(self, |a| a.raw_dim())
    }

    /// Number-of-bytes the data takes (excluding headers).
    #[cfg_attr(feature = "no-panic", no_panic::no_panic)]
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
            #[cfg_attr(feature = "no-panic", no_panic::no_panic)]
            #[inline]
            fn array_ref(a: &ArraysD) -> Option<&ArrayD<Self>> {
                match a {
                    ArraysD::$variant(x) => Some(x),
                    _ => None,
                }
            }
            #[cfg_attr(feature = "no-panic", no_panic::no_panic)]
            #[inline]
            fn array_mut(a: &mut ArraysD) -> Option<&mut ArrayD<Self>> {
                match a {
                    ArraysD::$variant(x) => Some(x),
                    _ => None,
                }
            }
            #[cfg_attr(feature = "no-panic", no_panic::no_panic)]
            #[inline]
            fn into_array(a: ArraysD) -> Result<ArrayD<Self>, ArraysD> {
                match a {
                    ArraysD::$variant(x) => Ok(x),
                    other => Err(other),
                }
            }
            #[cfg_attr(feature = "no-panic", no_panic::no_panic)]
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

    #[cfg_attr(feature = "no-panic", no_panic::no_panic)]
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
    // Numeric → numeric only goes through the per-target `mapv` arms.
    // Non-numeric variants either short-circuit (same dtype) or fall through
    // to a best-effort conversion (e.g. Str → Object).
    if !src.dtype().is_numeric() || !tgt.is_numeric() {
        return cast_nonnumeric(src, tgt);
    }
    match tgt {
        DType::Bool => ArraysD::Bool(dispatch_numeric!(src, |a| a.mapv(to_bool), _ => ArrayD::from_elem(src.raw_dim(), false))),
        DType::I8 => ArraysD::I8(dispatch_numeric!(src, |a| a.mapv(cast_to_i8), _ => ArrayD::from_elem(src.raw_dim(), 0i8))),
        DType::I16 => ArraysD::I16(dispatch_numeric!(src, |a| a.mapv(cast_to_i16), _ => ArrayD::from_elem(src.raw_dim(), 0i16))),
        DType::I32 => ArraysD::I32(dispatch_numeric!(src, |a| a.mapv(cast_to_i32), _ => ArrayD::from_elem(src.raw_dim(), 0i32))),
        DType::I64 => ArraysD::I64(dispatch_numeric!(src, |a| a.mapv(cast_to_i64), _ => ArrayD::from_elem(src.raw_dim(), 0i64))),
        DType::U8 => ArraysD::U8(dispatch_numeric!(src, |a| a.mapv(cast_to_u8), _ => ArrayD::from_elem(src.raw_dim(), 0u8))),
        DType::U16 => ArraysD::U16(dispatch_numeric!(src, |a| a.mapv(cast_to_u16), _ => ArrayD::from_elem(src.raw_dim(), 0u16))),
        DType::U32 => ArraysD::U32(dispatch_numeric!(src, |a| a.mapv(cast_to_u32), _ => ArrayD::from_elem(src.raw_dim(), 0u32))),
        DType::U64 => ArraysD::U64(dispatch_numeric!(src, |a| a.mapv(cast_to_u64), _ => ArrayD::from_elem(src.raw_dim(), 0u64))),
        DType::F16 => ArraysD::F16(dispatch_numeric!(src, |a| a.mapv(cast_to_f16), _ => ArrayD::from_elem(src.raw_dim(), f16::from_f32(0.0)))),
        DType::F32 => ArraysD::F32(dispatch_numeric!(src, |a| a.mapv(cast_to_f32), _ => ArrayD::from_elem(src.raw_dim(), 0f32))),
        DType::F64 => ArraysD::F64(dispatch_numeric!(src, |a| a.mapv(cast_to_f64), _ => ArrayD::from_elem(src.raw_dim(), 0f64))),
        DType::C64 => ArraysD::C64(dispatch_numeric!(src, |a| a.mapv(cast_to_c32), _ => ArrayD::from_elem(src.raw_dim(), C32::new(0.0, 0.0)))),
        DType::C128 => ArraysD::C128(dispatch_numeric!(src, |a| a.mapv(cast_to_c64), _ => ArrayD::from_elem(src.raw_dim(), C64::new(0.0, 0.0)))),
        // The is_numeric() short-circuit ensures we never reach here for
        // non-numeric targets, but the match must be exhaustive.
        DType::Object
        | DType::Str(_)
        | DType::Bytes(_)
        | DType::Datetime64(_)
        | DType::Timedelta64(_)
        | DType::Void(_) => cast_nonnumeric(src, tgt),
    }
}

/// Best-effort casts when either source or target is non-numeric. Numpy is
/// generally permissive here (`int → object` always works), and converting
/// strings / datetimes to numerics goes through a stringified form.
fn cast_nonnumeric(src: &ArraysD, tgt: DType) -> ArraysD {
    // Same dtype → just clone the storage.
    if src.dtype() == tgt {
        return src.clone();
    }
    match (src, tgt) {
        // Anything → Object: wrap each element as a Python int/float/etc.
        // We can't allocate PyObjectRefs without a `vm`, so build a placeholder
        // array of `None`s; the Python-side `astype('O')` path should call
        // `to_object_array` instead which takes a vm.
        (_, DType::Object) => {
            // Defer to a vm-aware caller. As a fallback return an empty
            // object array matching the source shape — better than panic.
            let shape = src.shape().to_vec();
            let elems = shape.iter().product::<usize>();
            // We need *some* PyObjectRef placeholder; we can't construct one
            // without a vm so we deliberately return an empty 1D array and
            // rely on the caller to use `to_object_array(vm)` instead.
            let _ = elems;
            ArraysD::Object(crate::internal::empty_array())
        }
        // Object → numeric: also requires a vm to interrogate each cell.
        // Return an empty array of the target dtype as a stand-in.
        (ArraysD::Object(_), _) => empty_for(tgt),
        // Datetime64 → Timedelta64 (and reverse) at same unit: zero-copy.
        (
            ArraysD::Datetime64 { unit, data },
            DType::Timedelta64(target_unit),
        ) if *unit == target_unit => ArraysD::Timedelta64 {
            unit: *unit,
            data: data.clone(),
        },
        (
            ArraysD::Timedelta64 { unit, data },
            DType::Datetime64(target_unit),
        ) if *unit == target_unit => ArraysD::Datetime64 {
            unit: *unit,
            data: data.clone(),
        },
        // Datetime/Timedelta → integer numeric: hand back the underlying i64
        // counter cast to the target.
        (ArraysD::Datetime64 { data, .. }, t) | (ArraysD::Timedelta64 { data, .. }, t) if t.is_numeric() => {
            ArraysD::I64(data.clone()).cast(t)
        }
        // Same-kind string width adjustment: preserve data, only the
        // metadata (declared width) changes. Truncation isn't applied here
        // — numpy stores the original strings and just reports the wider
        // dtype on the resulting array.
        (ArraysD::Str { data, .. }, DType::Str(n)) => ArraysD::Str {
            itemsize_chars: n,
            data: data.clone(),
        },
        (ArraysD::Bytes { data, .. }, DType::Bytes(n)) => ArraysD::Bytes {
            itemsize: n,
            data: data.clone(),
        },
        // Fallback: empty array of target dtype.
        _ => empty_for(tgt),
    }
}

fn empty_for(t: DType) -> ArraysD {
    let zero = IxDyn(&[0]);
    match t {
        DType::Bool => ArraysD::Bool(ArrayD::from_elem(zero, false)),
        DType::I8 => ArraysD::I8(ArrayD::from_elem(zero, 0)),
        DType::I16 => ArraysD::I16(ArrayD::from_elem(zero, 0)),
        DType::I32 => ArraysD::I32(ArrayD::from_elem(zero, 0)),
        DType::I64 => ArraysD::I64(ArrayD::from_elem(zero, 0)),
        DType::U8 => ArraysD::U8(ArrayD::from_elem(zero, 0)),
        DType::U16 => ArraysD::U16(ArrayD::from_elem(zero, 0)),
        DType::U32 => ArraysD::U32(ArrayD::from_elem(zero, 0)),
        DType::U64 => ArraysD::U64(ArrayD::from_elem(zero, 0)),
        DType::F16 => ArraysD::F16(ArrayD::from_elem(zero, f16::from_f32(0.0))),
        DType::F32 => ArraysD::F32(ArrayD::from_elem(zero, 0.0)),
        DType::F64 => ArraysD::F64(ArrayD::from_elem(zero, 0.0)),
        DType::C64 => ArraysD::C64(ArrayD::from_elem(zero, C32::new(0.0, 0.0))),
        DType::C128 => ArraysD::C128(ArrayD::from_elem(zero, C64::new(0.0, 0.0))),
        DType::Object => ArraysD::Object(crate::internal::empty_array()),
        DType::Str(n) => ArraysD::Str {
            itemsize_chars: n,
            data: crate::internal::empty_array(),
        },
        DType::Bytes(n) => ArraysD::Bytes {
            itemsize: n,
            data: crate::internal::empty_array(),
        },
        DType::Datetime64(u) => ArraysD::Datetime64 { unit: u, data: ArrayD::from_elem(zero, 0) },
        DType::Timedelta64(u) => ArraysD::Timedelta64 { unit: u, data: ArrayD::from_elem(zero, 0) },
        DType::Void(n) => ArraysD::Void {
            layout: Arc::new(StructLayout::new(Vec::new(), n as usize)),
            data: crate::internal::empty_array(),
        },
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
