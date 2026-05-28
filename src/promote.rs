//! Numpy-compatible type promotion (`numpy.result_type`).
//!
//! Numpy's rules — in summary:
//!
//!   * Same dtype → that dtype.
//!   * Mixed signed/unsigned ints → smallest signed int big enough to hold
//!     the unsigned range (e.g. u32+i32 → i64, u64+iX → f64).
//!   * Int + float → float (≥ float32 for small-int operands, otherwise the
//!     larger of the two).
//!   * Any + complex → complex of matching float width.
//!   * Bool acts as an unsigned 1-bit value: bool+i8 → i8, bool+f32 → f32.

use crate::dtype::DType;

/// `numpy.result_type(a, b)`.
#[cfg_attr(feature = "no-panic", no_panic::no_panic)]
#[inline]
pub fn promote(a: DType, b: DType) -> DType {
    if a == b {
        return a;
    }

    // String / bytes: same-kind widens to the wider operand. Mixing kinds
    // (or string with anything numeric) falls through to Object, matching
    // numpy's "common dtype" behaviour for incompatible flavours.
    match (a, b) {
        (DType::Str(x), DType::Str(y)) => return DType::Str(x.max(y)),
        (DType::Bytes(x), DType::Bytes(y)) => return DType::Bytes(x.max(y)),
        _ => {}
    }
    if matches!(a, DType::Str(_) | DType::Bytes(_) | DType::Object)
        || matches!(b, DType::Str(_) | DType::Bytes(_) | DType::Object)
    {
        return DType::Object;
    }

    // Complex: pick the float width that covers both real parts, then a
    // complex of that width.
    if a.is_complex() || b.is_complex() {
        let fw = float_width(a).max(float_width(b)).max(complex_real_width(a)).max(complex_real_width(b));
        return match fw {
            2 | 4 => DType::C64,   // complex64 has f32 components
            _ => DType::C128,
        };
    }

    // At least one float in the mix.
    if a.is_float() || b.is_float() {
        let lhs_w = effective_float_width(a);
        let rhs_w = effective_float_width(b);
        let w = lhs_w.max(rhs_w);
        return match w {
            2 => DType::F16,
            4 => DType::F32,
            _ => DType::F64,
        };
    }

    // Both integer-ish (bool acts as u1).
    let (a_sign, a_bits) = int_class(a);
    let (b_sign, b_bits) = int_class(b);
    if a_sign == b_sign {
        let bits = a_bits.max(b_bits);
        return int_dtype(a_sign, bits);
    }
    // Different signedness: numpy picks signed dtype with at least one more
    // bit than the unsigned operand, capped at i64.
    let (signed_bits, unsigned_bits) = if a_sign {
        (a_bits, b_bits)
    } else {
        (b_bits, a_bits)
    };
    let needed = signed_bits.max(unsigned_bits + 1).min(64);
    if unsigned_bits >= 64 {
        // u64 + signed → f64 (numpy's behaviour)
        return DType::F64;
    }
    int_dtype(true, needed)
}

/// `numpy.result_type` over arbitrarily-many dtypes (left fold). Returns
/// `DType::F64` (numpy's default-empty result type) if the input is empty
/// rather than panicking.
#[inline]
pub fn promote_many(types: &[DType]) -> DType {
    types
        .iter()
        .copied()
        .reduce(promote)
        .unwrap_or(DType::F64)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns (signed, bits). Bool is treated as unsigned 1-bit so it loses to
/// any other integer. Non-integer dtypes return (false, 0) as a defensive
/// fallback — `promote` only routes integers here, so this arm is logically
/// dead, but we don't want a panic in case of a future bug.
#[cfg_attr(feature = "no-panic", no_panic::no_panic)]
#[inline]
fn int_class(d: DType) -> (bool, u32) {
    match d {
        DType::Bool => (false, 1),
        DType::I8 => (true, 8),
        DType::I16 => (true, 16),
        DType::I32 => (true, 32),
        DType::I64 => (true, 64),
        DType::U8 => (false, 8),
        DType::U16 => (false, 16),
        DType::U32 => (false, 32),
        DType::U64 => (false, 64),
        _ => (false, 0),
    }
}

#[cfg_attr(feature = "no-panic", no_panic::no_panic)]
#[inline]
fn int_dtype(signed: bool, bits: u32) -> DType {
    let bits = bits.max(8);
    if signed {
        match bits {
            b if b <= 8 => DType::I8,
            b if b <= 16 => DType::I16,
            b if b <= 32 => DType::I32,
            _ => DType::I64,
        }
    } else {
        match bits {
            b if b <= 8 => DType::U8,
            b if b <= 16 => DType::U16,
            b if b <= 32 => DType::U32,
            _ => DType::U64,
        }
    }
}

#[cfg_attr(feature = "no-panic", no_panic::no_panic)]
#[inline]
fn float_width(d: DType) -> u32 {
    match d {
        DType::F16 => 2,
        DType::F32 => 4,
        DType::F64 => 8,
        _ => 0,
    }
}

#[cfg_attr(feature = "no-panic", no_panic::no_panic)]
#[inline]
fn complex_real_width(d: DType) -> u32 {
    match d {
        DType::C64 => 4,
        DType::C128 => 8,
        _ => 0,
    }
}

/// Numpy's rule for "what float width do I need to absorb this int":
/// small integers are absorbed into the existing float without growing it,
/// large integers force at least f64. This matches `result_type`'s output:
///
///   bool / u8 / i8        → 2  (covered by any float ≥ f16)
///   u16 / i16             → 2
///   u32 / i32             → 8  (forces ≥ f64 when mixed with f16/f32)
///   u64 / i64             → 8
#[cfg_attr(feature = "no-panic", no_panic::no_panic)]
#[inline]
fn effective_float_width(d: DType) -> u32 {
    match d {
        DType::Bool | DType::U8 | DType::I8 | DType::U16 | DType::I16 => 2,
        DType::U32 | DType::I32 | DType::U64 | DType::I64 => 8,
        DType::F16 => 2,
        DType::F32 => 4,
        DType::F64 => 8,
        _ => 8,
    }
}
