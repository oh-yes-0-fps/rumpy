//! Array constructors: zeros, ones, full, eye, arange, linspace.

use crate::dtype::{ArraysD, C32, C64, DType, StructLayout};
use half::f16;
use ndarray::{ArrayD, IxDyn};
use num_complex::Complex;
use std::sync::Arc;

pub fn zeros(shape: &[usize], dtype: DType) -> ArraysD {
    let s = IxDyn(shape);
    let nelem: usize = shape.iter().product();
    match dtype {
        DType::Bool => ArraysD::Bool(ArrayD::from_elem(s, false)),
        DType::I8 => ArraysD::I8(ArrayD::zeros(s)),
        DType::I16 => ArraysD::I16(ArrayD::zeros(s)),
        DType::I32 => ArraysD::I32(ArrayD::zeros(s)),
        DType::I64 => ArraysD::I64(ArrayD::zeros(s)),
        DType::U8 => ArraysD::U8(ArrayD::zeros(s)),
        DType::U16 => ArraysD::U16(ArrayD::zeros(s)),
        DType::U32 => ArraysD::U32(ArrayD::zeros(s)),
        DType::U64 => ArraysD::U64(ArrayD::zeros(s)),
        DType::F16 => ArraysD::F16(ArrayD::from_elem(s, f16::ZERO)),
        DType::F32 => ArraysD::F32(ArrayD::zeros(s)),
        DType::F64 => ArraysD::F64(ArrayD::zeros(s)),
        DType::C64 => ArraysD::C64(ArrayD::from_elem(s, C32::new(0.0, 0.0))),
        DType::C128 => ArraysD::C128(ArrayD::from_elem(s, C64::new(0.0, 0.0))),
        // Object zero-init: numpy uses None (not int 0). We can't construct a
        // PyObjectRef without a vm — callers needing object zeros should use
        // `zeros_object(vm, shape)` in lib.rs. As a fallback we return an
        // empty 1D array so downstream code at least sees the right dtype.
        DType::Object => ArraysD::Object(crate::internal::empty_array()),
        DType::Str(n) => ArraysD::Str {
            itemsize_chars: n,
            data: ArrayD::from_elem(s, String::new()),
        },
        DType::Bytes(n) => ArraysD::Bytes {
            itemsize: n,
            data: ArrayD::from_elem(s, vec![0u8; n as usize]),
        },
        DType::Datetime64(u) => ArraysD::Datetime64 {
            unit: u,
            data: ArrayD::zeros(s),
        },
        DType::Timedelta64(u) => ArraysD::Timedelta64 {
            unit: u,
            data: ArrayD::zeros(s),
        },
        DType::Void(n) => {
            let _ = nelem;
            ArraysD::Void {
                layout: Arc::new(StructLayout::new(Vec::new(), n as usize)),
                data: ArrayD::from_elem(s, vec![0u8; n as usize]),
            }
        }
    }
}

pub fn ones(shape: &[usize], dtype: DType) -> ArraysD {
    let s = IxDyn(shape);
    match dtype {
        DType::Bool => ArraysD::Bool(ArrayD::from_elem(s, true)),
        DType::I8 => ArraysD::I8(ArrayD::from_elem(s, 1)),
        DType::I16 => ArraysD::I16(ArrayD::from_elem(s, 1)),
        DType::I32 => ArraysD::I32(ArrayD::from_elem(s, 1)),
        DType::I64 => ArraysD::I64(ArrayD::from_elem(s, 1)),
        DType::U8 => ArraysD::U8(ArrayD::from_elem(s, 1)),
        DType::U16 => ArraysD::U16(ArrayD::from_elem(s, 1)),
        DType::U32 => ArraysD::U32(ArrayD::from_elem(s, 1)),
        DType::U64 => ArraysD::U64(ArrayD::from_elem(s, 1)),
        DType::F16 => ArraysD::F16(ArrayD::from_elem(s, f16::from_f32(1.0))),
        DType::F32 => ArraysD::F32(ArrayD::from_elem(s, 1.0)),
        DType::F64 => ArraysD::F64(ArrayD::from_elem(s, 1.0)),
        DType::C64 => ArraysD::C64(ArrayD::from_elem(s, C32::new(1.0, 0.0))),
        DType::C128 => ArraysD::C128(ArrayD::from_elem(s, C64::new(1.0, 0.0))),
        // For non-numeric types "one" isn't well-defined; numpy returns
        // arrays of "1"-ish defaults (e.g. b'1' for bytes). We use the
        // matching numeric meaning where possible.
        DType::Datetime64(u) => ArraysD::Datetime64 {
            unit: u,
            data: ArrayD::from_elem(s, 1),
        },
        DType::Timedelta64(u) => ArraysD::Timedelta64 {
            unit: u,
            data: ArrayD::from_elem(s, 1),
        },
        DType::Str(n) => ArraysD::Str {
            itemsize_chars: n,
            data: ArrayD::from_elem(s, "1".to_string()),
        },
        DType::Bytes(n) => {
            let mut buf = vec![0u8; n as usize];
            if !buf.is_empty() {
                buf[0] = b'1';
            }
            ArraysD::Bytes {
                itemsize: n,
                data: ArrayD::from_elem(s, buf),
            }
        }
        DType::Object => ArraysD::Object(crate::internal::empty_array()),
        DType::Void(n) => ArraysD::Void {
            layout: Arc::new(StructLayout::new(Vec::new(), n as usize)),
            data: ArrayD::from_elem(s, vec![0u8; n as usize]),
        },
    }
}

pub fn full_f64(shape: &[usize], value: f64, dtype: DType) -> ArraysD {
    let s = IxDyn(shape);
    match dtype {
        DType::Bool => ArraysD::Bool(ArrayD::from_elem(s, value != 0.0)),
        DType::I8 => ArraysD::I8(ArrayD::from_elem(s, value as i8)),
        DType::I16 => ArraysD::I16(ArrayD::from_elem(s, value as i16)),
        DType::I32 => ArraysD::I32(ArrayD::from_elem(s, value as i32)),
        DType::I64 => ArraysD::I64(ArrayD::from_elem(s, value as i64)),
        DType::U8 => ArraysD::U8(ArrayD::from_elem(s, value as u8)),
        DType::U16 => ArraysD::U16(ArrayD::from_elem(s, value as u16)),
        DType::U32 => ArraysD::U32(ArrayD::from_elem(s, value as u32)),
        DType::U64 => ArraysD::U64(ArrayD::from_elem(s, value as u64)),
        DType::F16 => ArraysD::F16(ArrayD::from_elem(s, f16::from_f64(value))),
        DType::F32 => ArraysD::F32(ArrayD::from_elem(s, value as f32)),
        DType::F64 => ArraysD::F64(ArrayD::from_elem(s, value)),
        DType::C64 => ArraysD::C64(ArrayD::from_elem(s, C32::new(value as f32, 0.0))),
        DType::C128 => ArraysD::C128(ArrayD::from_elem(s, C64::new(value, 0.0))),
        DType::Datetime64(u) => ArraysD::Datetime64 {
            unit: u,
            data: ArrayD::from_elem(s, value as i64),
        },
        DType::Timedelta64(u) => ArraysD::Timedelta64 {
            unit: u,
            data: ArrayD::from_elem(s, value as i64),
        },
        DType::Str(n) => ArraysD::Str {
            itemsize_chars: n,
            data: ArrayD::from_elem(s, format!("{value}")),
        },
        DType::Bytes(n) => {
            let mut bytes = format!("{value}").into_bytes();
            bytes.resize(n as usize, 0);
            ArraysD::Bytes {
                itemsize: n,
                data: ArrayD::from_elem(s, bytes),
            }
        }
        DType::Object => ArraysD::Object(crate::internal::empty_array()),
        DType::Void(n) => ArraysD::Void {
            layout: Arc::new(StructLayout::new(Vec::new(), n as usize)),
            data: ArrayD::from_elem(s, vec![0u8; n as usize]),
        },
    }
}

pub fn eye(n: usize, m: usize, dtype: DType) -> ArraysD {
    let mut a = zeros(&[n, m], dtype);
    for i in 0..n.min(m) {
        set_one(&mut a, &[i, i]);
    }
    a
}

fn set_one(a: &mut ArraysD, idx: &[usize]) {
    let ix = IxDyn(idx);
    match a {
        ArraysD::Bool(x) => x[ix] = true,
        ArraysD::I8(x) => x[ix] = 1,
        ArraysD::I16(x) => x[ix] = 1,
        ArraysD::I32(x) => x[ix] = 1,
        ArraysD::I64(x) => x[ix] = 1,
        ArraysD::U8(x) => x[ix] = 1,
        ArraysD::U16(x) => x[ix] = 1,
        ArraysD::U32(x) => x[ix] = 1,
        ArraysD::U64(x) => x[ix] = 1,
        ArraysD::F16(x) => x[ix] = f16::from_f32(1.0),
        ArraysD::F32(x) => x[ix] = 1.0,
        ArraysD::F64(x) => x[ix] = 1.0,
        ArraysD::C64(x) => x[ix] = Complex::new(1.0, 0.0),
        ArraysD::C128(x) => x[ix] = Complex::new(1.0, 0.0),
        // eye() is only meaningful for numeric dtypes — non-numeric fall
        // through to a no-op (the caller would have rejected the dtype
        // earlier in real numpy).
        _ => {}
    }
}

/// arange(start, stop, step) — numpy promotes to f64 if any argument is float.
pub fn arange(start: f64, stop: f64, step: f64, dtype: Option<DType>) -> ArraysD {
    let n = ((stop - start) / step).ceil().max(0.0) as usize;
    let mut data = Vec::with_capacity(n);
    let mut cur = start;
    for _ in 0..n {
        data.push(cur);
        cur += step;
    }
    let arr = ArraysD::F64(ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap_or_default());
    match dtype {
        Some(dt) => arr.cast(dt),
        None => {
            // Heuristic: if all three inputs are integer-valued, infer i64.
            if start.fract() == 0.0 && stop.fract() == 0.0 && step.fract() == 0.0 {
                arr.cast(DType::I64)
            } else {
                arr
            }
        }
    }
}

pub fn linspace(start: f64, stop: f64, num: usize, dtype: Option<DType>) -> ArraysD {
    let arr = if num == 0 {
        ArrayD::from_shape_vec(IxDyn(&[0]), Vec::<f64>::new()).unwrap_or_default()
    } else if num == 1 {
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![start]).unwrap_or_default()
    } else {
        let step = (stop - start) / (num - 1) as f64;
        let data: Vec<f64> = (0..num).map(|i| start + step * i as f64).collect();
        ArrayD::from_shape_vec(IxDyn(&[num]), data).unwrap_or_default()
    };
    let a = ArraysD::F64(arr);
    match dtype {
        Some(dt) => a.cast(dt),
        None => a,
    }
}
