//! Elementwise binary arithmetic with broadcasting + unary ufuncs.
//!
//! Binary ops follow numpy's `result_type` promotion: both operands are cast
//! to the promoted dtype, then the op runs in that dtype.
//!
//! Float-only ufuncs (sqrt, log, sin, …) promote integer / bool inputs to
//! float64 first, matching numpy.

use crate::dtype::{ArraysD, C32, C64, DType};
use crate::internal::{OptionExt, ResultExt, internal};
use crate::promote::promote;
use half::f16;
use ndarray::{ArrayD, IxDyn, Zip};
use num_complex::Complex;
use num_traits::Zero;
use rustpython_vm::{PyResult, VirtualMachine};

// ---------------------------------------------------------------------------
// Broadcasting
// ---------------------------------------------------------------------------

fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let nd = a.len().max(b.len());
    let mut out = vec![1usize; nd];
    for i in 0..nd {
        let da = if i + a.len() >= nd {
            a[i + a.len() - nd]
        } else {
            1
        };
        let db = if i + b.len() >= nd {
            b[i + b.len() - nd]
        } else {
            1
        };
        out[i] = match (da, db) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => return None,
        };
    }
    Some(out)
}

/// Run a binary op with numpy promotion + broadcasting.
pub fn binary_op<F>(a: &ArraysD, b: &ArraysD, vm: &VirtualMachine, op: F) -> PyResult<ArraysD>
where
    F: BinaryOp,
{
    let out_dtype = promote(a.dtype(), b.dtype());
    let a = a.cast_cow(out_dtype);
    let b = b.cast_cow(out_dtype);
    let shape = match broadcast_shape(a.shape(), b.shape()) {
        Some(s) => s,
        None => {
            return Err(vm.new_value_error(format!(
                "operands could not be broadcast together with shapes {:?} {:?}",
                a.shape(),
                b.shape()
            )));
        }
    };
    op.apply(&a, &b, &shape, vm)
}

/// Trait so the per-dtype impl of each binary op lives in one place.
pub trait BinaryOp {
    fn apply(
        &self,
        a: &ArraysD,
        b: &ArraysD,
        shape: &[usize],
        vm: &VirtualMachine,
    ) -> PyResult<ArraysD>;
}

macro_rules! impl_arith_op {
    ($name:ident, $int_f:expr, $float_f:expr, $bool_f:expr) => {
        pub struct $name;
        impl BinaryOp for $name {
            fn apply(
                &self,
                a: &ArraysD,
                b: &ArraysD,
                shape: &[usize],
                vm: &VirtualMachine,
            ) -> PyResult<ArraysD> {
                let s = IxDyn(shape);
                Ok(match (a, b) {
                    (ArraysD::Bool(x), ArraysD::Bool(y)) => {
                        let xv = x
                            .broadcast(s.clone())
                            .or_internal(vm, "broadcast bool lhs")?;
                        let yv = y
                            .broadcast(s.clone())
                            .or_internal(vm, "broadcast bool rhs")?;
                        let mut out = ArrayD::<bool>::from_elem(s.clone(), false);
                        Zip::from(&mut out)
                            .and(&xv)
                            .and(&yv)
                            .for_each(|o, &p, &q| *o = $bool_f(p, q));
                        ArraysD::Bool(out)
                    }
                    (ArraysD::I8(x), ArraysD::I8(y)) => ArraysD::I8(elem(x, y, &s, $int_f, vm)?),
                    (ArraysD::I16(x), ArraysD::I16(y)) => ArraysD::I16(elem(x, y, &s, $int_f, vm)?),
                    (ArraysD::I32(x), ArraysD::I32(y)) => ArraysD::I32(elem(x, y, &s, $int_f, vm)?),
                    (ArraysD::I64(x), ArraysD::I64(y)) => ArraysD::I64(elem(x, y, &s, $int_f, vm)?),
                    (ArraysD::U8(x), ArraysD::U8(y)) => ArraysD::U8(elem(x, y, &s, $int_f, vm)?),
                    (ArraysD::U16(x), ArraysD::U16(y)) => ArraysD::U16(elem(x, y, &s, $int_f, vm)?),
                    (ArraysD::U32(x), ArraysD::U32(y)) => ArraysD::U32(elem(x, y, &s, $int_f, vm)?),
                    (ArraysD::U64(x), ArraysD::U64(y)) => ArraysD::U64(elem(x, y, &s, $int_f, vm)?),
                    (ArraysD::F16(x), ArraysD::F16(y)) => {
                        ArraysD::F16(elem(x, y, &s, $float_f, vm)?)
                    }
                    (ArraysD::F32(x), ArraysD::F32(y)) => {
                        ArraysD::F32(elem(x, y, &s, $float_f, vm)?)
                    }
                    (ArraysD::F64(x), ArraysD::F64(y)) => {
                        ArraysD::F64(elem(x, y, &s, $float_f, vm)?)
                    }
                    (ArraysD::C64(x), ArraysD::C64(y)) => {
                        ArraysD::C64(elem(x, y, &s, $float_f, vm)?)
                    }
                    (ArraysD::C128(x), ArraysD::C128(y)) => {
                        ArraysD::C128(elem(x, y, &s, $float_f, vm)?)
                    }
                    _ => {
                        return Err(internal(
                            vm,
                            "operands not in the same dtype after promotion",
                        ));
                    }
                })
            }
        }
    };
}

fn elem<T, F>(
    a: &ArrayD<T>,
    b: &ArrayD<T>,
    shape: &IxDyn,
    f: F,
    vm: &VirtualMachine,
) -> PyResult<ArrayD<T>>
where
    T: Copy + Zero,
    F: Fn(T, T) -> T,
{
    let av = a
        .broadcast(shape.clone())
        .or_internal(vm, "elem broadcast lhs")?;
    let bv = b
        .broadcast(shape.clone())
        .or_internal(vm, "elem broadcast rhs")?;
    let mut out = ArrayD::<T>::zeros(shape.clone());
    Zip::from(&mut out)
        .and(&av)
        .and(&bv)
        .for_each(|o, &x, &y| *o = f(x, y));
    Ok(out)
}

// Integers use wrapping_* to match numpy's two's-complement overflow semantics
// (avoids the Rust debug-build overflow panic). Floats/complex use normal ops.
impl_arith_op!(
    Add,
    |x, y| num_traits::WrappingAdd::wrapping_add(&x, &y),
    |x, y| x + y,
    |x, y| x | y
); // bool + bool → OR
impl_arith_op!(
    Sub,
    |x, y| num_traits::WrappingSub::wrapping_sub(&x, &y),
    |x, y| x - y,
    |x, y| x ^ y
); // bool - bool → XOR
impl_arith_op!(
    Mul,
    |x, y| num_traits::WrappingMul::wrapping_mul(&x, &y),
    |x, y| x * y,
    |x, y| x & y
); // bool * bool → AND

// Division is special: numpy promotes integer-only division to float64.
pub fn true_divide(a: &ArraysD, b: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let mut out_dtype = promote(a.dtype(), b.dtype());
    if out_dtype.is_integer() {
        out_dtype = DType::F64;
    }
    let a = a.cast(out_dtype);
    let b = b.cast(out_dtype);
    let shape = broadcast_shape(a.shape(), b.shape()).ok_or_else(|| {
        vm.new_value_error(format!(
            "operands could not be broadcast together with shapes {:?} {:?}",
            a.shape(),
            b.shape()
        ))
    })?;
    let s = IxDyn(&shape);
    Ok(match (&a, &b) {
        (ArraysD::F16(x), ArraysD::F16(y)) => ArraysD::F16(elem(x, y, &s, |x, y| x / y, vm)?),
        (ArraysD::F32(x), ArraysD::F32(y)) => ArraysD::F32(elem(x, y, &s, |x, y| x / y, vm)?),
        (ArraysD::F64(x), ArraysD::F64(y)) => ArraysD::F64(elem(x, y, &s, |x, y| x / y, vm)?),
        (ArraysD::C64(x), ArraysD::C64(y)) => ArraysD::C64(elem(x, y, &s, |x, y| x / y, vm)?),
        (ArraysD::C128(x), ArraysD::C128(y)) => ArraysD::C128(elem(x, y, &s, |x, y| x / y, vm)?),
        _ => return Err(internal(vm, "true_divide promotion fell through")),
    })
}

pub fn floor_divide(a: &ArraysD, b: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let out_dtype = promote(a.dtype(), b.dtype());
    if out_dtype.is_complex() {
        return Err(vm.new_type_error("floor_divide not defined for complex numbers".to_string()));
    }
    let a = a.cast(out_dtype);
    let b = b.cast(out_dtype);
    let shape = broadcast_shape(a.shape(), b.shape()).ok_or_else(|| {
        vm.new_value_error(format!("broadcast {:?} vs {:?}", a.shape(), b.shape()))
    })?;
    let s = IxDyn(&shape);
    Ok(match (&a, &b) {
        (ArraysD::I8(x), ArraysD::I8(y)) => {
            ArraysD::I8(elem(x, y, &s, |a, b| a.div_euclid(b), vm)?)
        }
        (ArraysD::I16(x), ArraysD::I16(y)) => {
            ArraysD::I16(elem(x, y, &s, |a, b| a.div_euclid(b), vm)?)
        }
        (ArraysD::I32(x), ArraysD::I32(y)) => {
            ArraysD::I32(elem(x, y, &s, |a, b| a.div_euclid(b), vm)?)
        }
        (ArraysD::I64(x), ArraysD::I64(y)) => {
            ArraysD::I64(elem(x, y, &s, |a, b| a.div_euclid(b), vm)?)
        }
        (ArraysD::U8(x), ArraysD::U8(y)) => ArraysD::U8(elem(x, y, &s, |a, b| a / b, vm)?),
        (ArraysD::U16(x), ArraysD::U16(y)) => ArraysD::U16(elem(x, y, &s, |a, b| a / b, vm)?),
        (ArraysD::U32(x), ArraysD::U32(y)) => ArraysD::U32(elem(x, y, &s, |a, b| a / b, vm)?),
        (ArraysD::U64(x), ArraysD::U64(y)) => ArraysD::U64(elem(x, y, &s, |a, b| a / b, vm)?),
        (ArraysD::F16(x), ArraysD::F16(y)) => ArraysD::F16(elem(
            x,
            y,
            &s,
            |a, b| f16::from_f32((f32::from(a) / f32::from(b)).floor()),
            vm,
        )?),
        (ArraysD::F32(x), ArraysD::F32(y)) => {
            ArraysD::F32(elem(x, y, &s, |a, b| (a / b).floor(), vm)?)
        }
        (ArraysD::F64(x), ArraysD::F64(y)) => {
            ArraysD::F64(elem(x, y, &s, |a, b| (a / b).floor(), vm)?)
        }
        (ArraysD::Bool(x), ArraysD::Bool(y)) => ArraysD::I8(elem(
            &x.mapv(|v| v as i8),
            &y.mapv(|v| v as i8),
            &s,
            |a, b| a / b,
            vm,
        )?),
        _ => return Err(internal(vm, "floor_divide promotion fell through")),
    })
}

pub fn remainder(a: &ArraysD, b: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let out_dtype = promote(a.dtype(), b.dtype());
    if out_dtype.is_complex() {
        return Err(vm.new_type_error("remainder not defined for complex".to_string()));
    }
    let a = a.cast(out_dtype);
    let b = b.cast(out_dtype);
    let shape = broadcast_shape(a.shape(), b.shape()).ok_or_else(|| {
        vm.new_value_error(format!("broadcast {:?} vs {:?}", a.shape(), b.shape()))
    })?;
    let s = IxDyn(&shape);
    Ok(match (&a, &b) {
        (ArraysD::I8(x), ArraysD::I8(y)) => {
            ArraysD::I8(elem(x, y, &s, |a, b| a.rem_euclid(b), vm)?)
        }
        (ArraysD::I16(x), ArraysD::I16(y)) => {
            ArraysD::I16(elem(x, y, &s, |a, b| a.rem_euclid(b), vm)?)
        }
        (ArraysD::I32(x), ArraysD::I32(y)) => {
            ArraysD::I32(elem(x, y, &s, |a, b| a.rem_euclid(b), vm)?)
        }
        (ArraysD::I64(x), ArraysD::I64(y)) => {
            ArraysD::I64(elem(x, y, &s, |a, b| a.rem_euclid(b), vm)?)
        }
        (ArraysD::U8(x), ArraysD::U8(y)) => ArraysD::U8(elem(x, y, &s, |a, b| a % b, vm)?),
        (ArraysD::U16(x), ArraysD::U16(y)) => ArraysD::U16(elem(x, y, &s, |a, b| a % b, vm)?),
        (ArraysD::U32(x), ArraysD::U32(y)) => ArraysD::U32(elem(x, y, &s, |a, b| a % b, vm)?),
        (ArraysD::U64(x), ArraysD::U64(y)) => ArraysD::U64(elem(x, y, &s, |a, b| a % b, vm)?),
        (ArraysD::F16(x), ArraysD::F16(y)) => ArraysD::F16(elem(
            x,
            y,
            &s,
            |a, b| f16::from_f32(f32::from(a).rem_euclid(f32::from(b))),
            vm,
        )?),
        (ArraysD::F32(x), ArraysD::F32(y)) => {
            ArraysD::F32(elem(x, y, &s, |a, b| a.rem_euclid(b), vm)?)
        }
        (ArraysD::F64(x), ArraysD::F64(y)) => {
            ArraysD::F64(elem(x, y, &s, |a, b| a.rem_euclid(b), vm)?)
        }
        (ArraysD::Bool(x), ArraysD::Bool(y)) => ArraysD::I8(elem(
            &x.mapv(|v| v as i8),
            &y.mapv(|v| v as i8),
            &s,
            |a, b| a.rem_euclid(b),
            vm,
        )?),
        _ => return Err(internal(vm, "remainder promotion fell through")),
    })
}

pub fn power(a: &ArraysD, b: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let out_dtype = promote(a.dtype(), b.dtype());
    let a = a.cast(out_dtype);
    let b = b.cast(out_dtype);
    let shape = broadcast_shape(a.shape(), b.shape()).ok_or_else(|| {
        vm.new_value_error(format!(
            "operands could not be broadcast together with shapes {:?} {:?}",
            a.shape(),
            b.shape()
        ))
    })?;
    let s = IxDyn(&shape);
    Ok(match (&a, &b) {
        (ArraysD::F16(x), ArraysD::F16(y)) => ArraysD::F16(elem(
            x,
            y,
            &s,
            |a, b| f16::from_f32(f32::from(a).powf(f32::from(b))),
            vm,
        )?),
        (ArraysD::F32(x), ArraysD::F32(y)) => ArraysD::F32(elem(x, y, &s, |a, b| a.powf(b), vm)?),
        (ArraysD::F64(x), ArraysD::F64(y)) => ArraysD::F64(elem(x, y, &s, |a, b| a.powf(b), vm)?),
        (ArraysD::C64(x), ArraysD::C64(y)) => ArraysD::C64(elem(x, y, &s, |a, b| a.powc(b), vm)?),
        (ArraysD::C128(x), ArraysD::C128(y)) => {
            ArraysD::C128(elem(x, y, &s, |a, b| a.powc(b), vm)?)
        }
        (ArraysD::I8(x), ArraysD::I8(y)) => ArraysD::I8(elem(
            x,
            y,
            &s,
            |a, b| int_pow_i64(a as i64, b as i64) as i8,
            vm,
        )?),
        (ArraysD::I16(x), ArraysD::I16(y)) => ArraysD::I16(elem(
            x,
            y,
            &s,
            |a, b| int_pow_i64(a as i64, b as i64) as i16,
            vm,
        )?),
        (ArraysD::I32(x), ArraysD::I32(y)) => ArraysD::I32(elem(
            x,
            y,
            &s,
            |a, b| int_pow_i64(a as i64, b as i64) as i32,
            vm,
        )?),
        (ArraysD::I64(x), ArraysD::I64(y)) => ArraysD::I64(elem(x, y, &s, int_pow_i64, vm)?),
        (ArraysD::U8(x), ArraysD::U8(y)) => {
            ArraysD::U8(elem(x, y, &s, |a, b| a.pow(b as u32), vm)?)
        }
        (ArraysD::U16(x), ArraysD::U16(y)) => {
            ArraysD::U16(elem(x, y, &s, |a, b| a.pow(b as u32), vm)?)
        }
        (ArraysD::U32(x), ArraysD::U32(y)) => ArraysD::U32(elem(x, y, &s, |a, b| a.pow(b), vm)?),
        (ArraysD::U64(x), ArraysD::U64(y)) => {
            ArraysD::U64(elem(x, y, &s, |a, b| a.pow(b as u32), vm)?)
        }
        (ArraysD::Bool(x), ArraysD::Bool(y)) => {
            let av = x.broadcast(s.clone()).or_internal(vm, "power bool lhs")?;
            let bv = y.broadcast(s.clone()).or_internal(vm, "power bool rhs")?;
            let mut out = ArrayD::<bool>::from_elem(s.clone(), false);
            Zip::from(&mut out).and(&av).and(&bv).for_each(|o, &a, &b| {
                *o = if b { a } else { true };
            });
            ArraysD::Bool(out)
        }
        _ => return Err(internal(vm, "power promotion fell through")),
    })
}

fn int_pow_i64(base: i64, exp: i64) -> i64 {
    if exp < 0 {
        return 0;
    }
    base.wrapping_pow(exp as u32)
}

// ---------------------------------------------------------------------------
// Element-wise comparison (returns bool array)
// ---------------------------------------------------------------------------

pub fn compare(a: &ArraysD, b: &ArraysD, op: CmpOp, vm: &VirtualMachine) -> PyResult<ArraysD> {
    let pt = promote(a.dtype(), b.dtype());
    let a = a.cast(pt);
    let b = b.cast(pt);
    let shape = broadcast_shape(a.shape(), b.shape()).ok_or_else(|| {
        vm.new_value_error(format!("broadcast {:?} vs {:?}", a.shape(), b.shape()))
    })?;
    let s = IxDyn(&shape);
    let result = match (&a, &b) {
        (ArraysD::Bool(x), ArraysD::Bool(y)) => cmp_array(x, y, &s, op, vm)?,
        (ArraysD::I8(x), ArraysD::I8(y)) => cmp_array(x, y, &s, op, vm)?,
        (ArraysD::I16(x), ArraysD::I16(y)) => cmp_array(x, y, &s, op, vm)?,
        (ArraysD::I32(x), ArraysD::I32(y)) => cmp_array(x, y, &s, op, vm)?,
        (ArraysD::I64(x), ArraysD::I64(y)) => cmp_array(x, y, &s, op, vm)?,
        (ArraysD::U8(x), ArraysD::U8(y)) => cmp_array(x, y, &s, op, vm)?,
        (ArraysD::U16(x), ArraysD::U16(y)) => cmp_array(x, y, &s, op, vm)?,
        (ArraysD::U32(x), ArraysD::U32(y)) => cmp_array(x, y, &s, op, vm)?,
        (ArraysD::U64(x), ArraysD::U64(y)) => cmp_array(x, y, &s, op, vm)?,
        (ArraysD::F16(x), ArraysD::F16(y)) => cmp_array(x, y, &s, op, vm)?,
        (ArraysD::F32(x), ArraysD::F32(y)) => cmp_array(x, y, &s, op, vm)?,
        (ArraysD::F64(x), ArraysD::F64(y)) => cmp_array(x, y, &s, op, vm)?,
        (ArraysD::C64(x), ArraysD::C64(y)) => cmp_complex_c32(x, y, &s, op, vm)?,
        (ArraysD::C128(x), ArraysD::C128(y)) => cmp_complex_c64(x, y, &s, op, vm)?,
        (ArraysD::Str { data: x, .. }, ArraysD::Str { data: y, .. }) => {
            cmp_array_ref(x, y, &s, op, vm)?
        }
        (ArraysD::Bytes { data: x, .. }, ArraysD::Bytes { data: y, .. }) => {
            cmp_array_ref(x, y, &s, op, vm)?
        }
        _ => return Err(internal(vm, "compare promotion fell through")),
    };
    Ok(ArraysD::Bool(result))
}

#[derive(Copy, Clone)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

fn cmp_array<T: Copy + PartialOrd + PartialEq>(
    a: &ArrayD<T>,
    b: &ArrayD<T>,
    shape: &IxDyn,
    op: CmpOp,
    vm: &VirtualMachine,
) -> PyResult<ArrayD<bool>> {
    let av = a
        .broadcast(shape.clone())
        .or_internal(vm, "cmp broadcast lhs")?;
    let bv = b
        .broadcast(shape.clone())
        .or_internal(vm, "cmp broadcast rhs")?;
    let mut out = ArrayD::<bool>::from_elem(shape.clone(), false);
    Zip::from(&mut out).and(&av).and(&bv).for_each(|o, x, y| {
        *o = match op {
            CmpOp::Eq => x == y,
            CmpOp::Ne => x != y,
            CmpOp::Lt => x < y,
            CmpOp::Le => x <= y,
            CmpOp::Gt => x > y,
            CmpOp::Ge => x >= y,
        };
    });
    Ok(out)
}

/// Reference-comparison variant for non-Copy element types (String, Vec<u8>).
fn cmp_array_ref<T: PartialOrd + PartialEq + Clone>(
    a: &ArrayD<T>,
    b: &ArrayD<T>,
    shape: &IxDyn,
    op: CmpOp,
    vm: &VirtualMachine,
) -> PyResult<ArrayD<bool>> {
    let av = a
        .broadcast(shape.clone())
        .or_internal(vm, "cmp broadcast lhs")?;
    let bv = b
        .broadcast(shape.clone())
        .or_internal(vm, "cmp broadcast rhs")?;
    let mut out = ArrayD::<bool>::from_elem(shape.clone(), false);
    Zip::from(&mut out).and(&av).and(&bv).for_each(|o, x, y| {
        *o = match op {
            CmpOp::Eq => x == y,
            CmpOp::Ne => x != y,
            CmpOp::Lt => x < y,
            CmpOp::Le => x <= y,
            CmpOp::Gt => x > y,
            CmpOp::Ge => x >= y,
        };
    });
    Ok(out)
}

fn cmp_complex_c32(
    a: &ArrayD<C32>,
    b: &ArrayD<C32>,
    shape: &IxDyn,
    op: CmpOp,
    vm: &VirtualMachine,
) -> PyResult<ArrayD<bool>> {
    if !matches!(op, CmpOp::Eq | CmpOp::Ne) {
        return Err(vm.new_type_error("ordering comparison not defined for complex".to_string()));
    }
    let av = a.broadcast(shape.clone()).or_internal(vm, "cmp c32 lhs")?;
    let bv = b.broadcast(shape.clone()).or_internal(vm, "cmp c32 rhs")?;
    let mut out = ArrayD::<bool>::from_elem(shape.clone(), false);
    Zip::from(&mut out).and(&av).and(&bv).for_each(|o, x, y| {
        *o = match op {
            CmpOp::Eq => x == y,
            CmpOp::Ne => x != y,
            CmpOp::Lt | CmpOp::Le | CmpOp::Gt | CmpOp::Ge => false,
        };
    });
    Ok(out)
}

fn cmp_complex_c64(
    a: &ArrayD<C64>,
    b: &ArrayD<C64>,
    shape: &IxDyn,
    op: CmpOp,
    vm: &VirtualMachine,
) -> PyResult<ArrayD<bool>> {
    if !matches!(op, CmpOp::Eq | CmpOp::Ne) {
        return Err(vm.new_type_error("ordering comparison not defined for complex".to_string()));
    }
    let av = a.broadcast(shape.clone()).or_internal(vm, "cmp c64 lhs")?;
    let bv = b.broadcast(shape.clone()).or_internal(vm, "cmp c64 rhs")?;
    let mut out = ArrayD::<bool>::from_elem(shape.clone(), false);
    Zip::from(&mut out).and(&av).and(&bv).for_each(|o, x, y| {
        *o = match op {
            CmpOp::Eq => x == y,
            CmpOp::Ne => x != y,
            CmpOp::Lt | CmpOp::Le | CmpOp::Gt | CmpOp::Ge => false,
        };
    });
    Ok(out)
}

// ---------------------------------------------------------------------------
// Unary
// ---------------------------------------------------------------------------

pub fn negate(a: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    Ok(match a {
        ArraysD::Bool(_) => {
            return Err(vm.new_type_error(
                "negating a bool array is not supported; convert to int first".to_string(),
            ));
        }
        ArraysD::I8(a) => ArraysD::I8(a.mapv(|v| -v)),
        ArraysD::I16(a) => ArraysD::I16(a.mapv(|v| -v)),
        ArraysD::I32(a) => ArraysD::I32(a.mapv(|v| -v)),
        ArraysD::I64(a) => ArraysD::I64(a.mapv(|v| -v)),
        ArraysD::U8(a) => ArraysD::U8(a.mapv(|v| v.wrapping_neg())),
        ArraysD::U16(a) => ArraysD::U16(a.mapv(|v| v.wrapping_neg())),
        ArraysD::U32(a) => ArraysD::U32(a.mapv(|v| v.wrapping_neg())),
        ArraysD::U64(a) => ArraysD::U64(a.mapv(|v| v.wrapping_neg())),
        ArraysD::F16(a) => ArraysD::F16(a.mapv(|v| f16::from_f32(-f32::from(v)))),
        ArraysD::F32(a) => ArraysD::F32(a.mapv(|v| -v)),
        ArraysD::F64(a) => ArraysD::F64(a.mapv(|v| -v)),
        ArraysD::C64(a) => ArraysD::C64(a.mapv(|v| -v)),
        ArraysD::C128(a) => ArraysD::C128(a.mapv(|v| -v)),
        other => {
            return Err(crate::internal::unsupported_dtype(
                vm,
                "negate",
                other.dtype(),
            ));
        }
    })
}

pub fn absolute(a: &ArraysD) -> ArraysD {
    match a {
        ArraysD::Bool(a) => ArraysD::Bool(a.clone()),
        ArraysD::I8(a) => ArraysD::I8(a.mapv(|v| v.wrapping_abs())),
        ArraysD::I16(a) => ArraysD::I16(a.mapv(|v| v.wrapping_abs())),
        ArraysD::I32(a) => ArraysD::I32(a.mapv(|v| v.wrapping_abs())),
        ArraysD::I64(a) => ArraysD::I64(a.mapv(|v| v.wrapping_abs())),
        ArraysD::U8(a) => ArraysD::U8(a.clone()),
        ArraysD::U16(a) => ArraysD::U16(a.clone()),
        ArraysD::U32(a) => ArraysD::U32(a.clone()),
        ArraysD::U64(a) => ArraysD::U64(a.clone()),
        ArraysD::F16(a) => ArraysD::F16(a.mapv(|v| f16::from_f32(f32::from(v).abs()))),
        ArraysD::F32(a) => ArraysD::F32(a.mapv(|v| v.abs())),
        ArraysD::F64(a) => ArraysD::F64(a.mapv(|v| v.abs())),
        // Complex |z| is real
        ArraysD::C64(a) => ArraysD::F32(a.mapv(|v| v.norm())),
        ArraysD::C128(a) => ArraysD::F64(a.mapv(|v| v.norm())),
        // `abs()` of a non-numeric array isn't well-defined; numpy raises.
        // We can't return a Result here without changing the signature, and
        // `absolute` is called from a non-erroring slot — return the array
        // unchanged so the caller can still inspect it (matches numpy's
        // behaviour for `np.absolute(obj_arr)` which also no-ops on object).
        other => other.clone(),
    }
}

/// Apply a real-valued unary float function to an array.
/// Integer/bool inputs are promoted to f64; f32 stays f32; f64 stays f64;
/// f16 promotes to f32 for the calculation then narrows back.
/// Complex inputs use the supplied complex function.
pub fn unary_real_or_complex<FR, FC>(a: &ArraysD, fr: FR, fc: FC) -> ArraysD
where
    FR: Fn(f64) -> f64 + Copy,
    FC: Fn(C64) -> C64 + Copy,
{
    match a {
        ArraysD::F32(arr) => ArraysD::F32(arr.mapv(|v| fr(v as f64) as f32)),
        ArraysD::F64(arr) => ArraysD::F64(arr.mapv(fr)),
        ArraysD::F16(arr) => ArraysD::F16(arr.mapv(|v| f16::from_f64(fr(f32::from(v) as f64)))),
        ArraysD::C64(arr) => ArraysD::C64(arr.mapv(|v| {
            let c = fc(Complex::new(v.re as f64, v.im as f64));
            C32::new(c.re as f32, c.im as f32)
        })),
        ArraysD::C128(arr) => ArraysD::C128(arr.mapv(fc)),
        other => {
            // Bool / any integer dtype → widen to f64 first. `cast` always
            // produces an F64 variant for these inputs.
            let f = other.cast(DType::F64);
            match f {
                ArraysD::F64(arr) => ArraysD::F64(arr.mapv(fr)),
                // Cast post-condition: integer/bool always becomes F64. If
                // a future cast routes elsewhere, fall back to the original
                // array unchanged rather than panicking.
                ref _other => other.clone(),
            }
        }
    }
}

/// Apply a real-only unary function (errors on complex).
pub fn unary_real_only<FR>(
    a: &ArraysD,
    name: &str,
    fr: FR,
    vm: &VirtualMachine,
) -> PyResult<ArraysD>
where
    FR: Fn(f64) -> f64 + Copy,
{
    if a.dtype().is_complex() {
        return Err(vm.new_type_error(format!("{name} not defined for complex dtype")));
    }
    Ok(unary_real_or_complex(a, fr, |_| Complex::new(0.0, 0.0)))
}

/// `numpy.real` — return the real part. Complex → matching real dtype.
pub fn real_part(a: &ArraysD) -> ArraysD {
    match a {
        ArraysD::C64(arr) => ArraysD::F32(arr.mapv(|v| v.re)),
        ArraysD::C128(arr) => ArraysD::F64(arr.mapv(|v| v.re)),
        _ => a.clone(),
    }
}

/// `numpy.imag` — imag part; non-complex → zeros of same dtype (numpy returns
/// a real array of matching dtype).
pub fn imag_part(a: &ArraysD) -> ArraysD {
    match a {
        ArraysD::C64(arr) => ArraysD::F32(arr.mapv(|v| v.im)),
        ArraysD::C128(arr) => ArraysD::F64(arr.mapv(|v| v.im)),
        _ => {
            let shape = a.raw_dim();
            dispatch_zero_of(a.dtype(), shape)
        }
    }
}

/// `numpy.conjugate`.
pub fn conj(a: &ArraysD) -> ArraysD {
    match a {
        ArraysD::C64(arr) => ArraysD::C64(arr.mapv(|v| v.conj())),
        ArraysD::C128(arr) => ArraysD::C128(arr.mapv(|v| v.conj())),
        _ => a.clone(),
    }
}

fn dispatch_zero_of(dt: DType, shape: IxDyn) -> ArraysD {
    match dt {
        DType::Bool => ArraysD::Bool(ArrayD::from_elem(shape, false)),
        DType::I8 => ArraysD::I8(ArrayD::zeros(shape)),
        DType::I16 => ArraysD::I16(ArrayD::zeros(shape)),
        DType::I32 => ArraysD::I32(ArrayD::zeros(shape)),
        DType::I64 => ArraysD::I64(ArrayD::zeros(shape)),
        DType::U8 => ArraysD::U8(ArrayD::zeros(shape)),
        DType::U16 => ArraysD::U16(ArrayD::zeros(shape)),
        DType::U32 => ArraysD::U32(ArrayD::zeros(shape)),
        DType::U64 => ArraysD::U64(ArrayD::zeros(shape)),
        DType::F16 => ArraysD::F16(ArrayD::from_elem(shape, f16::ZERO)),
        DType::F32 => ArraysD::F32(ArrayD::zeros(shape)),
        DType::F64 => ArraysD::F64(ArrayD::zeros(shape)),
        DType::C64 => ArraysD::C64(ArrayD::from_elem(shape, C32::new(0.0, 0.0))),
        DType::C128 => ArraysD::C128(ArrayD::from_elem(shape, C64::new(0.0, 0.0))),
        // Non-numeric "zero" routes to the canonical empty/zero value for
        // each kind: Object → empty (caller should use a vm to build None),
        // Str/Bytes → empty string, Datetime/Timedelta → 0, Void → zero buffer.
        DType::Object => ArraysD::Object(crate::internal::empty_array()),
        DType::Str(n) => ArraysD::Str {
            itemsize_chars: n,
            data: ArrayD::from_elem(shape, String::new()),
        },
        DType::Bytes(n) => ArraysD::Bytes {
            itemsize: n,
            data: ArrayD::from_elem(shape, vec![0u8; n as usize]),
        },
        DType::Datetime64(u) => ArraysD::Datetime64 {
            unit: u,
            data: ArrayD::zeros(shape),
        },
        DType::Timedelta64(u) => ArraysD::Timedelta64 {
            unit: u,
            data: ArrayD::zeros(shape),
        },
        DType::Void(n) => ArraysD::Void {
            layout: std::sync::Arc::new(crate::dtype::StructLayout::new(Vec::new(), n as usize)),
            data: ArrayD::from_elem(shape, vec![0u8; n as usize]),
        },
    }
}

/// `numpy.maximum` / `numpy.minimum`. Promotes both operands.
pub fn elem_max(a: &ArraysD, b: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    binary_pair(a, b, vm, |x, y| if x > y { x } else { y }, |x, y| x | y)
}
pub fn elem_min(a: &ArraysD, b: &ArraysD, vm: &VirtualMachine) -> PyResult<ArraysD> {
    binary_pair(a, b, vm, |x, y| if x < y { x } else { y }, |x, y| x & y)
}

fn binary_pair<FR, FB>(
    a: &ArraysD,
    b: &ArraysD,
    vm: &VirtualMachine,
    fr: FR,
    fb: FB,
) -> PyResult<ArraysD>
where
    FR: Fn(f64, f64) -> f64 + Copy,
    FB: Fn(bool, bool) -> bool + Copy,
{
    let out_dtype = promote(a.dtype(), b.dtype());
    let a = a.cast(out_dtype);
    let b = b.cast(out_dtype);
    let shape = broadcast_shape(a.shape(), b.shape())
        .ok_or_else(|| vm.new_value_error("broadcast failure".to_string()))?;
    let s = IxDyn(&shape);
    Ok(match (&a, &b) {
        (ArraysD::Bool(x), ArraysD::Bool(y)) => {
            let xv = x
                .broadcast(s.clone())
                .or_internal(vm, "binary_pair bool lhs")?;
            let yv = y
                .broadcast(s.clone())
                .or_internal(vm, "binary_pair bool rhs")?;
            let mut out = ArrayD::from_elem(s, false);
            Zip::from(&mut out)
                .and(&xv)
                .and(&yv)
                .for_each(|o, &p, &q| *o = fb(p, q));
            ArraysD::Bool(out)
        }
        (ArraysD::I8(x), ArraysD::I8(y)) => ArraysD::I8(elem(
            x,
            y,
            &s,
            |a, b| {
                if fr(a as f64, b as f64) == a as f64 {
                    a
                } else {
                    b
                }
            },
            vm,
        )?),
        (ArraysD::I16(x), ArraysD::I16(y)) => ArraysD::I16(elem(
            x,
            y,
            &s,
            |a, b| {
                if fr(a as f64, b as f64) == a as f64 {
                    a
                } else {
                    b
                }
            },
            vm,
        )?),
        (ArraysD::I32(x), ArraysD::I32(y)) => ArraysD::I32(elem(
            x,
            y,
            &s,
            |a, b| {
                if fr(a as f64, b as f64) == a as f64 {
                    a
                } else {
                    b
                }
            },
            vm,
        )?),
        (ArraysD::I64(x), ArraysD::I64(y)) => ArraysD::I64(elem(
            x,
            y,
            &s,
            |a, b| {
                if fr(a as f64, b as f64) == a as f64 {
                    a
                } else {
                    b
                }
            },
            vm,
        )?),
        (ArraysD::U8(x), ArraysD::U8(y)) => ArraysD::U8(elem(
            x,
            y,
            &s,
            |a, b| {
                if fr(a as f64, b as f64) == a as f64 {
                    a
                } else {
                    b
                }
            },
            vm,
        )?),
        (ArraysD::U16(x), ArraysD::U16(y)) => ArraysD::U16(elem(
            x,
            y,
            &s,
            |a, b| {
                if fr(a as f64, b as f64) == a as f64 {
                    a
                } else {
                    b
                }
            },
            vm,
        )?),
        (ArraysD::U32(x), ArraysD::U32(y)) => ArraysD::U32(elem(
            x,
            y,
            &s,
            |a, b| {
                if fr(a as f64, b as f64) == a as f64 {
                    a
                } else {
                    b
                }
            },
            vm,
        )?),
        (ArraysD::U64(x), ArraysD::U64(y)) => ArraysD::U64(elem(
            x,
            y,
            &s,
            |a, b| {
                if fr(a as f64, b as f64) == a as f64 {
                    a
                } else {
                    b
                }
            },
            vm,
        )?),
        (ArraysD::F16(x), ArraysD::F16(y)) => ArraysD::F16(elem(
            x,
            y,
            &s,
            |a, b| f16::from_f64(fr(a.to_f64_(), b.to_f64_())),
            vm,
        )?),
        (ArraysD::F32(x), ArraysD::F32(y)) => {
            ArraysD::F32(elem(x, y, &s, |a, b| fr(a as f64, b as f64) as f32, vm)?)
        }
        (ArraysD::F64(x), ArraysD::F64(y)) => ArraysD::F64(elem(x, y, &s, fr, vm)?),
        (ArraysD::C64(_), ArraysD::C64(_)) | (ArraysD::C128(_), ArraysD::C128(_)) => {
            return Err(vm.new_type_error("maximum/minimum not defined for complex".to_string()));
        }
        _ => return Err(internal(vm, "binary_pair promotion fell through")),
    })
}

// f16 helper trait bridge
trait ToF64 {
    fn to_f64_(self) -> f64;
}
impl ToF64 for f16 {
    fn to_f64_(self) -> f64 {
        f32::from(self) as f64
    }
}

// `ResultExt::or_internal` is imported but only used inside the macro
// expansions above — silence the unused-trait-import lint without
// dropping the import (the macros need it visible at call sites).
#[allow(dead_code)]
fn _ensure_trait_visible() {
    let _ = <Result<(), &'static str> as ResultExt<_, _>>::or_internal;
}
