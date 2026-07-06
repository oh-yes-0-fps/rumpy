//! Small helpers for "this should not happen, but if it does, raise a clean
//! Python error instead of panicking."
//!
//! Everywhere we previously had `.unwrap()` or `unreachable!()` on an
//! invariant we believed to hold (post-conditions of `cast`, `broadcast_shape`,
//! `from_shape_vec` with matching length, …), we route through these
//! helpers so that a logic bug doesn't unwind across the FFI boundary.

use crate::dtype::DType;
use rustpython_vm::{PyRef, PyResult, VirtualMachine, builtins::PyBaseException};

/// Build a Python `TypeError` indicating that an operation isn't defined for
/// the given dtype. Used by all the numeric-only ops (FFT, linalg, reductions,
/// elementwise arith on non-numeric variants) to bubble a clean error back
/// to Python instead of `unreachable!()`'ing.
#[inline]
pub fn unsupported_dtype(vm: &VirtualMachine, op: &str, dt: DType) -> PyRef<PyBaseException> {
    vm.new_type_error(format!(
        "operation {op:?} not supported for dtype {}",
        dt.name_owned()
    ))
}

/// A panic-free length-0 `ArrayD<T>` for the non-numeric variants (where
/// `T` may not implement `Default`/`Zero` so the usual `ArrayD::zeros` /
/// `ArrayD::default` constructors don't apply).
///
/// `ndarray::ArrayD::from_shape_vec(IxDyn(&[0]), Vec::new())` is statically
/// guaranteed to succeed (shape product 0 == vec length 0); the
/// `unwrap_or_else` here is just defensive insurance in case the ndarray
/// API contract ever changes — the fallback path leaks no memory and
/// constructs a 0-element view that downstream code can safely walk.
#[inline]
pub fn empty_array<T: Clone>() -> ndarray::ArrayD<T> {
    ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[0]), Vec::<T>::new()).unwrap_or_else(|_| {
        // ndarray invariant violated — fall back to building from the
        // shape directly. `from_shape_simple_fn` skips the closure for
        // zero-element shapes, so this also can't panic.
        ndarray::ArrayD::from_shape_simple_fn(ndarray::IxDyn(&[0]), || {
            // Unreachable in practice (zero-element shape) — but if it
            // ever runs we have no element to return. Return a clone
            // of a heap-allocated `T` would require T: Default, which
            // we don't have. We can only get here through a fundamental
            // ndarray bug, in which case the abort path is acceptable.
            std::process::abort()
        })
    })
}

/// Build a Python `RuntimeError` for an "internal invariant violated" case
/// — i.e. something that should be guaranteed by surrounding code and
/// would represent a bug if it ever fired.
#[inline]
pub fn internal(vm: &VirtualMachine, what: impl AsRef<str>) -> PyRef<PyBaseException> {
    vm.new_runtime_error(format!("rumpy internal: {}", what.as_ref()))
}

/// Convenience: turn an `Option<T>` into a `PyResult<T>` carrying the
/// internal-error message.
pub trait OptionExt<T> {
    fn or_internal(self, vm: &VirtualMachine, what: &str) -> PyResult<T>;
}

impl<T> OptionExt<T> for Option<T> {
    #[inline]
    fn or_internal(self, vm: &VirtualMachine, what: &str) -> PyResult<T> {
        match self {
            Some(v) => Ok(v),
            None => Err(internal(vm, what)),
        }
    }
}

/// Convenience: turn a `Result<T, E: Display>` into a `PyResult<T>` carrying
/// the internal-error message. Errors arising from external libraries
/// (`ndarray::ShapeError`, `std::io::Error`, etc.) get their Display string
/// preserved.
pub trait ResultExt<T, E> {
    fn or_internal(self, vm: &VirtualMachine, what: &str) -> PyResult<T>;
}

impl<T, E: std::fmt::Display> ResultExt<T, E> for Result<T, E> {
    #[inline]
    fn or_internal(self, vm: &VirtualMachine, what: &str) -> PyResult<T> {
        self.map_err(|e| internal(vm, format!("{what}: {e}")))
    }
}
