//! Small helpers for "this should not happen, but if it does, raise a clean
//! Python error instead of panicking."
//!
//! Everywhere we previously had `.unwrap()` or `unreachable!()` on an
//! invariant we believed to hold (post-conditions of `cast`, `broadcast_shape`,
//! `from_shape_vec` with matching length, …), we route through these
//! helpers so that a logic bug doesn't unwind across the FFI boundary.

use rustpython_vm::{
    PyResult, VirtualMachine, builtins::PyBaseException, PyRef,
};

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
