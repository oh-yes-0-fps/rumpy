//! Additional numpy cross-validation tests covering functions that were
//! sparsely exercised by the other test files: covariance/correlation,
//! `hypot`, stacking variants, axis manipulators, set/search ops, the
//! polynomial family, and FFT helpers.
//!
//! Each test snippet runs in both runtimes (RustPython+rumpy and
//! CPython+numpy) and asserts the `.tolist()` outputs match element-wise.

use approx::assert_abs_diff_eq;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyList, PyModule};
use rustpython_vm::{AsObject, Interpreter, builtins::PyList as RpyList};

// ---------------------------------------------------------------------------
// Shared scaffolding — same shape as compare_numpy.rs's helpers.
// ---------------------------------------------------------------------------

fn run_in_rumpy(source: &str) -> RumpyResult {
    let interp = {
        let builder = Interpreter::builder(Default::default());
        let def = rumpy::module_def(&builder.ctx);
        builder.add_native_module(def).build()
    };
    let outcome = interp.enter(|vm| -> Result<RumpyResult, String> {
        let scope = vm.new_scope_with_builtins();
        let code = vm
            .compile(source, rustpython_vm::compiler::Mode::Exec, "<test>".into())
            .map_err(|e| format!("compile error: {e}"))?;
        vm.run_code_obj(code, scope.clone())
            .map_err(|e| py_err_string(vm, &e))?;
        let result = scope
            .globals
            .get_item("result", vm)
            .expect("snippet must set `result`");
        extract_rumpy_value(&result, vm).map_err(|e| py_err_string(vm, &e))
    });
    outcome.unwrap_or_else(|e| panic!("rumpy snippet failed: {e}\n--- source ---\n{source}"))
}

fn py_err_string(
    vm: &rustpython_vm::VirtualMachine,
    e: &rustpython_vm::PyRef<rustpython_vm::builtins::PyBaseException>,
) -> String {
    let mut out = String::new();
    if vm.write_exception(&mut out, e).is_err() {
        return format!("<unprintable exception of type {}>", e.class().name());
    }
    out
}

struct RumpyResult {
    shape: Option<Vec<usize>>,
    data: Vec<f64>,
}

fn extract_rumpy_value(
    obj: &rustpython_vm::PyObjectRef,
    vm: &rustpython_vm::VirtualMachine,
) -> rustpython_vm::PyResult<RumpyResult> {
    use rumpy::{ArraysD, PyNdArray};
    if let Some(arr) = obj.downcast_ref::<PyNdArray>() {
        let g = arr.view().cast(rumpy::DType::F64);
        let f = match &g {
            ArraysD::F64(x) => x,
            _ => unreachable!(),
        };
        return Ok(RumpyResult {
            shape: Some(f.shape().to_vec()),
            data: f.iter().copied().collect(),
        });
    }
    if let Ok(f) = obj.try_float(vm) {
        return Ok(RumpyResult { shape: None, data: vec![f.to_f64()] });
    }
    if let Some(t) = obj.downcast_ref::<rustpython_vm::builtins::PyTuple>() {
        let mut data = Vec::with_capacity(t.len());
        for it in t.as_slice() {
            data.push(it.try_float(vm)?.to_f64());
        }
        return Ok(RumpyResult { shape: Some(vec![data.len()]), data });
    }
    if let Some(l) = obj.downcast_ref::<RpyList>() {
        let mut shape = Vec::new();
        let mut data = Vec::new();
        flatten_pylist(l, &mut shape, &mut data, vm, 0)?;
        return Ok(RumpyResult { shape: Some(shape), data });
    }
    Err(vm.new_type_error(format!(
        "cannot extract rumpy result of type {}",
        obj.class().name()
    )))
}

fn flatten_pylist(
    list: &RpyList,
    shape: &mut Vec<usize>,
    data: &mut Vec<f64>,
    vm: &rustpython_vm::VirtualMachine,
    depth: usize,
) -> rustpython_vm::PyResult<()> {
    let items = list.borrow_vec();
    if depth == shape.len() {
        shape.push(items.len());
    }
    for it in items.iter() {
        if let Some(sub) = it.downcast_ref::<RpyList>() {
            flatten_pylist(sub, shape, data, vm, depth + 1)?;
        } else {
            data.push(it.try_float(vm)?.to_f64());
        }
    }
    Ok(())
}

fn run_in_numpy(source: &str) -> NumpyResult {
    Python::attach(|py| -> PyResult<NumpyResult> {
        let globals = pyo3::types::PyDict::new(py);
        let numpy = PyModule::import(py, "numpy")?;
        globals.set_item("numpy", &numpy)?;
        globals.set_item("np", &numpy)?;
        py.run(&std::ffi::CString::new(source).unwrap(), Some(&globals), None)?;
        let result = globals.get_item("result")?.unwrap();
        let arr = numpy.getattr("asarray")?.call1((result,))?;
        let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
        let flat = arr.call_method0("ravel")?.call_method0("tolist")?;
        let data: Vec<f64> = flat.cast::<PyList>()?.extract()?;
        Ok(NumpyResult { shape, data })
    })
    .expect("numpy snippet failed")
}

struct NumpyResult {
    shape: Vec<usize>,
    data: Vec<f64>,
}

fn assert_same_eps(snippet: &str, eps: f64) {
    let r = run_in_rumpy(snippet);
    let n = run_in_numpy(snippet);
    match r.shape {
        None => {
            assert!(
                n.shape.is_empty(),
                "rumpy returned scalar but numpy returned shape {:?} for snippet:\n{snippet}",
                n.shape
            );
        }
        Some(rs) => {
            assert_eq!(
                rs, n.shape,
                "shape mismatch (rumpy={:?}, numpy={:?}) for snippet:\n{snippet}",
                rs, n.shape
            );
        }
    }
    assert_eq!(
        r.data.len(),
        n.data.len(),
        "element count mismatch (rumpy={}, numpy={}) for snippet:\n{snippet}",
        r.data.len(),
        n.data.len()
    );
    for (a, b) in r.data.iter().zip(n.data.iter()) {
        if a.is_nan() && b.is_nan() {
            continue;
        }
        assert_abs_diff_eq!(*a, *b, epsilon = eps);
    }
}

fn assert_same(snippet: &str) {
    assert_same_eps(snippet, 1e-9);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// --- covariance / correlation ---

#[test]
fn cov_two_rows() {
    assert_same_eps(
        r#"
import numpy as np
m = np.array([[1.0, 2.0, 3.0, 4.0],
              [2.0, 4.0, 6.0, 8.0]])
result = np.cov(m)
"#,
        1e-9,
    );
}

#[test]
fn cov_anticorrelated() {
    assert_same_eps(
        r#"
import numpy as np
m = np.array([[1.0, 2.0, 3.0, 4.0],
              [4.0, 3.0, 2.0, 1.0]])
result = np.cov(m)
"#,
        1e-9,
    );
}

#[test]
fn corrcoef_perfect_correlation() {
    assert_same_eps(
        r#"
import numpy as np
m = np.array([[1.0, 2.0, 3.0, 4.0],
              [2.0, 4.0, 6.0, 8.0]])
result = np.corrcoef(m)
"#,
        1e-9,
    );
}

#[test]
fn corrcoef_three_rows() {
    assert_same_eps(
        r#"
import numpy as np
m = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
              [5.0, 4.0, 3.0, 2.0, 1.0],
              [1.0, 3.0, 2.0, 4.0, 0.0]])
result = np.corrcoef(m)
"#,
        1e-9,
    );
}

// --- hypot ---

#[test]
fn hypot_pythagorean_triple() {
    assert_same(
        r#"
import numpy as np
result = np.hypot(np.array([3.0, 5.0, 8.0]), np.array([4.0, 12.0, 15.0]))
"#,
    );
}

#[test]
fn hypot_broadcasts() {
    assert_same(
        r#"
import numpy as np
result = np.hypot(np.array([[3.0], [6.0]]), np.array([4.0, 8.0, 0.0]))
"#,
    );
}

#[test]
fn hypot_scalar_and_array() {
    assert_same(
        r#"
import numpy as np
result = np.hypot(3.0, np.array([4.0, 0.0, -4.0]))
"#,
    );
}

// --- stacking family ---

#[test]
fn hstack_1d_and_2d() {
    assert_same(
        r#"
import numpy as np
result = np.hstack([np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])])
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0], [2.0]])
b = np.array([[3.0, 4.0], [5.0, 6.0]])
result = np.hstack([a, b])
"#,
    );
}

#[test]
fn vstack_1d_and_2d() {
    assert_same(
        r#"
import numpy as np
result = np.vstack([np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])])
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.array([[5.0, 6.0]])
result = np.vstack([a, b])
"#,
    );
}

#[test]
fn dstack_1d_inputs() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
result = np.dstack([a, b])
"#,
    );
}

#[test]
fn column_stack_1d() {
    assert_same(
        r#"
import numpy as np
result = np.column_stack([np.array([1.0, 2.0, 3.0]),
                          np.array([4.0, 5.0, 6.0]),
                          np.array([7.0, 8.0, 9.0])])
"#,
    );
}

#[test]
fn stack_new_axis() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
result = np.stack([a, b], axis=0)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
result = np.stack([a, b], axis=1)
"#,
    );
}

#[test]
fn block_nested() {
    assert_same(
        r#"
import numpy as np
A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.array([[5.0, 6.0], [7.0, 8.0]])
result = np.block([[A, B], [B, A]])
"#,
    );
}

// --- tile / repeat / kron ---

#[test]
fn tile_1d_and_2d() {
    assert_same(
        r#"
import numpy as np
result = np.tile(np.array([1.0, 2.0, 3.0]), 3)
"#,
    );
    assert_same(
        r#"
import numpy as np
result = np.tile(np.array([[1.0, 2.0], [3.0, 4.0]]), (2, 3))
"#,
    );
}

#[test]
fn repeat_along_axes() {
    assert_same(
        r#"
import numpy as np
result = np.repeat(np.array([1.0, 2.0, 3.0]), 2)
"#,
    );
    assert_same(
        r#"
import numpy as np
result = np.repeat(np.array([[1.0, 2.0], [3.0, 4.0]]), 3, axis=0)
"#,
    );
    assert_same(
        r#"
import numpy as np
result = np.repeat(np.array([[1.0, 2.0], [3.0, 4.0]]), [2, 3], axis=1)
"#,
    );
}

#[test]
fn kron_outer_product_like() {
    assert_same(
        r#"
import numpy as np
result = np.kron(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0]))
"#,
    );
    assert_same(
        r#"
import numpy as np
A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.array([[0.0, 1.0], [1.0, 0.0]])
result = np.kron(A, B)
"#,
    );
}

// --- axis manipulation ---

#[test]
fn expand_dims_and_squeeze_round_trip() {
    assert_same(
        r#"
import numpy as np
a = np.arange(6.0).reshape(2, 3)
b = np.expand_dims(a, axis=1)
result = np.squeeze(b, axis=1)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.arange(6.0).reshape(2, 3)
result = np.expand_dims(a, axis=(0, 3))
"#,
    );
}

#[test]
fn moveaxis_and_swapaxes() {
    assert_same(
        r#"
import numpy as np
a = np.arange(24.0).reshape(2, 3, 4)
result = np.moveaxis(a, 0, -1)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.arange(24.0).reshape(2, 3, 4)
result = np.swapaxes(a, 0, 2)
"#,
    );
}

#[test]
fn flip_family() {
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
result = np.flip(a, axis=0)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
result = np.fliplr(a)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
result = np.flipud(a)
"#,
    );
}

#[test]
fn rot90_quarter_turns() {
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
result = np.rot90(a)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
result = np.rot90(a, k=2)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
result = np.rot90(a, k=-1)
"#,
    );
}

#[test]
fn roll_along_axes() {
    assert_same(
        r#"
import numpy as np
result = np.roll(np.arange(10.0), 3)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
result = np.roll(a, shift=1, axis=0)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
result = np.roll(a, shift=(1, 2), axis=(0, 1))
"#,
    );
}

// --- search / unique / sort ---

#[test]
fn unique_returns_sorted_distinct() {
    assert_same(
        r#"
import numpy as np
result = np.unique(np.array([3.0, 1.0, 2.0, 3.0, 2.0, 1.0, 4.0, 5.0, 5.0]))
"#,
    );
}

#[test]
fn searchsorted_left_and_right() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
result = np.searchsorted(a, np.array([0.0, 2.0, 5.0, 9.0]))
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
result = np.searchsorted(a, np.array([2.0, 5.0]), side='right')
"#,
    );
}

#[test]
fn partition_kth_separates() {
    // After partition(a, k), every element before index k is <= a[k] and every
    // element after is >= a[k]. Compare by sorting both partitions, which is
    // implementation-independent.
    assert_same(
        r#"
import numpy as np
a = np.array([7.0, 1.0, 5.0, 2.0, 9.0, 3.0, 8.0, 4.0, 6.0])
p = np.partition(a, 3)
result = np.array([np.sort(p[:3]), np.sort(p[3:])][0]) if False else np.concatenate([np.sort(p[:3]), p[3:4], np.sort(p[4:])])
"#,
    );
}

#[test]
fn lexsort_orders_by_last_key_first() {
    // Tie-break on lexsort: last key is primary.
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 5.0, 1.0, 4.0, 3.0])
b = np.array([9.0, 2.0, 0.0, 4.0, 0.0])
# Primary: a. Tie-breaker: b.
result = np.lexsort([b, a])
"#,
    );
}

#[test]
fn sort_along_axes() {
    assert_same(
        r#"
import numpy as np
a = np.array([[3.0, 1.0, 2.0], [9.0, 7.0, 8.0]])
result = np.sort(a, axis=1)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([[3.0, 1.0, 2.0], [9.0, 7.0, 8.0]])
result = np.sort(a, axis=0)
"#,
    );
}

// --- polynomial ---

#[test]
fn polyval_horner() {
    // p(x) = 2x^2 + 3x + 1, evaluated at multiple x.
    assert_same(
        r#"
import numpy as np
p = np.array([2.0, 3.0, 1.0])
result = np.polyval(p, np.array([0.0, 1.0, 2.0, -1.0]))
"#,
    );
}

#[test]
fn polyder_first_and_second() {
    assert_same(
        r#"
import numpy as np
p = np.array([1.0, -2.0, 3.0, -4.0, 5.0])  # x^4 - 2x^3 + 3x^2 - 4x + 5
result = np.polyder(p)
"#,
    );
    assert_same(
        r#"
import numpy as np
p = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
result = np.polyder(p, m=2)
"#,
    );
}

#[test]
fn polyint_with_constant() {
    assert_same(
        r#"
import numpy as np
p = np.array([3.0, 2.0, 1.0])  # 3x^2 + 2x + 1
# Integral with C=0 is x^3 + x^2 + x.
result = np.polyint(p)
"#,
    );
}

// --- nan-aware reductions ---

#[test]
fn nansum_ignores_nans() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, np.nan, 4.0, np.nan, 5.0])
result = np.array([np.nansum(a), np.nanmean(a), np.nanmin(a), np.nanmax(a)])
"#,
    );
}

#[test]
fn nan_reductions_axis() {
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, 6.0]])
result = np.nansum(a, axis=1)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, 6.0]])
result = np.nanmean(a, axis=0)
"#,
    );
}

#[test]
fn nan_to_num_replaces() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, np.nan, np.inf, -np.inf, 2.0])
result = np.nan_to_num(a, nan=0.0, posinf=1e10, neginf=-1e10)
"#,
    );
}

// --- quantile / percentile ---

#[test]
fn percentile_basic() {
    assert_same_eps(
        r#"
import numpy as np
a = np.arange(11.0)
result = np.percentile(a, np.array([0.0, 25.0, 50.0, 75.0, 100.0]))
"#,
        1e-9,
    );
}

#[test]
fn quantile_axis() {
    assert_same_eps(
        r#"
import numpy as np
a = np.array([[1.0, 7.0, 4.0], [3.0, 2.0, 8.0]])
result = np.quantile(a, 0.5, axis=1)
"#,
        1e-9,
    );
}

// --- FFT helpers ---

#[test]
fn fftshift_round_trip() {
    assert_same(
        r#"
import numpy as np
a = np.arange(8.0)
result = np.ifftshift(np.fftshift(a))
"#,
    );
}

#[test]
fn fftfreq_even_and_odd() {
    assert_same_eps(
        r#"
import numpy as np
result = np.fftfreq(8, d=0.5)
"#,
        1e-12,
    );
    assert_same_eps(
        r#"
import numpy as np
result = np.fftfreq(9)
"#,
        1e-12,
    );
}

// --- modf / frexp / ldexp ---

#[test]
fn modf_splits_fractional_and_integer() {
    // modf returns a 2-tuple. Compare via stacking so the helper sees an array.
    assert_same(
        r#"
import numpy as np
a = np.array([1.25, -2.75, 3.0, -0.5])
frac, whole = np.modf(a)
result = np.stack([frac, whole])
"#,
    );
}

#[test]
fn ldexp_inverse_of_frexp_on_floats() {
    assert_same(
        r#"
import numpy as np
a = np.array([0.75, 1.5, -2.25, 10.0])
m, e = np.frexp(a)
result = np.ldexp(m, e)
"#,
    );
}

// --- gcd / lcm ---

#[test]
fn gcd_lcm_pairs() {
    assert_same(
        r#"
import numpy as np
a = np.array([12, 18, 9, 100], dtype=np.int64)
b = np.array([8, 24, 3, 75], dtype=np.int64)
result = np.stack([np.gcd(a, b), np.lcm(a, b)])
"#,
    );
}

// --- gradient / diff / interp ---

#[test]
fn gradient_uniform_spacing() {
    assert_same_eps(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 4.0, 7.0, 11.0, 16.0])
result = np.gradient(a)
"#,
        1e-9,
    );
}

#[test]
fn diff_axis_and_n() {
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, 2.0, 4.0, 7.0],
              [10.0, 14.0, 19.0, 25.0]])
result = np.diff(a, axis=1)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
result = np.diff(a, n=2)
"#,
    );
}

#[test]
fn interp_linear() {
    assert_same_eps(
        r#"
import numpy as np
xp = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
fp = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
x = np.array([0.5, 1.5, 2.5, 3.5])
result = np.interp(x, xp, fp)
"#,
        1e-12,
    );
}

// --- histogram / bincount ---

#[test]
fn histogram_with_bin_edges() {
    // Compare the counts only (the second return is the edges).
    assert_same(
        r#"
import numpy as np
a = np.array([0.1, 0.5, 0.9, 1.2, 1.7, 2.0, 2.5, 2.9])
counts, _edges = np.histogram(a, bins=np.array([0.0, 1.0, 2.0, 3.0]))
result = counts.astype(np.float64)
"#,
    );
}

#[test]
fn bincount_with_weights() {
    assert_same(
        r#"
import numpy as np
ints = np.array([0, 1, 1, 2, 2, 2, 3], dtype=np.int64)
weights = np.array([0.5, 1.0, 2.0, 0.25, 0.25, 0.5, 4.0])
result = np.bincount(ints, weights=weights)
"#,
    );
}

// --- where / clip / select-ish ---

#[test]
fn clip_broadcasts() {
    assert_same(
        r#"
import numpy as np
a = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
result = np.clip(a, -1.0, 1.5)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.arange(6.0).reshape(2, 3)
lo = np.array([0.0, 1.0, 2.0])
hi = np.array([2.0, 3.0, 4.0])
result = np.clip(a, lo, hi)
"#,
    );
}

#[test]
fn where_three_arg() {
    assert_same(
        r#"
import numpy as np
a = np.arange(6.0)
result = np.where(a % 2 == 0, a, -a)
"#,
    );
}

// --- nonzero / argwhere / flatnonzero ---

#[test]
fn nonzero_and_argwhere_agreement() {
    // nonzero returns a tuple of arrays; argwhere returns shape (N, ndim).
    // Stack the nonzero arrays to get a comparable 2D result.
    assert_same(
        r#"
import numpy as np
a = np.array([[0.0, 1.0, 0.0],
              [2.0, 0.0, 3.0]])
rows, cols = np.nonzero(a)
result = np.stack([rows.astype(np.float64), cols.astype(np.float64)])
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([0.0, 0.0, 1.0, 0.0, 2.0, 3.0])
result = np.flatnonzero(a).astype(np.float64)
"#,
    );
}

// --- pad ---

#[test]
fn pad_constant_and_edge() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
result = np.pad(a, (2, 1), mode='constant', constant_values=0.0)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, 2.0], [3.0, 4.0]])
result = np.pad(a, 1, mode='edge')
"#,
    );
}

// --- unwrap ---

#[test]
fn unwrap_phase() {
    assert_same_eps(
        r#"
import numpy as np
# Phase with a 2*pi jump that unwrap should remove.
phase = np.array([0.0, 1.0, 2.0, -2.0, -1.0, 0.0])
result = np.unwrap(phase)
"#,
        1e-9,
    );
}

// --- deg2rad / rad2deg ---

#[test]
fn deg2rad_rad2deg_inverse() {
    assert_same_eps(
        r#"
import numpy as np
d = np.array([0.0, 30.0, 45.0, 60.0, 90.0, 180.0, 360.0])
result = np.rad2deg(np.deg2rad(d))
"#,
        1e-12,
    );
}

// --- matrix_power / det / slogdet / trace ---

#[test]
fn matrix_power_squared() {
    assert_same_eps(
        r#"
import numpy as np
A = np.array([[1.0, 2.0], [3.0, 4.0]])
result = np.linalg.matrix_power(A, 3)
"#,
        1e-9,
    );
}

#[test]
fn det_and_trace() {
    assert_same_eps(
        r#"
import numpy as np
A = np.array([[2.0, 0.0, 1.0],
              [0.0, 3.0, 0.0],
              [4.0, 0.0, 5.0]])
result = np.array([np.linalg.det(A), np.trace(A)])
"#,
        1e-9,
    );
}

#[test]
fn slogdet_positive() {
    // Positive determinant: sign should be 1.
    assert_same_eps(
        r#"
import numpy as np
A = np.array([[3.0, 1.0], [2.0, 4.0]])  # det = 10
sign, logabs = np.linalg.slogdet(A)
result = np.array([sign, logabs])
"#,
        1e-9,
    );
}

// --- outer / inner / vdot ---

#[test]
fn outer_inner_vdot() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0])
result = np.outer(a, b)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
result = np.array([np.inner(a, b), np.vdot(a, b)])
"#,
    );
}

// --- broadcast_to / broadcast_arrays ---

#[test]
fn broadcast_to_explicit_shape() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
result = np.broadcast_to(a, (2, 3))
"#,
    );
}

// --- meshgrid / indices ---

#[test]
fn meshgrid_xy_indexing() {
    assert_same(
        r#"
import numpy as np
x = np.array([1.0, 2.0, 3.0])
y = np.array([10.0, 20.0])
X, Y = np.meshgrid(x, y)
result = np.stack([X, Y])
"#,
    );
}

#[test]
fn indices_grid() {
    assert_same(
        r#"
import numpy as np
result = np.indices((2, 3)).astype(np.float64)
"#,
    );
}
