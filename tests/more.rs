//! Cross-validation tests for the third-tier numpy API.

use approx::assert_abs_diff_eq;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyList, PyModule};
use rustpython_vm::{AsObject, Interpreter, builtins::PyList as RpyList};

fn rumpy_interp() -> Interpreter {
    let b = Interpreter::builder(Default::default());
    let def = rumpy::module_def(&b.ctx);
    b.add_native_module(def).build()
}

#[derive(Debug)]
struct Out {
    shape: Vec<usize>,
    data: Vec<f64>,
}

fn rumpy_run(source: &str) -> Out {
    let interp = rumpy_interp();
    interp
        .enter(|vm| -> Result<Out, String> {
            let scope = vm.new_scope_with_builtins();
            let code = vm
                .compile(source, rustpython_vm::compiler::Mode::Exec, "<t>".into())
                .map_err(|e| format!("compile: {e}"))?;
            vm.run_code_obj(code, scope.clone()).map_err(|e| pyerr(vm, &e))?;
            let r = scope.globals.get_item("result", vm).expect("set result");
            extract(&r, vm).map_err(|e| pyerr(vm, &e))
        })
        .unwrap_or_else(|e| panic!("rumpy: {e}\n--- src ---\n{source}"))
}

fn pyerr(
    vm: &rustpython_vm::VirtualMachine,
    e: &rustpython_vm::PyRef<rustpython_vm::builtins::PyBaseException>,
) -> String {
    let mut s = String::new();
    let _ = vm.write_exception(&mut s, e);
    s
}

fn extract(
    obj: &rustpython_vm::PyObjectRef,
    vm: &rustpython_vm::VirtualMachine,
) -> rustpython_vm::PyResult<Out> {
    use rumpy::{ArraysD, DType, PyNdArray};
    if let Some(a) = obj.downcast_ref::<PyNdArray>() {
        let f = a.view().cast(DType::F64);
        let ArraysD::F64(x) = f else { unreachable!() };
        return Ok(Out {
            shape: x.shape().to_vec(),
            data: x.iter().copied().collect(),
        });
    }
    if obj.is(&vm.ctx.true_value) {
        return Ok(Out { shape: vec![], data: vec![1.0] });
    }
    if obj.is(&vm.ctx.false_value) {
        return Ok(Out { shape: vec![], data: vec![0.0] });
    }
    if let Ok(f) = obj.try_float(vm) {
        return Ok(Out { shape: vec![], data: vec![f.to_f64()] });
    }
    if let Some(l) = obj.downcast_ref::<RpyList>() {
        let mut shape = Vec::new();
        let mut data = Vec::new();
        flatten(l, &mut shape, &mut data, vm, 0)?;
        return Ok(Out { shape, data });
    }
    Err(vm.new_type_error(format!("bad result {}", obj.class().name())))
}

fn flatten(
    l: &RpyList,
    shape: &mut Vec<usize>,
    data: &mut Vec<f64>,
    vm: &rustpython_vm::VirtualMachine,
    depth: usize,
) -> rustpython_vm::PyResult<()> {
    let items = l.borrow_vec();
    if depth == shape.len() {
        shape.push(items.len());
    }
    for it in items.iter() {
        if let Some(s) = it.downcast_ref::<RpyList>() {
            flatten(s, shape, data, vm, depth + 1)?;
        } else if it.is(&vm.ctx.true_value) {
            data.push(1.0);
        } else if it.is(&vm.ctx.false_value) {
            data.push(0.0);
        } else {
            data.push(it.try_float(vm)?.to_f64());
        }
    }
    Ok(())
}

fn numpy_run(source: &str) -> Out {
    Python::attach(|py| -> PyResult<Out> {
        let g = pyo3::types::PyDict::new(py);
        let np = PyModule::import(py, "numpy")?;
        g.set_item("numpy", &np)?;
        g.set_item("np", &np)?;
        py.run(&std::ffi::CString::new(source).unwrap(), Some(&g), None)?;
        let result = g.get_item("result")?.unwrap();
        let arr = np.getattr("asarray")?.call1((result,))?;
        let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
        let flat = arr.call_method0("ravel")?.call_method0("tolist")?;
        let data: Vec<f64> = flat
            .cast::<PyList>()?
            .iter()
            .map(|x| {
                if let Ok(b) = x.extract::<bool>() {
                    Ok(if b { 1.0 } else { 0.0 })
                } else {
                    x.extract::<f64>()
                }
            })
            .collect::<PyResult<_>>()?;
        Ok(Out { shape, data })
    })
    .expect("numpy snippet failed")
}

fn assert_same(s: &str) {
    let r = rumpy_run(s);
    let n = numpy_run(s);
    assert_eq!(r.shape, n.shape, "shape mismatch:\n{s}");
    assert_eq!(r.data.len(), n.data.len(), "len mismatch:\n{s}");
    for (a, b) in r.data.iter().zip(n.data.iter()) {
        if a.is_nan() && b.is_nan() {
            continue;
        }
        assert_abs_diff_eq!(*a, *b, epsilon = 1e-7);
    }
}

// ---- flip / roll / rot90 ----

#[test]
fn flip_family() {
    assert_same(
        r#"
import numpy as np
a = np.arange(6).reshape(2, 3)
result = np.flip(a)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.arange(6).reshape(2, 3)
result = np.flipud(a)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.arange(6).reshape(2, 3)
result = np.fliplr(a)
"#,
    );
}

#[test]
fn roll_op() {
    assert_same(
        r#"
import numpy as np
a = np.arange(6)
result = np.roll(a, 2)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.arange(6)
result = np.roll(a, -2)
"#,
    );
}

#[test]
fn rot90_op() {
    assert_same(
        r#"
import numpy as np
a = np.arange(6).reshape(2, 3)
result = np.rot90(a)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.arange(6).reshape(2, 3)
result = np.rot90(a, 2)
"#,
    );
}

// ---- column_stack / dstack ----

#[test]
fn column_dstack() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
result = np.column_stack([a, b])
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.array([[5.0, 6.0], [7.0, 8.0]])
result = np.dstack([a, b])
"#,
    );
}

// ---- diag / triu / tril / tri ----

#[test]
fn diag_family() {
    assert_same(
        r#"
import numpy as np
a = np.arange(9).reshape(3, 3)
result = np.diag(a)
"#,
    );
    assert_same(
        r#"
import numpy as np
result = np.diag(np.array([1.0, 2.0, 3.0]))
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.arange(16).reshape(4, 4)
result = np.triu(a)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.arange(16).reshape(4, 4)
result = np.tril(a)
"#,
    );
    assert_same(
        r#"
import numpy as np
result = np.tri(4, dtype="float64")
"#,
    );
}

// ---- atleast_Nd ----

#[test]
fn atleast_dims() {
    assert_same(
        r#"
import numpy as np
result = np.atleast_2d(np.array([1.0, 2.0, 3.0]))
"#,
    );
}

// ---- count_nonzero / bincount ----

#[test]
fn count_and_bin() {
    assert_same(
        r#"
import numpy as np
a = np.array([0, 1, 0, 2, 0, 3])
result = np.array([int(np.count_nonzero(a))])
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([0, 1, 1, 2, 2, 2, 3])
result = np.bincount(a)
"#,
    );
}

// ---- nan reductions ----

#[test]
fn nan_reductions() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, float('nan'), 4.0])
result = np.array([float(np.nansum(a)), float(np.nanmean(a)),
                   float(np.nanmin(a)), float(np.nanmax(a)),
                   float(np.nanmedian(a))])
"#,
    );
}

// ---- searchsorted / meshgrid / interp / trapz / gradient ----

#[test]
fn searchsorted_op() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
v = np.array([0.5, 2.5, 4.0, 10.0])
result = np.searchsorted(a, v).astype(float)
"#,
    );
}

#[test]
fn meshgrid_op() {
    assert_same(
        r#"
import numpy as np
x = np.array([1.0, 2.0, 3.0])
y = np.array([10.0, 20.0])
xx, yy = np.meshgrid(x, y)
result = xx + yy
"#,
    );
}

#[test]
fn interp_linear() {
    assert_same(
        r#"
import numpy as np
xp = np.array([0.0, 1.0, 2.0, 3.0])
fp = np.array([0.0, 2.0, 4.0, 6.0])
x = np.array([0.5, 1.5, 2.5, -1.0, 5.0])
result = np.interp(x, xp, fp)
"#,
    );
}

#[test]
fn trapz_op() {
    // numpy 2.0 renamed trapz → trapezoid; use the new name.
    assert_same(
        r#"
import numpy as np
y = np.array([1.0, 2.0, 3.0, 4.0])
result = np.array([float(np.trapezoid(y))])
"#,
    );
}

#[test]
fn gradient_1d() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
result = np.gradient(a)
"#,
    );
}

// ---- append / delete ----

#[test]
fn append_delete() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0])
result = np.append(a, b)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.arange(6.0)
result = np.delete(a, 2)
"#,
    );
}

// ---- linalg.cholesky / qr ----

#[test]
fn cholesky_roundtrip() {
    // Cholesky returns L; we verify by reconstructing A.
    assert_same(
        r#"
import numpy as np
A = np.array([[4.0, 2.0, 0.0], [2.0, 5.0, 1.0], [0.0, 1.0, 3.0]])
L = np.linalg.cholesky(A)
result = L @ L.T
"#,
    );
}

#[test]
fn qr_roundtrip() {
    // Verify Q @ R reconstructs A.
    assert_same(
        r#"
import numpy as np
A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
Q, R = np.linalg.qr(A)
result = Q @ R
"#,
    );
}

// ---- operator overloads on arrays ----

#[test]
fn comparison_operators_on_array() {
    assert_same(
        r#"
import numpy as np
a = np.arange(5)
result = (a > 2).astype(int)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0, 2.0, 4.0])
result = (a == b).astype(int)
"#,
    );
}

#[test]
fn xor_operator() {
    assert_same(
        r#"
import numpy as np
a = np.array([0b1100, 0b1010], dtype="int32")
b = np.array([0b1010, 0b1111], dtype="int32")
result = a ^ b
"#,
    );
}

// --- inplace operators ---

#[test]
fn iadd_array() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
a += np.array([10.0, 20.0, 30.0])
result = a
"#,
    );
}

#[test]
fn iadd_scalar() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
a += 5.0
result = a
"#,
    );
}

#[test]
fn isub_array() {
    assert_same(
        r#"
import numpy as np
a = np.arange(6.0).reshape(2, 3)
a -= np.array([1.0, 1.0, 1.0])
result = a
"#,
    );
}

#[test]
fn imul_scalar() {
    assert_same(
        r#"
import numpy as np
a = np.arange(5.0)
a *= 3.0
result = a
"#,
    );
}

#[test]
fn idiv_array() {
    assert_same(
        r#"
import numpy as np
a = np.array([6.0, 12.0, 24.0])
a /= np.array([2.0, 3.0, 4.0])
result = a
"#,
    );
}

#[test]
fn ifloordiv() {
    assert_same(
        r#"
import numpy as np
a = np.array([7, 11, 13], dtype="int32")
a //= 2
result = a
"#,
    );
}

#[test]
fn imod() {
    assert_same(
        r#"
import numpy as np
a = np.array([7, 11, 13], dtype="int32")
a %= 4
result = a
"#,
    );
}

#[test]
fn ipow() {
    assert_same(
        r#"
import numpy as np
a = np.array([2.0, 3.0, 4.0])
a **= 2
result = a
"#,
    );
}

#[test]
fn iand_ior_ixor() {
    assert_same(
        r#"
import numpy as np
a = np.array([0b1100, 0b1010], dtype="int32")
a &= np.array([0b1010, 0b1111], dtype="int32")
result = a
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([0b1100, 0b1010], dtype="int32")
a |= np.array([0b0011, 0b0101], dtype="int32")
result = a
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([0b1100, 0b1010], dtype="int32")
a ^= np.array([0b1010, 0b1111], dtype="int32")
result = a
"#,
    );
}

// --- new ndarray methods ---

#[test]
fn method_squeeze() {
    assert_same(
        r#"
import numpy as np
a = np.zeros((1, 3, 1, 4))
result = a.squeeze()
"#,
    );
}

#[test]
fn method_swapaxes() {
    assert_same(
        r#"
import numpy as np
a = np.arange(24.0).reshape(2, 3, 4)
result = a.swapaxes(0, 2)
"#,
    );
}

#[test]
fn method_diagonal() {
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
result = a.diagonal()
"#,
    );
}

#[test]
fn method_trace() {
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
result = np.array([a.trace()])
"#,
    );
}

#[test]
fn method_clip() {
    assert_same(
        r#"
import numpy as np
a = np.array([-2.0, 0.0, 1.5, 3.0, 5.0])
result = a.clip(0.0, 3.0)
"#,
    );
}

#[test]
fn method_round() {
    assert_same(
        r#"
import numpy as np
a = np.array([0.4, 0.6, 1.5, -0.5, -0.4])
result = a.round()
"#,
    );
}

#[test]
fn method_cumsum_cumprod() {
    assert_same(
        r#"
import numpy as np
a = np.arange(1.0, 5.0)
result = a.cumsum()
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.arange(1.0, 5.0)
result = a.cumprod()
"#,
    );
}

#[test]
fn method_ptp() {
    // numpy 2.x removed `arr.ptp()` — we still keep the method on our side
    // (it's a useful convenience) and test it explicitly.
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, 5.0, 3.0, 8.0, 2.0])
result = np.array([a.ptp()])
"#,
    );
    assert_eq!(r.data, vec![7.0]);
}

#[test]
fn method_any_all() {
    assert_same(
        r#"
import numpy as np
a = np.array([0.0, 1.0, 2.0])
result = np.array([1.0 if a.any() else 0.0])
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
result = np.array([1.0 if a.all() else 0.0])
"#,
    );
}

#[test]
fn method_argsort() {
    assert_same(
        r#"
import numpy as np
a = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0])
result = a.argsort()
"#,
    );
}

#[test]
fn method_sort_inplace() {
    assert_same(
        r#"
import numpy as np
a = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0])
a.sort()
result = a
"#,
    );
}

#[test]
fn method_fill() {
    assert_same(
        r#"
import numpy as np
a = np.zeros((3, 4))
a.fill(7.5)
result = a
"#,
    );
}

#[test]
fn method_item() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([42.5])
result = np.array([a.item()])
"#,
    );
    assert!((r.data[0] - 42.5).abs() < 1e-10);
}

#[test]
fn method_repeat_tile() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
result = a.repeat(2)
"#,
    );
    // numpy 2.x has no .tile() method (only np.tile). We expose `.tile` as a
    // rumpy extension; verify it directly.
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
result = a.tile(3)
"#,
    );
    assert_eq!(r.data, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
}

#[test]
fn method_view_returns_copy() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
result = a.view()
"#,
    );
    assert_eq!(r.data, vec![1.0, 2.0, 3.0]);
}

#[test]
fn method_flat_is_1d() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.arange(6.0).reshape(2, 3)
result = a.flat
"#,
    );
    assert_eq!(r.shape, vec![6]);
}

// --- keepdims & tuple axis on reductions ---

#[test]
fn sum_keepdims() {
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
result = a.sum(axis=1, keepdims=True)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
result = a.sum(axis=0, keepdims=True)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.arange(24.0).reshape(2, 3, 4)
result = a.sum(keepdims=True)
"#,
    );
}

#[test]
fn mean_keepdims() {
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
result = a.mean(axis=1, keepdims=True)
"#,
    );
}

#[test]
fn sum_tuple_axis() {
    assert_same(
        r#"
import numpy as np
a = np.arange(24.0).reshape(2, 3, 4)
result = a.sum(axis=(0, 2))
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.arange(24.0).reshape(2, 3, 4)
result = a.sum(axis=(0, 1))
"#,
    );
}

#[test]
fn mean_tuple_axis() {
    assert_same(
        r#"
import numpy as np
a = np.arange(24.0).reshape(2, 3, 4)
result = a.mean(axis=(1, 2))
"#,
    );
}

#[test]
fn max_min_keepdims() {
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])
result = a.max(axis=1, keepdims=True)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])
result = a.min(axis=0, keepdims=True)
"#,
    );
}

#[test]
fn any_all_tuple_axis() {
    assert_same(
        r#"
import numpy as np
a = np.array([[[True, False], [True, True]], [[False, False], [True, True]]])
result = np.any(a, axis=(0, 2)).astype(int)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([[[True, False], [True, True]], [[True, True], [True, True]]])
result = np.all(a, axis=(1,)).astype(int)
"#,
    );
}

#[test]
fn any_all_keepdims() {
    assert_same(
        r#"
import numpy as np
a = np.array([[True, False, True], [False, True, True]])
result = np.any(a, axis=1, keepdims=True).astype(int)
"#,
    );
}

#[test]
fn var_std_keepdims() {
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
result = a.std(axis=1, keepdims=True)
"#,
    );
}

#[test]
fn iadd_preserves_dtype() {
    // int32 += int python scalar — must stay int32.
    assert_same(
        r#"
import numpy as np
a = np.array([1, 2, 3], dtype="int32")
a += 5
result = a
"#,
    );
    // int16 += int16 array — stays int16.
    assert_same(
        r#"
import numpy as np
a = np.array([1, 2, 3], dtype="int16")
a += np.array([10, 20, 30], dtype="int16")
result = a
"#,
    );
}

// =====================================================================
// Expanded test coverage
// =====================================================================

// ---- broadcasting edge cases ----

#[test]
fn broadcast_scalar_plus_array() {
    assert_same(
        r#"
import numpy as np
a = np.arange(6.0).reshape(2, 3)
result = a + 100.0
"#,
    );
}

#[test]
fn broadcast_row_and_col() {
    assert_same(
        r#"
import numpy as np
a = np.arange(3.0).reshape(1, 3)
b = np.arange(2.0).reshape(2, 1)
result = a + b
"#,
    );
}

#[test]
fn broadcast_3d_with_1d() {
    assert_same(
        r#"
import numpy as np
a = np.arange(24.0).reshape(2, 3, 4)
b = np.arange(4.0)
result = a * b
"#,
    );
}

#[test]
fn broadcast_compatible_singleton() {
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
b = np.ones((3, 1))
result = a / b
"#,
    );
}

// ---- empty array operations ----

#[test]
fn empty_sum() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([], dtype="float64")
result = np.array([np.sum(a)])
"#,
    );
    assert_eq!(r.data, vec![0.0]);
}

#[test]
fn empty_prod() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([], dtype="float64")
result = np.array([np.prod(a)])
"#,
    );
    assert_eq!(r.data, vec![1.0]);
}

#[test]
fn empty_concatenate() {
    assert_same(
        r#"
import numpy as np
a = np.arange(3.0)
b = np.array([], dtype="float64")
result = np.concatenate([a, b])
"#,
    );
}

// ---- nan propagation ----

#[test]
fn nan_propagates_through_arithmetic() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, float("nan"), 3.0])
b = a + 1.0
result = b
"#,
    );
    assert_eq!(r.data[0], 2.0);
    assert!(r.data[1].is_nan());
    assert_eq!(r.data[2], 4.0);
}

#[test]
fn nan_breaks_mean() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, float("nan"), 3.0])
result = np.array([float(a.mean())])
"#,
    );
    assert!(r.data[0].is_nan());
}

#[test]
fn nansum_skips_nans() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, float("nan"), 3.0, 5.0, float("nan")])
result = np.array([float(np.nansum(a))])
"#,
    );
}

// ---- complex arithmetic ----

#[test]
fn complex_add_mul() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1+1j, 2+0j, 0+3j])
b = a * (1+1j)
result = np.real(b)
"#,
    );
    // (1+1j)*(1+1j) = 2j, real=0
    // (2)*(1+1j) = 2+2j, real=2
    // (3j)*(1+1j) = -3+3j, real=-3
    assert!((r.data[0] - 0.0).abs() < 1e-10);
    assert!((r.data[1] - 2.0).abs() < 1e-10);
    assert!((r.data[2] - -3.0).abs() < 1e-10);
}

#[test]
fn complex_abs_returns_float() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([3+4j, 0+0j, 1+0j])
result = np.abs(a)
"#,
    );
    assert!((r.data[0] - 5.0).abs() < 1e-9);
    assert!((r.data[1] - 0.0).abs() < 1e-9);
    assert!((r.data[2] - 1.0).abs() < 1e-9);
}

#[test]
fn complex_conjugate() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1+2j, 3-4j])
b = np.conj(a)
result = np.array([b[0].imag, b[1].imag])
"#,
    );
    assert!((r.data[0] - -2.0).abs() < 1e-9);
    assert!((r.data[1] - 4.0).abs() < 1e-9);
}

// ---- reversed / negative-step slicing ----

#[test]
fn reverse_slice_1d() {
    assert_same(
        r#"
import numpy as np
a = np.arange(10.0)
result = a[::-1]
"#,
    );
}

#[test]
fn reverse_slice_2d_axis_0() {
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
result = a[::-1]
"#,
    );
}

#[test]
fn step_2_slice() {
    assert_same(
        r#"
import numpy as np
a = np.arange(10.0)
result = a[1::2]
"#,
    );
}

// ---- mixed integer + slice + newaxis indexing ----

#[test]
fn mixed_int_slice_newaxis() {
    assert_same(
        r#"
import numpy as np
a = np.arange(24.0).reshape(2, 3, 4)
result = a[1, :, None, 2]
"#,
    );
}

#[test]
fn negative_int_with_slice() {
    assert_same(
        r#"
import numpy as np
a = np.arange(24.0).reshape(2, 3, 4)
result = a[-1, ::-1]
"#,
    );
}

// ---- predicate operations ----

#[test]
fn where_op() {
    assert_same(
        r#"
import numpy as np
a = np.arange(10.0)
result = np.where(a > 5, a, -a)
"#,
    );
}

#[test]
fn select_with_compound_condition() {
    assert_same(
        r#"
import numpy as np
a = np.arange(10.0)
mask = (a > 2) & (a < 8)
result = a[mask]
"#,
    );
}

// ---- searchsorted ----

#[test]
fn searchsorted_basic() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
v = np.array([0.0, 4.0, 10.0, 6.0])
result = np.searchsorted(a, v)
"#,
    );
}

// ---- ops with python scalars ----

#[test]
fn array_minus_python_int() {
    assert_same(
        r#"
import numpy as np
a = np.arange(6.0)
result = a - 3
"#,
    );
}

#[test]
fn python_int_div_array() {
    assert_same(
        r#"
import numpy as np
a = np.array([2.0, 4.0, 8.0])
result = 16 / a
"#,
    );
}

// ---- reduction return types ----

#[test]
fn argmin_argmax_1d() {
    assert_same(
        r#"
import numpy as np
a = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
result = np.array([np.argmin(a), np.argmax(a)])
"#,
    );
}

// ---- where with array branches ----

#[test]
fn where_array_branches() {
    assert_same(
        r#"
import numpy as np
a = np.arange(10.0)
b = -a
mask = a > 5
result = np.where(mask, a, b)
"#,
    );
}

// ---- nonzero ----

#[test]
fn nonzero_1d_flat_indices() {
    // `np.nonzero(a)` returns a tuple of per-axis index arrays — for 1-D
    // input it's a 1-tuple. Unwrap to match numpy's shape.
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0])
result = np.nonzero(a)[0].astype("float64")
"#,
    );
    assert_eq!(r.data, vec![1.0, 3.0, 5.0]);
}

// ---- integer overflow / wrap-around ----

#[test]
fn int8_overflow_wraps() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([100, 50], dtype="int8")
b = a + a  # 200 overflows int8 -> -56 (wrap)
result = b.astype("int64")
"#,
    );
    // Expect wrap: 100+100 = 200 wraps to -56; 50+50 = 100
    assert_eq!(r.data, vec![-56.0, 100.0]);
}

// ---- FFT round-trips ----

#[test]
fn fft_ifft_roundtrip() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0, 4.0])
b = np.fft.fft(a)
c = np.fft.ifft(b)
result = np.real(c)
"#,
    );
    let expected = [1.0, 2.0, 3.0, 4.0];
    for (a, b) in r.data.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-9, "ifft(fft(x)) != x: {a} vs {b}");
    }
}

#[test]
fn rfft_irfft_roundtrip() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
b = np.fft.rfft(a)
c = np.fft.irfft(b)
result = c
"#,
    );
    let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    for (a, b) in r.data.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-9, "irfft(rfft(x)) != x: {a} vs {b}");
    }
}

// ---- einsum patterns ----

#[test]
fn einsum_trace() {
    assert_same(
        r#"
import numpy as np
a = np.arange(9.0).reshape(3, 3)
result = np.array([np.einsum('ii', a)])
"#,
    );
}

#[test]
fn einsum_matmul() {
    assert_same(
        r#"
import numpy as np
a = np.arange(6.0).reshape(2, 3)
b = np.arange(12.0).reshape(3, 4)
result = np.einsum('ij,jk->ik', a, b)
"#,
    );
}

#[test]
fn einsum_inner_product() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
result = np.array([np.einsum('i,i->', a, b)])
"#,
    );
}

// ---- mixed-dtype broadcasting and promotion ----

#[test]
fn add_int_and_float_arrays() {
    assert_same(
        r#"
import numpy as np
a = np.array([1, 2, 3], dtype="int32")
b = np.array([0.5, 1.5, 2.5], dtype="float64")
result = a + b
"#,
    );
}

// ---- cumulative reductions on 2D ----

#[test]
fn cumsum_axis_0_2d() {
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
result = np.cumsum(a, axis=0)
"#,
    );
}

#[test]
fn cumprod_axis_1_2d() {
    assert_same(
        r#"
import numpy as np
a = np.arange(1.0, 13.0).reshape(3, 4)
result = np.cumprod(a, axis=1)
"#,
    );
}

// ---- diff family ----

#[test]
fn diff_2d() {
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, 3.0, 6.0], [10.0, 15.0, 21.0]])
result = np.diff(a)
"#,
    );
}

// ---- transpose / reshape interaction ----

#[test]
fn transpose_then_reshape() {
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
result = a.T.reshape(2, 6)
"#,
    );
}

// ---- division by zero ----

#[test]
fn division_by_zero_returns_inf() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, -1.0, 0.0])
b = np.array([0.0, 0.0, 0.0])
result = a / b
"#,
    );
    assert!(r.data[0].is_infinite() && r.data[0] > 0.0);
    assert!(r.data[1].is_infinite() && r.data[1] < 0.0);
    assert!(r.data[2].is_nan());
}

// ---- float16 conversions ----

#[test]
fn float16_arithmetic() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0], dtype="float16")
b = np.array([0.5, 0.5, 0.5], dtype="float16")
result = (a + b).astype("float64")
"#,
    );
}

// ---- arr equality returns bool array ----

#[test]
fn equality_op_returns_bool_array() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0, 4.0])
b = np.array([1.0, 5.0, 3.0, 7.0])
result = (a == b).astype(int)
"#,
    );
}

// ---- ascontiguous / from list of lists / tuple shape ----

#[test]
fn array_from_jagged_python_list_2d() {
    assert_same(
        r#"
import numpy as np
data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
result = np.array(data)
"#,
    );
}

#[test]
fn array_from_tuple_shape() {
    assert_same(
        r#"
import numpy as np
result = np.zeros((2, 3, 4))
"#,
    );
}

// ---- linspace endpoint=False ----

#[test]
fn linspace_no_endpoint() {
    assert_same(
        r#"
import numpy as np
result = np.linspace(0.0, 1.0, 5, endpoint=False)
"#,
    );
}

// ---- arange with float step ----

#[test]
fn arange_float_step() {
    assert_same(
        r#"
import numpy as np
result = np.arange(0.0, 2.0, 0.25)
"#,
    );
}

// ---- sort on 2D matrix along axis ----

#[test]
fn sort_2d_axis_0() {
    assert_same(
        r#"
import numpy as np
a = np.array([[3.0, 1.0, 4.0], [1.0, 5.0, 9.0], [2.0, 6.0, 5.0]])
result = np.sort(a, axis=0)
"#,
    );
}

#[test]
fn sort_2d_axis_minus_1() {
    assert_same(
        r#"
import numpy as np
a = np.array([[3.0, 1.0, 4.0], [1.0, 5.0, 9.0], [2.0, 6.0, 5.0]])
result = np.sort(a, axis=-1)
"#,
    );
}

// ---- median ----

#[test]
fn median_flat() {
    // rumpy's median is flat-only (axis= kwarg not yet implemented).
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 3.0, 5.0, 2.0, 4.0, 6.0])
result = np.array([float(np.median(a))])
"#,
    );
}

// ---- clip with scalar bounds ----

#[test]
fn clip_function() {
    assert_same(
        r#"
import numpy as np
a = np.linspace(-3.0, 3.0, 7)
result = np.clip(a, -1.0, 1.0)
"#,
    );
}

// ---- unique drops duplicates and sorts ----

#[test]
fn unique_drops_duplicates() {
    assert_same(
        r#"
import numpy as np
a = np.array([3.0, 1.0, 2.0, 1.0, 3.0, 4.0])
result = np.unique(a)
"#,
    );
}

// =====================================================================
// Round 2: deep operator semantics
// =====================================================================

// ---- comparison operators ----

#[test]
fn eq_ne_lt_le_gt_ge() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0, 4.0])
b = np.array([1.0, 3.0, 2.0, 4.0])
result = (a == b).astype(int) + 2 * (a != b).astype(int)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0, 4.0])
b = np.array([1.0, 3.0, 2.0, 4.0])
result = (a < b).astype(int) + 2 * (a <= b).astype(int) + 4 * (a > b).astype(int) + 8 * (a >= b).astype(int)
"#,
    );
}

// ---- unary negate / positive / abs ----

#[test]
fn unary_neg_pos_abs() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, -2.0, 3.0])
result = -a
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, -2.0, 3.0])
result = +a
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, -2.0, 3.0])
result = abs(a)
"#,
    );
}

// ---- bitwise on bool ----

#[test]
fn bool_and_or() {
    assert_same(
        r#"
import numpy as np
a = np.array([True, True, False, False])
b = np.array([True, False, True, False])
result = (a & b).astype(int) + 2 * (a | b).astype(int) + 4 * (a ^ b).astype(int)
"#,
    );
}

// ---- floor / ceil / trunc / rint ----

#[test]
fn rounding_family() {
    assert_same(
        r#"
import numpy as np
a = np.array([-1.7, -0.5, 0.5, 1.7, 2.5])
result = np.array([np.floor(a), np.ceil(a), np.rint(a)])
"#,
    );
}

// ---- sin/cos/tan ----

#[test]
fn trig_pi_quarter() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([0.0, np.pi/4, np.pi/2, np.pi])
result = np.sin(a)
"#,
    );
    let expected = [0.0, 2f64.sqrt() / 2.0, 1.0, 0.0];
    for (a, b) in r.data.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-9, "sin: {a} vs {b}");
    }
}

#[test]
fn cos_inverse_arcos() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([0.0, 0.5, 1.0])
result = np.cos(np.arccos(a))
"#,
    );
    let expected = [0.0, 0.5, 1.0];
    for (a, b) in r.data.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-9);
    }
}

// ---- tanh sinh cosh ----

#[test]
fn hyperbolic_family() {
    assert_same(
        r#"
import numpy as np
a = np.array([-1.0, 0.0, 1.0])
result = np.array([np.sinh(a), np.cosh(a), np.tanh(a)])
"#,
    );
}

// ---- reductions with all-positive / all-negative ----

#[test]
fn sum_all_negative() {
    assert_same(
        r#"
import numpy as np
a = np.array([-1.0, -2.0, -3.0, -4.0])
result = np.array([float(a.sum())])
"#,
    );
}

#[test]
fn prod_with_zero() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 0.0, 4.0])
result = np.array([float(a.prod())])
"#,
    );
}

// ---- ndim and size ----

#[test]
fn ndim_size_shape() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.arange(24.0).reshape(2, 3, 4)
result = np.array([a.ndim, a.size, a.shape[0], a.shape[1], a.shape[2]]).astype(float)
"#,
    );
    assert_eq!(r.data, vec![3.0, 24.0, 2.0, 3.0, 4.0]);
}

// ---- transpose with explicit axes ----

#[test]
fn transpose_attribute() {
    assert_same(
        r#"
import numpy as np
a = np.arange(6.0).reshape(2, 3)
result = a.T
"#,
    );
}

// ---- reshape with -1 ----

#[test]
fn reshape_neg_one() {
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0)
result = a.reshape(3, -1)
"#,
    );
}

#[test]
fn reshape_neg_one_first() {
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0)
result = a.reshape(-1, 4)
"#,
    );
}

// ---- repeated assignments mutate in place ----

#[test]
fn repeated_iadd() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.zeros(3)
for i in range(5):
    a += 1.0
result = a
"#,
    );
    assert_eq!(r.data, vec![5.0, 5.0, 5.0]);
}

// ---- bool indexing assignment ----

#[test]
fn bool_assign_negates_negatives() {
    assert_same(
        r#"
import numpy as np
a = np.array([-1.0, 2.0, -3.0, 4.0, -5.0])
a[a < 0] = -a[a < 0]
result = a
"#,
    );
}

// ---- chained ops preserve dtype ----

#[test]
fn chained_ops_preserve_dtype() {
    assert_same(
        r#"
import numpy as np
a = np.array([1, 2, 3, 4], dtype="int32")
b = (a * 2 + 1).astype("float64")
result = b
"#,
    );
}

// ---- floor_divide ----

#[test]
fn floor_divide_op() {
    assert_same(
        r#"
import numpy as np
a = np.array([7.0, -7.0, 10.0, -10.0])
b = np.array([3.0, 3.0, 3.0, 3.0])
result = np.floor_divide(a, b)
"#,
    );
}

// ---- modulo with sign mix ----

#[test]
fn modulo_sign_handling() {
    assert_same(
        r#"
import numpy as np
a = np.array([7.0, -7.0, 10.0, -10.0])
b = np.array([3.0, 3.0, 3.0, 3.0])
result = np.mod(a, b)
"#,
    );
}

// ---- view through arithmetic preserves shape ----

#[test]
fn shape_preserved_through_chain() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
b = (a + 1) * 2 - 3
result = b
"#,
    );
    assert_eq!(r.shape, vec![3, 4]);
}

// ---- broadcast against (1,) ----

#[test]
fn broadcast_with_size_1() {
    assert_same(
        r#"
import numpy as np
a = np.arange(6.0).reshape(2, 3)
b = np.array([10.0])
result = a + b
"#,
    );
}

// ---- multi-step reshape via T ----

#[test]
fn t_then_t_returns_original() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
result = a.T.T
"#,
    );
    let expected: Vec<f64> = (0..12).map(|i| i as f64).collect();
    assert_eq!(r.data, expected);
}

// ---- conjugate of real array is identity ----

#[test]
fn conj_real() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
result = np.conj(a)
"#,
    );
}

// ---- sign of mixed values ----

#[test]
fn sign_op() {
    assert_same(
        r#"
import numpy as np
a = np.array([-2.0, -0.0, 0.0, 0.5, 3.0])
result = np.sign(a)
"#,
    );
}

// ---- power on negative base ----

#[test]
fn pow_negative_base() {
    assert_same(
        r#"
import numpy as np
a = np.array([-2.0, 2.0])
result = a ** 3
"#,
    );
}

// ---- string-input dtype creation ----

#[test]
fn full_with_str_dtype() {
    assert_same(
        r#"
import numpy as np
result = np.full((2, 2), 7).astype("float64")
"#,
    );
}

// ---- arange typed ----

#[test]
fn arange_with_dtype_int8() {
    assert_same(
        r#"
import numpy as np
a = np.arange(0, 5, dtype="int8")
result = a.astype("float64")
"#,
    );
}

// ---- multiple chained slicings ----

#[test]
fn chained_slice_2d() {
    assert_same(
        r#"
import numpy as np
a = np.arange(24.0).reshape(4, 6)
result = a[1:3, 2:5]
"#,
    );
}

// ---- equality between integer arrays ----

#[test]
fn equality_between_int_arrays() {
    assert_same(
        r#"
import numpy as np
a = np.array([1, 2, 3, 4], dtype="int32")
b = np.array([1, 3, 3, 5], dtype="int32")
result = (a == b).astype(int)
"#,
    );
}

// ---- copy is independent ----

#[test]
fn copy_is_independent() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = a.copy()
a[0] = 99.0
result = b
"#,
    );
    assert_eq!(r.data, vec![1.0, 2.0, 3.0]);
}

// ---- ravel returns a flat array ----

#[test]
fn ravel_2d() {
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
result = a.ravel()
"#,
    );
}

// ---- arr + arr.T (square matrix) ----

#[test]
fn add_with_transpose() {
    assert_same(
        r#"
import numpy as np
a = np.arange(9.0).reshape(3, 3)
result = a + a.T
"#,
    );
}

// ---- elementwise log on large values ----

#[test]
fn log_large() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, 1000.0, 1e6])
result = np.log10(a)
"#,
    );
    let expected = [0.0, 3.0, 6.0];
    for (a, b) in r.data.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-9);
    }
}
