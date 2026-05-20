//! Cross-validation tests for advanced indexing + item assignment.

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

// --- bool-mask indexing -------------------------------------------------

#[test]
fn bool_mask_read() {
    assert_same(
        r#"
import numpy as np
a = np.arange(6.0)
mask = np.array([True, False, True, False, True, True])
result = a[mask]
"#,
    );
}

#[test]
fn bool_mask_via_comparison() {
    assert_same(
        r#"
import numpy as np
a = np.arange(10.0)
result = a[a > 5]
"#,
    );
}

// --- integer-array (fancy) indexing -------------------------------------

#[test]
fn int_array_read() {
    assert_same(
        r#"
import numpy as np
a = np.arange(10.0)
idx = np.array([0, 2, 5, 9, 9])
result = a[idx]
"#,
    );
}

#[test]
fn int_array_2d_rows() {
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0).reshape(4, 3)
idx = np.array([2, 0, 3])
result = a[idx]
"#,
    );
}

#[test]
fn list_index_works_like_array() {
    assert_same(
        r#"
import numpy as np
a = np.arange(10.0)
result = a[[1, 3, 5, 7]]
"#,
    );
}

// --- scalar item assignment ---------------------------------------------

#[test]
fn scalar_setitem_1d() {
    assert_same(
        r#"
import numpy as np
a = np.arange(5.0)
a[0] = 99.0
a[-1] = -1.0
result = a
"#,
    );
}

#[test]
fn scalar_setitem_2d() {
    assert_same(
        r#"
import numpy as np
a = np.zeros((3, 4))
a[1, 2] = 7.5
a[0, 0] = 1.0
a[2, 3] = -2.0
result = a
"#,
    );
}

// --- slice assignment ---------------------------------------------------

#[test]
fn slice_setitem_scalar_broadcast() {
    assert_same(
        r#"
import numpy as np
a = np.zeros((4, 4))
a[1:3, 1:3] = 9.0
result = a
"#,
    );
}

#[test]
fn slice_setitem_array() {
    assert_same(
        r#"
import numpy as np
a = np.zeros((3, 4))
a[1] = np.array([10.0, 20.0, 30.0, 40.0])
result = a
"#,
    );
}

// --- bool-mask assignment -----------------------------------------------

#[test]
fn bool_mask_setitem_scalar() {
    assert_same(
        r#"
import numpy as np
a = np.arange(10.0)
a[a > 5] = 0.0
result = a
"#,
    );
}

#[test]
fn bool_mask_setitem_array() {
    assert_same(
        r#"
import numpy as np
a = np.arange(6.0)
mask = np.array([True, False, True, False, True, False])
a[mask] = np.array([10.0, 30.0, 50.0])
result = a
"#,
    );
}

// --- fancy-int-array assignment -----------------------------------------

#[test]
fn fancy_int_setitem_scalar() {
    assert_same(
        r#"
import numpy as np
a = np.arange(10.0)
a[[1, 3, 5]] = -1.0
result = a
"#,
    );
}

// --- Ellipsis (`...`) -----------------------------------------------------

#[test]
fn ellipsis_full() {
    assert_same(
        r#"
import numpy as np
a = np.arange(24.0).reshape(2, 3, 4)
result = a[...]
"#,
    );
}

#[test]
fn ellipsis_trailing_int() {
    assert_same(
        r#"
import numpy as np
a = np.arange(24.0).reshape(2, 3, 4)
result = a[..., 0]
"#,
    );
}

#[test]
fn ellipsis_leading_int() {
    assert_same(
        r#"
import numpy as np
a = np.arange(24.0).reshape(2, 3, 4)
result = a[1, ...]
"#,
    );
}

#[test]
fn ellipsis_middle() {
    assert_same(
        r#"
import numpy as np
a = np.arange(120.0).reshape(2, 3, 4, 5)
result = a[1, ..., 2]
"#,
    );
}

// --- newaxis / None -------------------------------------------------------

#[test]
fn newaxis_leading() {
    assert_same(
        r#"
import numpy as np
a = np.arange(6.0)
result = a[None, :]
"#,
    );
}

#[test]
fn newaxis_trailing() {
    assert_same(
        r#"
import numpy as np
a = np.arange(6.0)
result = a[:, None]
"#,
    );
}

#[test]
fn newaxis_with_int() {
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
result = a[None, 0]
"#,
    );
}

#[test]
fn newaxis_between() {
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
result = a[:, None, :]
"#,
    );
}

#[test]
fn newaxis_with_ellipsis() {
    assert_same(
        r#"
import numpy as np
a = np.arange(24.0).reshape(2, 3, 4)
result = a[..., None]
"#,
    );
}

#[test]
fn newaxis_multi() {
    assert_same(
        r#"
import numpy as np
a = np.arange(6.0)
result = a[None, :, None]
"#,
    );
}

// --- Ellipsis in setitem -----------------------------------------------

#[test]
fn ellipsis_setitem_scalar() {
    assert_same(
        r#"
import numpy as np
a = np.zeros((2, 3, 4))
a[..., 0] = 5.0
result = a
"#,
    );
}

#[test]
fn ellipsis_setitem_full() {
    assert_same(
        r#"
import numpy as np
a = np.zeros((2, 3))
a[...] = 7.0
result = a
"#,
    );
}

// =====================================================================
// Expanded indexing coverage
// =====================================================================

#[test]
fn negative_step_2d_axis_0() {
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
result = a[::-1, :]
"#,
    );
}

#[test]
fn negative_step_2d_axis_1() {
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
result = a[:, ::-1]
"#,
    );
}

#[test]
fn step_3_slice() {
    assert_same(
        r#"
import numpy as np
a = np.arange(10.0)
result = a[::3]
"#,
    );
}

#[test]
fn boolean_index_2d_flat() {
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
mask = a > 5
result = a[mask]
"#,
    );
}

#[test]
fn newaxis_after_int() {
    // arr[i, None] preserves a length-1 axis after collapsing axis 0.
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
result = a[1, None, :]
"#,
    );
}

#[test]
fn ellipsis_alone_returns_full() {
    assert_same(
        r#"
import numpy as np
a = np.arange(24.0).reshape(2, 3, 4)
result = a[...]
"#,
    );
}

#[test]
fn negative_int_slicing() {
    assert_same(
        r#"
import numpy as np
a = np.arange(10.0)
result = a[-3:]
"#,
    );
}

#[test]
fn slice_start_only() {
    assert_same(
        r#"
import numpy as np
a = np.arange(10.0)
result = a[3:]
"#,
    );
}

#[test]
fn slice_stop_only() {
    assert_same(
        r#"
import numpy as np
a = np.arange(10.0)
result = a[:4]
"#,
    );
}

#[test]
fn slice_with_large_stop() {
    // Numpy clamps the stop index to len.
    assert_same(
        r#"
import numpy as np
a = np.arange(5.0)
result = a[1:100]
"#,
    );
}

#[test]
fn empty_slice() {
    assert_same(
        r#"
import numpy as np
a = np.arange(10.0)
result = a[5:5]
"#,
    );
}

#[test]
fn setitem_slice_writes_back() {
    assert_same(
        r#"
import numpy as np
a = np.zeros(6)
a[1:4] = np.array([1.0, 2.0, 3.0])
result = a
"#,
    );
}

#[test]
fn setitem_reverse_slice() {
    assert_same(
        r#"
import numpy as np
a = np.zeros(5)
a[::-1] = np.arange(5.0)
result = a
"#,
    );
}

// =====================================================================
// Round 2: indexing edges
// =====================================================================

#[test]
fn slice_only_step() {
    assert_same(
        r#"
import numpy as np
a = np.arange(20.0)
result = a[::4]
"#,
    );
}

#[test]
fn negative_step_with_bounds() {
    assert_same(
        r#"
import numpy as np
a = np.arange(20.0)
result = a[15:5:-2]
"#,
    );
}

#[test]
fn boolean_index_assignment_array() {
    assert_same(
        r#"
import numpy as np
a = np.arange(10.0)
mask = a > 4
a[mask] = -a[mask]
result = a
"#,
    );
}

#[test]
fn ellipsis_only_3d() {
    assert_same(
        r#"
import numpy as np
a = np.arange(24.0).reshape(2, 3, 4)
result = a[Ellipsis]
"#,
    );
}

#[test]
fn double_newaxis_then_scalar() {
    assert_same(
        r#"
import numpy as np
a = np.arange(6.0)
result = a[None, None, 2]
"#,
    );
}

#[test]
fn slice_3d_axis_0_and_1() {
    assert_same(
        r#"
import numpy as np
a = np.arange(24.0).reshape(2, 3, 4)
result = a[0:1, 1:3]
"#,
    );
}

#[test]
fn integer_array_index_with_negatives() {
    assert_same(
        r#"
import numpy as np
a = np.arange(10.0)
idx = np.array([0, -1, -2, 3])
result = a[idx]
"#,
    );
}

#[test]
fn empty_int_array_index() {
    assert_same(
        r#"
import numpy as np
a = np.arange(5.0)
idx = np.array([], dtype="int64")
result = a[idx]
"#,
    );
}

#[test]
fn single_int_index_negative() {
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
result = a[-1]
"#,
    );
}

#[test]
fn scalar_index_returns_python_float() {
    // a[i, j] on a 2-D array returns a 0-D scalar that gets unwrapped.
    let r = rumpy_run(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
result = np.array([a[1, 2]])
"#,
    );
    assert_eq!(r.data, vec![6.0]);
}

#[test]
fn step_slice_2d_both_axes() {
    assert_same(
        r#"
import numpy as np
a = np.arange(24.0).reshape(4, 6)
result = a[::2, ::3]
"#,
    );
}

#[test]
fn full_setitem_via_slice() {
    assert_same(
        r#"
import numpy as np
a = np.arange(6.0)
a[:] = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
result = a
"#,
    );
}
