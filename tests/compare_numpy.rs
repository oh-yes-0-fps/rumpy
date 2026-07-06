//! Cross-validation tests: every operation is run under both
//!
//!   * **rumpy** — our `numpy` module embedded in a RustPython VM
//!   * **numpy** — the real CPython numpy library accessed via pyo3
//!
//! and the resulting `.tolist()` outputs are compared element-wise.
//!
//! We share the exact Python source between the two implementations so the
//! only thing that varies is the runtime. If a test fails it almost
//! certainly means our implementation diverged from numpy's semantics.

use approx::assert_abs_diff_eq;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyList, PyModule};
use rustpython_vm::{AsObject, Interpreter, builtins::PyList as RpyList};

/// Run a Python snippet in a RustPython VM that has our `numpy` module
/// pre-registered. The snippet must end by assigning the value of interest
/// to the global name `result`.
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
    // Array case — collapse dtype to f64 for cross-checking against numpy.
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
    // Scalar (int/float)
    if let Ok(f) = obj.try_float(vm) {
        return Ok(RumpyResult {
            shape: None,
            data: vec![f.to_f64()],
        });
    }
    // Tuple of ints (e.g. .shape)
    if let Some(t) = obj.downcast_ref::<rustpython_vm::builtins::PyTuple>() {
        let mut data = Vec::with_capacity(t.len());
        for it in t.as_slice() {
            data.push(it.try_float(vm)?.to_f64());
        }
        return Ok(RumpyResult {
            shape: Some(vec![data.len()]),
            data,
        });
    }
    // Nested list — recursively flatten
    if let Some(l) = obj.downcast_ref::<RpyList>() {
        let mut shape = Vec::new();
        let mut data = Vec::new();
        flatten_pylist(l, &mut shape, &mut data, vm, 0)?;
        return Ok(RumpyResult {
            shape: Some(shape),
            data,
        });
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

/// Run the same source in real CPython+numpy. Same convention: snippet
/// assigns to `result`. We coerce the result into `(shape, flat_data)`
/// using `numpy.asarray(result).ravel().tolist()`.
fn run_in_numpy(source: &str) -> NumpyResult {
    Python::attach(|py| -> PyResult<NumpyResult> {
        let globals = pyo3::types::PyDict::new(py);
        let numpy = PyModule::import(py, "numpy")?;
        globals.set_item("numpy", &numpy)?;
        globals.set_item("np", &numpy)?;

        py.run(
            &std::ffi::CString::new(source).unwrap(),
            Some(&globals),
            None,
        )?;
        let result = globals.get_item("result")?.unwrap();
        // Use numpy itself to canonicalize: shape + flat list.
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

/// Compare a rumpy run with a numpy run for the same snippet.
fn assert_same(snippet: &str) {
    let r = run_in_rumpy(snippet);
    let n = run_in_numpy(snippet);
    match r.shape {
        None => {
            // Scalar from rumpy; numpy shape should be `()`.
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
    for (i, (a, b)) in r.data.iter().zip(n.data.iter()).enumerate() {
        if a.is_nan() && b.is_nan() {
            continue;
        }
        assert_abs_diff_eq!(*a, *b, epsilon = 1e-9);
        let _ = i;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn zeros_and_shape() {
    assert_same("import numpy as np\nresult = np.zeros((3, 4)).shape");
    assert_same("import numpy as np\nresult = np.zeros((3, 4))");
}

#[test]
fn ones_and_full() {
    assert_same("import numpy as np\nresult = np.ones((2, 3))");
    assert_same("import numpy as np\nresult = np.full((2, 2), 7.5)");
}

#[test]
fn arange() {
    assert_same("import numpy as np\nresult = np.arange(10)");
    assert_same("import numpy as np\nresult = np.arange(2, 8)");
    assert_same("import numpy as np\nresult = np.arange(0.0, 1.0, 0.25)");
}

#[test]
fn linspace() {
    assert_same("import numpy as np\nresult = np.linspace(0.0, 1.0, 5)");
    assert_same("import numpy as np\nresult = np.linspace(-1.0, 1.0, 11)");
}

#[test]
fn eye_and_identity() {
    assert_same("import numpy as np\nresult = np.eye(4)");
    assert_same("import numpy as np\nresult = np.identity(3)");
}

#[test]
fn array_from_nested_list() {
    assert_same("import numpy as np\nresult = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])");
}

#[test]
fn arithmetic_broadcasting() {
    assert_same(
        r#"
import numpy as np
a = np.arange(6).reshape(2, 3)
b = np.array([10.0, 20.0, 30.0])
result = a + b
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.arange(6).reshape(2, 3)
result = a * 2.0 - 1.0
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.ones((3, 1))
b = np.arange(4.0)
result = a * b
"#,
    );
}

#[test]
fn elementwise_ufuncs() {
    assert_same(
        r#"
import numpy as np
x = np.linspace(0.1, 1.0, 5)
result = np.sqrt(x) + np.log(x) * np.sin(x)
"#,
    );
    assert_same(
        r#"
import numpy as np
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
result = np.exp(x) - np.abs(x)
"#,
    );
}

#[test]
fn reductions_full() {
    assert_same(
        r#"
import numpy as np
a = np.arange(12).reshape(3, 4)
result = np.array([a.sum(), a.mean(), a.min(), a.max(), a.prod()])
"#,
    );
}

#[test]
fn reductions_axis() {
    assert_same(
        r#"
import numpy as np
a = np.arange(12).reshape(3, 4)
result = a.sum(axis=0)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.arange(12).reshape(3, 4)
result = a.mean(axis=1)
"#,
    );
}

#[test]
fn variance_and_std() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
result = np.array([a.var(), a.std()])
"#,
    );
}

#[test]
fn reshape_transpose_flatten() {
    assert_same(
        r#"
import numpy as np
a = np.arange(12).reshape(3, 4)
result = a.T
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.arange(24).reshape(2, 3, 4)
result = a.reshape(-1)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.arange(12).reshape(3, 4)
result = a.flatten()
"#,
    );
}

#[test]
fn dot_and_matmul() {
    assert_same(
        r#"
import numpy as np
a = np.arange(6).reshape(2, 3).astype(float) if False else np.arange(6.0).reshape(2, 3)
b = np.arange(12.0).reshape(3, 4)
result = a @ b
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.arange(4.0)
b = np.arange(4.0) * 2.0
result = np.dot(a, b)
"#,
    );
    assert_same(
        r#"
import numpy as np
A = np.array([[1.0, 2.0], [3.0, 4.0]])
v = np.array([1.0, -1.0])
result = A.dot(v)
"#,
    );
}

#[test]
fn indexing_scalar_and_slice() {
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
result = float(a[1, 2])
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
result = a[1]
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
result = a[1:, 1:3]
"#,
    );
}

#[test]
fn module_constants() {
    assert_same("import numpy as np\nresult = np.pi");
    assert_same("import numpy as np\nresult = np.e");
}

#[test]
fn power_and_negation() {
    assert_same(
        r#"
import numpy as np
a = np.linspace(0.5, 3.0, 6)
result = -a ** 2 + 1
"#,
    );
}

#[test]
fn concatenate_axis() {
    // axis=0 (default) — joins rows
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.array([[5.0, 6.0], [7.0, 8.0]])
result = np.concatenate([a, b])
"#,
    );
    // axis=1 — joins columns (this is the bug that was silently wrong)
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.array([[5.0, 6.0], [7.0, 8.0]])
result = np.concatenate([a, b], axis=1)
"#,
    );
    // axis=-1 — last axis
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = np.array([[7.0, 8.0], [9.0, 10.0]])
result = np.concatenate([a, b], axis=-1)
"#,
    );
    // axis=None — flatten then concatenate
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.array([[5.0, 6.0], [7.0, 8.0]])
result = np.concatenate([a, b], axis=None)
"#,
    );
}
