//! Parity checks for the linalg paths now backed by faer:
//!   * `linalg.eig` on a non-symmetric matrix (returns complex output).
//!   * Matrix `norm(ord=2)` / `ord=-2` / `ord='nuc'` (SVD-based).
//!
//! Each test runs the same Python snippet under both rumpy (in a RustPython
//! VM) and CPython numpy (via pyo3) and asserts the scalar outputs match.

use approx::assert_abs_diff_eq;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyList, PyModule};
use rustpython_vm::{AsObject, Interpreter};

/// Run `source` and return the global `result`, coerced to a flat Vec<f64>
/// via the snippet itself (so the caller controls how complex / array
/// outputs collapse to a comparable scalar list).
fn run_in_rumpy(source: &str) -> Vec<f64> {
    let interp = {
        let builder = Interpreter::builder(Default::default());
        let def = rumpy::module_def(&builder.ctx);
        builder.add_native_module(def).build()
    };
    interp
        .enter(|vm| -> Result<Vec<f64>, String> {
            let scope = vm.new_scope_with_builtins();
            let code = vm
                .compile(source, rustpython_vm::compiler::Mode::Exec, "<test>".into())
                .map_err(|e| format!("compile: {e}"))?;
            vm.run_code_obj(code, scope.clone()).map_err(|e| {
                let mut s = String::new();
                let _ = vm.write_exception(&mut s, &e);
                format!("run: {s}\n[name {}]", e.as_object().class().name())
            })?;
            let result = scope
                .globals
                .get_item("result", vm)
                .expect("snippet must set `result`");
            // result must be a list of floats.
            let lst = result
                .downcast_ref::<rustpython_vm::builtins::PyList>()
                .ok_or_else(|| "result must be a list".to_string())?;
            let mut out = Vec::with_capacity(lst.borrow_vec().len());
            for it in lst.borrow_vec().iter() {
                let f = it.try_float(vm).map_err(|e| {
                    let mut s = String::new();
                    let _ = vm.write_exception(&mut s, &e);
                    s
                })?;
                out.push(f.to_f64());
            }
            Ok(out)
        })
        .unwrap_or_else(|e| panic!("rumpy snippet failed: {e}\n--- source ---\n{source}"))
}

fn run_in_numpy(source: &str) -> Vec<f64> {
    Python::attach(|py| -> PyResult<Vec<f64>> {
        let globals = PyModule::import(py, "builtins")?.dict();
        let numpy = PyModule::import(py, "numpy")?;
        globals.set_item("np", numpy)?;
        py.run(&std::ffi::CString::new(source).unwrap(), Some(&globals), None)?;
        let result = globals.get_item("result")?.expect("snippet sets result");
        let lst = result.cast_into::<PyList>()?;
        let mut out = Vec::with_capacity(lst.len());
        for it in lst.iter() {
            out.push(it.extract::<f64>()?);
        }
        Ok(out)
    })
    .expect("numpy snippet failed")
}

fn assert_same(src: &str, eps: f64) {
    let r = run_in_rumpy(src);
    let n = run_in_numpy(src);
    assert_eq!(r.len(), n.len(), "len mismatch\nsrc: {src}\nrumpy: {r:?}\nnumpy: {n:?}");
    for (i, (a, b)) in r.iter().zip(n.iter()).enumerate() {
        assert_abs_diff_eq!(*a, *b, epsilon = eps);
        let _ = i;
    }
}

// ----- eig on non-symmetric matrices ---------------------------------------

#[test]
fn eig_nonsymmetric_real_eigenvalues() {
    // [[1,2,3],[4,5,6],[7,8,9]] has all-real eigenvalues — rank-2 so one is 0.
    let src = r#"
import numpy as np
A = np.array([[1.0,2,3],[4,5,6],[7,8,9]])
w, _ = np.linalg.eig(A)
mags = sorted(round(float(abs(w[i])), 9) for i in range(3))
result = mags
"#;
    assert_same(src, 1e-9);
}

#[test]
fn eig_nonsymmetric_complex_eigenvalues() {
    // 2D rotation by 45°: eigenvalues are e^(±iπ/4), both with |λ|=1.
    let src = r#"
import numpy as np
c = (2 ** 0.5) / 2
A = np.array([[c, -c],[c, c]])
w, _ = np.linalg.eig(A)
mags = sorted(round(float(abs(w[i])), 9) for i in range(2))
result = mags
"#;
    assert_same(src, 1e-9);
}

#[test]
fn eigvals_nonsymmetric_matches() {
    let src = r#"
import numpy as np
A = np.array([[0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0],
              [-1.0, 0.0, 0.0]])  # companion of x^3 = -1, three cube roots of -1
w = np.linalg.eigvals(A)
mags = sorted(round(float(abs(w[i])), 9) for i in range(3))
result = mags
"#;
    assert_same(src, 1e-9);
}

// ----- matrix-norm via SVD --------------------------------------------------

#[test]
fn norm_matrix_2() {
    let src = r#"
import numpy as np
A = np.array([[1.0, 2, 3],[4, 5, 6],[7, 8, 9.5]])
result = [float(np.linalg.norm(A, ord=2))]
"#;
    assert_same(src, 1e-9);
}

#[test]
fn norm_matrix_neg2() {
    let src = r#"
import numpy as np
A = np.array([[1.0, 2, 3],[4, 5, 6],[7, 8, 9.5]])
result = [float(np.linalg.norm(A, ord=-2))]
"#;
    assert_same(src, 1e-9);
}

#[test]
fn norm_matrix_nuc() {
    let src = r#"
import numpy as np
A = np.array([[1.0, 2, 3],[4, 5, 6],[7, 8, 9.5]])
result = [float(np.linalg.norm(A, ord='nuc'))]
"#;
    assert_same(src, 1e-9);
}

#[test]
fn norm_matrix_2_rectangular() {
    let src = r#"
import numpy as np
A = np.array([[1.0, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12]])
result = [float(np.linalg.norm(A, ord=2))]
"#;
    assert_same(src, 1e-9);
}
