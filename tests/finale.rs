//! Cross-validation tests for batches D (text/raw I/O), E (linalg lstsq/
//! pinv/eigvalsh), F (FFT), G (polynomial), H (einsum).

use approx::assert_abs_diff_eq;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyList, PyModule};
use rustpython_vm::{AsObject, Interpreter, builtins::PyList as RpyList};
use std::sync::atomic::{AtomicU64, Ordering};

fn rumpy_interp() -> Interpreter {
    let b = Interpreter::builder(Default::default());
    let def = rumpy::module_def(&b.ctx);
    b.add_native_module(def).build()
}

fn tmp_path(suffix: &str) -> std::path::PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let id = format!(
        "rumpy_{}_{}_{n}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    std::env::temp_dir().join(format!("{id}{suffix}"))
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
        } else {
            data.push(it.try_float(vm)?.to_f64());
        }
    }
    Ok(())
}

fn numpy_run(source: &str) -> Out {
    Python::attach(|py| -> PyResult<Out> {
        // Use a fresh dict per snippet — sharing `builtins.__dict__` across
        // parallel tests creates a data race on `result`.
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
            .map(|x| x.extract::<f64>())
            .collect::<PyResult<_>>()?;
        Ok(Out { shape, data })
    })
    .expect("numpy snippet failed")
}

fn assert_close(r: &Out, n: &Out, eps: f64) {
    assert_eq!(r.shape, n.shape, "shape mismatch");
    assert_eq!(r.data.len(), n.data.len(), "length mismatch");
    for (a, b) in r.data.iter().zip(n.data.iter()) {
        if a.is_nan() && b.is_nan() {
            continue;
        }
        assert_abs_diff_eq!(*a, *b, epsilon = eps);
    }
}

fn assert_same(s: &str) {
    let r = rumpy_run(s);
    let n = numpy_run(s);
    assert_close(&r, &n, 1e-7);
}

// =====================================================================
// Batch D — savetxt / loadtxt / tofile / fromfile
// =====================================================================

#[test]
fn savetxt_then_numpy_loadtxt() {
    let p = tmp_path(".txt");
    let path = p.to_string_lossy().into_owned();
    rumpy_run(&format!(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
np.savetxt({path:?}, a, delimiter=",")
result = np.array([0.0])
"#
    ));
    // Read via numpy.
    Python::attach(|py| {
        let g = PyModule::import(py, "builtins").unwrap().dict();
        let np = PyModule::import(py, "numpy").unwrap();
        g.set_item("np", &np).unwrap();
        g.set_item("path", &path).unwrap();
        py.run(
            c"
import numpy as np
arr = np.loadtxt(path, delimiter=\",\")
result_shape = list(arr.shape)
result = arr.ravel().tolist()
",
            Some(&g),
            None,
        )
        .unwrap();
        let shape: Vec<usize> =
            g.get_item("result_shape").unwrap().unwrap().extract().unwrap();
        let data: Vec<f64> = g
            .get_item("result")
            .unwrap()
            .unwrap()
            .cast::<PyList>()
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(shape, vec![3, 4]);
        let expected: Vec<f64> = (0..12).map(|i| i as f64).collect();
        for (a, b) in data.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-10);
        }
    });
}

#[test]
fn numpy_savetxt_then_rumpy_loadtxt() {
    let p = tmp_path(".txt");
    let path = p.to_string_lossy().into_owned();
    Python::attach(|py| {
        let g = PyModule::import(py, "builtins").unwrap().dict();
        let np = PyModule::import(py, "numpy").unwrap();
        g.set_item("np", &np).unwrap();
        g.set_item("path", &path).unwrap();
        py.run(
            c"
import numpy as np
np.savetxt(path, np.array([[1.5, 2.5], [3.0, 4.0]]), delimiter=\",\")
",
            Some(&g),
            None,
        )
        .unwrap();
    });
    let r = rumpy_run(&format!(
        r#"
import numpy as np
result = np.loadtxt({path:?}, delimiter=",")
"#
    ));
    assert_eq!(r.shape, vec![2, 2]);
    assert_eq!(r.data, vec![1.5, 2.5, 3.0, 4.0]);
}

#[test]
fn tofile_fromfile_roundtrip() {
    let p = tmp_path(".bin");
    let path = p.to_string_lossy().into_owned();
    rumpy_run(&format!(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
a.astype("float64") if False else None  # noop
np.tofile({path:?}, a)
b = np.fromfile({path:?}, dtype="float64")
result = b
"#
    ));
    // Just verify the file is readable in real numpy too.
    Python::attach(|py| {
        let g = PyModule::import(py, "builtins").unwrap().dict();
        let np = PyModule::import(py, "numpy").unwrap();
        g.set_item("np", &np).unwrap();
        g.set_item("path", &path).unwrap();
        py.run(
            c"
import numpy as np
arr = np.fromfile(path, dtype=\"float64\")
result = arr.tolist()
",
            Some(&g),
            None,
        )
        .unwrap();
        let data: Vec<f64> = g
            .get_item("result")
            .unwrap()
            .unwrap()
            .cast::<PyList>()
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    });
}

// =====================================================================
// Batch E — linalg.lstsq / pinv / eigvalsh
// =====================================================================

#[test]
fn lstsq_overdetermined() {
    // numpy.linalg.lstsq returns (solution, residuals, rank, s); rumpy
    // returns just the solution. So pull element [0] when running in real
    // numpy.
    let r = rumpy_run(
        r#"
import numpy as np
A = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
b = np.array([6.0, 5.0, 7.0, 10.0])
x, residuals, rank, s = np.linalg.lstsq(A, b)
result = A @ x
"#,
    );
    let n = numpy_run(
        r#"
import numpy as np
A = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
b = np.array([6.0, 5.0, 7.0, 10.0])
x, *_ = np.linalg.lstsq(A, b, rcond=None)
result = A @ x
"#,
    );
    assert_close(&r, &n, 1e-9);
}

#[test]
fn pinv_left_inverse_identity() {
    // For tall A: pinv(A) @ A ≈ I.
    assert_same(
        r#"
import numpy as np
A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
inv = np.linalg.pinv(A)
result = inv @ A
"#,
    );
}

#[test]
fn eigvalsh_diagonal() {
    // A symmetric diagonal matrix → diagonal values are the eigenvalues.
    let r = rumpy_run(
        r#"
import numpy as np
A = np.array([[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
result = np.linalg.eigvalsh(A)
"#,
    );
    assert_eq!(r.shape, vec![3]);
    let mut sorted = r.data.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_abs_diff_eq!(sorted[0], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(sorted[1], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(sorted[2], 3.0, epsilon = 1e-10);
}

#[test]
fn eigvalsh_symmetric() {
    // Eigenvalues of [[2,1],[1,2]] are 1 and 3.
    let r = rumpy_run(
        r#"
import numpy as np
A = np.array([[2.0, 1.0], [1.0, 2.0]])
result = np.linalg.eigvalsh(A)
"#,
    );
    let mut sorted = r.data.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_abs_diff_eq!(sorted[0], 1.0, epsilon = 1e-9);
    assert_abs_diff_eq!(sorted[1], 3.0, epsilon = 1e-9);
}

// =====================================================================
// Batch G — polyval / roots / polyfit
// =====================================================================

#[test]
fn polyval_matches_numpy() {
    assert_same(
        r#"
import numpy as np
p = np.array([1.0, -3.0, 2.0])   # x^2 - 3x + 2
x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
result = np.polyval(p, x)
"#,
    );
}

#[test]
fn roots_quadratic() {
    // roots of x^2 - 3x + 2  → 1 and 2.
    let r = rumpy_run(
        r#"
import numpy as np
result = np.roots(np.array([1.0, -3.0, 2.0]))
"#,
    );
    let mut sorted = r.data.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_abs_diff_eq!(sorted[0], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(sorted[1], 2.0, epsilon = 1e-6);
}

#[test]
fn polyfit_recovers_line() {
    // y = 2x + 1 exactly → polyfit deg=1 should give [2, 1].
    let r = rumpy_run(
        r#"
import numpy as np
x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
y = 2 * x + 1
result = np.polyfit(x, y, 1)
"#,
    );
    assert_eq!(r.shape, vec![2]);
    assert_abs_diff_eq!(r.data[0], 2.0, epsilon = 1e-8);
    assert_abs_diff_eq!(r.data[1], 1.0, epsilon = 1e-8);
}

// =====================================================================
// Batch H — einsum
// =====================================================================

#[test]
fn einsum_matmul() {
    assert_same(
        r#"
import numpy as np
a = np.arange(6.0).reshape(2, 3)
b = np.arange(12.0).reshape(3, 4)
result = np.einsum("ij,jk->ik", a, b)
"#,
    );
}

#[test]
fn einsum_trace() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.arange(9.0).reshape(3, 3)
result = float(np.einsum("ii->", a))
"#,
    );
    // diag(0,4,8) → trace = 12
    assert_abs_diff_eq!(r.data[0], 12.0, epsilon = 1e-10);
}

#[test]
fn einsum_transpose() {
    assert_same(
        r#"
import numpy as np
a = np.arange(6.0).reshape(2, 3)
result = np.einsum("ij->ji", a)
"#,
    );
}

#[test]
fn einsum_dot() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
result = float(np.einsum("i,i->", a, b))
"#,
    );
    assert_abs_diff_eq!(r.data[0], 32.0, epsilon = 1e-10);
}

#[test]
fn einsum_outer() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0])
result = np.einsum("i,j->ij", a, b)
"#,
    );
}

// =====================================================================
// Batch F — FFT
// =====================================================================

#[test]
fn fft_then_ifft_roundtrip() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0, 4.0])
recovered = np.fft.ifft(np.fft.fft(a)).real
result = recovered
"#,
    );
    let expected = [1.0, 2.0, 3.0, 4.0];
    for (a, b) in r.data.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(*a, *b, epsilon = 1e-8);
    }
}

#[test]
fn rfft_matches_numpy() {
    // Compare |rfft| since the complex output extraction is fiddly.
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
result = np.abs(np.fft.rfft(a))
"#,
    );
}

#[test]
fn fftfreq_matches() {
    assert_same(
        r#"
import numpy as np
result = np.fft.fftfreq(8, d=0.5)
"#,
    );
}

#[test]
fn rfftfreq_matches() {
    assert_same(
        r#"
import numpy as np
result = np.fft.rfftfreq(10, d=0.1)
"#,
    );
}

#[test]
fn fftshift_1d() {
    assert_same(
        r#"
import numpy as np
a = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
result = np.fft.fftshift(a)
"#,
    );
}

// =====================================================================
// Caveat fixes — lstsq tuple, qr modes, N-D fftshift, complex roots
// =====================================================================

#[test]
fn lstsq_returns_full_tuple() {
    let r = rumpy_run(
        r#"
import numpy as np
A = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
b = np.array([6.0, 5.0, 7.0, 10.0])
x, residuals, rank, s = np.linalg.lstsq(A, b)
# Pack the rank scalar into the result for comparison.
result = np.array([float(rank)])
"#,
    );
    // For this full-rank 4×2 problem the rank should be 2.
    assert_abs_diff_eq!(r.data[0], 2.0, epsilon = 0.0);
}

#[test]
fn qr_complete_mode() {
    assert_same(
        r#"
import numpy as np
A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
Q, R = np.linalg.qr(A, mode="complete")
# Q should now be 3x3, R should be 3x2.
result = Q @ R
"#,
    );
}

#[test]
fn qr_r_only_mode() {
    let r = rumpy_run(
        r#"
import numpy as np
A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
R = np.linalg.qr(A, mode="r")
result = R
"#,
    );
    // R has shape (2,2) for this 3×2 input.
    assert_eq!(r.shape, vec![2, 2]);
}

#[test]
fn fftshift_2d() {
    assert_same(
        r#"
import numpy as np
a = np.arange(16.0).reshape(4, 4)
result = np.fft.fftshift(a)
"#,
    );
}

#[test]
fn fftshift_axes_kwarg() {
    assert_same(
        r#"
import numpy as np
a = np.arange(16.0).reshape(4, 4)
result = np.fft.fftshift(a, axes=0)
"#,
    );
}

#[test]
fn roots_complex_pair() {
    // x^2 + 1 → roots ±i.  Pull |im|² + re² to compare a single number.
    let r = rumpy_run(
        r#"
import numpy as np
roots = np.roots(np.array([1.0, 0.0, 1.0]))
result = np.abs(roots) ** 2
"#,
    );
    assert_eq!(r.shape, vec![2]);
    for v in &r.data {
        assert_abs_diff_eq!(*v, 1.0, epsilon = 1e-6);
    }
}

#[test]
fn einsum_three_operand_chain() {
    // Three-operand contraction — exercises the greedy pair-selection code.
    assert_same(
        r#"
import numpy as np
a = np.arange(6.0).reshape(2, 3)
b = np.arange(12.0).reshape(3, 4)
c = np.arange(20.0).reshape(4, 5)
result = np.einsum("ij,jk,kl->il", a, b, c)
"#,
    );
}
