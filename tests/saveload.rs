//! Save/load tests — round-trip arrays across rumpy ↔ real numpy via
//! tempfiles.

use approx::assert_abs_diff_eq;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyList, PyModule};
use rustpython_vm::{Interpreter, PyObjectRef};
use std::path::PathBuf;

fn rumpy_interp() -> Interpreter {
    let b = Interpreter::builder(Default::default());
    let def = rumpy::module_def(&b.ctx);
    b.add_native_module(def).build()
}

fn tmp_path(suffix: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
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

fn rumpy_run_str(source: &str) {
    let interp = rumpy_interp();
    interp
        .enter(|vm| -> Result<(), String> {
            let scope = vm.new_scope_with_builtins();
            let code = vm
                .compile(source, rustpython_vm::compiler::Mode::Exec, "<t>".into())
                .map_err(|e| format!("compile: {e}"))?;
            vm.run_code_obj(code, scope.clone())
                .map_err(|e| pyerr(vm, &e))?;
            Ok(())
        })
        .unwrap_or_else(|e| panic!("rumpy: {e}\n--- src ---\n{source}"));
}

fn rumpy_load_arr(path: &str) -> Vec<f64> {
    let interp = rumpy_interp();
    interp
        .enter(|vm| -> Result<Vec<f64>, String> {
            let scope = vm.new_scope_with_builtins();
            let snippet = format!(
                r#"
import numpy as np
arr = np.load({:?}).astype("float64")
result = arr.ravel().tolist()
"#,
                path
            );
            let code = vm
                .compile(&snippet, rustpython_vm::compiler::Mode::Exec, "<t>".into())
                .map_err(|e| format!("compile: {e}"))?;
            vm.run_code_obj(code, scope.clone())
                .map_err(|e| pyerr(vm, &e))?;
            let result = scope.globals.get_item("result", vm).expect("result");
            extract_flat(&result, vm).map_err(|e| pyerr(vm, &e))
        })
        .unwrap_or_else(|e| panic!("rumpy load: {e}"))
}

fn extract_flat(
    obj: &PyObjectRef,
    vm: &rustpython_vm::VirtualMachine,
) -> rustpython_vm::PyResult<Vec<f64>> {
    use rustpython_vm::builtins::PyList as RpyList;
    let l = obj.downcast_ref::<RpyList>().expect("expected list result");
    let mut out = Vec::new();
    for it in l.borrow_vec().iter() {
        out.push(it.try_float(vm)?.to_f64());
    }
    Ok(out)
}

fn pyerr(
    vm: &rustpython_vm::VirtualMachine,
    e: &rustpython_vm::PyRef<rustpython_vm::builtins::PyBaseException>,
) -> String {
    let mut s = String::new();
    let _ = vm.write_exception(&mut s, e);
    s
}

// -- helper: drive numpy via pyo3 and assert -----------------------------

fn numpy_load_flat(path: &str) -> (Vec<usize>, Vec<f64>) {
    Python::attach(|py| -> PyResult<(Vec<usize>, Vec<f64>)> {
        let g = pyo3::types::PyDict::new(py);
        let np = PyModule::import(py, "numpy")?;
        g.set_item("np", &np)?;
        g.set_item("path", path)?;
        let src = r#"
import numpy as np
arr = np.load(path).astype("float64")
result_shape = list(arr.shape)
result = arr.ravel().tolist()
"#;
        py.run(&std::ffi::CString::new(src).unwrap(), Some(&g), None)?;
        let shape: Vec<usize> = g.get_item("result_shape")?.unwrap().extract()?;
        let data: Vec<f64> = g.get_item("result")?.unwrap().cast::<PyList>()?.extract()?;
        Ok((shape, data))
    })
    .expect("numpy.load failed")
}

fn numpy_load_npz(path: &str) -> Vec<(String, Vec<f64>)> {
    Python::attach(|py| -> PyResult<Vec<(String, Vec<f64>)>> {
        let g = pyo3::types::PyDict::new(py);
        let np = PyModule::import(py, "numpy")?;
        g.set_item("np", &np)?;
        g.set_item("path", path)?;
        let src = r#"
import numpy as np
z = np.load(path)
keys = sorted(list(z.keys()))
result = [(k, np.asarray(z[k]).astype("float64").ravel().tolist()) for k in keys]
"#;
        py.run(&std::ffi::CString::new(src).unwrap(), Some(&g), None)?;
        let items = g.get_item("result")?.unwrap();
        let list = items.cast::<PyList>()?;
        let mut out = Vec::new();
        for item in list.iter() {
            let t = item.cast::<pyo3::types::PyTuple>()?;
            let k: String = t.get_item(0)?.extract()?;
            let v: Vec<f64> = t.get_item(1)?.cast::<PyList>()?.extract()?;
            out.push((k, v));
        }
        Ok(out)
    })
    .expect("numpy load npz failed")
}

fn numpy_save_then(path: &str, src_python: &str) {
    Python::attach(|py| -> PyResult<()> {
        let g = pyo3::types::PyDict::new(py);
        let np = PyModule::import(py, "numpy")?;
        g.set_item("np", &np)?;
        g.set_item("path", path)?;
        py.run(&std::ffi::CString::new(src_python).unwrap(), Some(&g), None)?;
        Ok(())
    })
    .expect("numpy save failed");
}

// -----------------------------------------------------------------------
// rumpy → numpy   (save in rumpy, read in numpy)
// -----------------------------------------------------------------------

#[test]
fn rumpy_save_then_numpy_load_f64() {
    let p = tmp_path(".npy");
    let path = p.to_string_lossy().into_owned();
    rumpy_run_str(&format!(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
np.save({path:?}, a)
"#
    ));
    let (shape, data) = numpy_load_flat(&path);
    assert_eq!(shape, vec![3, 4]);
    for (i, v) in data.iter().enumerate() {
        assert_abs_diff_eq!(*v, i as f64, epsilon = 0.0);
    }
}

#[test]
fn rumpy_save_then_numpy_load_all_dtypes() {
    // Round-trip every dtype rumpy supports.
    for dt in [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ] {
        let p = tmp_path(".npy");
        let path = p.to_string_lossy().into_owned();
        rumpy_run_str(&format!(
            r#"
import numpy as np
a = np.array([1, 2, 3, 4, 5, 0], dtype={dt:?})
np.save({path:?}, a)
"#
        ));
        let (shape, data) = numpy_load_flat(&path);
        assert_eq!(shape, vec![6], "dtype {dt}");
        // For bool, numpy collapses every non-zero to True.
        let expected: Vec<f64> = if dt == "bool" {
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
        } else {
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.0]
        };
        for (i, (a, b)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 0.0,
                "dtype={dt} idx={i}: rumpy={a} numpy={b}"
            );
        }
    }
}

#[test]
fn rumpy_save_no_extension_added() {
    // `np.save("foo", a)` should write to `foo.npy`.
    let p = tmp_path("");
    let path = p.to_string_lossy().into_owned();
    rumpy_run_str(&format!(
        r#"
import numpy as np
a = np.arange(3.0)
np.save({path:?}, a)
"#
    ));
    let final_path = format!("{path}.npy");
    assert!(
        std::path::Path::new(&final_path).exists(),
        "expected file at {final_path}"
    );
    let (shape, data) = numpy_load_flat(&final_path);
    assert_eq!(shape, vec![3]);
    assert_eq!(data, vec![0.0, 1.0, 2.0]);
}

// -----------------------------------------------------------------------
// numpy → rumpy   (save in numpy, read in rumpy)
// -----------------------------------------------------------------------

#[test]
fn numpy_save_then_rumpy_load() {
    let p = tmp_path(".npy");
    let path = p.to_string_lossy().into_owned();
    numpy_save_then(
        &path,
        r#"
import numpy as np
a = np.arange(24.0).reshape(2, 3, 4)
np.save(path, a)
"#,
    );
    let data = rumpy_load_arr(&path);
    let expected: Vec<f64> = (0..24).map(|i| i as f64).collect();
    assert_eq!(data, expected);
}

#[test]
fn numpy_save_each_dtype_rumpy_load() {
    for dt in [
        "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
        "float16", "float32", "float64",
    ] {
        let p = tmp_path(".npy");
        let path = p.to_string_lossy().into_owned();
        let src = format!(
            r#"
import numpy as np
a = np.array([0, 1, 2, 3, 4], dtype={dt:?})
np.save(path, a)
"#
        );
        numpy_save_then(&path, &src);
        let data = rumpy_load_arr(&path);
        let expected: Vec<f64> = if dt == "bool" {
            vec![0.0, 1.0, 1.0, 1.0, 1.0]
        } else {
            vec![0.0, 1.0, 2.0, 3.0, 4.0]
        };
        assert_eq!(data, expected, "dtype {dt} round-trip mismatch");
    }
}

// -----------------------------------------------------------------------
// savez / npz
// -----------------------------------------------------------------------

#[test]
fn rumpy_savez_numpy_load() {
    let p = tmp_path(".npz");
    let path = p.to_string_lossy().into_owned();
    rumpy_run_str(&format!(
        r#"
import numpy as np
np.savez({path:?}, x=np.arange(3.0), y=np.array([10.0, 20.0]))
"#
    ));
    let entries = numpy_load_npz(&path);
    let map: std::collections::HashMap<String, Vec<f64>> = entries.into_iter().collect();
    assert_eq!(map.get("x").unwrap(), &vec![0.0, 1.0, 2.0]);
    assert_eq!(map.get("y").unwrap(), &vec![10.0, 20.0]);
}

#[test]
fn numpy_savez_rumpy_load() {
    let p = tmp_path(".npz");
    let path = p.to_string_lossy().into_owned();
    numpy_save_then(
        &path,
        r#"
import numpy as np
np.savez(path, a=np.array([1.5, 2.5, 3.5]), b=np.arange(6).reshape(2, 3))
"#,
    );
    let interp = rumpy_interp();
    let (a, b) = interp
        .enter(|vm| -> Result<(Vec<f64>, Vec<f64>), String> {
            let scope = vm.new_scope_with_builtins();
            let snip = format!(
                r#"
import numpy as np
d = np.load({path:?})
a_list = d["a"].astype("float64").ravel().tolist()
b_list = d["b"].astype("float64").ravel().tolist()
"#
            );
            let code = vm
                .compile(&snip, rustpython_vm::compiler::Mode::Exec, "<t>".into())
                .map_err(|e| format!("compile: {e}"))?;
            vm.run_code_obj(code, scope.clone())
                .map_err(|e| pyerr(vm, &e))?;
            let a_obj = scope.globals.get_item("a_list", vm).expect("a_list");
            let b_obj = scope.globals.get_item("b_list", vm).expect("b_list");
            Ok((
                extract_flat(&a_obj, vm).map_err(|e| pyerr(vm, &e))?,
                extract_flat(&b_obj, vm).map_err(|e| pyerr(vm, &e))?,
            ))
        })
        .unwrap_or_else(|e| panic!("rumpy npz load: {e}"));
    assert_eq!(a, vec![1.5, 2.5, 3.5]);
    assert_eq!(b, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}
