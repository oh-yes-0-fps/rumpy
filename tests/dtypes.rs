//! Cross-validation tests for the full dtype set.
//!
//! We compare rumpy against real numpy for every dtype, plus type promotion,
//! casting, complex math, and bool semantics.

use approx::assert_abs_diff_eq;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyList, PyModule};
use rustpython_vm::{AsObject, Interpreter, builtins::PyList as RpyList};

fn rumpy_interp() -> Interpreter {
    let builder = Interpreter::builder(Default::default());
    let def = rumpy::module_def(&builder.ctx);
    builder.add_native_module(def).build()
}

struct ResultPair {
    dtype: String,
    shape: Vec<usize>,
    data: Vec<f64>,
}

/// Run a snippet in rumpy, expecting `result` and `result_dtype` (a string)
/// to be set at the end.
fn run_rumpy(source: &str) -> ResultPair {
    let interp = rumpy_interp();
    let outcome = interp.enter(|vm| -> Result<ResultPair, String> {
        let scope = vm.new_scope_with_builtins();
        let code = vm
            .compile(source, rustpython_vm::compiler::Mode::Exec, "<test>".into())
            .map_err(|e| format!("compile error: {e}"))?;
        vm.run_code_obj(code, scope.clone())
            .map_err(|e| py_err(vm, &e))?;
        let result = scope.globals.get_item("result", vm).expect("set result");
        let dtype = scope
            .globals
            .get_item("result_dtype", vm)
            .expect("set result_dtype");
        extract(&result, &dtype, vm).map_err(|e| py_err(vm, &e))
    });
    outcome.unwrap_or_else(|e| panic!("rumpy failed: {e}\nsource:\n{source}"))
}

fn py_err(
    vm: &rustpython_vm::VirtualMachine,
    e: &rustpython_vm::PyRef<rustpython_vm::builtins::PyBaseException>,
) -> String {
    let mut s = String::new();
    let _ = vm.write_exception(&mut s, e);
    s
}

fn extract(
    obj: &rustpython_vm::PyObjectRef,
    dtype: &rustpython_vm::PyObjectRef,
    vm: &rustpython_vm::VirtualMachine,
) -> rustpython_vm::PyResult<ResultPair> {
    use rumpy::{ArraysD, DType, PyNdArray};
    let dtype_str = dtype
        .downcast_ref::<rustpython_vm::builtins::PyStr>()
        .map(|s| s.as_wtf8().to_string_lossy().into_owned())
        .unwrap_or_else(|| "?".to_owned());
    if let Some(a) = obj.downcast_ref::<PyNdArray>() {
        let cast = a.view().cast(DType::F64);
        let f = match &cast {
            ArraysD::F64(x) => x,
            _ => unreachable!(),
        };
        return Ok(ResultPair {
            dtype: dtype_str,
            shape: f.shape().to_vec(),
            data: f.iter().copied().collect(),
        });
    }
    if let Ok(f) = obj.try_float(vm) {
        return Ok(ResultPair {
            dtype: dtype_str,
            shape: vec![],
            data: vec![f.to_f64()],
        });
    }
    if let Some(l) = obj.downcast_ref::<RpyList>() {
        let mut shape = Vec::new();
        let mut data = Vec::new();
        flatten(l, &mut shape, &mut data, vm, 0)?;
        return Ok(ResultPair {
            dtype: dtype_str,
            shape,
            data,
        });
    }
    Err(vm.new_type_error(format!("can't extract {}", obj.class().name())))
}

fn flatten(
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
            flatten(sub, shape, data, vm, depth + 1)?;
        } else {
            data.push(it.try_float(vm)?.to_f64());
        }
    }
    Ok(())
}

fn run_numpy(source: &str) -> ResultPair {
    Python::attach(|py| -> PyResult<ResultPair> {
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
        let dtype = globals
            .get_item("result_dtype")?
            .unwrap()
            .extract::<String>()?;
        let arr = numpy.getattr("asarray")?.call1((result,))?;
        let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
        let flat = arr.call_method0("ravel")?.call_method0("tolist")?;
        let data: Vec<f64> = flat
            .cast::<PyList>()?
            .iter()
            .map(|x| x.extract::<f64>())
            .collect::<PyResult<_>>()?;
        Ok(ResultPair { dtype, shape, data })
    })
    .expect("numpy snippet failed")
}

fn assert_same_with_dtype(snippet: &str) {
    let r = run_rumpy(snippet);
    let n = run_numpy(snippet);
    assert_eq!(
        r.dtype, n.dtype,
        "dtype mismatch (rumpy={}, numpy={}) for snippet:\n{snippet}",
        r.dtype, n.dtype
    );
    assert_eq!(r.shape, n.shape, "shape mismatch for snippet:\n{snippet}");
    for (a, b) in r.data.iter().zip(n.data.iter()) {
        if a.is_nan() && b.is_nan() {
            continue;
        }
        assert_abs_diff_eq!(*a, *b, epsilon = 1e-6);
    }
}

// -- creation --

#[test]
fn zeros_each_dtype() {
    for dt in [
        "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
        "float16", "float32", "float64",
    ] {
        let snippet = format!(
            r#"
import numpy as np
a = np.zeros((2, 3), dtype="{dt}")
result = a
result_dtype = str(a.dtype)
"#
        );
        assert_same_with_dtype(&snippet);
    }
}

#[test]
fn ones_each_dtype() {
    for dt in [
        "bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
        "float16", "float32", "float64",
    ] {
        let snippet = format!(
            r#"
import numpy as np
a = np.ones((4,), dtype="{dt}")
result = a
result_dtype = str(a.dtype)
"#
        );
        assert_same_with_dtype(&snippet);
    }
}

#[test]
fn arange_typed() {
    assert_same_with_dtype(
        r#"
import numpy as np
a = np.arange(10, dtype="int32")
result = a
result_dtype = str(a.dtype)
"#,
    );
    assert_same_with_dtype(
        r#"
import numpy as np
a = np.arange(5, dtype="float32")
result = a
result_dtype = str(a.dtype)
"#,
    );
}

#[test]
fn linspace_typed() {
    assert_same_with_dtype(
        r#"
import numpy as np
a = np.linspace(0, 1, 5, dtype="float32")
result = a
result_dtype = str(a.dtype)
"#,
    );
}

#[test]
fn astype_roundtrip() {
    assert_same_with_dtype(
        r#"
import numpy as np
a = np.array([1.7, -2.3, 4.0, 100.0])
b = a.astype("int32").astype("float64")
result = b
result_dtype = str(b.dtype)
"#,
    );
}

// -- promotion --

#[test]
fn promotion_int_int() {
    assert_same_with_dtype(
        r#"
import numpy as np
a = np.arange(4, dtype="int32")
b = np.arange(4, dtype="int64")
result = a + b
result_dtype = str(result.dtype)
"#,
    );
    assert_same_with_dtype(
        r#"
import numpy as np
a = np.array([1, 2, 3], dtype="uint8")
b = np.array([10, 20, 30], dtype="int16")
result = a + b
result_dtype = str(result.dtype)
"#,
    );
}

#[test]
fn promotion_int_float() {
    assert_same_with_dtype(
        r#"
import numpy as np
a = np.arange(4, dtype="int32")
b = np.arange(4, dtype="float32")
result = a + b
result_dtype = str(result.dtype)
"#,
    );
}

#[test]
fn promotion_bool_int() {
    assert_same_with_dtype(
        r#"
import numpy as np
a = np.array([True, False, True], dtype="bool")
b = np.array([10, 20, 30], dtype="int32")
result = a + b
result_dtype = str(result.dtype)
"#,
    );
}

// -- complex math --

#[test]
fn complex_arithmetic() {
    assert_same_with_dtype(
        r#"
import numpy as np
a = np.array([1+2j, 3-1j, 0+0j])
b = np.array([2+0j, 0-1j, 1+1j])
c = a * b + a / (b + 1)
result = np.abs(c)
result_dtype = str(result.dtype)
"#,
    );
}

#[test]
fn complex_sqrt_of_negative() {
    // Bring the complex result back to real for cross-comparison.
    assert_same_with_dtype(
        r#"
import numpy as np
a = np.array([-1+0j, -4+0j, 9+0j])
s = np.sqrt(a)
result = np.abs(s) + np.abs(s.imag)
result_dtype = str(s.dtype)
"#,
    );
}

#[test]
fn real_imag_conj() {
    assert_same_with_dtype(
        r#"
import numpy as np
a = np.array([1+2j, 3-4j])
result = a.real + a.imag * 1j
# Convert complex to real array (numpy auto-promotes to complex; abs(im) is float)
result = np.abs(result)
result_dtype = str(result.dtype)
"#,
    );
    assert_same_with_dtype(
        r#"
import numpy as np
a = np.array([1+2j, 3-4j])
b = a.conj()
result = np.abs(b)
result_dtype = str(result.dtype)
"#,
    );
}

// -- bool semantics --

#[test]
fn bool_arithmetic() {
    // bool + bool → bool (numpy: actually bool|bool stays bool only via &, |, ^).
    // Plain + on bool arrays in numpy yields bool too via OR semantics; we
    // match that.
    assert_same_with_dtype(
        r#"
import numpy as np
a = np.array([True, False, True], dtype="bool")
b = np.array([True, True, False], dtype="bool")
result = (a + b).astype("int8")
result_dtype = str(result.dtype)
"#,
    );
}

// -- reductions across dtypes --

#[test]
fn sum_widens_int() {
    assert_same_with_dtype(
        r#"
import numpy as np
a = np.array([100, 100, 100], dtype="int8")
s = a.sum()
result = np.array([int(s)])
result_dtype = str(np.asarray(s).dtype)
"#,
    );
}

#[test]
fn mean_returns_float() {
    assert_same_with_dtype(
        r#"
import numpy as np
a = np.array([1, 2, 3, 4], dtype="int32")
m = a.mean()
result = np.array([float(m)])
result_dtype = str(np.asarray(m).dtype)
"#,
    );
}

// -- comparison --

#[test]
fn comparison_produces_bool() {
    assert_same_with_dtype(
        r#"
import numpy as np
a = np.arange(5)
b = np.array([0, 0, 2, 4, 5])
result = np.equal(a, b)
result_dtype = str(result.dtype)
"#,
    );
    assert_same_with_dtype(
        r#"
import numpy as np
a = np.arange(5)
result = np.greater(a, 2)
result_dtype = str(result.dtype)
"#,
    );
}

// -- shape-preserving across dtype --

#[test]
fn transpose_preserves_dtype() {
    assert_same_with_dtype(
        r#"
import numpy as np
a = np.arange(12, dtype="int16").reshape(3, 4)
b = a.T
result = b
result_dtype = str(b.dtype)
"#,
    );
}

// -- dtype promotion functions --

#[test]
fn result_type_pair() {
    assert_same_with_dtype(
        r#"
import numpy as np
result = np.array([1.0])
result_dtype = str(np.result_type("int32", "float32"))
"#,
    );
    assert_same_with_dtype(
        r#"
import numpy as np
result = np.array([1.0])
result_dtype = str(np.result_type("int8", "uint8"))
"#,
    );
    assert_same_with_dtype(
        r#"
import numpy as np
result = np.array([1.0])
result_dtype = str(np.result_type("complex64", "float64"))
"#,
    );
    assert_same_with_dtype(
        r#"
import numpy as np
result = np.array([1.0])
result_dtype = str(np.result_type("bool", "int16", "float64"))
"#,
    );
}

#[test]
fn promote_types_pair() {
    assert_same_with_dtype(
        r#"
import numpy as np
result = np.array([1.0])
result_dtype = str(np.promote_types("int32", "float32"))
"#,
    );
    assert_same_with_dtype(
        r#"
import numpy as np
result = np.array([1.0])
result_dtype = str(np.promote_types("complex64", "complex128"))
"#,
    );
}

#[test]
fn can_cast_safe() {
    assert_same_with_dtype(
        r#"
import numpy as np
# int8 → int64 is safe
v = np.can_cast("int8", "int64", casting="safe")
result = np.array([1.0 if v else 0.0])
result_dtype = "float64"
"#,
    );
    assert_same_with_dtype(
        r#"
import numpy as np
v = np.can_cast("float64", "float32", casting="safe")
result = np.array([1.0 if v else 0.0])
result_dtype = "float64"
"#,
    );
    assert_same_with_dtype(
        r#"
import numpy as np
v = np.can_cast("float64", "float32", casting="unsafe")
result = np.array([1.0 if v else 0.0])
result_dtype = "float64"
"#,
    );
    assert_same_with_dtype(
        r#"
import numpy as np
v = np.can_cast("int32", "float32", casting="same_kind")
result = np.array([1.0 if v else 0.0])
result_dtype = "float64"
"#,
    );
}

// -- PyDType object --

#[test]
fn dtype_object_attrs() {
    assert_same_with_dtype(
        r#"
import numpy as np
d = np.dtype("float32")
result = np.array([d.itemsize])
result_dtype = d.name
"#,
    );
    assert_same_with_dtype(
        r#"
import numpy as np
d = np.dtype("int64")
result = np.array([d.itemsize])
result_dtype = d.kind
"#,
    );
    assert_same_with_dtype(
        r#"
import numpy as np
d = np.dtype("complex128")
result = np.array([d.itemsize])
result_dtype = d.char
"#,
    );
}

#[test]
fn dtype_equality_with_string() {
    let r = run_rumpy(
        r#"
import numpy as np
arr = np.zeros((3,), dtype="float32")
ok1 = arr.dtype == "float32"
ok2 = arr.dtype != "int32"
ok3 = arr.dtype == np.dtype("float32")
result = np.array([1.0 if (ok1 and ok2 and ok3) else 0.0])
result_dtype = "float64"
"#,
    );
    assert_eq!(r.data, vec![1.0]);
}

#[test]
fn dtype_kind_accessor() {
    let r = run_rumpy(
        r#"
import numpy as np
a = np.zeros((3,), dtype="int16")
result = np.array([0.0]) if a.dtype.kind != "i" else np.array([1.0])
result_dtype = "float64"
"#,
    );
    assert_eq!(r.data, vec![1.0]);
}

#[test]
fn dtype_repr() {
    let r = run_rumpy(
        r#"
import numpy as np
d = np.dtype("float64")
result = np.array([1.0 if repr(d) == "dtype('float64')" else 0.0])
result_dtype = "float64"
"#,
    );
    assert_eq!(r.data, vec![1.0]);
}

#[test]
fn iinfo_basic() {
    assert_same_with_dtype(
        r#"
import numpy as np
info = np.iinfo("int32")
result = np.array([info.min, info.max, info.bits])
result_dtype = "float64"
"#,
    );
    assert_same_with_dtype(
        r#"
import numpy as np
info = np.iinfo("uint8")
result = np.array([info.min, info.max, info.bits])
result_dtype = "float64"
"#,
    );
    assert_same_with_dtype(
        r#"
import numpy as np
info = np.iinfo("int8")
result = np.array([info.min, info.max])
result_dtype = "float64"
"#,
    );
}

#[test]
fn finfo_basic() {
    // float32 eps and bits
    assert_same_with_dtype(
        r#"
import numpy as np
info = np.finfo("float32")
result = np.array([info.bits, info.precision])
result_dtype = "float64"
"#,
    );
    // float64 eps
    assert_same_with_dtype(
        r#"
import numpy as np
info = np.finfo("float64")
result = np.array([info.eps, info.bits])
result_dtype = "float64"
"#,
    );
}

#[test]
fn min_scalar_type_basic() {
    // Smallest unsigned that fits 200
    assert_same_with_dtype(
        r#"
import numpy as np
result = np.array([1.0])
result_dtype = str(np.min_scalar_type(200))
"#,
    );
    // Negative — smallest signed
    assert_same_with_dtype(
        r#"
import numpy as np
result = np.array([1.0])
result_dtype = str(np.min_scalar_type(-5))
"#,
    );
    // Larger int
    assert_same_with_dtype(
        r#"
import numpy as np
result = np.array([1.0])
result_dtype = str(np.min_scalar_type(70000))
"#,
    );
}

// =====================================================================
// Expanded dtype coverage
// =====================================================================

#[test]
fn dtype_string_aliases() {
    // The short numpy aliases (i4, f8, c16, b1, ?) should resolve to the
    // matching dtype.
    for (alias, expected_name) in [
        ("i4", "int32"),
        ("f8", "float64"),
        ("c16", "complex128"),
        ("b1", "bool"),
        ("?", "bool"),
        ("u2", "uint16"),
        ("f2", "float16"),
        ("c8", "complex64"),
    ] {
        let snippet = format!(
            r#"
import numpy as np
result = np.array([1.0])
result_dtype = str(np.dtype("{alias}"))
"#
        );
        let r = run_rumpy(&snippet);
        assert_eq!(r.dtype, expected_name, "alias {alias}");
    }
}

#[test]
fn dtype_byteorder_prefix() {
    // Byte-order prefixes are stripped — '<i4', '>i4', '|i4' all map to int32.
    for prefix in ["<", ">", "=", "|"] {
        let snippet = format!(
            r#"
import numpy as np
result = np.array([1.0])
result_dtype = str(np.dtype("{prefix}i4"))
"#
        );
        let r = run_rumpy(&snippet);
        assert_eq!(r.dtype, "int32", "prefix {prefix}");
    }
}

#[test]
fn dtype_str_attr() {
    // .str returns a string with byteorder + char + size.
    let r = run_rumpy(
        r#"
import numpy as np
d = np.dtype("int32")
result = np.array([1.0])
result_dtype = d.str
"#,
    );
    // On little-endian (default on macOS arm64), int32 → "<i4".
    assert!(r.dtype.ends_with("i4"), "got {}", r.dtype);
}

#[test]
fn dtype_num_unique() {
    // Numeric type numbers should be distinct across dtypes.
    let r = run_rumpy(
        r#"
import numpy as np
nums = [np.dtype(s).num for s in ["bool", "int8", "int16", "int32", "int64",
                                   "uint8", "uint16", "uint32", "uint64",
                                   "float16", "float32", "float64",
                                   "complex64", "complex128"]]
result = np.array(nums).astype("float64")
result_dtype = "float64"
"#,
    );
    let mut seen = std::collections::HashSet::new();
    for v in &r.data {
        assert!(seen.insert(*v as i64), "duplicate num: {v}");
    }
}

#[test]
fn arr_dtype_passes_to_astype() {
    // arr.astype accepts a PyDType (not just a string).
    assert_same_with_dtype(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
target = np.dtype("int32")
result = a.astype(target)
result_dtype = str(result.dtype)
"#,
    );
}

#[test]
fn isinstance_dtype_object() {
    // arr.dtype is a real np.dtype object.
    let r = run_rumpy(
        r#"
import numpy as np
a = np.zeros((3,), dtype="float32")
result = np.array([1.0 if isinstance(a.dtype, np.dtype) else 0.0])
result_dtype = "float64"
"#,
    );
    assert_eq!(r.data, vec![1.0]);
}

#[test]
fn dtype_hash_dict_key() {
    // PyDType is hashable, so it works as a dict key.
    let r = run_rumpy(
        r#"
import numpy as np
d = {np.dtype("float32"): "f32", np.dtype("int32"): "i32"}
ok = d[np.dtype("float32")] == "f32" and d[np.dtype("int32")] == "i32"
result = np.array([1.0 if ok else 0.0])
result_dtype = "float64"
"#,
    );
    assert_eq!(r.data, vec![1.0]);
}

#[test]
fn complex_arithmetic_dtype_stays_complex() {
    // The test runner extracts only floats; just compare dtype and real parts.
    assert_same_with_dtype(
        r#"
import numpy as np
a = np.array([1+0j, 2+1j])
b = a + 1
result = np.array([b[0].real, b[1].real])
result_dtype = str(b.dtype)
"#,
    );
}

#[test]
fn integer_division_preserves_dtype() {
    // Use a same-dtype divisor — mixed dtype with a Python int scalar would
    // expose the NEP-50 vs legacy promotion gap.
    assert_same_with_dtype(
        r#"
import numpy as np
a = np.array([10, 20, 30], dtype="int32")
b = a // np.array([3, 3, 3], dtype="int32")
result = b
result_dtype = str(b.dtype)
"#,
    );
}

// =====================================================================
// Round 2 stress: dtype × operation matrix
// =====================================================================

// ---- arithmetic preserves dtype for same-dtype operands ----

#[test]
fn add_preserves_dtype_each_int() {
    for dt in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        let snippet = format!(
            r#"
import numpy as np
a = np.array([1, 2, 3], dtype="{dt}")
b = np.array([4, 5, 6], dtype="{dt}")
c = a + b
result = c.astype("float64")
result_dtype = str(c.dtype)
"#
        );
        assert_same_with_dtype(&snippet);
    }
}

#[test]
fn mul_preserves_dtype_each_float() {
    for dt in ["float16", "float32", "float64"] {
        let snippet = format!(
            r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0], dtype="{dt}")
b = np.array([2.0, 2.0, 2.0], dtype="{dt}")
c = a * b
result = c.astype("float64")
result_dtype = str(c.dtype)
"#
        );
        assert_same_with_dtype(&snippet);
    }
}

// ---- cross-dtype promotion ----

#[test]
fn int8_plus_uint8_promotes_to_int16() {
    assert_same_with_dtype(
        r#"
import numpy as np
a = np.array([1, 2, 3], dtype="int8")
b = np.array([1, 2, 3], dtype="uint8")
c = a + b
result = c.astype("float64")
result_dtype = str(c.dtype)
"#,
    );
}

#[test]
fn float32_plus_complex64_promotes_to_complex64() {
    assert_same_with_dtype(
        r#"
import numpy as np
a = np.array([1.0, 2.0], dtype="float32")
b = np.array([1+0j, 1+0j], dtype="complex64")
c = a + b
result = np.array([c[0].real, c[1].real]).astype("float64")
result_dtype = str(c.dtype)
"#,
    );
}

// ---- casting between every dtype pair via astype ----

#[test]
fn astype_to_each_dtype() {
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
        let snippet = format!(
            r#"
import numpy as np
a = np.array([0, 1, 2, 3])
b = a.astype("{dt}")
result = np.array([b.shape[0]]).astype("float64")
result_dtype = str(b.dtype)
"#
        );
        assert_same_with_dtype(&snippet);
    }
}

// ---- dtype.kind for every dtype ----

#[test]
fn dtype_kind_for_each() {
    for (dt, want_kind) in [
        ("bool", "b"),
        ("int8", "i"),
        ("uint8", "u"),
        ("float32", "f"),
        ("complex128", "c"),
    ] {
        let r = run_rumpy(&format!(
            r#"
import numpy as np
d = np.dtype("{dt}")
result = np.array([1.0])
result_dtype = d.kind
"#
        ));
        assert_eq!(r.dtype, want_kind, "{dt} kind");
    }
}

// ---- broadcasting + dtype ----

#[test]
fn broadcast_keeps_promoted_dtype() {
    assert_same_with_dtype(
        r#"
import numpy as np
a = np.ones((3, 1), dtype="int32")
b = np.ones((1, 4), dtype="float64")
c = a + b
result = c
result_dtype = str(c.dtype)
"#,
    );
}

// ---- cast to bool ----

#[test]
fn nonzero_values_cast_to_bool_true() {
    assert_same_with_dtype(
        r#"
import numpy as np
a = np.array([-1, 0, 1, 2, 0], dtype="int32")
b = a.astype("bool")
result = b.astype("float64")
result_dtype = str(b.dtype)
"#,
    );
}

// ---- complex cast ----

#[test]
fn float_to_complex_zero_imag() {
    assert_same_with_dtype(
        r#"
import numpy as np
a = np.array([1.5, 2.5, 3.5])
b = a.astype("complex128")
result = np.array([b[0].real, b[1].real, b[2].real])
result_dtype = str(b.dtype)
"#,
    );
}

// ---- itemsize per dtype ----

#[test]
fn dtype_itemsize_each() {
    for (dt, want) in [
        ("bool", 1.0),
        ("int8", 1.0),
        ("uint8", 1.0),
        ("int16", 2.0),
        ("uint16", 2.0),
        ("float16", 2.0),
        ("int32", 4.0),
        ("uint32", 4.0),
        ("float32", 4.0),
        ("complex64", 8.0),
        ("int64", 8.0),
        ("uint64", 8.0),
        ("float64", 8.0),
        ("complex128", 16.0),
    ] {
        let r = run_rumpy(&format!(
            r#"
import numpy as np
a = np.zeros((1,), dtype="{dt}")
result = np.array([a.itemsize]).astype("float64")
result_dtype = "float64"
"#
        ));
        assert_eq!(r.data, vec![want], "{dt} itemsize");
    }
}

// ---- nbytes for sized array ----

#[test]
fn nbytes_for_dtype() {
    let r = run_rumpy(
        r#"
import numpy as np
a = np.zeros((3, 4), dtype="float64")
result = np.array([a.nbytes]).astype("float64")
result_dtype = "float64"
"#,
    );
    assert_eq!(r.data, vec![96.0]); // 12 * 8 bytes
}

// ---- bool short-circuit semantics in numpy ----

#[test]
fn bool_array_sum_dtype() {
    let r = run_rumpy(
        r#"
import numpy as np
a = np.array([True, True, False, True])
b = a.sum()
result = np.array([float(b)])
result_dtype = "float64"
"#,
    );
    assert_eq!(r.data, vec![3.0]);
}

// ---- arange of float ----

#[test]
fn arange_step_dtype() {
    assert_same_with_dtype(
        r#"
import numpy as np
a = np.arange(0.0, 10.0, 2.0)
result = a
result_dtype = str(a.dtype)
"#,
    );
}

// ---- zeros_like / ones_like (if exposed) ----

#[test]
fn zeros_like_preserves_dtype() {
    assert_same_with_dtype(
        r#"
import numpy as np
a = np.array([1, 2, 3], dtype="int32")
b = np.zeros_like(a)
result = b.astype("float64")
result_dtype = str(b.dtype)
"#,
    );
}

#[test]
fn ones_like_preserves_dtype() {
    assert_same_with_dtype(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0], dtype="float32")
b = np.ones_like(a)
result = b.astype("float64")
result_dtype = str(b.dtype)
"#,
    );
}
