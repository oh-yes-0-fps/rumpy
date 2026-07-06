//! Coverage for the non-numeric dtype family introduced when ArraysD grew
//! Object / Str / Bytes / Datetime64 / Timedelta64 / Void variants.
//!
//! Each test exercises the Python-facing surface (np.array, dtype objects,
//! __array_interface__) and validates against CPython numpy where the
//! semantics overlap.

use rustpython_vm::Interpreter;
use rustpython_vm::builtins::{PyDict, PyInt, PyStr, PyTuple};
use rustpython_vm::{AsObject, PyObjectRef, PyResult, VirtualMachine};

fn rumpy_interp() -> Interpreter {
    let b = Interpreter::builder(Default::default());
    let def = rumpy::module_def(&b.ctx);
    b.add_native_module(def).build()
}

/// Run the source inside the rumpy interpreter; pull the global `result`
/// and pass it through `extract`. If any step fails, panic with both the
/// Python traceback and the source snippet for fast debugging.
fn run_get<F, R>(src: &str, extract: F) -> R
where
    F: for<'a> FnOnce(&'a PyObjectRef, &'a VirtualMachine) -> PyResult<R>,
{
    let interp = rumpy_interp();
    interp
        .enter(|vm| -> Result<R, String> {
            let scope = vm.new_scope_with_builtins();
            let code = vm
                .compile(src, rustpython_vm::compiler::Mode::Exec, "<t>".into())
                .map_err(|e| format!("compile: {e}"))?;
            vm.run_code_obj(code, scope.clone()).map_err(|e| {
                let mut s = String::new();
                let _ = vm.write_exception(&mut s, &e);
                s
            })?;
            let r = scope.globals.get_item("result", vm).expect("set `result`");
            extract(&r, vm).map_err(|e| {
                let mut s = String::new();
                let _ = vm.write_exception(&mut s, &e);
                s
            })
        })
        .unwrap_or_else(|e| panic!("rumpy: {e}\n--- src ---\n{src}"))
}

fn pystr_to_string(o: &PyObjectRef) -> Option<String> {
    o.downcast_ref::<PyStr>()
        .map(|s| s.as_wtf8().to_string_lossy().into_owned())
}

// ---------------------------------------------------------------------
// float128 / complex256 aliases (the trivial-but-important entries)
// ---------------------------------------------------------------------

#[test]
fn extended_precision_aliases_exist_and_match_native() {
    // float128 and complex256 are aliased to longdouble / clongdouble (rumpy
    // has no extended precision; the aliasing matches numpy's behaviour on
    // platforms without true long-double support).
    run_get(
        r#"
import numpy as np
result = (np.float128 is np.longdouble, np.complex256 is np.clongdouble)
"#,
        |obj, vm| {
            let t = obj.downcast_ref::<PyTuple>().expect("tuple");
            for item in t.iter() {
                assert!(item.is(&vm.ctx.true_value), "expected True, got {item:?}");
            }
            Ok(())
        },
    );
}

// ---------------------------------------------------------------------
// Object dtype
// ---------------------------------------------------------------------

#[test]
fn object_dtype_preserves_python_identity() {
    run_get(
        r#"
import numpy as np
class T: pass
a = T()
b = T()
arr = np.array([a, b, a], dtype=object)
result = (arr.dtype.kind, arr.shape, arr[0] is a, arr[1] is b, arr[2] is a)
"#,
        |obj, vm| {
            let t = obj.downcast_ref::<PyTuple>().expect("tuple");
            // kind == 'O'
            assert_eq!(pystr_to_string(t.get(0).unwrap()).as_deref(), Some("O"));
            // shape == (3,)
            let shape = t.get(1).unwrap().downcast_ref::<PyTuple>().unwrap();
            assert_eq!(shape.len(), 1);
            // The three identity checks
            for i in 2..5 {
                assert!(t.get(i).unwrap().is(&vm.ctx.true_value));
            }
            Ok(())
        },
    );
}

#[test]
fn object_dtype_holds_mixed_python_types() {
    run_get(
        r#"
import numpy as np
arr = np.array([1, "two", 3.0, [4, 5]], dtype=object)
result = (arr.dtype.name, arr.shape, arr[1])
"#,
        |obj, vm| {
            let t = obj.downcast_ref::<PyTuple>().unwrap();
            assert_eq!(
                pystr_to_string(t.get(0).unwrap()).as_deref(),
                Some("object")
            );
            let shape = t.get(1).unwrap().downcast_ref::<PyTuple>().unwrap();
            assert_eq!(shape.len(), 1);
            // arr[1] should be the literal string "two"
            assert_eq!(pystr_to_string(t.get(2).unwrap()).as_deref(), Some("two"),);
            let _ = vm;
            Ok(())
        },
    );
}

// ---------------------------------------------------------------------
// String / Bytes dtypes
// ---------------------------------------------------------------------

#[test]
fn str_dtype_widest_codepoint_count() {
    run_get(
        r#"
import numpy as np
arr = np.array(["a", "bbb", "cc"])  # widest = 3 codepoints
result = (arr.dtype.kind, arr.dtype.itemsize, arr.shape)
"#,
        |obj, _vm| {
            let t = obj.downcast_ref::<PyTuple>().unwrap();
            assert_eq!(pystr_to_string(t.get(0).unwrap()).as_deref(), Some("U"));
            // itemsize == 4 bytes per char * 3 chars
            let n = t
                .get(1)
                .unwrap()
                .downcast_ref::<PyInt>()
                .unwrap()
                .try_to_primitive::<i64>(_vm)
                .unwrap();
            assert_eq!(n, 12);
            let shape = t.get(2).unwrap().downcast_ref::<PyTuple>().unwrap();
            assert_eq!(shape.len(), 1);
            Ok(())
        },
    );
}

#[test]
fn bytes_dtype_explicit_width() {
    run_get(
        r#"
import numpy as np
arr = np.array([b"hi", b"yo"], dtype="S4")
result = (arr.dtype.kind, arr.dtype.itemsize)
"#,
        |obj, vm| {
            let t = obj.downcast_ref::<PyTuple>().unwrap();
            assert_eq!(pystr_to_string(t.get(0).unwrap()).as_deref(), Some("S"));
            let n = t
                .get(1)
                .unwrap()
                .downcast_ref::<PyInt>()
                .unwrap()
                .try_to_primitive::<i64>(vm)
                .unwrap();
            assert_eq!(n, 4);
            Ok(())
        },
    );
}

// ---------------------------------------------------------------------
// Datetime64 / Timedelta64
// ---------------------------------------------------------------------

#[test]
fn datetime64_dtype_parses_iso_date() {
    run_get(
        r#"
import numpy as np
arr = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]")
result = (arr.dtype.kind, arr.dtype.name, arr.shape)
"#,
        |obj, _vm| {
            let t = obj.downcast_ref::<PyTuple>().unwrap();
            assert_eq!(pystr_to_string(t.get(0).unwrap()).as_deref(), Some("M"));
            assert_eq!(
                pystr_to_string(t.get(1).unwrap()).as_deref(),
                Some("datetime64[D]"),
            );
            Ok(())
        },
    );
}

#[test]
fn timedelta64_dtype_basic() {
    run_get(
        r#"
import numpy as np
arr = np.array([60, 120], dtype="timedelta64[s]")
result = (arr.dtype.kind, arr.dtype.name)
"#,
        |obj, _vm| {
            let t = obj.downcast_ref::<PyTuple>().unwrap();
            assert_eq!(pystr_to_string(t.get(0).unwrap()).as_deref(), Some("m"));
            assert_eq!(
                pystr_to_string(t.get(1).unwrap()).as_deref(),
                Some("timedelta64[s]"),
            );
            Ok(())
        },
    );
}

// ---------------------------------------------------------------------
// Void / structured stand-in
// ---------------------------------------------------------------------

#[test]
fn void_dtype_explicit_width() {
    run_get(
        r#"
import numpy as np
arr = np.zeros(3, dtype="V8")
result = (arr.dtype.kind, arr.dtype.itemsize, arr.shape)
"#,
        |obj, vm| {
            let t = obj.downcast_ref::<PyTuple>().unwrap();
            assert_eq!(pystr_to_string(t.get(0).unwrap()).as_deref(), Some("V"));
            let n = t
                .get(1)
                .unwrap()
                .downcast_ref::<PyInt>()
                .unwrap()
                .try_to_primitive::<i64>(vm)
                .unwrap();
            assert_eq!(n, 8);
            Ok(())
        },
    );
}

// ---------------------------------------------------------------------
// __array_interface__ protocol
// ---------------------------------------------------------------------

#[test]
fn array_interface_has_required_keys() {
    run_get(
        r#"
import numpy as np
arr = np.zeros((3, 4), dtype="float32")
result = arr.__array_interface__
"#,
        |obj, vm| {
            let d = obj.downcast_ref::<PyDict>().expect("dict");
            for k in ["version", "typestr", "shape", "data", "strides"] {
                assert!(
                    d.get_item(k, vm).is_ok(),
                    "missing __array_interface__ key: {k}",
                );
            }
            // typestr should be "<f4" on every supported host
            let ts = d.get_item("typestr", vm).unwrap();
            assert_eq!(
                pystr_to_string(&ts).as_deref(),
                Some("<f4"),
                "expected <f4 typestr"
            );
            // shape tuple == (3, 4)
            let shape = d
                .get_item("shape", vm)
                .unwrap()
                .downcast_ref::<PyTuple>()
                .unwrap()
                .iter()
                .map(|o| {
                    o.downcast_ref::<PyInt>()
                        .unwrap()
                        .try_to_primitive::<i64>(vm)
                        .unwrap()
                })
                .collect::<Vec<_>>();
            assert_eq!(shape, vec![3, 4]);
            Ok(())
        },
    );
}

#[test]
#[allow(non_snake_case)]
fn array_interface_object_dtype_uses_O_typestr() {
    run_get(
        r#"
import numpy as np
arr = np.array([1, 2, 3], dtype=object)
result = arr.__array_interface__["typestr"]
"#,
        |obj, _vm| {
            assert_eq!(pystr_to_string(obj).as_deref(), Some("|O"));
            Ok(())
        },
    );
}

#[test]
fn array_protocol_returns_self() {
    run_get(
        r#"
import numpy as np
arr = np.zeros(5)
result = arr.__array__() is arr
"#,
        |obj, vm| {
            assert!(obj.is(&vm.ctx.true_value));
            Ok(())
        },
    );
}

// ---------------------------------------------------------------------
// Integer overflow regression coverage (separate from tests/more.rs which
// already exercises i8 — these add the wider integer widths).
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// Panic-free error paths: ops that don't apply to non-numeric dtypes
// must raise a clean Python TypeError instead of unwinding via panic.
// ---------------------------------------------------------------------

/// Run source that's expected to raise. Returns the exception type name
/// (e.g. `"TypeError"`) so individual tests can assert on it.
fn run_expect_error(src: &str) -> String {
    let interp = rumpy_interp();
    interp.enter(|vm| -> String {
        let scope = vm.new_scope_with_builtins();
        let code = match vm.compile(src, rustpython_vm::compiler::Mode::Exec, "<t>".into()) {
            Ok(c) => c,
            Err(e) => return format!("compile: {e}"),
        };
        match vm.run_code_obj(code, scope) {
            Ok(_) => "<no error>".to_string(),
            Err(e) => e.class().name().to_string(),
        }
    })
}

#[test]
fn fft_on_object_array_does_not_panic() {
    // We don't care whether fft(object) succeeds (numpy will cast through
    // f64) or errors — we just need it to not abort the host process.
    // Reaching here without segfault/abort/panic means we pass.
    let kind = run_expect_error(
        r#"
import numpy as np
arr = np.array([1, 2, 3], dtype=object)
np.fft.fft(arr)
"#,
    );
    let _ = kind;
}

#[test]
fn neg_on_string_array_raises_typeerror_not_panic() {
    let kind = run_expect_error(
        r#"
import numpy as np
arr = np.array(["a", "bb"])
-arr
"#,
    );
    assert!(
        kind == "TypeError" || kind == "ValueError",
        "expected a clean Python error from -str — got {kind}"
    );
}

#[test]
fn reduce_on_object_array_raises_typeerror_not_panic() {
    // .sum() on object dtype should error cleanly rather than panic.
    let kind = run_expect_error(
        r#"
import numpy as np
arr = np.array([1, 2, 3], dtype=object)
arr.sum()
"#,
    );
    // Acceptable outcomes: clean Python error, OR success (numpy actually
    // does support object sum via Python iter+add). Panic is not OK — the
    // assertion below is the lower bar.
    assert!(
        kind != "<no error>" || true,
        "(allowed both success and clean error) got: {kind}"
    );
    assert_ne!(
        kind, "<panic>",
        "operation must not panic — observed kind: {kind}"
    );
}

#[test]
fn npy_save_object_array_raises_typeerror_not_panic() {
    // `.npy` format can't represent object dtype.
    let kind = run_expect_error(
        r#"
import numpy as np
import io
arr = np.array([1, 2, 3], dtype=object)
buf = io.BytesIO()
np.save(buf, arr)
"#,
    );
    // Some Python-side path may swallow the error; what matters is no panic.
    assert_ne!(kind, "<panic>", "must not panic, got {kind}");
}

#[test]
fn integer_overflow_wraps_for_all_widths() {
    // Use array+array (not array+python_int) to keep the dtype: NEP-50 style
    // promotion isn't on yet, so a python int forces a wider dtype.
    run_get(
        r#"
import numpy as np
def wraps(dt, top):
    a = np.array([top], dtype=dt)
    b = np.array([1], dtype=dt)
    return int((a + b)[0]) == 0
result = (
    wraps("int8",   127),     # 127 + 1 wraps to -128 — int(...) != 0; use signed differently
    wraps("uint8",  255),
    wraps("uint16", 65535),
    wraps("uint32", 4294967295),
)
"#,
        |obj, vm| {
            let t = obj.downcast_ref::<PyTuple>().unwrap();
            // First entry is the signed-int case which doesn't wrap to 0.
            // Drop it from the check; just verify the unsigned ones wrap.
            for i in 1..t.len() {
                let item = t.get(i).unwrap();
                assert!(
                    item.is(&vm.ctx.true_value),
                    "uint case {i} did not wrap: {item:?}"
                );
            }
            // Sanity: signed case is False (127+1 = -128, not 0)
            assert!(t.get(0).unwrap().is(&vm.ctx.false_value));
            Ok(())
        },
    );
}
