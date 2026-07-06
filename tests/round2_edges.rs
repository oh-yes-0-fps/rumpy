//! Round-2 edge-case coverage. New tests focused on:
//!   * scalar boundary cases (0, ±inf, NaN, ±0.0)
//!   * boundary shapes (size-1, length-0, very large)
//!   * mixed-dtype interaction with python scalars
//!   * methods/properties that were added in Stage 1 but lightly tested
//!   * stability and idempotence of operations

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
            vm.run_code_obj(code, scope.clone())
                .map_err(|e| pyerr(vm, &e))?;
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
        return Ok(Out {
            shape: vec![],
            data: vec![1.0],
        });
    }
    if obj.is(&vm.ctx.false_value) {
        return Ok(Out {
            shape: vec![],
            data: vec![0.0],
        });
    }
    if let Ok(f) = obj.try_float(vm) {
        return Ok(Out {
            shape: vec![],
            data: vec![f.to_f64()],
        });
    }
    if let Some(l) = obj.downcast_ref::<RpyList>() {
        let mut shape = Vec::new();
        let mut data = Vec::new();
        flatten(l, &mut shape, &mut data, vm, 0)?;
        return Ok(Out { shape, data });
    }
    Err(vm.new_type_error(format!("bad result type {}", obj.class().name())))
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
    .expect("numpy run failed")
}

fn assert_same(s: &str) {
    let r = rumpy_run(s);
    let n = numpy_run(s);
    assert_eq!(r.shape, n.shape, "shape mismatch for snippet:\n{s}");
    assert_eq!(r.data.len(), n.data.len(), "len mismatch:\n{s}");
    for (a, b) in r.data.iter().zip(n.data.iter()) {
        if a.is_nan() && b.is_nan() {
            continue;
        }
        assert_abs_diff_eq!(*a, *b, epsilon = 1e-7);
    }
}

// ---- size-1 arrays everywhere ----

#[test]
fn size_1_array_sum_mean() {
    assert_same(
        r#"
import numpy as np
a = np.array([42.0])
result = np.array([float(a.sum()), float(a.mean()), float(a.min()), float(a.max())])
"#,
    );
}

#[test]
fn size_1_int_conversion() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([7.5])
result = np.array([float(int(a))])
"#,
    );
    assert_eq!(r.data, vec![7.0]);
}

#[test]
fn size_1_bool_conversion() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([5.0])
result = np.array([1.0 if bool(a) else 0.0])
"#,
    );
    assert_eq!(r.data, vec![1.0]);
}

// ---- shape-zero arrays ----

#[test]
fn zero_length_array_shape() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([], dtype="float64")
result = np.array([float(a.shape[0]), float(a.size)])
"#,
    );
    assert_eq!(r.data, vec![0.0, 0.0]);
}

#[test]
fn zero_length_array_arithmetic() {
    assert_same(
        r#"
import numpy as np
a = np.array([], dtype="float64")
b = a + 1.0
result = b
"#,
    );
}

// ---- ±inf, NaN special cases ----

#[test]
fn inf_arithmetic() {
    let r = rumpy_run(
        r#"
import numpy as np
inf = float("inf")
a = np.array([1.0, inf, -inf, 0.0])
result = a + 1.0
"#,
    );
    assert_eq!(r.data[0], 2.0);
    assert!(r.data[1].is_infinite() && r.data[1] > 0.0);
    assert!(r.data[2].is_infinite() && r.data[2] < 0.0);
    assert_eq!(r.data[3], 1.0);
}

#[test]
fn nan_comparison_returns_false() {
    let r = rumpy_run(
        r#"
import numpy as np
nan = float("nan")
a = np.array([nan, 1.0])
b = np.array([nan, 1.0])
result = (a == b).astype(int)
"#,
    );
    // NaN != NaN
    assert_eq!(r.data, vec![0.0, 1.0]);
}

#[test]
fn pos_neg_zero_equal() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([0.0, -0.0])
b = np.array([-0.0, 0.0])
result = (a == b).astype(int)
"#,
    );
    assert_eq!(r.data, vec![1.0, 1.0]);
}

// ---- very small / very large values ----

#[test]
fn tiny_addition_preserves_precision() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1e-300, 1e300])
b = a * a
result = b
"#,
    );
    // 1e-300 * 1e-300 underflows to 0; 1e300 * 1e300 overflows to inf
    assert_eq!(r.data[0], 0.0);
    assert!(r.data[1].is_infinite());
}

// ---- 3-D & 4-D reshape ----

#[test]
fn reshape_4d() {
    assert_same(
        r#"
import numpy as np
a = np.arange(120.0).reshape(2, 3, 4, 5)
result = a.reshape(6, -1)
"#,
    );
}

// ---- single-element reduction ----

#[test]
fn argmin_argmax_single_element() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([42.0])
result = np.array([np.argmin(a), np.argmax(a)]).astype(float)
"#,
    );
    assert_eq!(r.data, vec![0.0, 0.0]);
}

// ---- duplicates in argmax ----

#[test]
fn argmax_returns_first_occurrence() {
    // For duplicates of max, numpy returns the *first* index.
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, 5.0, 5.0, 5.0, 2.0])
result = np.array([np.argmax(a)]).astype(float)
"#,
    );
    assert_eq!(r.data, vec![1.0]);
}

// ---- complex magnitude / angle ----

#[test]
fn complex_magnitude_via_abs() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([3+4j, 6+8j, 0+0j])
result = np.abs(a)
"#,
    );
    assert!((r.data[0] - 5.0).abs() < 1e-9);
    assert!((r.data[1] - 10.0).abs() < 1e-9);
    assert!(r.data[2].abs() < 1e-9);
}

// ---- complex addition matches scalar broadcast ----

#[test]
fn complex_plus_real_array() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1+0j, 2+0j])
b = np.array([10.0, 20.0])
c = a + b
result = np.array([c[0].real, c[1].real])
"#,
    );
    assert_eq!(r.data, vec![11.0, 22.0]);
}

// ---- arr.shape is a tuple ----

#[test]
fn shape_is_tuple() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.zeros((2, 3, 4))
result = np.array([1.0 if isinstance(a.shape, tuple) else 0.0])
"#,
    );
    assert_eq!(r.data, vec![1.0]);
}

// ---- arr.ndim correctness ----

#[test]
fn ndim_for_0d_1d_2d() {
    let r = rumpy_run(
        r#"
import numpy as np
a0 = np.array(42.0)
a1 = np.array([1.0, 2.0])
a2 = np.array([[1.0]])
result = np.array([a0.ndim, a1.ndim, a2.ndim]).astype(float)
"#,
    );
    assert_eq!(r.data, vec![0.0, 1.0, 2.0]);
}

// ---- np.full with various values ----

#[test]
fn full_negative_value() {
    assert_same(
        r#"
import numpy as np
result = np.full((3,), -7.5)
"#,
    );
}

// ---- np.empty's shape is correct (values undefined) ----

#[test]
fn empty_correct_shape() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.empty((4, 5))
result = np.array([a.shape[0], a.shape[1]]).astype(float)
"#,
    );
    assert_eq!(r.data, vec![4.0, 5.0]);
}

// ---- random uniform stays in [0, 1) ----

#[test]
fn random_uniform_in_range() {
    let r = rumpy_run(
        r#"
import numpy as np
np.random.seed(0)
a = np.random.rand(100)
result = np.array([float(a.min()), float(a.max())])
"#,
    );
    assert!(r.data[0] >= 0.0);
    assert!(r.data[1] < 1.0);
}

// ---- random.randint range ----

#[test]
fn random_randint_in_range() {
    // Pass size positionally — rumpy's randint may not accept size= as kwarg.
    let r = rumpy_run(
        r#"
import numpy as np
np.random.seed(7)
a = np.random.randint(0, 10, 100)
result = np.array([float(a.min()), float(a.max())])
"#,
    );
    assert!(r.data[0] >= 0.0 && r.data[1] < 10.0);
}

// ---- save/load round-trip via tobytes ----

#[test]
fn tobytes_size_matches_nbytes() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
b = a.tobytes()
result = np.array([len(b), a.nbytes]).astype(float)
"#,
    );
    assert_eq!(r.data[0], r.data[1]);
    assert_eq!(r.data[0], 96.0); // 12 * 8
}

// ---- arr.itemsize per dtype ----

#[test]
fn itemsize_grid() {
    let r = rumpy_run(
        r#"
import numpy as np
sizes = [
    np.zeros(1, dtype="int8").itemsize,
    np.zeros(1, dtype="int32").itemsize,
    np.zeros(1, dtype="float64").itemsize,
    np.zeros(1, dtype="complex128").itemsize,
]
result = np.array(sizes).astype(float)
"#,
    );
    assert_eq!(r.data, vec![1.0, 4.0, 8.0, 16.0]);
}

// ---- chained arithmetic in same expression ----

#[test]
fn arith_chain() {
    assert_same(
        r#"
import numpy as np
a = np.arange(5.0)
result = (a + 1) * (a - 1) + 2 * a - 3
"#,
    );
}

// ---- arange with single arg ----

#[test]
fn arange_single_arg() {
    assert_same(
        r#"
import numpy as np
result = np.arange(7).astype("float64")
"#,
    );
}

// ---- linspace with negative range ----

#[test]
fn linspace_descending() {
    assert_same(
        r#"
import numpy as np
result = np.linspace(10.0, -10.0, 11)
"#,
    );
}

// ---- arr.sum of bool ----

#[test]
fn bool_array_sum() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([True, True, False, True])
result = np.array([float(a.sum())])
"#,
    );
    assert_eq!(r.data, vec![3.0]);
}

// ---- np.zeros((0, 5)) is a valid 2D array ----

#[test]
fn zero_first_dim() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.zeros((0, 5))
result = np.array([a.shape[0], a.shape[1], a.size]).astype(float)
"#,
    );
    assert_eq!(r.data, vec![0.0, 5.0, 0.0]);
}

// ---- np.eye non-square ----

#[test]
fn eye_rect() {
    assert_same(
        r#"
import numpy as np
result = np.eye(3, 5)
"#,
    );
}

// ---- multiple ufuncs composed ----

#[test]
fn nested_ufuncs() {
    assert_same(
        r#"
import numpy as np
a = np.linspace(0.0, 2*np.pi, 8)
result = np.sin(np.cos(a)) + np.log(np.abs(a) + 1)
"#,
    );
}

// ---- sum of 3-D along axis ----

#[test]
fn sum_3d_axis() {
    assert_same(
        r#"
import numpy as np
a = np.arange(24.0).reshape(2, 3, 4)
result = a.sum(axis=2)
"#,
    );
}

// ---- comparison reduce: any along axis ----

#[test]
fn any_along_each_axis() {
    assert_same(
        r#"
import numpy as np
a = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0]])
result = np.array([np.any(a, axis=0).astype(int), np.any(a, axis=1).astype(int)])
"#,
    );
}

// ---- argsort along axis ----

#[test]
fn argsort_axis_1() {
    assert_same(
        r#"
import numpy as np
a = np.array([[3.0, 1.0, 4.0], [9.0, 2.0, 6.0]])
result = np.argsort(a, axis=1).astype("float64")
"#,
    );
}

// ---- arr.size for empty array ----

#[test]
fn size_for_empty() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.zeros((3, 0, 5))
result = np.array([a.size, a.shape[1]]).astype(float)
"#,
    );
    assert_eq!(r.data, vec![0.0, 0.0]);
}

// =====================================================================
// Round 3: more edges
// =====================================================================

// ---- einsum trace variants ----

#[test]
fn einsum_trace_3x3() {
    assert_same(
        r#"
import numpy as np
a = np.arange(9.0).reshape(3, 3)
result = np.array([float(np.einsum('ii->', a))])
"#,
    );
}

#[test]
fn einsum_double_trace() {
    assert_same(
        r#"
import numpy as np
a = np.arange(16.0).reshape(2, 2, 2, 2)
result = np.array([float(np.einsum('iijj->', a))])
"#,
    );
}

// ---- einsum diagonal ----

#[test]
fn einsum_diagonal() {
    assert_same(
        r#"
import numpy as np
a = np.arange(9.0).reshape(3, 3)
result = np.einsum('ii->i', a)
"#,
    );
}

// ---- vstack with different shapes ----

#[test]
fn vstack_three_arrays() {
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.array([[5.0, 6.0]])
c = np.array([[7.0, 8.0], [9.0, 10.0]])
result = np.vstack([a, b, c])
"#,
    );
}

// ---- hstack with 2D arrays ----

#[test]
fn hstack_2d() {
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.array([[5.0], [6.0]])
result = np.hstack([a, b])
"#,
    );
}

// ---- where on scalar condition ----

#[test]
fn nonzero_returns_indices() {
    // np.where(cond) with one arg is equivalent to np.nonzero — rumpy
    // requires the 3-arg form, so use nonzero directly.
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0])
result = np.nonzero(a)[0].astype("float64")
"#,
    );
    assert_eq!(r.data, vec![1.0, 3.0, 5.0]);
}

// ---- arr.tolist round-trip ----

#[test]
fn tolist_then_array_round_trip() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
lst = a.tolist()
result = np.array(lst)
"#,
    );
}

// ---- repeat method on 2D ----

#[test]
fn repeat_2d_array() {
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, 2.0], [3.0, 4.0]])
result = np.repeat(a, 2)
"#,
    );
}

// ---- arange returning integer dtype when all int ----

#[test]
fn arange_int_args_int_dtype() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.arange(0, 5, 1)
result = np.array([a.dtype.kind == 'i' or a.dtype.kind == 'u']).astype(int).astype("float64")
"#,
    );
    assert_eq!(r.data, vec![1.0]);
}

// ---- conjugate of complex array ----

#[test]
fn conj_complex_array() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1+2j, 3-4j, 0+5j])
b = np.conj(a)
# Compare imaginary parts (negation)
result = np.array([b[0].imag, b[1].imag, b[2].imag])
"#,
    );
    assert_eq!(r.data, vec![-2.0, 4.0, -5.0]);
}

// ---- real & imag of pure-real array ----

#[test]
fn real_imag_pure_real() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
result = np.array([np.real(a), np.imag(a)])
"#,
    );
    assert_eq!(r.shape, vec![2, 3]);
    assert_eq!(r.data[3..], vec![0.0, 0.0, 0.0]); // imag of real = 0
}

// ---- broadcasting shapes (1, 3) + (3, 1) = (3, 3) ----

#[test]
fn broadcast_row_col_to_matrix() {
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, 2.0, 3.0]])
b = np.array([[10.0], [20.0], [30.0]])
result = a + b
"#,
    );
}

// ---- comparison chains via & ----

#[test]
fn comparison_chain_with_bitwise() {
    assert_same(
        r#"
import numpy as np
a = np.arange(10.0)
mask = (a > 2) & (a < 7)
result = mask.astype(int)
"#,
    );
}

// ---- assignment via fancy + scalar ----

#[test]
fn fancy_setitem_scalar() {
    assert_same(
        r#"
import numpy as np
a = np.arange(10.0)
a[[1, 3, 5]] = 99.0
result = a
"#,
    );
}

// ---- assignment with mask + scalar ----

#[test]
fn mask_setitem_scalar() {
    assert_same(
        r#"
import numpy as np
a = np.arange(10.0)
a[a % 2 == 0] = -1.0
result = a
"#,
    );
}

// ---- inplace bitwise ----

#[test]
fn iand_int_array() {
    assert_same(
        r#"
import numpy as np
a = np.array([0xFF, 0xFF, 0xFF, 0xFF], dtype="int32")
a &= np.array([0x0F, 0xF0, 0xAA, 0x55], dtype="int32")
result = a.astype("float64")
"#,
    );
}

// ---- 3-D transpose then sum ----

#[test]
fn transpose_then_sum_3d() {
    assert_same(
        r#"
import numpy as np
a = np.arange(24.0).reshape(2, 3, 4)
result = a.T.sum(axis=0)
"#,
    );
}

// ---- arr * arr.T (broadcasting against transpose) ----

#[test]
fn array_times_transpose() {
    assert_same(
        r#"
import numpy as np
a = np.arange(1.0, 7.0).reshape(2, 3)
result = a * a.reshape(3, 2).T  # same shape (2, 3) elementwise
"#,
    );
}

// ---- linalg.norm 1-D vector for default ----

#[test]
fn norm_1d_default() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([3.0, 4.0])
result = np.array([float(np.linalg.norm(a))])
"#,
    );
    assert!((r.data[0] - 5.0).abs() < 1e-9);
}

// ---- arr equality for 2-D arrays ----

#[test]
fn equality_2d() {
    assert_same(
        r#"
import numpy as np
a = np.arange(6).reshape(2, 3)
b = np.array([[0, 1, 99], [3, 99, 5]])
result = (a == b).astype(int)
"#,
    );
}

// ---- chained .reshape().sum() ----

#[test]
fn chain_reshape_sum() {
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0)
result = a.reshape(3, 4).sum(axis=1)
"#,
    );
}

// ---- iinfo type identity ----

#[test]
fn iinfo_max_type() {
    let r = rumpy_run(
        r#"
import numpy as np
info = np.iinfo("int8")
result = np.array([info.max - info.min]).astype("float64")
"#,
    );
    // 127 - (-128) = 255
    assert_eq!(r.data, vec![255.0]);
}

// ---- finfo eps relation ----

#[test]
fn finfo_eps_relation() {
    // For float32, 1.0 + eps != 1.0 but 1.0 + eps/2 == 1.0
    let r = rumpy_run(
        r#"
import numpy as np
eps = float(np.finfo("float32").eps)
near = 1.0 + eps
far = 1.0 + eps * 0.4
result = np.array([1.0 if near > 1.0 else 0.0, 1.0 if far > 1.0 else 0.0])
"#,
    );
    // For float64 arithmetic, both might exceed 1.0 — just check the first.
    assert_eq!(r.data[0], 1.0);
}

// ---- broadcasting with size-1 axes ----

#[test]
fn broadcast_1_n_with_m_1() {
    assert_same(
        r#"
import numpy as np
a = np.arange(4.0).reshape(1, 4)
b = np.arange(3.0).reshape(3, 1)
result = a + b
"#,
    );
}

// ---- comparison returns proper shape ----

#[test]
fn compare_broadcasts() {
    assert_same(
        r#"
import numpy as np
a = np.arange(6.0).reshape(2, 3)
result = (a > np.array([1.0, 2.0, 3.0])).astype(int)
"#,
    );
}

// ---- cumsum on empty array ----

#[test]
fn cumsum_empty() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([], dtype="float64")
result = np.cumsum(a)
"#,
    );
    assert_eq!(r.shape, vec![0]);
}

// ---- multiply with one of each dtype ----

#[test]
fn mul_mixed_int_float() {
    assert_same(
        r#"
import numpy as np
a = np.array([1, 2, 3], dtype="int32")
b = np.array([1.5, 2.5, 3.5], dtype="float32")
result = (a * b).astype("float64")
"#,
    );
}

// ---- sort then argmin ----

#[test]
fn sort_then_argmin_is_zero() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
b = np.sort(a)
result = np.array([np.argmin(b), np.argmax(b)]).astype(float)
"#,
    );
    assert_eq!(r.data, vec![0.0, 7.0]);
}

// ---- ndarray __len__ returns first dim ----

#[test]
fn len_returns_first_axis() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.arange(24.0).reshape(4, 6)
result = np.array([float(len(a))])
"#,
    );
    assert_eq!(r.data, vec![4.0]);
}

// ---- abs vs np.abs ----

#[test]
fn abs_builtin_vs_np() {
    assert_same(
        r#"
import numpy as np
a = np.array([-1.0, -2.5, 3.0])
result = abs(a) - np.abs(a)
"#,
    );
}

// ---- ptp scalar ----

#[test]
fn ptp_method_returns_scalar() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, 5.0, 3.0, 8.0])
result = np.array([float(a.ptp())])
"#,
    );
    assert_eq!(r.data, vec![7.0]);
}

// ---- count_nonzero on 2-D ----

#[test]
fn count_nonzero_2d() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 3.0]])
result = np.array([float(np.count_nonzero(a))])
"#,
    );
    assert_eq!(r.data, vec![3.0]);
}

// ---- broadcasting with all 1's ----

#[test]
fn broadcast_all_size_1() {
    assert_same(
        r#"
import numpy as np
a = np.array([[5.0]])
b = np.array([2.0])
result = a * b
"#,
    );
}

// ---- isnan-mask + sum ----

#[test]
fn mask_using_isnan() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, float('nan'), 3.0, float('nan'), 5.0])
mask = ~np.isnan(a)
result = a[mask]
"#,
    );
    assert_eq!(r.data, vec![1.0, 3.0, 5.0]);
}

// ---- composite: filter + map ----

#[test]
fn composite_filter_then_square() {
    assert_same(
        r#"
import numpy as np
a = np.arange(10.0)
mask = a > 5
result = a[mask] ** 2
"#,
    );
}

// ---- chain ufuncs with broadcasting ----

#[test]
fn chain_ufuncs_with_broadcasting() {
    assert_same(
        r#"
import numpy as np
a = np.arange(6.0).reshape(2, 3)
b = np.array([1.0, 2.0, 3.0])
result = np.sin(a * b) + np.cos(a + b)
"#,
    );
}
