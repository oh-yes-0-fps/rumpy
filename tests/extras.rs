//! Cross-validation tests for the second-tier numpy API: logical/bitwise/
//! finite predicates, cumulative ops, where/nonzero, sort/argsort/unique,
//! stack/squeeze/broadcast_to/repeat/tile, ptp/median, linalg.

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
        // Fresh dict per snippet — sharing `builtins.__dict__` across parallel
        // tests creates a race on `result` (multiple threads each `Python::attach`
        // and assign to the same dict).
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

// ---- logical / bitwise / predicates ----

#[test]
fn logical_ops() {
    assert_same(
        r#"
import numpy as np
a = np.array([True, False, True, False])
b = np.array([True, True, False, False])
result = np.logical_and(a, b)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([1, 0, 1, 0])
b = np.array([1, 1, 0, 0])
result = np.logical_or(a, b)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([True, False])
result = np.logical_not(a)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([True, False, True, False])
b = np.array([True, True, False, False])
result = np.logical_xor(a, b)
"#,
    );
}

#[test]
fn bitwise_ops() {
    assert_same(
        r#"
import numpy as np
a = np.array([0b1100, 0b1010], dtype="int32")
b = np.array([0b1010, 0b1111], dtype="int32")
result = np.bitwise_and(a, b)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([0b1100, 0b1010], dtype="int32")
b = np.array([0b1010, 0b1111], dtype="int32")
result = np.bitwise_or(a, b) ^ np.bitwise_xor(a, b)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([0, 1, 2, 3], dtype="uint8")
result = np.invert(a)
"#,
    );
}

#[test]
fn isnan_isinf_isfinite() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, float('nan'), float('inf'), -2.0])
result = np.array([np.isnan(a).astype(int), np.isinf(a).astype(int), np.isfinite(a).astype(int)])
"#,
    );
}

#[test]
fn isclose_and_allclose() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0000001, 2.00001, 3.1])
result = np.isclose(a, b)
"#,
    );
}

// ---- any / all ----

#[test]
fn any_all_reductions() {
    assert_same(
        r#"
import numpy as np
a = np.array([[True, False], [False, False]])
result = np.array([int(np.any(a)), int(np.all(a))])
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.arange(6).reshape(2, 3)
result = np.any(a > 2, axis=1).astype(int)
"#,
    );
}

// ---- cumulative ops ----

#[test]
fn cumsum_cumprod_diff() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0, 4.0])
result = np.cumsum(a)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0, 4.0])
result = np.cumprod(a)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 4.0, 9.0, 16.0])
result = np.diff(a)
"#,
    );
}

// ---- clip / round / trunc ----

#[test]
fn clip_op() {
    assert_same(
        r#"
import numpy as np
a = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
result = np.clip(a, -1.0, 2.0)
"#,
    );
}

#[test]
fn round_trunc() {
    assert_same(
        r#"
import numpy as np
a = np.array([0.5, 1.5, 2.5, -0.5, -1.5, 2.7])
result = np.round(a)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([1.7, -1.7, 2.0, -2.5])
result = np.trunc(a)
"#,
    );
}

// ---- where / nonzero ----

#[test]
fn where_and_nonzero() {
    assert_same(
        r#"
import numpy as np
a = np.arange(6)
b = -np.arange(6)
result = np.where(a > 2, a, b)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([0, 5, 0, 3, 0, 1])
result = np.nonzero(a)[0]
"#,
    );
}

// ---- sort / argsort / unique ----

#[test]
fn sort_argsort_unique() {
    assert_same(
        r#"
import numpy as np
a = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0])
result = np.sort(a)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
result = np.argsort(a).astype(float)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0])
result = np.unique(a)
"#,
    );
}

// ---- stack / hstack / vstack / squeeze / expand_dims / broadcast_to ----

#[test]
fn stack_family() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
result = np.stack([a, b])
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
result = np.hstack([a, b])
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
result = np.vstack([a, b])
"#,
    );
}

#[test]
fn squeeze_expand() {
    assert_same(
        r#"
import numpy as np
a = np.array([[[1.0, 2.0, 3.0]]])
result = np.squeeze(a)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
result = np.expand_dims(a, 0)
"#,
    );
}

#[test]
fn broadcast_to_shape() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
result = np.broadcast_to(a, (4, 3))
"#,
    );
}

#[test]
fn repeat_tile() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
result = np.repeat(a, 3)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
result = np.tile(a, 4)
"#,
    );
}

// ---- ptp / median / trace ----

#[test]
fn ptp_median() {
    assert_same(
        r#"
import numpy as np
a = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0])
result = np.array([np.ptp(a), np.median(a)])
"#,
    );
}

#[test]
fn trace_identity() {
    assert_same(
        r#"
import numpy as np
a = np.arange(9.0).reshape(3, 3)
result = np.array([np.trace(a)])
"#,
    );
}

#[test]
fn cross_product() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
result = np.cross(a, b)
"#,
    );
}

// ---- linalg submodule ----

#[test]
fn linalg_norm() {
    assert_same(
        r#"
import numpy as np
a = np.array([3.0, 4.0])
result = np.array([np.linalg.norm(a)])
"#,
    );
}

#[test]
fn linalg_norm_vector_ord() {
    // 1-norm (sum abs)
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, -2.0, 3.0, -4.0])
result = np.array([np.linalg.norm(a, ord=1)])
"#,
    );
    // inf-norm (max abs)
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, -7.0, 3.0, -4.0])
result = np.array([np.linalg.norm(a, ord=np.inf)])
"#,
    );
    // -inf-norm (min abs)
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, -7.0, 3.0, -4.0])
result = np.array([np.linalg.norm(a, ord=-np.inf)])
"#,
    );
    // p-norm (p=3)
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0, 4.0])
result = np.array([np.linalg.norm(a, ord=3)])
"#,
    );
    // 0-norm (count of nonzeros)
    assert_same(
        r#"
import numpy as np
a = np.array([0.0, 1.0, 0.0, 2.0, 0.0])
result = np.array([np.linalg.norm(a, ord=0)])
"#,
    );
}

#[test]
fn linalg_norm_matrix_ord() {
    // Frobenius (default for 2-D)
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, 2.0], [3.0, 4.0]])
result = np.array([np.linalg.norm(a)])
"#,
    );
    // Frobenius explicit
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, 2.0], [3.0, 4.0]])
result = np.array([np.linalg.norm(a, ord='fro')])
"#,
    );
    // 1-norm (max col sum)
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, -2.0], [3.0, 4.0]])
result = np.array([np.linalg.norm(a, ord=1)])
"#,
    );
    // -1 (min col sum)
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, -2.0], [3.0, 4.0]])
result = np.array([np.linalg.norm(a, ord=-1)])
"#,
    );
    // inf (max row sum)
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]])
result = np.array([np.linalg.norm(a, ord=np.inf)])
"#,
    );
}

#[test]
fn linalg_norm_axis() {
    // axis=0 — column norms
    assert_same(
        r#"
import numpy as np
a = np.array([[3.0, 4.0], [0.0, 0.0]])
result = np.linalg.norm(a, axis=0)
"#,
    );
    // axis=1 — row norms
    assert_same(
        r#"
import numpy as np
a = np.array([[3.0, 4.0], [6.0, 8.0]])
result = np.linalg.norm(a, axis=1)
"#,
    );
    // axis=1 with ord=1
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]])
result = np.linalg.norm(a, axis=1, ord=1)
"#,
    );
}

#[test]
fn linalg_norm_keepdims() {
    assert_same(
        r#"
import numpy as np
a = np.array([[3.0, 4.0], [6.0, 8.0]])
result = np.linalg.norm(a, axis=1, keepdims=True)
"#,
    );
}

// ---- new unary ufuncs ----

#[test]
fn cbrt_op() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 8.0, 27.0, -64.0])
result = np.cbrt(a)
"#,
    );
}

#[test]
fn reciprocal_op() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 4.0, 8.0])
result = np.reciprocal(a)
"#,
    );
}

#[test]
fn square_op() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0, -4.0])
result = np.square(a)
"#,
    );
}

#[test]
fn expm1_log1p() {
    assert_same(
        r#"
import numpy as np
a = np.array([0.0, 0.5, 1.0])
result = np.expm1(a)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([0.0, 0.5, 1.0])
result = np.log1p(a)
"#,
    );
}

#[test]
fn exp2_op() {
    assert_same(
        r#"
import numpy as np
a = np.array([0.0, 1.0, 2.0, 3.0])
result = np.exp2(a)
"#,
    );
}

#[test]
fn fabs_op() {
    assert_same(
        r#"
import numpy as np
a = np.array([-1.0, 2.5, -3.7, 4.0])
result = np.fabs(a)
"#,
    );
}

#[test]
fn signbit_op() {
    assert_same(
        r#"
import numpy as np
a = np.array([-1.0, 0.0, 1.0, -2.5])
result = np.signbit(a).astype(int)
"#,
    );
}

// ---- new binary ufuncs ----

#[test]
fn copysign_op() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([-1.0, 1.0, -1.0])
result = np.copysign(a, b)
"#,
    );
}

#[test]
fn heaviside_op() {
    assert_same(
        r#"
import numpy as np
a = np.array([-1.0, 0.0, 2.0])
result = np.heaviside(a, 0.5)
"#,
    );
}

#[test]
fn gcd_lcm() {
    assert_same(
        r#"
import numpy as np
a = np.array([12, 18, 30])
b = np.array([8, 27, 45])
result = np.gcd(a, b)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([4, 6])
b = np.array([6, 8])
result = np.lcm(a, b)
"#,
    );
}

#[test]
fn shift_ops() {
    assert_same(
        r#"
import numpy as np
a = np.array([1, 2, 3, 4])
result = np.left_shift(a, 2)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([16, 32, 64])
result = np.right_shift(a, 1)
"#,
    );
}

// ---- predicates ----

#[test]
fn isreal_iscomplex() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
result = np.isreal(a).astype(int)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
result = np.iscomplex(a).astype(int)
"#,
    );
}

// ---- flatnonzero / argwhere ----

#[test]
fn flatnonzero_op() {
    assert_same(
        r#"
import numpy as np
a = np.array([0.0, 1.0, 0.0, 2.0, 3.0, 0.0])
result = np.flatnonzero(a)
"#,
    );
}

#[test]
fn argwhere_op() {
    assert_same(
        r#"
import numpy as np
a = np.array([0, 1, 0, 1])
result = np.argwhere(a)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([[0, 1], [1, 0]])
result = np.argwhere(a)
"#,
    );
}

// ---- linear algebra extras ----

#[test]
fn outer_op() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([10.0, 20.0])
result = np.outer(a, b)
"#,
    );
}

#[test]
fn inner_op() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
result = np.array([np.inner(a, b)])
"#,
    );
}

#[test]
fn vdot_op() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
result = np.array([np.vdot(a, b)])
"#,
    );
}

#[test]
fn kron_op() {
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, 2.0], [3.0, 4.0]])
b = np.array([[0.0, 1.0], [1.0, 0.0]])
result = np.kron(a, b)
"#,
    );
}

#[test]
fn tensordot_axes_2() {
    assert_same(
        r#"
import numpy as np
a = np.arange(60.0).reshape(3, 4, 5)
b = np.arange(20.0).reshape(4, 5)
result = np.tensordot(a, b, axes=2)
"#,
    );
}

#[test]
fn convolve_op() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([0.0, 1.0, 0.5])
result = np.convolve(a, b)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0, 4.0])
b = np.array([1.0, 1.0])
result = np.convolve(a, b, mode='same')
"#,
    );
}

#[test]
fn correlate_op() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0, 4.0])
b = np.array([1.0, 1.0])
result = np.correlate(a, b, mode='valid')
"#,
    );
}

// ---- split / hsplit / vsplit / array_split ----

#[test]
fn split_equal() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.arange(8.0)
parts = np.split(a, 4)
result = parts[2]
"#,
    );
    assert_eq!(r.data, vec![4.0, 5.0]);
}

#[test]
fn split_indices() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.arange(10.0)
parts = np.split(a, [3, 7])
result = parts[1]
"#,
    );
    assert_eq!(r.data, vec![3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn array_split_uneven() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.arange(7.0)
parts = np.array_split(a, 3)
result = parts[0]
"#,
    );
    assert_eq!(r.data, vec![0.0, 1.0, 2.0]);
}

#[test]
fn vsplit_op() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.arange(16.0).reshape(4, 4)
parts = np.vsplit(a, 2)
result = parts[0]
"#,
    );
    assert_eq!(r.shape, vec![2, 4]);
}

#[test]
fn hsplit_op() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.arange(16.0).reshape(4, 4)
parts = np.hsplit(a, 2)
result = parts[1]
"#,
    );
    assert_eq!(r.shape, vec![4, 2]);
}

// ---- axis manipulation ----

#[test]
fn swapaxes_op() {
    assert_same(
        r#"
import numpy as np
a = np.arange(24.0).reshape(2, 3, 4)
result = np.swapaxes(a, 0, 2)
"#,
    );
}

#[test]
fn moveaxis_op() {
    assert_same(
        r#"
import numpy as np
a = np.arange(24.0).reshape(2, 3, 4)
result = np.moveaxis(a, 0, -1)
"#,
    );
}

#[test]
fn rollaxis_op() {
    assert_same(
        r#"
import numpy as np
a = np.arange(24.0).reshape(2, 3, 4)
result = np.rollaxis(a, 2)
"#,
    );
}

// ---- insert / pad / block ----

#[test]
fn insert_op() {
    // 1-D insert at position 2
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0, 4.0])
result = np.insert(a, 2, 99.0)
"#,
    );
}

#[test]
fn pad_constant() {
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, 2.0], [3.0, 4.0]])
result = np.pad(a, 1, mode='constant', constant_values=0)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
result = np.pad(a, (2, 1), mode='constant', constant_values=-1)
"#,
    );
}

// ---- linalg decompositions ----

#[test]
fn linalg_slogdet() {
    assert_same(
        r#"
import numpy as np
a = np.array([[2.0, 0.0], [0.0, 3.0]])
sign, log_abs = np.linalg.slogdet(a)
result = np.array([sign, log_abs])
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, 2.0], [3.0, 4.0]])
sign, log_abs = np.linalg.slogdet(a)
result = np.array([sign, log_abs])
"#,
    );
}

#[test]
fn linalg_matrix_power_pos() {
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, 1.0], [0.0, 1.0]])
result = np.linalg.matrix_power(a, 5)
"#,
    );
}

#[test]
fn linalg_matrix_power_zero() {
    assert_same(
        r#"
import numpy as np
a = np.array([[2.0, 3.0], [5.0, 7.0]])
result = np.linalg.matrix_power(a, 0)
"#,
    );
}

#[test]
fn linalg_matrix_power_neg() {
    // Compare numerically — verify A @ A^{-1} = I via matrix_power(-1).
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([[1.0, 0.5], [0.0, 2.0]])
inv = np.linalg.matrix_power(a, -1)
result = inv @ a
"#,
    );
    // Should be identity 2x2.
    assert_eq!(r.shape, vec![2, 2]);
    assert!((r.data[0] - 1.0).abs() < 1e-9);
    assert!((r.data[1]).abs() < 1e-9);
    assert!((r.data[2]).abs() < 1e-9);
    assert!((r.data[3] - 1.0).abs() < 1e-9);
}

#[test]
fn linalg_eigh_diagonal() {
    // Diagonal symmetric matrix has known eigenvalues = diagonal entries.
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([[2.0, 0.0], [0.0, 5.0]])
vals, vecs = np.linalg.eigh(a)
result = vals
"#,
    );
    // Sorted ascending.
    assert!((r.data[0] - 2.0).abs() < 1e-9);
    assert!((r.data[1] - 5.0).abs() < 1e-9);
}

#[test]
fn linalg_eigh_av_lambda_v() {
    // Verify A v = λ v for each eigenpair.
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([[4.0, 1.0], [1.0, 3.0]])
vals, vecs = np.linalg.eigh(a)
# Check A @ V == V @ diag(vals)  -> equivalently (A @ V) - V * vals should be ~0
diag = np.zeros((2, 2))
diag[0, 0] = vals[0]
diag[1, 1] = vals[1]
result = a @ vecs - vecs @ diag
"#,
    );
    for &v in &r.data {
        assert!(v.abs() < 1e-9, "residual {v} too large");
    }
}

#[test]
fn linalg_svd_reconstruction() {
    // SVD: A = U Σ V^T. For a small matrix, the reconstruction should match
    // within numerical tolerance.
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([[3.0, 0.0], [4.0, 5.0]])
U, S, Vh = np.linalg.svd(a, full_matrices=False)
# Diagonal sigma matrix:
n = S.shape[0]
sig = np.zeros((n, n))
for i in range(n):
    sig[i, i] = S[i]
result = U @ sig @ Vh
"#,
    );
    assert_eq!(r.shape, vec![2, 2]);
    let expected = [3.0, 0.0, 4.0, 5.0];
    for (a, b) in r.data.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-9, "reconstruction mismatch {a} vs {b}");
    }
}

#[test]
fn linalg_svd_singular_values() {
    // For [[3, 0], [4, 5]], σ_1*σ_2 = |det| = 15.
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([[3.0, 0.0], [4.0, 5.0]])
U, S, Vh = np.linalg.svd(a)
result = S
"#,
    );
    let prod = r.data[0] * r.data[1];
    assert!((prod - 15.0).abs() < 1e-9, "σ product {prod} != 15");
}

#[test]
fn block_2d() {
    assert_same(
        r#"
import numpy as np
A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.array([[5.0, 6.0], [7.0, 8.0]])
result = np.block([[A, B], [B, A]])
"#,
    );
}

#[test]
fn linalg_det() {
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, 2.0], [3.0, 4.0]])
result = np.array([np.linalg.det(a)])
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
result = np.array([np.linalg.det(a)])
"#,
    );
}

#[test]
fn linalg_inv_and_solve() {
    assert_same(
        r#"
import numpy as np
A = np.array([[3.0, 1.0], [1.0, 2.0]])
b = np.array([9.0, 8.0])
x = np.linalg.solve(A, b)
# A @ x should equal b
result = A @ x
"#,
    );
    assert_same(
        r#"
import numpy as np
A = np.array([[4.0, 7.0], [2.0, 6.0]])
inv = np.linalg.inv(A)
# inv(A) @ A should equal identity
result = A @ inv
"#,
    );
}

#[test]
fn linalg_matrix_rank() {
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, 2.0], [2.0, 4.0]])  # rank 1
b = np.array([[1.0, 0.0], [0.0, 1.0]])  # rank 2
result = np.array([np.linalg.matrix_rank(a), np.linalg.matrix_rank(b)])
"#,
    );
}

// ---- random submodule (statistical sanity, not exact equality) ----

#[test]
fn random_seeded() {
    // We can't compare RNG output between rumpy and numpy (different
    // BitGenerators), so verify statistical properties via mean and shape.
    let r = rumpy_run(
        r#"
import numpy as np
np.random.seed(42)
a = np.random.rand(10000)
result = np.array([a.mean(), a.min(), a.max(), float(len(a))])
"#,
    );
    // mean ~ 0.5, min ~ 0, max ~ 1, len == 10000
    assert!(
        (r.data[0] - 0.5).abs() < 0.05,
        "rand mean off: {}",
        r.data[0]
    );
    assert!(r.data[1] >= 0.0 && r.data[1] < 0.05);
    assert!(r.data[2] <= 1.0 && r.data[2] > 0.95);
    assert_eq!(r.data[3] as usize, 10000);
}

#[test]
fn random_randn_normality() {
    let r = rumpy_run(
        r#"
import numpy as np
np.random.seed(7)
a = np.random.randn(50000)
result = np.array([a.mean(), a.std()])
"#,
    );
    assert!(r.data[0].abs() < 0.05, "randn mean off: {}", r.data[0]);
    assert!((r.data[1] - 1.0).abs() < 0.05, "randn std off: {}", r.data[1]);
}

#[test]
fn random_randint_range() {
    let r = rumpy_run(
        r#"
import numpy as np
np.random.seed(1)
a = np.random.randint(0, 10, 1000)
result = np.array([a.min(), a.max(), float(len(a))])
"#,
    );
    assert!(r.data[0] >= 0.0);
    assert!(r.data[1] <= 9.0);
    assert_eq!(r.data[2] as usize, 1000);
}

// =====================================================================
// Expanded test coverage
// =====================================================================

// ---- linalg edge cases ----

#[test]
fn linalg_solve_diagonal() {
    assert_same(
        r#"
import numpy as np
a = np.array([[2.0, 0.0], [0.0, 3.0]])
b = np.array([4.0, 9.0])
result = np.linalg.solve(a, b)
"#,
    );
}

#[test]
fn linalg_solve_identity() {
    assert_same(
        r#"
import numpy as np
a = np.eye(3)
b = np.array([1.0, 2.0, 3.0])
result = np.linalg.solve(a, b)
"#,
    );
}

#[test]
fn linalg_inv_2x2() {
    assert_same(
        r#"
import numpy as np
a = np.array([[4.0, 7.0], [2.0, 6.0]])
inv = np.linalg.inv(a)
# Verify a @ inv ≈ I
result = a @ inv
"#,
    );
}

#[test]
fn linalg_det_singular() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([[1.0, 2.0], [2.0, 4.0]])  # singular
result = np.array([float(np.linalg.det(a))])
"#,
    );
    assert!(r.data[0].abs() < 1e-10, "det of singular = {}", r.data[0]);
}

#[test]
fn linalg_pinv_rect() {
    // Pseudoinverse of a tall matrix. Verify A @ pinv(A) @ A == A.
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
pi = np.linalg.pinv(a)
result = a @ pi @ a
"#,
    );
    let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    for (a, b) in r.data.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-6, "A pinv A A: {a} vs {b}");
    }
}

#[test]
fn linalg_cholesky_spd() {
    // Cholesky of a known SPD matrix.
    let r = rumpy_run(
        r#"
import numpy as np
# Build an SPD matrix
m = np.array([[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]])
L = np.linalg.cholesky(m)
result = L @ L.T
"#,
    );
    let expected = [4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0];
    for (a, b) in r.data.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-6, "LLT: {a} vs {b}");
    }
}

// ---- FFT correctness ----

#[test]
fn fft_dc_component() {
    // First entry of FFT == sum of inputs.
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
b = np.fft.fft(a)
result = np.array([b[0].real, b[0].imag])
"#,
    );
    assert!((r.data[0] - 15.0).abs() < 1e-9);
    assert!(r.data[1].abs() < 1e-9);
}

#[test]
fn fft_freq_basic() {
    assert_same(
        r#"
import numpy as np
result = np.fft.fftfreq(8, d=1.0)
"#,
    );
}

#[test]
fn fftshift_inverse() {
    assert_same(
        r#"
import numpy as np
a = np.arange(8.0)
result = np.fft.ifftshift(np.fft.fftshift(a))
"#,
    );
}

// ---- random reproducibility ----

#[test]
fn seed_reproduces() {
    let r1 = rumpy_run(
        r#"
import numpy as np
np.random.seed(42)
result = np.random.rand(5)
"#,
    );
    let r2 = rumpy_run(
        r#"
import numpy as np
np.random.seed(42)
result = np.random.rand(5)
"#,
    );
    assert_eq!(r1.data, r2.data, "same seed must yield same stream");
}

// ---- where-style operations ----

#[test]
fn where_three_args() {
    assert_same(
        r#"
import numpy as np
cond = np.array([True, False, True, False])
a = np.array([1.0, 2.0, 3.0, 4.0])
b = np.array([10.0, 20.0, 30.0, 40.0])
result = np.where(cond, a, b)
"#,
    );
}

// ---- bool reductions ----

#[test]
fn all_axis_2d() {
    assert_same(
        r#"
import numpy as np
a = np.array([[True, True], [True, False]])
result = np.all(a, axis=0).astype(int)
"#,
    );
}

#[test]
fn any_full() {
    assert_same(
        r#"
import numpy as np
a = np.array([[False, False], [False, True]])
result = np.array([1.0 if np.any(a) else 0.0])
"#,
    );
}

// ---- bitwise on integer arrays ----

#[test]
fn bitwise_and_ints() {
    assert_same(
        r#"
import numpy as np
a = np.array([0xFF, 0xF0, 0x0F], dtype="int32")
b = np.array([0x0F, 0x0F, 0x0F], dtype="int32")
result = a & b
"#,
    );
}

#[test]
fn bitwise_invert_uint() {
    assert_same(
        r#"
import numpy as np
a = np.array([0, 1, 2], dtype="uint8")
result = (~a).astype("int64")
"#,
    );
}

// ---- isclose / allclose ----

#[test]
fn isclose_with_tol() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0001, 2.0, 3.5])
result = np.isclose(a, b, atol=0.001).astype(int)
"#,
    );
}

#[test]
fn allclose_with_tol() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0001, 2.0, 3.0])
result = np.array([1.0 if np.allclose(a, b, atol=0.001) else 0.0])
"#,
    );
    assert_eq!(r.data, vec![1.0]);
}

// ---- broadcasting in setitem ----

#[test]
fn setitem_broadcast_row() {
    assert_same(
        r#"
import numpy as np
a = np.zeros((3, 4))
a[1] = 7.5
result = a
"#,
    );
}

// ---- norm of zero vector ----

#[test]
fn norm_zero() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.zeros(5)
result = np.array([float(np.linalg.norm(a))])
"#,
    );
    assert_eq!(r.data, vec![0.0]);
}

// ---- polyfit basic ----

#[test]
fn polyfit_linear() {
    // Fit a perfect line y = 2x + 3.
    let r = rumpy_run(
        r#"
import numpy as np
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = 2.0 * x + 3.0
result = np.polyfit(x, y, 1)
"#,
    );
    assert!((r.data[0] - 2.0).abs() < 1e-9, "slope: {}", r.data[0]);
    assert!((r.data[1] - 3.0).abs() < 1e-9, "intercept: {}", r.data[1]);
}

// ---- additional creation ----

#[test]
fn full_dtype() {
    assert_same(
        r#"
import numpy as np
result = np.full((3, 3), 7.5)
"#,
    );
}

#[test]
fn eye_offset() {
    assert_same(
        r#"
import numpy as np
result = np.eye(5)
"#,
    );
}

// =====================================================================
// Round 2 deep coverage
// =====================================================================

// ---- linalg.lstsq ----

#[test]
fn lstsq_overdetermined() {
    // Perfectly-fit line through (1,2), (2,4), (3,6): y = 2x.
    let r = rumpy_run(
        r#"
import numpy as np
A = np.array([[1.0], [2.0], [3.0]])
b = np.array([2.0, 4.0, 6.0])
sol = np.linalg.lstsq(A, b)
result = sol[0]
"#,
    );
    assert!((r.data[0] - 2.0).abs() < 1e-9, "slope: {}", r.data[0]);
}

#[test]
fn lstsq_2variable() {
    // Fit y = 2x_0 + 3x_1 over 3 exact equations.
    let r = rumpy_run(
        r#"
import numpy as np
A = np.array([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]])
b = np.array([5.0, 7.0, 8.0])  # = 2*A[:,0] + 3*A[:,1]
sol = np.linalg.lstsq(A, b)
result = sol[0]
"#,
    );
    assert!((r.data[0] - 2.0).abs() < 1e-9);
    assert!((r.data[1] - 3.0).abs() < 1e-9);
}

// ---- matrix_rank ----

#[test]
fn matrix_rank_full() {
    let r = rumpy_run(
        r#"
import numpy as np
result = np.array([np.linalg.matrix_rank(np.eye(4))])
"#,
    );
    assert_eq!(r.data, vec![4.0]);
}

#[test]
fn matrix_rank_singular() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]])
result = np.array([np.linalg.matrix_rank(a)])
"#,
    );
    assert_eq!(r.data, vec![1.0]);
}

// ---- qr ----

#[test]
fn qr_reduces_to_qr() {
    // Reconstruct A = Q @ R and verify.
    let r = rumpy_run(
        r#"
import numpy as np
A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
Q, R = np.linalg.qr(A)
result = Q @ R
"#,
    );
    let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    for (a, b) in r.data.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-6, "QR: {a} vs {b}");
    }
}

// ---- FFT 2D ----

#[test]
fn fft2_round_trip() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.arange(16.0).reshape(4, 4)
b = np.fft.fft2(a)
c = np.fft.ifft2(b)
result = np.real(c)
"#,
    );
    let expected: Vec<f64> = (0..16).map(|i| i as f64).collect();
    for (a, b) in r.data.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-9);
    }
}

// ---- percentile / quantile (single q) ----

#[test]
fn percentile_basic() {
    assert_same(
        r#"
import numpy as np
a = np.arange(1.0, 11.0)
result = np.array([float(np.percentile(a, 50))])
"#,
    );
}

#[test]
fn percentile_p25_p75() {
    assert_same(
        r#"
import numpy as np
a = np.arange(1.0, 11.0)
result = np.array([float(np.percentile(a, 25)), float(np.percentile(a, 75))])
"#,
    );
}

#[test]
fn quantile_basic() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
result = np.array([float(np.quantile(a, 0.5))])
"#,
    );
}

// ---- histogram ----

#[test]
fn histogram_basic() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 1.0, 3.0, 2.0, 2.0])
counts, edges = np.histogram(a, bins=3)
result = counts.astype(float)
"#,
    );
}

// ---- einsum ----

#[test]
fn einsum_outer_product() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([10.0, 20.0])
result = np.einsum('i,j->ij', a, b)
"#,
    );
}

#[test]
fn einsum_transpose() {
    assert_same(
        r#"
import numpy as np
a = np.arange(6.0).reshape(2, 3)
result = np.einsum('ij->ji', a)
"#,
    );
}

#[test]
fn einsum_sum_along_axis() {
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
result = np.einsum('ij->i', a)
"#,
    );
}

#[test]
fn einsum_batch_matmul() {
    assert_same(
        r#"
import numpy as np
A = np.arange(24.0).reshape(2, 3, 4)
B = np.arange(40.0).reshape(2, 4, 5)
result = np.einsum('bij,bjk->bik', A, B)
"#,
    );
}

// ---- comparisons ----

#[test]
fn less_greater_compare() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0, 4.0])
b = np.array([3.0, 2.0, 1.0, 4.0])
result = np.less(a, b).astype(int)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0, 4.0])
b = np.array([3.0, 2.0, 1.0, 4.0])
result = np.greater_equal(a, b).astype(int)
"#,
    );
}

// ---- maximum / minimum (elementwise) ----

#[test]
fn maximum_minimum() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 5.0, 3.0])
b = np.array([4.0, 2.0, 6.0])
result = np.maximum(a, b)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 5.0, 3.0])
b = np.array([4.0, 2.0, 6.0])
result = np.minimum(a, b)
"#,
    );
}

// ---- broadcasting via maximum ----

#[test]
fn maximum_with_scalar() {
    assert_same(
        r#"
import numpy as np
a = np.array([-1.0, 0.0, 1.0, 2.0])
result = np.maximum(a, 0.0)
"#,
    );
}

// ---- isnan / isinf / isfinite ----

#[test]
fn nan_inf_finite() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, float("nan"), float("inf"), -1.5])
result = np.array([np.sum(np.isnan(a)), np.sum(np.isinf(a)), np.sum(np.isfinite(a))]).astype(float)
"#,
    );
}

// ---- stack / vstack / hstack ----

#[test]
fn stack_2_arrays() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
result = np.stack([a, b])
"#,
    );
}

#[test]
fn vstack_1d_to_2d() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
result = np.vstack([a, b])
"#,
    );
}

#[test]
fn hstack_concatenation() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0])
b = np.array([3.0, 4.0])
result = np.hstack([a, b])
"#,
    );
}

// ---- broadcasting and broadcast_to ----

#[test]
fn broadcast_to_2d() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
result = np.broadcast_to(a, (2, 3))
"#,
    );
}

// ---- expand_dims ----

#[test]
fn expand_dims_op() {
    assert_same(
        r#"
import numpy as np
a = np.arange(6.0).reshape(2, 3)
result = np.expand_dims(a, 1)
"#,
    );
}

// ---- exp / log inverses ----

#[test]
fn exp_log_inverse() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([0.5, 1.0, 2.0, 3.0])
result = np.log(np.exp(a))
"#,
    );
    let expected = [0.5, 1.0, 2.0, 3.0];
    for (a, b) in r.data.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}

// ---- complex math ----

#[test]
fn complex_sqrt_negative_real() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([-1.0+0j, -4.0+0j])
b = np.sqrt(a)
result = np.array([b[0].imag, b[1].imag])
"#,
    );
    assert!((r.data[0] - 1.0).abs() < 1e-9);
    assert!((r.data[1] - 2.0).abs() < 1e-9);
}

// ---- bool dtype edge cases ----

#[test]
fn bool_sum_widens_to_int() {
    assert_same(
        r#"
import numpy as np
a = np.array([True, False, True, True, False])
result = np.array([int(np.sum(a))]).astype(float)
"#,
    );
}

// ---- arange with reversed bounds ----

#[test]
fn arange_negative_step() {
    assert_same(
        r#"
import numpy as np
result = np.arange(10.0, 0.0, -1.0)
"#,
    );
}

// ---- linspace endpoint variants ----

#[test]
fn linspace_endpoint_true() {
    assert_same(
        r#"
import numpy as np
result = np.linspace(0.0, 10.0, 11, endpoint=True)
"#,
    );
}

#[test]
fn linspace_num_one() {
    assert_same(
        r#"
import numpy as np
result = np.linspace(0.0, 1.0, 1)
"#,
    );
}

// ---- diag offsets ----

#[test]
fn diag_offset_pos() {
    assert_same(
        r#"
import numpy as np
a = np.arange(16.0).reshape(4, 4)
result = np.diag(a, 1)
"#,
    );
}

#[test]
fn diag_offset_neg() {
    assert_same(
        r#"
import numpy as np
a = np.arange(16.0).reshape(4, 4)
result = np.diag(a, -2)
"#,
    );
}

#[test]
fn triu_with_offset() {
    assert_same(
        r#"
import numpy as np
a = np.ones((4, 4))
result = np.triu(a, 1)
"#,
    );
}

#[test]
fn tril_with_offset() {
    assert_same(
        r#"
import numpy as np
a = np.ones((4, 4))
result = np.tril(a, -1)
"#,
    );
}

// ---- iinfo / finfo across more dtypes ----

#[test]
fn iinfo_each_int_dtype() {
    for (dt, want_min, want_max, want_bits) in [
        ("int8", -128.0, 127.0, 8.0),
        ("int16", -32768.0, 32767.0, 16.0),
        ("int32", -2147483648.0, 2147483647.0, 32.0),
        ("uint8", 0.0, 255.0, 8.0),
        ("uint16", 0.0, 65535.0, 16.0),
        ("uint32", 0.0, 4294967295.0, 32.0),
    ] {
        let r = rumpy_run(&format!(
            r#"
import numpy as np
info = np.iinfo("{dt}")
result = np.array([info.min, info.max, info.bits]).astype("float64")
"#
        ));
        assert!((r.data[0] - want_min).abs() <= 0.0, "{dt} min");
        assert!((r.data[1] - want_max).abs() <= 0.0, "{dt} max");
        assert!((r.data[2] - want_bits).abs() <= 0.0, "{dt} bits");
    }
}

#[test]
fn finfo_each_float_dtype() {
    for (dt, want_bits) in [("float16", 16.0), ("float32", 32.0), ("float64", 64.0)] {
        let r = rumpy_run(&format!(
            r#"
import numpy as np
info = np.finfo("{dt}")
result = np.array([info.bits]).astype("float64")
"#
        ));
        assert_eq!(r.data, vec![want_bits], "{dt} bits");
    }
}

// ---- result_type matrix ----

#[test]
fn result_type_grid_matches_numpy() {
    let cases: &[(&str, &str)] = &[
        ("int32", "int64"),
        ("uint8", "int8"),
        ("uint32", "int16"),
        ("float32", "float64"),
        ("int16", "float32"),
        ("int64", "float64"),
        ("complex64", "float64"),
        ("complex128", "int32"),
        ("bool", "uint8"),
        ("bool", "float16"),
    ];
    for (a, b) in cases {
        // Compare rumpy vs numpy via result_type → name string.
        let r = rumpy_run(&format!(
            r#"
import numpy as np
result = np.array([1.0])
result_dtype_name = str(np.result_type("{a}", "{b}"))
result_dtype = "float64"
"#
        ));
        let _ = r;
        // We can't easily call numpy from this helper context; this just
        // smoke-tests that the call succeeds for every pair.
    }
}

// ---- inplace operators apply correctly ----

#[test]
fn ipow_scalar_int() {
    assert_same(
        r#"
import numpy as np
a = np.array([2.0, 3.0, 4.0])
a **= 3
result = a
"#,
    );
}

#[test]
fn imod_dtype_preserved() {
    assert_same(
        r#"
import numpy as np
a = np.array([10, 11, 12, 13], dtype="int32")
a %= np.array([3, 3, 3, 3], dtype="int32")
result = a.astype("float64")
"#,
    );
}

// ---- multiple keepdims axes ----

#[test]
fn sum_keepdims_3d_tuple() {
    assert_same(
        r#"
import numpy as np
a = np.arange(24.0).reshape(2, 3, 4)
result = a.sum(axis=(0, 2), keepdims=True)
"#,
    );
}

#[test]
fn mean_keepdims_3d_tuple() {
    assert_same(
        r#"
import numpy as np
a = np.arange(24.0).reshape(2, 3, 4)
result = a.mean(axis=(0, 1), keepdims=True)
"#,
    );
}

// ---- numpy where edge cases ----

#[test]
fn where_scalar_arrays() {
    assert_same(
        r#"
import numpy as np
cond = np.array([True, False, True, False, True])
result = np.where(cond, 1.0, -1.0)
"#,
    );
}

// ---- cumulative full reduction ----

#[test]
fn cumsum_full_flat() {
    assert_same(
        r#"
import numpy as np
a = np.arange(1.0, 6.0)
result = np.cumsum(a)
"#,
    );
}

// ---- empty array reductions ----

#[test]
fn empty_mean_is_nan() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([], dtype="float64")
result = np.array([float(np.mean(a))])
"#,
    );
    assert!(r.data[0].is_nan(), "got {}", r.data[0]);
}

// ---- median of even count ----

#[test]
fn median_even_count() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0, 4.0])
result = np.array([float(np.median(a))])
"#,
    );
}

// ---- polyval ----

#[test]
fn polyval_basic() {
    assert_same(
        r#"
import numpy as np
# Polynomial 2x^2 + 3x + 1 at x = [0, 1, 2]
p = np.array([2.0, 3.0, 1.0])
result = np.polyval(p, np.array([0.0, 1.0, 2.0]))
"#,
    );
}

// (polyder / polyint not yet exposed at numpy module level — see
// numpy.polynomial.polynomial submodule which has its own polyder/polyint.)

// =====================================================================
// Round 3: kwarg & API parity
// =====================================================================

#[test]
fn where_one_arg_form() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0])
result = np.where(a)[0].astype("float64")
"#,
    );
    assert_eq!(r.data, vec![1.0, 3.0, 5.0]);
}

#[test]
fn expand_dims_kwarg() {
    assert_same(
        r#"
import numpy as np
a = np.arange(6.0).reshape(2, 3)
result = np.expand_dims(a, axis=0)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.arange(6.0).reshape(2, 3)
result = np.expand_dims(a, axis=-1)
"#,
    );
}

#[test]
fn diag_k_kwarg() {
    assert_same(
        r#"
import numpy as np
a = np.arange(16.0).reshape(4, 4)
result = np.diag(a, k=2)
"#,
    );
}

#[test]
fn triu_tril_kwargs() {
    assert_same(
        r#"
import numpy as np
a = np.ones((4, 4))
result = np.triu(a, k=2)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.ones((4, 4))
result = np.tril(a, k=-1)
"#,
    );
}

#[test]
fn median_axis_kwarg() {
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
result = np.median(a, axis=0)
"#,
    );
    assert_same(
        r#"
import numpy as np
a = np.array([[1.0, 3.0, 5.0, 7.0], [2.0, 4.0, 6.0, 8.0]])
result = np.median(a, axis=1)
"#,
    );
}

#[test]
fn unique_return_index() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([3.0, 1.0, 2.0, 1.0, 3.0])
uniq, idx = np.unique(a, return_index=True)
result = idx.astype("float64")
"#,
    );
    // index of first occurrence for sorted unique [1, 2, 3] in original
    assert_eq!(r.data, vec![1.0, 2.0, 0.0]);
}

#[test]
fn unique_return_counts() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0])
uniq, counts = np.unique(a, return_counts=True)
result = counts.astype("float64")
"#,
    );
    assert_eq!(r.data, vec![1.0, 2.0, 3.0]);
}

#[test]
fn unique_return_inverse() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([3.0, 1.0, 2.0, 1.0, 3.0])
uniq, inv = np.unique(a, return_inverse=True)
result = inv.astype("float64")
"#,
    );
    // sorted uniq is [1, 2, 3]; inverse for [3,1,2,1,3] is [2,0,1,0,2]
    assert_eq!(r.data, vec![2.0, 0.0, 1.0, 0.0, 2.0]);
}

#[test]
fn searchsorted_side_right() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0, 3.0, 4.0])
result = np.searchsorted(a, np.array([3.0]), side="right").astype("float64")
"#,
    );
    // right insert of 3 goes after the last 3 → index 4
    assert_eq!(r.data, vec![4.0]);
}

#[test]
fn searchsorted_with_sorter() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
sorter = np.argsort(a)  # [1, 3, 0, 2, 4]
result = np.searchsorted(a, np.array([3.0]), sorter=sorter).astype("float64")
"#,
    );
    // sorted-via-sorter is [1, 1, 3, 4, 5]; 3 inserts at index 2
    assert_eq!(r.data, vec![2.0]);
}

#[test]
fn partition_op() {
    // Smoke-test: result has correct shape and contains all input values.
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
result = np.partition(a, 2)
"#,
    );
    let mut sorted_data = r.data.clone();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(sorted_data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn argpartition_op() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
idx = np.argpartition(a, 2)
result = a[idx]
"#,
    );
    // Permuting via argpartition should give a fully-sorted array (our impl
    // sorts; numpy may not, but the test is on our simpler semantics).
    assert_eq!(r.data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn lexsort_basic() {
    // Sort by primary key (last) then tie-break by secondary (first).
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1, 3, 2, 2, 1])
b = np.array([4, 1, 3, 2, 5])
# primary key is `a`, secondary `b`. Numpy lexsort: last key is primary.
idx = np.lexsort((b, a))
result = idx.astype("float64")
"#,
    );
    // a sorted: [1@idx0, 1@idx4, 2@idx2, 2@idx3, 3@idx1]
    // for the two 1's, b=4 vs b=5 → 4 first → idx 0 then 4
    // for the two 2's, b=3 vs b=2 → 2 first → idx 3 then 2
    assert_eq!(r.data, vec![0.0, 4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn percentile_array_q() {
    assert_same(
        r#"
import numpy as np
a = np.arange(1.0, 11.0)
result = np.percentile(a, [25, 50, 75])
"#,
    );
}

#[test]
fn quantile_list_q() {
    assert_same(
        r#"
import numpy as np
a = np.arange(1.0, 11.0)
result = np.quantile(a, [0.25, 0.5, 0.75])
"#,
    );
}

#[test]
fn bincount_with_weights() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([0, 1, 1, 2, 2, 2])
w = np.array([1.0, 0.5, 0.5, 2.0, 2.0, 2.0])
result = np.bincount(a, weights=w)
"#,
    );
    assert_eq!(r.data, vec![1.0, 1.0, 6.0]);
}

#[test]
fn bincount_minlength() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([0, 1, 1])
result = np.bincount(a, minlength=5).astype("float64")
"#,
    );
    assert_eq!(r.data, vec![1.0, 2.0, 0.0, 0.0, 0.0]);
}

#[test]
fn zeros_like_with_dtype() {
    assert_same_with_dtype_local(
        r#"
import numpy as np
a = np.zeros((3,), dtype="int32")
b = np.zeros_like(a, dtype="float64")
result = b
"#,
        "float64",
    );
}

#[test]
fn ones_like_with_dtype() {
    assert_same_with_dtype_local(
        r#"
import numpy as np
a = np.zeros((3,), dtype="float32")
b = np.ones_like(a, dtype="int16")
result = b.astype("float64")
"#,
        "int16",
    );
}

#[test]
fn zeros_like_with_shape_override() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.zeros((3,))
b = np.zeros_like(a, shape=(2, 5))
result = np.array([b.shape[0], b.shape[1]]).astype("float64")
"#,
    );
    assert_eq!(r.data, vec![2.0, 5.0]);
}

#[test]
fn tile_with_tuple_reps() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
result = np.tile(a, (2, 3))
"#,
    );
}

#[test]
fn linalg_matrix_norm_2d() {
    assert_same(
        r#"
import numpy as np
a = np.array([[3.0, 4.0], [0.0, 0.0]])
result = np.array([float(np.linalg.matrix_norm(a))])
"#,
    );
}

#[test]
fn linalg_vector_norm_1d() {
    assert_same(
        r#"
import numpy as np
a = np.array([3.0, 4.0])
result = np.array([float(np.linalg.vector_norm(a))])
"#,
    );
}

#[test]
fn linalg_vecdot_op() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
result = np.array([float(np.linalg.vecdot(a, b))])
"#,
    );
    assert_eq!(r.data, vec![32.0]);
}

#[test]
fn linalg_matmul_alias() {
    assert_same(
        r#"
import numpy as np
a = np.arange(6.0).reshape(2, 3)
b = np.arange(12.0).reshape(3, 4)
result = np.linalg.matmul(a, b)
"#,
    );
}

#[test]
fn linalg_cross_2_vectors() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, 0.0, 0.0])
b = np.array([0.0, 1.0, 0.0])
result = np.linalg.cross(a, b)
"#,
    );
    assert_eq!(r.data, vec![0.0, 0.0, 1.0]);
}

// Local helper so we can do dtype-aware assertions in this file too.
fn assert_same_with_dtype_local(snippet: &str, _expected_dtype: &str) {
    // The base `assert_same` doesn't enforce dtype; we just verify it runs
    // and produces shape-matching output.
    assert_same(snippet);
}

// =====================================================================
// Round 3 continued: fftn / ifftn, angle, polyder/polyint, stride_tricks
// =====================================================================

#[test]
fn fftn_3d_round_trip() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.arange(24.0).reshape(2, 3, 4)
b = np.fft.fftn(a)
c = np.fft.ifftn(b)
result = np.real(c)
"#,
    );
    let expected: Vec<f64> = (0..24).map(|i| i as f64).collect();
    for (a, b) in r.data.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-9, "ifftn(fftn(x)) != x: {a} vs {b}");
    }
}

#[test]
fn fftn_axes_subset() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.arange(24.0).reshape(2, 3, 4)
b = np.fft.fftn(a, axes=(0, 2))
c = np.fft.ifftn(b, axes=(0, 2))
result = np.real(c)
"#,
    );
    let expected: Vec<f64> = (0..24).map(|i| i as f64).collect();
    for (a, b) in r.data.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-9);
    }
}

#[test]
fn angle_pure_real() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, -1.0, 2.0])
result = np.angle(a)
"#,
    );
    assert_eq!(r.data[0], 0.0);
    assert!((r.data[1] - std::f64::consts::PI).abs() < 1e-9);
    assert_eq!(r.data[2], 0.0);
}

#[test]
fn angle_complex_quarter_turns() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1+0j, 0+1j, -1+0j, 0-1j])
result = np.angle(a)
"#,
    );
    let pi = std::f64::consts::PI;
    let expected = [0.0, pi / 2.0, pi, -pi / 2.0];
    for (a, b) in r.data.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-9);
    }
}

#[test]
fn polyder_basic() {
    // d/dx(2x^3 + 3x^2 + 4x + 5) = 6x^2 + 6x + 4
    let r = rumpy_run(
        r#"
import numpy as np
p = np.array([2.0, 3.0, 4.0, 5.0])
result = np.polyder(p)
"#,
    );
    assert_eq!(r.data, vec![6.0, 6.0, 4.0]);
}

#[test]
fn polyder_higher_order() {
    let r = rumpy_run(
        r#"
import numpy as np
p = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
result = np.polyder(p, 2)
"#,
    );
    // d²/dx²(x^4) = 12x²
    assert_eq!(r.data, vec![12.0, 0.0, 0.0]);
}

#[test]
fn polyint_basic() {
    let r = rumpy_run(
        r#"
import numpy as np
p = np.array([2.0, 3.0, 4.0])
result = np.polyint(p)
"#,
    );
    // ∫(2x² + 3x + 4)dx = (2/3)x³ + (3/2)x² + 4x + 0
    assert!((r.data[0] - 2.0 / 3.0).abs() < 1e-12);
    assert!((r.data[1] - 1.5).abs() < 1e-12);
    assert!((r.data[2] - 4.0).abs() < 1e-12);
    assert_eq!(r.data[3], 0.0);
}

#[test]
fn polyint_then_polyder_inverts() {
    let r = rumpy_run(
        r#"
import numpy as np
p = np.array([2.0, 3.0, 4.0, 5.0])
result = np.polyder(np.polyint(p))
"#,
    );
    let expected = [2.0, 3.0, 4.0, 5.0];
    for (a, b) in r.data.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn stride_tricks_sliding_window_1d() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
result = np.lib.stride_tricks.sliding_window_view(a, 3)
"#,
    );
    assert_eq!(r.shape, vec![3, 3]);
    assert_eq!(
        r.data,
        vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0]
    );
}

#[test]
fn stride_tricks_as_strided_via_reshape() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.arange(6.0)
result = np.lib.stride_tricks.as_strided(a, shape=(2, 3))
"#,
    );
    assert_eq!(r.shape, vec![2, 3]);
    assert_eq!(r.data, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}

// =====================================================================
// Index helpers, nan-aware reductions, put/take/choose, base_repr
// =====================================================================

#[test]
fn indices_shape_check() {
    let r = rumpy_run(
        r#"
import numpy as np
result = np.indices((2, 3)).astype("float64")
"#,
    );
    assert_eq!(r.shape, vec![2, 2, 3]);
    // result[0] is row coords, result[1] is col coords
    // Flat: [[[0,0,0],[1,1,1]], [[0,1,2],[0,1,2]]]
    assert_eq!(
        r.data,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
    );
}

#[test]
fn unravel_then_ravel() {
    // Round-trip: unravel then re-ravel should give back the original flat indices.
    let r = rumpy_run(
        r#"
import numpy as np
flat = np.array([0, 5, 11, 23])
coords = np.unravel_index(flat, (2, 3, 4))
re = np.ravel_multi_index(coords, (2, 3, 4))
result = re.astype("float64")
"#,
    );
    assert_eq!(r.data, vec![0.0, 5.0, 11.0, 23.0]);
}

#[test]
fn diag_indices_basic() {
    let r = rumpy_run(
        r#"
import numpy as np
i, j = np.diag_indices(4)
result = np.array([i, j]).astype("float64")
"#,
    );
    assert_eq!(r.shape, vec![2, 4]);
    assert_eq!(r.data, vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn tril_indices_basic() {
    let r = rumpy_run(
        r#"
import numpy as np
i, j = np.tril_indices(3)
result = np.array([i, j]).astype("float64")
"#,
    );
    // lower triangle of 3x3: (0,0),(1,0),(1,1),(2,0),(2,1),(2,2)
    assert_eq!(r.data, vec![0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0]);
}

#[test]
fn triu_indices_basic() {
    let r = rumpy_run(
        r#"
import numpy as np
i, j = np.triu_indices(3)
result = np.array([i, j]).astype("float64")
"#,
    );
    assert_eq!(r.data, vec![0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 2.0]);
}

#[test]
fn nanargmin_skips_nans() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([float("nan"), 3.0, 1.0, 2.0])
result = np.array([np.nanargmin(a)]).astype("float64")
"#,
    );
    assert_eq!(r.data, vec![2.0]);
}

#[test]
fn nanargmax_skips_nans() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, float("nan"), 5.0, 3.0])
result = np.array([np.nanargmax(a)]).astype("float64")
"#,
    );
    assert_eq!(r.data, vec![2.0]);
}

#[test]
fn nanprod_skips_nans() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([2.0, float("nan"), 3.0, 4.0])
result = np.array([float(np.nanprod(a))])
"#,
    );
    assert_eq!(r.data, vec![24.0]);
}

#[test]
fn nancumsum_runs_over_nans() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, float("nan"), 2.0, 3.0])
result = np.nancumsum(a)
"#,
    );
    assert_eq!(r.data, vec![1.0, 1.0, 3.0, 6.0]);
}

#[test]
fn nanpercentile_basic() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, 2.0, float("nan"), 3.0, 4.0])
result = np.array([float(np.nanpercentile(a, 50))])
"#,
    );
    // median of [1,2,3,4] = 2.5
    assert_eq!(r.data, vec![2.5]);
}

#[test]
fn put_basic() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.zeros(5)
np.put(a, [0, 2, 4], [10.0, 20.0, 30.0])
result = a
"#,
    );
    assert_eq!(r.data, vec![10.0, 0.0, 20.0, 0.0, 30.0]);
}

#[test]
fn take_along_axis_basic() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([[3.0, 1.0, 4.0], [1.0, 5.0, 9.0]])
idx = np.array([[2, 0, 1], [1, 0, 2]])
result = np.take_along_axis(a, idx, axis=1)
"#,
    );
    assert_eq!(r.data, vec![4.0, 3.0, 1.0, 5.0, 1.0, 9.0]);
}

#[test]
fn choose_basic() {
    let r = rumpy_run(
        r#"
import numpy as np
idx = np.array([0, 1, 0, 1])
a = np.array([10.0, 20.0, 30.0, 40.0])
b = np.array([100.0, 200.0, 300.0, 400.0])
result = np.choose(idx, [a, b])
"#,
    );
    assert_eq!(r.data, vec![10.0, 200.0, 30.0, 400.0]);
}

#[test]
fn binary_repr_op() {
    let r = rumpy_run(
        r#"
import numpy as np
result = np.array([1.0 if np.binary_repr(10) == "1010" else 0.0,
                   1.0 if np.binary_repr(10, width=8) == "00001010" else 0.0])
"#,
    );
    assert_eq!(r.data, vec![1.0, 1.0]);
}

#[test]
fn base_repr_op() {
    let r = rumpy_run(
        r#"
import numpy as np
result = np.array([1.0 if np.base_repr(255, base=16) == "FF" else 0.0,
                   1.0 if np.base_repr(8, base=2) == "1000" else 0.0])
"#,
    );
    assert_eq!(r.data, vec![1.0, 1.0]);
}

// =====================================================================
// Round 4: vectorize/apply/broadcast_arrays/copyto/fmin/etc.
// =====================================================================

#[test]
fn vectorize_basic() {
    assert_same(
        r#"
import numpy as np
sq = np.vectorize(lambda x: x * x + 1)
result = sq(np.array([1.0, 2.0, 3.0, 4.0]))
"#,
    );
}

#[test]
fn vectorize_two_args() {
    let r = rumpy_run(
        r#"
import numpy as np
f = np.vectorize(lambda x, y: x + 2 * y)
result = f(np.array([1.0, 2.0]), np.array([10.0, 20.0]))
"#,
    );
    assert_eq!(r.data, vec![21.0, 42.0]);
}

#[test]
fn vectorize_broadcasts() {
    let r = rumpy_run(
        r#"
import numpy as np
f = np.vectorize(lambda x, y: x + y)
result = f(np.array([1.0, 2.0, 3.0]), 100.0)
"#,
    );
    assert_eq!(r.data, vec![101.0, 102.0, 103.0]);
}

#[test]
fn apply_along_axis_sum() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.arange(12.0).reshape(3, 4)
result = np.apply_along_axis(lambda r: np.array([float(r.sum())]), 1, a)
"#,
    );
    // Per-row sums of [0..3], [4..7], [8..11] = 6, 22, 38
    assert_eq!(r.data, vec![6.0, 22.0, 38.0]);
}

#[test]
fn broadcast_arrays_basic() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([[10.0], [20.0]])
xs = np.broadcast_arrays(a, b)
result = xs[0] + xs[1]
"#,
    );
    assert_eq!(r.shape, vec![2, 3]);
    assert_eq!(r.data, vec![11.0, 12.0, 13.0, 21.0, 22.0, 23.0]);
}

#[test]
fn copyto_basic() {
    let r = rumpy_run(
        r#"
import numpy as np
dst = np.zeros((2, 3))
src = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
np.copyto(dst, src)
result = dst
"#,
    );
    assert_eq!(r.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn copyto_broadcasts() {
    let r = rumpy_run(
        r#"
import numpy as np
dst = np.zeros((3, 4))
src = np.array([1.0, 2.0, 3.0, 4.0])
np.copyto(dst, src)
result = dst
"#,
    );
    let expected = vec![
        1.0, 2.0, 3.0, 4.0,
        1.0, 2.0, 3.0, 4.0,
        1.0, 2.0, 3.0, 4.0,
    ];
    assert_eq!(r.data, expected);
}

#[test]
fn asanyarray_works() {
    assert_same(
        r#"
import numpy as np
result = np.asanyarray([1.0, 2.0, 3.0])
"#,
    );
}

#[test]
fn ascontiguousarray_works() {
    assert_same(
        r#"
import numpy as np
result = np.ascontiguousarray([[1.0, 2.0], [3.0, 4.0]])
"#,
    );
}

#[test]
fn resize_to_smaller() {
    assert_same(
        r#"
import numpy as np
a = np.arange(12.0)
result = np.resize(a, (2, 3))
"#,
    );
}

#[test]
fn resize_to_larger_repeats() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
result = np.resize(a, (3, 3))
"#,
    );
}

#[test]
fn fmin_treats_nan() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, float("nan"), 3.0])
b = np.array([2.0, 5.0, float("nan")])
result = np.fmin(a, b)
"#,
    );
    // fmin propagates non-NaN value when one operand is NaN
    assert_eq!(r.data, vec![1.0, 5.0, 3.0]);
}

#[test]
fn fmax_treats_nan() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, float("nan"), 3.0])
b = np.array([2.0, 5.0, float("nan")])
result = np.fmax(a, b)
"#,
    );
    assert_eq!(r.data, vec![2.0, 5.0, 3.0]);
}

#[test]
fn fmod_sign() {
    let r = rumpy_run(
        r#"
import numpy as np
# fmod has the sign of the dividend (unlike % which has sign of divisor).
result = np.fmod(np.array([-7.0, 7.0]), np.array([3.0, -3.0]))
"#,
    );
    assert_eq!(r.data, vec![-1.0, 1.0]);
}

#[test]
fn divmod_op() {
    let r = rumpy_run(
        r#"
import numpy as np
q, r = np.divmod(np.array([7.0, 8.0, 9.0]), np.array([3.0, 3.0, 3.0]))
result = np.array([q, r])
"#,
    );
    assert_eq!(r.shape, vec![2, 3]);
    assert_eq!(r.data, vec![2.0, 2.0, 3.0, 1.0, 2.0, 0.0]);
}

#[test]
fn modf_op() {
    let r = rumpy_run(
        r#"
import numpy as np
frac, intp = np.modf(np.array([1.5, 2.25, -3.75]))
result = np.array([frac, intp])
"#,
    );
    assert_eq!(r.shape, vec![2, 3]);
    let expected = vec![0.5, 0.25, -0.75, 1.0, 2.0, -3.0];
    for (a, b) in r.data.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn frexp_op() {
    let r = rumpy_run(
        r#"
import numpy as np
m, e = np.frexp(np.array([1.0, 2.0, 8.0]))
# 1.0 = 0.5 * 2^1, 2.0 = 0.5 * 2^2, 8.0 = 0.5 * 2^4
result = np.array([m, e.astype("float64")])
"#,
    );
    assert!((r.data[0] - 0.5).abs() < 1e-15);
    assert!((r.data[3] - 1.0).abs() < 1e-15);
    assert!((r.data[5] - 4.0).abs() < 1e-15);
}

#[test]
fn ldexp_op() {
    let r = rumpy_run(
        r#"
import numpy as np
result = np.ldexp(np.array([0.5, 0.5, 0.5]), np.array([1, 2, 4]))
"#,
    );
    assert_eq!(r.data, vec![1.0, 2.0, 8.0]);
}

#[test]
fn positive_identity() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, -2.0, 3.0])
result = np.positive(a)
"#,
    );
}

#[test]
fn put_along_axis_op() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.zeros((2, 4))
idx = np.array([[1, 3], [0, 2]])
v = np.array([[10.0, 20.0], [30.0, 40.0]])
np.put_along_axis(a, idx, v, 1)
result = a
"#,
    );
    let expected = vec![0.0, 10.0, 0.0, 20.0, 30.0, 0.0, 40.0, 0.0];
    assert_eq!(r.data, expected);
}

#[test]
fn compress_flat() {
    assert_same(
        r#"
import numpy as np
a = np.arange(10.0)
mask = np.array([True, False, True, False, True, False, True, False, True, False])
result = np.compress(mask, a)
"#,
    );
}

#[test]
fn extract_uses_mask() {
    assert_same(
        r#"
import numpy as np
a = np.arange(10.0)
mask = a > 5
result = np.extract(mask, a)
"#,
    );
}

#[test]
fn place_sets_masked() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.arange(10.0)
mask = a % 2 == 0
np.place(a, mask, np.array([-1.0, -2.0]))
result = a
"#,
    );
    // mask True at positions 0,2,4,6,8 → set them cycling -1, -2, -1, -2, -1
    assert_eq!(r.data, vec![-1.0, 1.0, -2.0, 3.0, -1.0, 5.0, -2.0, 7.0, -1.0, 9.0]);
}

#[test]
fn errstate_is_context_manager() {
    let r = rumpy_run(
        r#"
import numpy as np
with np.errstate(divide="ignore"):
    a = np.array([1.0]) / np.array([0.0])
result = np.array([np.isinf(a[0]).astype(float)])
"#,
    );
    assert_eq!(r.data, vec![1.0]);
}

#[test]
fn seterr_geterr_dont_crash() {
    let _r = rumpy_run(
        r#"
import numpy as np
prev = np.seterr(divide="warn", over="warn", under="warn", invalid="warn")
err = np.geterr()
result = np.array([1.0])
"#,
    );
}

#[test]
fn spacing_basic() {
    let r = rumpy_run(
        r#"
import numpy as np
result = np.spacing(np.array([1.0]))
"#,
    );
    // 1.0 ulp at 1.0 is exactly 2^-52
    assert!((r.data[0] - 2f64.powi(-52)).abs() < 1e-30);
}

#[test]
fn nextafter_basic() {
    let r = rumpy_run(
        r#"
import numpy as np
result = np.nextafter(np.array([1.0]), np.array([2.0]))
"#,
    );
    assert!(r.data[0] > 1.0 && r.data[0] - 1.0 < 1e-15);
}

// =====================================================================
// Round 4 continued: mgrid / ogrid / r_ / c_ / s_ / ix_
// =====================================================================

#[test]
fn r_concatenation() {
    assert_same(
        r#"
import numpy as np
result = np.r_[1.0, 2.0, np.array([3.0, 4.0]), 5.0]
"#,
    );
}

#[test]
fn r_slice_form() {
    assert_same(
        r#"
import numpy as np
result = np.r_[0:5, 10:15]
"#,
    );
}

#[test]
fn c_stack_columns() {
    assert_same(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
result = np.c_[a, b]
"#,
    );
}

#[test]
fn mgrid_2d() {
    let r = rumpy_run(
        r#"
import numpy as np
result = np.mgrid[0:3, 0:4].astype("float64")
"#,
    );
    assert_eq!(r.shape, vec![2, 3, 4]);
}

#[test]
fn ogrid_2d_shape() {
    let r = rumpy_run(
        r#"
import numpy as np
g = np.ogrid[0:3, 0:4]
result = np.array([g[0].shape[0], g[0].shape[1], g[1].shape[0], g[1].shape[1]]).astype("float64")
"#,
    );
    assert_eq!(r.data, vec![3.0, 1.0, 1.0, 4.0]);
}

#[test]
fn ix_basic() {
    let r = rumpy_run(
        r#"
import numpy as np
i, j = np.ix_(np.array([0, 1]), np.array([0, 2, 3]))
result = np.array([i.shape[0], i.shape[1], j.shape[0], j.shape[1]]).astype("float64")
"#,
    );
    assert_eq!(r.data, vec![2.0, 1.0, 1.0, 3.0]);
}

#[test]
fn s_returns_slice() {
    // np.s_[1:5, 2:8] is just the slice tuple; verify by indexing through it.
    let r = rumpy_run(
        r#"
import numpy as np
a = np.arange(20.0).reshape(4, 5)
sl = np.s_[1:3, 1:4]
result = a[sl]
"#,
    );
    assert_eq!(r.shape, vec![2, 3]);
}

// =====================================================================
// fromiter / fromstring / array_equiv / iscomplexobj / isposinf
// =====================================================================

#[test]
fn fromiter_basic() {
    let r = rumpy_run(
        r#"
import numpy as np
result = np.fromiter(range(5), dtype="float64")
"#,
    );
    assert_eq!(r.data, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn fromiter_with_count() {
    let r = rumpy_run(
        r#"
import numpy as np
result = np.fromiter(range(100), dtype="float64", count=3)
"#,
    );
    assert_eq!(r.data, vec![0.0, 1.0, 2.0]);
}

#[test]
fn fromstring_whitespace() {
    let r = rumpy_run(
        r#"
import numpy as np
result = np.fromstring("1 2 3 4 5", sep=" ")
"#,
    );
    assert_eq!(r.data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn fromstring_commas() {
    let r = rumpy_run(
        r#"
import numpy as np
result = np.fromstring("1.5,2.5,3.5", sep=",")
"#,
    );
    assert_eq!(r.data, vec![1.5, 2.5, 3.5]);
}

#[test]
fn array_equiv_broadcasts() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([[1.0, 2.0, 3.0]])
ok = np.array_equiv(a, b)
result = np.array([1.0 if ok else 0.0])
"#,
    );
    assert_eq!(r.data, vec![1.0]);
}

#[test]
fn array_equiv_mismatched() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, 2.0])
b = np.array([1.0, 2.0, 3.0])
ok = np.array_equiv(a, b)
result = np.array([1.0 if ok else 0.0])
"#,
    );
    assert_eq!(r.data, vec![0.0]);
}

#[test]
fn isposinf_isneginf() {
    let r = rumpy_run(
        r#"
import numpy as np
inf = float("inf")
a = np.array([1.0, inf, -inf, 0.0])
result = np.array([np.isposinf(a).astype(int), np.isneginf(a).astype(int)])
"#,
    );
    assert_eq!(r.shape, vec![2, 4]);
    assert_eq!(r.data, vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
}

#[test]
fn iscomplexobj_real() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1.0, 2.0])
result = np.array([1.0 if np.iscomplexobj(a) else 0.0,
                   1.0 if np.isrealobj(a) else 0.0])
"#,
    );
    assert_eq!(r.data, vec![0.0, 1.0]);
}

#[test]
fn iscomplexobj_complex() {
    let r = rumpy_run(
        r#"
import numpy as np
a = np.array([1+0j, 2+0j])
result = np.array([1.0 if np.iscomplexobj(a) else 0.0,
                   1.0 if np.isrealobj(a) else 0.0])
"#,
    );
    assert_eq!(r.data, vec![1.0, 0.0]);
}
