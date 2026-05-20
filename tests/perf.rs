//! Smoke-tests that exercise the perf-sensitive paths on larger arrays.
//! Not a benchmark — just sanity-checks that the optimised reduction +
//! Cow-cast paths still produce numerically correct results when the data
//! is non-trivial.

use approx::assert_abs_diff_eq;
use rustpython_vm::Interpreter;

fn rumpy_interp() -> Interpreter {
    let b = Interpreter::builder(Default::default());
    let def = rumpy::module_def(&b.ctx);
    b.add_native_module(def).build()
}

#[test]
fn large_sum_axis_correct() {
    // 1000-element reduction along axis 1, then sum the axis-0 result.
    let interp = rumpy_interp();
    let r = interp.enter(|vm| -> rustpython_vm::PyResult<f64> {
        let scope = vm.new_scope_with_builtins();
        let src = r#"
import numpy as np
a = np.arange(10000.0).reshape(100, 100)
result = float(a.sum(axis=1).sum())
"#;
        let code = vm
            .compile(src, rustpython_vm::compiler::Mode::Exec, "<t>".into())
            .map_err(|e| vm.new_syntax_error(&e, Some(src)))?;
        vm.run_code_obj(code, scope.clone())?;
        let r = scope.globals.get_item("result", vm).unwrap();
        Ok(r.try_float(vm)?.to_f64())
    });
    let r = r.expect("rumpy");
    // sum(0..10000) = 49995000
    assert_abs_diff_eq!(r, 49995000.0, epsilon = 1.0);
}

#[test]
fn large_binary_op_same_dtype_no_clone() {
    // Adding two f64 arrays of the promoted dtype should hit the no-clone
    // fast path inside binary_op. We can't observe the allocation count
    // directly but we can verify correctness on a sizable input.
    let interp = rumpy_interp();
    let r = interp.enter(|vm| -> rustpython_vm::PyResult<f64> {
        let scope = vm.new_scope_with_builtins();
        let src = r#"
import numpy as np
a = np.arange(50000.0)
b = np.arange(50000.0)
c = a + b
result = float(c.sum())
"#;
        let code = vm
            .compile(src, rustpython_vm::compiler::Mode::Exec, "<t>".into())
            .map_err(|e| vm.new_syntax_error(&e, Some(src)))?;
        vm.run_code_obj(code, scope.clone())?;
        let r = scope.globals.get_item("result", vm).unwrap();
        Ok(r.try_float(vm)?.to_f64())
    });
    // sum_{i=0}^{49999} (2i) = 2 · 49999·50000/2 = 49999·50000 = 2_499_950_000
    let r = r.expect("rumpy");
    assert_abs_diff_eq!(r, 2_499_950_000.0, epsilon = 100.0);
}

#[test]
fn cumsum_axis_correct_on_2d() {
    // Axis-aware cumsum on a 50×50 array.
    let interp = rumpy_interp();
    let r = interp.enter(|vm| -> rustpython_vm::PyResult<f64> {
        let scope = vm.new_scope_with_builtins();
        let src = r#"
import numpy as np
a = np.arange(2500.0).reshape(50, 50)
result = float(np.cumsum(a, axis=1).sum())
"#;
        let code = vm
            .compile(src, rustpython_vm::compiler::Mode::Exec, "<t>".into())
            .map_err(|e| vm.new_syntax_error(&e, Some(src)))?;
        vm.run_code_obj(code, scope.clone())?;
        let r = scope.globals.get_item("result", vm).unwrap();
        Ok(r.try_float(vm)?.to_f64())
    });
    // Row i (0..49) has values [50i, 50i+1, …, 50i+49]; row cumsum sum is
    //   sum_{j=0..49} [(j+1)·50i + j(j+1)/2] = 63750·i + 20825.
    // Summed over i in 0..49 → 78_093_750 + 50·20_825 = 79_135_000.
    let r = r.expect("rumpy");
    assert_abs_diff_eq!(r, 79_135_000.0, epsilon = 1.0);
}
