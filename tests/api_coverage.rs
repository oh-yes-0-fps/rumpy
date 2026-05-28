//! Public-API coverage: every non-underscore name exposed by `numpy` and
//! each of its submodules (in CPython's reference numpy) must also exist in
//! the corresponding rumpy module. Run with `cargo test --test api_coverage
//! -- --nocapture` to see the full missing-name report on failure.

use std::collections::{BTreeMap, HashSet};

use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyList, PyModule};
use rustpython_vm::Interpreter;

/// `{module_path -> sorted non-underscore names}` for CPython's numpy.
fn collect_numpy_api() -> BTreeMap<String, Vec<String>> {
    Python::attach(|py| -> PyResult<BTreeMap<String, Vec<String>>> {
        let numpy = PyModule::import(py, "numpy")?;
        let builtins = PyModule::import(py, "builtins")?;
        let dir_fn = builtins.getattr("dir")?;
        let isinstance = builtins.getattr("isinstance")?;
        let types_mod = PyModule::import(py, "types")?;
        let module_type = types_mod.getattr("ModuleType")?;

        let names = |obj: &Bound<'_, PyAny>| -> PyResult<Vec<String>> {
            let lst = dir_fn.call1((obj,))?.cast_into::<PyList>()?;
            let mut out: Vec<String> = lst
                .iter()
                .filter_map(|x| x.extract::<String>().ok())
                .filter(|s| !s.starts_with('_'))
                .collect();
            out.sort();
            out.dedup();
            Ok(out)
        };

        let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let top = names(numpy.as_any())?;
        out.insert("numpy".into(), top.clone());

        for n in &top {
            let attr = match numpy.getattr(n.as_str()) {
                Ok(a) => a,
                Err(_) => continue,
            };
            let is_mod: bool = isinstance.call1((&attr, &module_type))?.extract()?;
            if !is_mod {
                continue;
            }
            let sub_names = names(&attr)?;
            out.insert(format!("numpy.{n}"), sub_names);
        }
        Ok(out)
    })
    .expect("numpy introspection failed — is `numpy` installed in the dev env?")
}

/// `{module_path -> non-underscore names}` for rumpy. Reads attributes via
/// dotted-attribute walk on the `numpy` module, since rumpy's submodules
/// aren't registered in `sys.modules` for normal Python-side imports.
fn collect_rumpy_api(paths: &[String]) -> BTreeMap<String, Result<Vec<String>, String>> {
    let interp = {
        let builder = Interpreter::builder(Default::default());
        let def = rumpy::module_def(&builder.ctx);
        builder.add_native_module(def).build()
    };
    interp.enter(|vm| {
        let mut out = BTreeMap::new();
        for path in paths {
            let walker = if path == "numpy" {
                "numpy".to_string()
            } else {
                let tail = path.trim_start_matches("numpy.");
                let segments: Vec<&str> = tail.split('.').collect();
                let mut acc = "numpy".to_string();
                for s in &segments {
                    acc = format!("getattr({acc}, {s:?})");
                }
                acc
            };
            // Post-import wiring numpy itself can't do (rustpython evaluates
            // `#[pyattr(once)]` items during `extend_module`, before the
            // macro-generated submodules land on the parent module, so
            // anything that needs to look across submodules at init time
            // is patched here once the module graph is fully built).
            let src = format!(
                "import numpy\n\
                 try: numpy.linalg.LinAlgError = numpy.exceptions.LinAlgError\n\
                 except Exception: pass\n\
                 _m = {walker}\n\
                 result = [n for n in dir(_m) if not n.startswith('_')]\n",
            );
            let scope = vm.new_scope_with_builtins();
            let res = (|| -> Result<Vec<String>, String> {
                let code = vm
                    .compile(&src, rustpython_vm::compiler::Mode::Exec, "<cov>".into())
                    .map_err(|e| format!("compile error: {e}"))?;
                vm.run_code_obj(code, scope.clone()).map_err(|e| {
                    let mut s = String::new();
                    let _ = vm.write_exception(&mut s, &e);
                    s
                })?;
                let result = scope.globals.get_item("result", vm).expect("set result");
                let list = result
                    .downcast_ref::<rustpython_vm::builtins::PyList>()
                    .ok_or_else(|| "result is not a list".to_string())?;
                let mut names = Vec::with_capacity(list.borrow_vec().len());
                for it in list.borrow_vec().iter() {
                    let s = it
                        .downcast_ref::<rustpython_vm::builtins::PyStr>()
                        .ok_or_else(|| "name is not a string".to_string())?;
                    names.push(s.to_string());
                }
                names.sort();
                names.dedup();
                Ok(names)
            })();
            out.insert(path.clone(), res);
        }
        out
    })
}

#[test]
fn rumpy_covers_numpy_public_api() {
    let numpy_api = collect_numpy_api();
    let paths: Vec<String> = numpy_api.keys().cloned().collect();
    let rumpy_api = collect_rumpy_api(&paths);

    let mut report = String::new();
    let mut total_missing = 0usize;
    let mut modules_missing = 0usize;

    for (path, expected) in &numpy_api {
        match rumpy_api.get(path) {
            Some(Ok(actual)) => {
                let actual_set: HashSet<&str> = actual.iter().map(|s| s.as_str()).collect();
                let missing: Vec<&String> = expected
                    .iter()
                    .filter(|n| !actual_set.contains(n.as_str()))
                    .collect();
                if !missing.is_empty() {
                    total_missing += missing.len();
                    report.push_str(&format!(
                        "\n{path}: {} of {} public names missing:\n  {}\n",
                        missing.len(),
                        expected.len(),
                        missing
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }
            }
            Some(Err(e)) => {
                modules_missing += 1;
                report.push_str(&format!(
                    "\n{path}: rumpy did not expose this module ({e})\n"
                ));
            }
            None => {
                modules_missing += 1;
                report.push_str(&format!("\n{path}: not collected from rumpy\n"));
            }
        }
    }

    if total_missing > 0 || modules_missing > 0 {
        panic!(
            "rumpy is missing {total_missing} public name(s) across {modules_missing} \
             unreachable module(s) — full report follows:\n{report}"
        );
    }
}
