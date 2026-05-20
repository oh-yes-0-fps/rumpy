# rumpy

A numpy-compatible Python module implemented in Rust on top of [`ndarray`],
exposed to [`rustpython-vm`] as the module `numpy`.

## What it is

Drop-in `import numpy as np` for embedded [RustPython] interpreters — no
CPython, no PyO3, no C extension. Pure Rust array core with a Python surface
that matches numpy's API where it makes sense.

## Dtypes

`bool`, `int8/16/32/64`, `uint8/16/32/64`, `float16/32/64`, `complex64/128`.

Promotion follows `numpy.result_type` (see `src/promote.rs`).

## Usage

```rust
use rustpython_vm::Interpreter;

let interp = Interpreter::with_init(Default::default(), |vm| {
    vm.add_native_module("numpy".to_owned(), Box::new(rumpy::module_def));
});

interp.enter(|vm| {
    let scope = vm.new_scope_with_builtins();
    vm.run_code_string(scope, "import numpy as np; print(np.arange(6).reshape(2,3))", "<embed>".into())
        .unwrap();
});
```

## Layout

- `src/` — Rust array core, ops, linalg, fft, npy/npz I/O, indexing, einsum.
- `py-src/` — Python shims loaded into the `numpy` module namespace (dtypes,
  polynomial, testing, etc.).
- `tests/` — Rust-side correctness tests, including parity checks against
  CPython numpy via `pyo3`.

## Features

- `safe-locks` — wraps array storage with a real lock (via
  `rustpython-vm/threading`). Off by default; embedders that don't run the VM
  on multiple OS threads can skip it.

## Status

Pre-1.0. The numeric core and most numpy surface area work; gaps and rough
edges exist. File issues with a failing snippet.

[`ndarray`]: https://docs.rs/ndarray
[`rustpython-vm`]: https://docs.rs/rustpython-vm
[RustPython]: https://rustpython.github.io/
