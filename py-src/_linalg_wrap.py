"""Python wrapper over the Rust-backed ``numpy._linalg_rust`` submodule.

Real numpy exposes ``LinAlgError`` as a top-level attribute of
``numpy.linalg``. The ``#[pymodule]`` macro in rustpython doesn't accept
nested pyattrs, so we materialise ``numpy.linalg`` as this Python module
that re-exports every public name from the Rust core plus the canonical
``LinAlgError`` class from ``numpy.exceptions``.

``_rust_linalg`` and ``_exceptions`` are injected from Rust.
"""


# Re-export every public Rust-side function from `numpy.linalg_rust`.
for _name in dir(_rust_linalg):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_rust_linalg, _name)


# Bring in the canonical LinAlgError.
LinAlgError = _exceptions.LinAlgError


# `numpy.linalg.test()` — no-op test runner.
def test(*args, **kwargs):
    _ = (args, kwargs)
    return True
