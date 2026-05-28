"""``numpy.ctypeslib`` — ctypes interop helpers (stub).

rumpy doesn't implement ctypes integration. The functions below exist so
that ``from numpy.ctypeslib import ...`` resolves; calling them raises
``NotImplementedError``.
"""


def _stub(name):
    def _f(*args, **kwargs):
        raise NotImplementedError(
            f"numpy.ctypeslib.{name} is not implemented in rumpy"
        )

    _f.__name__ = name
    return _f


as_array = _stub("as_array")
as_ctypes = _stub("as_ctypes")
as_ctypes_type = _stub("as_ctypes_type")
ndpointer = _stub("ndpointer")
load_library = _stub("load_library")


# Real numpy re-exports the standard ``ctypes`` module here and exposes
# ``c_intp`` as the C-level integer type wide enough to hold a pointer.
#
# Embedders can ship the Python-side ``ctypes`` package (from
# ``rustpython-pylib``) alongside rumpy; when present, we re-export it
# verbatim. When absent, ``import ctypes`` fails and we expose a thin
# placeholder built from the low-level ``_ctypes`` Rust module — enough
# for attribute lookups and ``c_intp(value)`` round-trips.

try:
    import ctypes as _ctypes_module  # noqa: F401 — re-exported below
    ctypes = _ctypes_module
    c_intp = ctypes.c_int64 if hasattr(ctypes, "c_int64") else ctypes.c_long
except ImportError:
    try:
        import _ctypes as _lowlevel
    except ImportError:
        _lowlevel = None

    class _CTypesShim:
        """Best-effort ``ctypes`` module placeholder.

        Delegates attribute lookup to ``_ctypes`` when available so the
        low-level Rust types (``_ctypes.Array``, ``_ctypes.Pointer``, …)
        remain reachable. Returns ``None`` for unknown attributes.
        """

        def __getattr__(self, name):
            if _lowlevel is not None and hasattr(_lowlevel, name):
                return getattr(_lowlevel, name)
            raise AttributeError(
                f"ctypes.{name} is not available in this rumpy build "
                "(install rustpython-pylib for full ctypes support)"
            )

    ctypes = _CTypesShim()

    class c_intp:
        """Placeholder for ``ctypes.c_intp`` (64-bit on every rumpy target)."""

        def __init__(self, value=0):
            self.value = int(value)


__all__ = [
    "as_array",
    "as_ctypes",
    "as_ctypes_type",
    "ndpointer",
    "load_library",
    "ctypes",
    "c_intp",
]
