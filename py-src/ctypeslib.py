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


__all__ = [
    "as_array",
    "as_ctypes",
    "as_ctypes_type",
    "ndpointer",
    "load_library",
]
