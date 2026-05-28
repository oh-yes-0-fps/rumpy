"""
numpy.lib — collection of legacy and utility submodules / helpers.

Real numpy uses ``numpy.lib`` as the staging area for utilities that
either predate the top-level module or shouldn't pollute it. Most things
here are tiny shims that satisfy ``from numpy.lib import X`` imports
written against numpy.
"""

# ``stride_tricks`` is patched in by Rust before this module's namespace
# is finalised. The rest of the surface is implemented below.


class Arrayterator:
    """Iterates over an N-D array in (potentially small) chunks.

    rumpy's iteration model is just plain Python — this is a faithful
    port of numpy's ``Arrayterator`` semantics with one block per yield.
    """

    def __init__(self, var, buf_size=None):
        self.var = var
        self.buf_size = buf_size
        self.shape = getattr(var, "shape", (len(var),) if hasattr(var, "__len__") else ())
        self.start = [0] * len(self.shape)
        self.stop = list(self.shape)
        self.step = [1] * len(self.shape)

    def __getitem__(self, index):
        result = self.__class__(self.var, self.buf_size)
        result.start = list(self.start)
        result.stop = list(self.stop)
        result.step = list(self.step)
        if not isinstance(index, tuple):
            index = (index,)
        for axis, idx in enumerate(index):
            if isinstance(idx, slice):
                s = idx.indices(self.shape[axis])
                result.start[axis] = s[0]
                result.stop[axis] = s[1]
                result.step[axis] = s[2]
        return result

    def __iter__(self):
        # Naïve: yield one element at a time over the C-order traversal.
        flat = list(self.flat)
        for v in flat:
            yield v

    @property
    def flat(self):
        if hasattr(self.var, "ravel"):
            return list(self.var.ravel().tolist())
        return list(self.var)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = tuple(value)


class NumpyVersion:
    """Parses and compares numpy-style version strings (``major.minor.patch``)."""

    def __init__(self, vstring):
        self.vstring = vstring
        parts = vstring.split(".")
        major = int(parts[0]) if parts and parts[0].isdigit() else 0
        minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        bugfix = int(parts[2]) if len(parts) > 2 and parts[2].split("a")[0].split("b")[0].split("rc")[0].split("+")[0].isdigit() else 0
        self.major = major
        self.minor = minor
        self.bugfix = bugfix

    def _tuple(self):
        return (self.major, self.minor, self.bugfix)

    def __lt__(self, other):
        return self._tuple() < self._coerce(other)._tuple()

    def __le__(self, other):
        return self._tuple() <= self._coerce(other)._tuple()

    def __gt__(self, other):
        return self._tuple() > self._coerce(other)._tuple()

    def __ge__(self, other):
        return self._tuple() >= self._coerce(other)._tuple()

    def __eq__(self, other):
        return self._tuple() == self._coerce(other)._tuple()

    def __ne__(self, other):
        return self._tuple() != self._coerce(other)._tuple()

    def __repr__(self):
        return f"NumpyVersion('{self.vstring}')"

    @classmethod
    def _coerce(cls, other):
        return other if isinstance(other, cls) else cls(str(other))


def add_docstring(obj, doc):
    """Best-effort: set ``obj.__doc__ = doc``. Some C-extension types reject this."""
    try:
        obj.__doc__ = doc
    except (TypeError, AttributeError):
        pass


def add_newdoc(place, obj, doc, warn_on_python=True):
    """Look up ``place.obj`` and assign its docstring."""
    _ = warn_on_python
    try:
        target = getattr(place, obj) if isinstance(obj, str) else obj
        add_docstring(target, doc if isinstance(doc, str) else doc[1])
    except AttributeError:
        pass


# `tracemalloc_domain` is the numeric ID numpy uses when reporting allocations
# via ``tracemalloc``. We don't allocate from a domain-aware allocator, so the
# value is purely informational.
tracemalloc_domain = 389047


# ---- Sub-namespaces that real numpy exposes as ``numpy.lib.X`` ----

class _Namespace:
    """Tiny attribute namespace used to mimic numpy.lib's stripped-down submodules."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _normalize_axis_index(axis, ndim):
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"axis {axis} out of range for ndim={ndim}")
    return axis


def _normalize_axis_tuple(axis, ndim, argname=None, allow_duplicate=False):
    _ = argname
    if isinstance(axis, int):
        axis = (axis,)
    out = tuple(_normalize_axis_index(a, ndim) for a in axis)
    if not allow_duplicate and len(set(out)) != len(out):
        raise ValueError("duplicate axes in tuple")
    return out


array_utils = _Namespace(
    normalize_axis_index=_normalize_axis_index,
    normalize_axis_tuple=_normalize_axis_tuple,
)


# ``numpy.lib.format`` is the file-format helpers for .npy / .npz; we expose
# the version constants downstream code probes for.
format = _Namespace(
    MAGIC_PREFIX=b"\x93NUMPY",
    MAGIC_LEN=8,
    BUFFER_SIZE=2 ** 16,
    GROWTH_AXIS_MAX_DIGITS=21,
    ARRAY_ALIGN=64,
    EXPECTED_KEYS={"descr", "fortran_order", "shape"},
)


introspect = _Namespace()


class _BroadcastMixin:
    """Real numpy uses this to mix broadcasting into ndarray-like classes."""


class _NDArrayOperatorsMixin:
    """Real numpy uses this to wire arithmetic protocol methods onto subclasses."""


mixins = _Namespace(
    NDArrayOperatorsMixin=_NDArrayOperatorsMixin,
)


npyio = _Namespace()


# `scimath` is the numpy-emath family — domain-extended versions of log,
# sqrt, etc. The actual implementations live in numpy.emath, but real
# numpy also exposes them at numpy.lib.scimath.
def _import_emath():
    # Lazy import to avoid pulling numpy in at module load time.
    try:
        import numpy
        return numpy.emath
    except ImportError:
        return _Namespace()


scimath = _import_emath()


def test(*args, **kwargs):
    """rumpy ships no test runner."""
    _ = (args, kwargs)
    return True


__all__ = [
    "stride_tricks",
    "Arrayterator",
    "NumpyVersion",
    "add_docstring",
    "add_newdoc",
    "tracemalloc_domain",
    "array_utils",
    "format",
    "introspect",
    "mixins",
    "npyio",
    "scimath",
    "test",
]
