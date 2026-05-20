"""``numpy.emath`` — math functions that auto-promote real inputs to
complex when the real-domain answer would be undefined.

``emath.sqrt(-1) == 1j`` where regular ``math.sqrt(-1)`` raises and
``numpy.sqrt(-1)`` returns ``nan``.

Implementations are pure Python and route through Rust-supplied math
primitives (``_sqrt``, ``_log``, ``_atan2``, …) injected by rumpy at
module-init time. This sidesteps the ``import math`` requirement so the
module works in minimal rustpython builds.
"""


# Primitives ``_sqrt``, ``_log``, … are injected from Rust before this
# file executes; the assignments below are static-analysis fallbacks.
try:
    _sqrt  # noqa: F821
except NameError:  # pragma: no cover
    _sqrt = lambda x: x ** 0.5
    _log = lambda x: 0.0
    _log10 = lambda x: 0.0
    _log2 = lambda x: 0.0
    _atan2 = lambda y, x: 0.0
    _hypot = lambda x, y: 0.0
    _acos = lambda x: 0.0
    _asin = lambda x: 0.0
    _atanh = lambda x: 0.0


def _map(x, f):
    if isinstance(x, (list, tuple)):
        return [_map(v, f) for v in x]
    return f(x)


def _to_complex(x):
    if isinstance(x, complex):
        return x
    return complex(x, 0.0)


def _csqrt(z):
    if isinstance(z, complex):
        x, y = float(z.real), float(z.imag)
    else:
        x, y = float(z), 0.0
    if x == 0.0 and y == 0.0:
        return complex(0.0, 0.0)
    r = _hypot(x, y)
    re = _sqrt((r + x) / 2.0)
    im_mag = _sqrt((r - x) / 2.0)
    im = im_mag if y >= 0.0 else -im_mag
    return complex(re, im)


def _clog(z):
    if not isinstance(z, complex):
        z = complex(z, 0.0)
    r = _hypot(float(z.real), float(z.imag))
    theta = _atan2(float(z.imag), float(z.real))
    return complex(_log(r), theta)


def sqrt(x):
    """Element-wise square root, promoting negatives to complex."""

    def f(v):
        if isinstance(v, complex):
            return _csqrt(v)
        if v < 0:
            return _csqrt(complex(v, 0.0))
        return _sqrt(float(v))

    return _map(x, f)


def log(x):
    """Natural log, promoting non-positive reals to complex."""

    def f(v):
        if isinstance(v, complex):
            return _clog(v)
        if v <= 0:
            return _clog(complex(v, 0.0))
        return _log(float(v))

    return _map(x, f)


def log2(x):
    ln2 = _log(2.0)

    def f(v):
        if isinstance(v, complex):
            return _clog(v) / ln2
        if v <= 0:
            return _clog(complex(v, 0.0)) / ln2
        return _log(float(v)) / ln2

    return _map(x, f)


def log10(x):
    ln10 = _log(10.0)

    def f(v):
        if isinstance(v, complex):
            return _clog(v) / ln10
        if v <= 0:
            return _clog(complex(v, 0.0)) / ln10
        return _log10(float(v))

    return _map(x, f)


def logn(n, x):
    lnn = _log(float(n))

    def f(v):
        if isinstance(v, complex):
            return _clog(v) / lnn
        if v <= 0:
            return _clog(complex(v, 0.0)) / lnn
        return _log(float(v)) / lnn

    return _map(x, f)


def power(x, p):
    """``x ** p`` with complex promotion for fractional powers of negatives."""

    def f(v):
        if isinstance(v, complex):
            return v ** p
        if v < 0 and not float(p).is_integer():
            return complex(v, 0.0) ** p
        return v ** p

    return _map(x, f)


def arccos(x):
    def f(v):
        if isinstance(v, complex) or abs(v) > 1:
            z = _to_complex(v)
            inside = z + complex(0, 1) * _csqrt(complex(1, 0) - z * z)
            return complex(0, -1) * _clog(inside)
        return _acos(float(v))

    return _map(x, f)


def arcsin(x):
    def f(v):
        if isinstance(v, complex) or abs(v) > 1:
            z = _to_complex(v)
            inside = complex(0, 1) * z + _csqrt(complex(1, 0) - z * z)
            return complex(0, -1) * _clog(inside)
        return _asin(float(v))

    return _map(x, f)


def arctanh(x):
    def f(v):
        if isinstance(v, complex) or abs(v) >= 1:
            z = _to_complex(v)
            return 0.5 * (_clog(complex(1, 0) + z) - _clog(complex(1, 0) - z))
        return _atanh(float(v))

    return _map(x, f)


__all__ = [
    "sqrt",
    "log",
    "log2",
    "log10",
    "logn",
    "power",
    "arccos",
    "arcsin",
    "arctanh",
]
