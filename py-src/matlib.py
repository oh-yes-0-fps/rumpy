"""``numpy.matlib`` — matrix-returning shims for common constructors.

This is the legacy companion to :mod:`numpy.matrixlib`. It re-exports the
familiar numpy constructors (``zeros``, ``ones``, ``eye``, ``identity``,
``empty``, ``rand``, ``randn``) but coerces the result to a ``matrix``
so the rest of the matrix arithmetic API kicks in (``*`` as matmul, etc.).
``repmat`` is also provided.

``_np`` is the rumpy numpy module, injected from Rust. ``matrix``,
``asmatrix``, ``mat``, and ``bmat`` are also injected (sourced from
``numpy.matrixlib``) since rumpy's submodules aren't registered in
``sys.modules`` for normal Python-side imports.
"""


def _as_shape(shape):
    if isinstance(shape, int):
        return (shape, shape)
    if isinstance(shape, (list, tuple)):
        if len(shape) == 1:
            return (1, int(shape[0]))
        if len(shape) == 2:
            return (int(shape[0]), int(shape[1]))
        raise ValueError("matlib: shape must be 1- or 2-dimensional")
    raise TypeError("matlib: shape must be int or sequence of ints")


def empty(shape, dtype=None, order="C"):
    _ = order
    return matrix(_np.empty(_as_shape(shape), dtype=dtype))


def zeros(shape, dtype=None, order="C"):
    _ = order
    return matrix(_np.zeros(_as_shape(shape), dtype=dtype))


def ones(shape, dtype=None, order="C"):
    _ = order
    return matrix(_np.ones(_as_shape(shape), dtype=dtype))


def eye(n, M=None, k=0, dtype=None, order="C"):
    _ = order
    _ = k  # rumpy's np.eye is an unshifted identity; ignore offset
    m = n if M is None else M
    return matrix(_np.eye(n, m, dtype=dtype))


def identity(n, dtype=None):
    return matrix(_np.identity(n, dtype=dtype))


def rand(*shape):
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    shape = _as_shape(shape if len(shape) > 1 else shape[0]) if shape else (1, 1)
    return matrix(_np.random.rand(*shape))


def randn(*shape):
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    shape = _as_shape(shape if len(shape) > 1 else shape[0]) if shape else (1, 1)
    return matrix(_np.random.randn(*shape))


def repmat(a, m, n):
    """Repeat a matrix ``m`` times along axis 0 and ``n`` times along axis 1."""
    a = asmatrix(a)._a
    return matrix(_np.tile(a, (m, n)))


__all__ = [
    "matrix",
    "asmatrix",
    "mat",
    "bmat",
    "empty",
    "zeros",
    "ones",
    "eye",
    "identity",
    "rand",
    "randn",
    "repmat",
]
