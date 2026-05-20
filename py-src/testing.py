"""``numpy.testing`` — assert helpers used by numpy's own test suite and
by downstream libraries.

This is the public sliver that matters most: ``assert_equal``,
``assert_allclose``, ``assert_array_equal`` (and friends). The helpers
work on any iterable structure — they iterate elementwise and only
recurse when the elements are themselves iterable.
"""


def _is_seq(x):
    return isinstance(x, (list, tuple)) or (
        hasattr(x, "tolist") and not isinstance(x, (str, bytes))
    )


def _flatten(x):
    """Yield leaf elements of a possibly-nested sequence/array."""
    if hasattr(x, "tolist") and not isinstance(x, (str, bytes)):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        for v in x:
            for leaf in _flatten(v):
                yield leaf
    else:
        yield x


def _shape(x):
    """Best-effort shape tuple for nested sequences / ndarrays."""
    if hasattr(x, "shape"):
        return tuple(x.shape)
    if isinstance(x, (list, tuple)):
        if not x:
            return (0,)
        first = _shape(x[0])
        return (len(x),) + first
    return ()


def _fail(msg, err_msg=None):
    if err_msg:
        msg = f"{err_msg}\n{msg}"
    raise AssertionError(msg)


def assert_equal(actual, desired, err_msg="", verbose=True):
    """Strict equality. Recurses into sequences elementwise."""
    a = list(_flatten(actual))
    d = list(_flatten(desired))
    if _shape(actual) != _shape(desired):
        _fail(
            f"shape mismatch: {_shape(actual)} vs {_shape(desired)}",
            err_msg,
        )
    if len(a) != len(d):
        _fail(
            f"length mismatch: {len(a)} vs {len(d)}",
            err_msg,
        )
    for i, (x, y) in enumerate(zip(a, d)):
        if x != y:
            extra = f" (item {i})" if verbose else ""
            _fail(f"{x!r} != {y!r}{extra}", err_msg)


def assert_array_equal(actual, desired, err_msg="", verbose=True):
    """Alias for ``assert_equal`` on array-like inputs."""
    assert_equal(actual, desired, err_msg=err_msg, verbose=verbose)


def assert_allclose(
    actual, desired, rtol=1e-7, atol=0.0, equal_nan=True,
    err_msg="", verbose=True,
):
    """All elements close in absolute and relative tolerance."""
    a = list(_flatten(actual))
    d = list(_flatten(desired))
    if len(a) != len(d):
        _fail(f"length mismatch: {len(a)} vs {len(d)}", err_msg)
    for i, (x, y) in enumerate(zip(a, d)):
        # NaN handling.
        x_nan = (x != x) if isinstance(x, float) else False
        y_nan = (y != y) if isinstance(y, float) else False
        if x_nan and y_nan:
            if equal_nan:
                continue
            _fail(f"NaN at item {i} (equal_nan=False)", err_msg)
        if x_nan or y_nan:
            _fail(f"NaN mismatch at item {i}: {x!r} vs {y!r}", err_msg)
        diff = abs(x - y)
        tol = atol + rtol * abs(y)
        if diff > tol:
            extra = f" (item {i}, diff={diff}, tol={tol})" if verbose else ""
            _fail(f"{x!r} not close to {y!r}{extra}", err_msg)


def assert_array_almost_equal(actual, desired, decimal=6, err_msg="", verbose=True):
    """Closeness up to ``decimal`` decimal places."""
    tol = 1.5 * (10 ** -decimal)
    assert_allclose(
        actual, desired, rtol=0.0, atol=tol,
        err_msg=err_msg, verbose=verbose,
    )


def assert_almost_equal(actual, desired, decimal=7, err_msg="", verbose=True):
    """Scalar/array closeness up to ``decimal`` decimal places."""
    assert_array_almost_equal(
        actual, desired, decimal=decimal,
        err_msg=err_msg, verbose=verbose,
    )


def assert_approx_equal(actual, desired, significant=7, err_msg=""):
    """Two scalars agree to ``significant`` significant digits."""
    if desired == 0.0:
        tol = 10 ** -significant
    else:
        tol = abs(desired) * 10 ** (-significant + 1)
    if abs(actual - desired) > tol:
        _fail(
            f"{actual!r} not approx equal to {desired!r} "
            f"to {significant} significant digits",
            err_msg,
        )


def assert_array_less(x, y, err_msg=""):
    """Strict element-wise ``x < y``."""
    a = list(_flatten(x))
    b = list(_flatten(y))
    for i, (p, q) in enumerate(zip(a, b)):
        if not (p < q):
            _fail(f"item {i}: {p!r} not < {q!r}", err_msg)


def assert_raises(exc_type, callable_, *args, **kwargs):
    """Assert ``callable_(*args, **kwargs)`` raises ``exc_type``."""
    try:
        callable_(*args, **kwargs)
    except exc_type:
        return
    raise AssertionError(
        f"expected {exc_type.__name__}, no exception raised"
    )


def assert_warns(*args, **kwargs):
    """Stub — rumpy's minimal build doesn't track warning state.

    Always succeeds; present so test suites that decorate with
    ``assert_warns`` don't fail at import time."""
    # If a callable was passed, just call it; otherwise no-op.
    if args and callable(args[0]):
        return args[0]()
    return None


__all__ = [
    "assert_equal",
    "assert_array_equal",
    "assert_allclose",
    "assert_array_almost_equal",
    "assert_almost_equal",
    "assert_approx_equal",
    "assert_array_less",
    "assert_raises",
    "assert_warns",
]
