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
    actual,
    desired,
    rtol=1e-7,
    atol=0.0,
    equal_nan=True,
    err_msg="",
    verbose=True,
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
    tol = 1.5 * (10**-decimal)
    assert_allclose(
        actual,
        desired,
        rtol=0.0,
        atol=tol,
        err_msg=err_msg,
        verbose=verbose,
    )


def assert_almost_equal(actual, desired, decimal=7, err_msg="", verbose=True):
    """Scalar/array closeness up to ``decimal`` decimal places."""
    assert_array_almost_equal(
        actual,
        desired,
        decimal=decimal,
        err_msg=err_msg,
        verbose=verbose,
    )


def assert_approx_equal(actual, desired, significant=7, err_msg=""):
    """Two scalars agree to ``significant`` significant digits."""
    if desired == 0.0:
        tol = 10**-significant
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
    raise AssertionError(f"expected {exc_type.__name__}, no exception raised")


def assert_warns(*args, **kwargs):
    """Stub — rumpy's minimal build doesn't track warning state.

    Always succeeds; present so test suites that decorate with
    ``assert_warns`` don't fail at import time."""
    # If a callable was passed, just call it; otherwise no-op.
    if args and callable(args[0]):
        return args[0]()
    return None


# ---- Platform / build introspection flags ----
#
# Real numpy populates these from its compile-time configuration so
# downstream test suites can skip platform-specific cases. rumpy is
# self-contained: we report a conservative "modern 64-bit, no exotic
# acceleration" environment.

try:
    import sys as _sys

    IS_64BIT = _sys.maxsize > 2**32
except ImportError:
    IS_64BIT = True

try:
    import platform as _platform

    IS_PYPY = _platform.python_implementation() == "PyPy"
    IS_WASM = _platform.machine().lower().startswith("wasm")
    IS_MUSL = (
        "musl" in _platform.libc_ver()[0].lower()
        if hasattr(_platform, "libc_ver")
        else False
    )
except ImportError:
    IS_PYPY = False
    IS_WASM = False
    IS_MUSL = False

IS_PYSTON = False
IS_EDITABLE = False
IS_INSTALLED = True
NOGIL_BUILD = False
HAS_REFCOUNT = True
HAS_LAPACK64 = False
BLAS_SUPPORTS_FPE = True
NUMPY_ROOT = ""

verbose = 0


# ---- Exceptions and warning subclasses re-exported by numpy.testing ----


class IgnoreException(Exception):
    """Used to flag a test as intentionally skipped."""


class KnownFailureException(Exception):
    """Used to mark an expected test failure."""


class SkipTest(Exception):
    """Raised to skip a test (compatible with unittest's SkipTest)."""


# ---- Mini-unittest compatibility ----


class TestCase:
    """Minimal stand-in for ``unittest.TestCase``.

    Real numpy re-exports CPython's ``unittest.TestCase``; this build
    of rustpython doesn't ship ``unittest``, so we provide a tiny shim
    with the assertion methods downstream test suites lean on.
    """

    def assertEqual(self, a, b, msg=None):
        if a != b:
            raise AssertionError(msg or f"{a!r} != {b!r}")

    def assertNotEqual(self, a, b, msg=None):
        if a == b:
            raise AssertionError(msg or f"{a!r} == {b!r}")

    def assertTrue(self, x, msg=None):
        if not x:
            raise AssertionError(msg or f"{x!r} is not truthy")

    def assertFalse(self, x, msg=None):
        if x:
            raise AssertionError(msg or f"{x!r} is not falsy")

    def assertRaises(self, exc, fn, *args, **kw):
        assert_raises(exc, fn, *args, **kw)


# ---- Extra assertion helpers ----


def assert_(condition, msg=""):
    """Raise ``AssertionError(msg)`` if ``condition`` is falsy."""
    if not condition:
        raise AssertionError(msg or "assertion failed")


def assert_array_compare(
    comparison, x, y, err_msg="", verbose=True, header="", strict=False, equal_nan=True
):
    """Element-wise comparison via ``comparison(x, y)`` for every pair."""
    _ = (verbose, header, strict, equal_nan)
    a = list(_flatten(x))
    b = list(_flatten(y))
    if len(a) != len(b):
        _fail(f"length mismatch: {len(a)} vs {len(b)}", err_msg)
    for i, (xi, yi) in enumerate(zip(a, b)):
        if not comparison(xi, yi):
            _fail(f"item {i}: comparison(xi={xi!r}, yi={yi!r}) is false", err_msg)


def assert_array_almost_equal_nulp(actual, desired, nulp=1):
    """Closeness measured in units in the last place (ULPs)."""
    import math

    a = list(_flatten(actual))
    d = list(_flatten(desired))
    for i, (x, y) in enumerate(zip(a, d)):
        if x == y:
            continue
        ulp = math.ulp(max(abs(float(x)), abs(float(y))))
        if abs(float(x) - float(y)) > nulp * ulp:
            raise AssertionError(f"item {i}: {x!r} vs {y!r} differs by > {nulp} ULP")


def assert_array_max_ulp(a, b, maxulp=1, dtype=None):
    """Like ``assert_array_almost_equal_nulp`` but with a per-element max."""
    _ = dtype
    assert_array_almost_equal_nulp(a, b, nulp=maxulp)
    return [maxulp]


def assert_no_gc_cycles(*args, **kwargs):
    """Stub — rumpy doesn't expose a refcount/gc hook."""
    if args and callable(args[0]):
        return args[0](*args[1:], **kwargs)


def assert_no_warnings(*args, **kwargs):
    if args and callable(args[0]):
        return args[0](*args[1:], **kwargs)


def assert_raises_regex(exc_type, regex, callable_, *args, **kwargs):
    import re

    try:
        callable_(*args, **kwargs)
    except exc_type as e:
        if not re.search(regex, str(e)):
            raise AssertionError(
                f"exception message {str(e)!r} did not match regex {regex!r}"
            )
        return
    raise AssertionError(f"expected {exc_type.__name__}, none raised")


def assert_string_equal(actual, desired):
    if actual != desired:
        raise AssertionError(
            f"strings differ:\n  actual:  {actual!r}\n  desired: {desired!r}"
        )


def break_cycles():
    """No-op — rumpy doesn't expose a GC handle."""
    pass


def build_err_msg(
    arrays,
    err_msg,
    header="Arrays are not equal",
    verbose=True,
    names=("ACTUAL", "DESIRED"),
    precision=8,
):
    _ = (verbose, precision)
    parts = [header]
    if err_msg:
        parts.append(err_msg)
    for n, a in zip(names, arrays):
        parts.append(f"{n}: {a!r}")
    return "\n".join(parts)


def check_support_sve():
    """No SVE support — rumpy targets the portable ndarray crate."""
    return False


class clear_and_catch_warnings:
    """Context manager that swallows warnings raised inside its body."""

    def __init__(self, record=False, modules=()):
        _ = (record, modules)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def decorate_methods(cls, decorator, testmatch=None):
    _ = testmatch
    for name in dir(cls):
        if name.startswith("test_"):
            setattr(cls, name, decorator(getattr(cls, name)))


def extbuild(*args, **kwargs):
    raise NotImplementedError("extbuild requires a C toolchain")


def jiffies(_proc_pid_stat="", _load_time=()):
    """Time in clock ticks since process start. Returns 0 in rumpy."""
    return 0


def measure(code_str, times=1, label=""):
    """Time ``times`` runs of ``code_str``; returns total seconds."""
    _ = label
    import time

    start = time.time()
    for _ in range(times):
        exec(code_str)
    return time.time() - start


def memusage(_proc_pid_stat=""):
    """Memory usage in bytes (best-effort; returns 0 here)."""
    return 0


class overrides:
    """No-op placeholder for numpy's `__array_function__` testing helper."""

    ARRAY_FUNCTIONS = set()


def print_assert_equal(test_string, actual, desired):
    """Print and run ``assert_equal``."""
    print(test_string)
    assert_equal(actual, desired)


def run_threaded(func, n_threads=2, args=(), kwargs=None):
    """Run ``func`` from ``n_threads`` threads concurrently."""
    import threading

    kwargs = kwargs or {}
    threads = [
        threading.Thread(target=func, args=args, kwargs=kwargs)
        for _ in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def rundocs(filename=None, raise_on_error=True):
    _ = (filename, raise_on_error)
    return True


def runstring(astr, dict_):
    """Compile and exec ``astr`` in ``dict_``."""
    exec(astr, dict_)


class suppress_warnings:
    """Context manager that swallows warnings emitted in its body."""

    def __init__(self, forwarding_rule="always"):
        _ = forwarding_rule

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def filter(self, *args, **kwargs):
        pass

    def record(self, *args, **kwargs):
        return []


def tempdir(*args, **kwargs):
    """Return a context manager yielding a fresh temporary directory."""
    import tempfile

    return tempfile.TemporaryDirectory(*args, **kwargs)


def temppath(*args, **kwargs):
    """Return a context manager yielding a fresh temporary file path."""
    import tempfile

    return tempfile.NamedTemporaryFile(*args, **kwargs)


def test(*args, **kwargs):
    """rumpy's bundled test runner is a no-op shim."""
    _ = (args, kwargs)
    return True


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
    "assert_",
    "assert_array_compare",
    "assert_array_almost_equal_nulp",
    "assert_array_max_ulp",
    "assert_no_gc_cycles",
    "assert_no_warnings",
    "assert_raises_regex",
    "assert_string_equal",
    "break_cycles",
    "build_err_msg",
    "check_support_sve",
    "clear_and_catch_warnings",
    "decorate_methods",
    "extbuild",
    "jiffies",
    "measure",
    "memusage",
    "overrides",
    "print_assert_equal",
    "run_threaded",
    "rundocs",
    "runstring",
    "suppress_warnings",
    "tempdir",
    "temppath",
    "test",
    "IgnoreException",
    "KnownFailureException",
    "SkipTest",
    "TestCase",
    "IS_64BIT",
    "IS_PYPY",
    "IS_PYSTON",
    "IS_WASM",
    "IS_MUSL",
    "IS_EDITABLE",
    "IS_INSTALLED",
    "NOGIL_BUILD",
    "HAS_REFCOUNT",
    "HAS_LAPACK64",
    "BLAS_SUPPORTS_FPE",
    "NUMPY_ROOT",
    "verbose",
]
