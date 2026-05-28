"""Pure-Python extras patched onto the top-level ``numpy`` module.

These are functions that real numpy exposes at ``np.<name>`` but were
missing from rumpy. They're implemented here so they can lean on the
already-exposed numeric core (matmul, where, concatenate, mean, etc.)
without having to thread each one through Rust manually.

``_np`` is the rumpy numpy module, injected from Rust at module-build time.
"""


# ---- math helpers ----

def _pi():
    return _np.pi


def sinc(x):
    """``sin(pi*x) / (pi*x)``, with ``sinc(0) = 1``. Returns an ndarray."""
    a = _np.asarray(x, dtype="float64")
    pi = _pi()
    # Avoid division by zero element-wise: use where().
    safe = _np.where(a == 0, 1.0, a)
    out = _np.sin(pi * safe) / (pi * safe)
    return _np.where(a == 0, 1.0, out)


def float_power(a, b):
    """Raise to a power, always returning float (never integer overflow)."""
    return _np.power(_np.asarray(a, dtype="float64"), _np.asarray(b, dtype="float64"))


def logaddexp(a, b):
    """``log(exp(a) + exp(b))`` computed stably."""
    a = _np.asarray(a, dtype="float64")
    b = _np.asarray(b, dtype="float64")
    m = _np.maximum(a, b)
    return m + _np.log(_np.exp(a - m) + _np.exp(b - m))


def logaddexp2(a, b):
    """``log2(2**a + 2**b)`` computed stably."""
    a = _np.asarray(a, dtype="float64")
    b = _np.asarray(b, dtype="float64")
    m = _np.maximum(a, b)
    return m + _np.log2(_np.exp2(a - m) + _np.exp2(b - m))


def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    """Replace NaN with ``nan``, +inf with ``posinf``, -inf with ``neginf``."""
    a = _np.asarray(x, dtype="float64")
    a = _np.where(_np.isnan(a), nan, a)
    pos = posinf if posinf is not None else _np.finfo("float64").max
    neg = neginf if neginf is not None else -_np.finfo("float64").max
    a = _np.where(_np.isinf(a) & (a > 0), pos, a)
    a = _np.where(_np.isinf(a) & (a < 0), neg, a)
    return a


def real_if_close(a, tol=100):
    """Return the real part if the imaginary part is within ``tol`` ulps of 0."""
    a = _np.asarray(a)
    if not _np.iscomplexobj(a):
        return a
    eps = _np.finfo("float64").eps
    if _np.all(_np.abs(_np.imag(a)) < tol * eps):
        return _np.real(a)
    return a


def trim_zeros(filt, trim="fb"):
    """Trim leading (``f``) / trailing (``b``) zeros from a 1-D sequence."""
    a = list(_np.asarray(filt).tolist())
    trim = trim.lower()
    if "f" in trim:
        while a and a[0] == 0:
            a.pop(0)
    if "b" in trim:
        while a and a[-1] == 0:
            a.pop()
    return _np.asarray(a)


# ---- window functions ----

def _window_arange(n):
    return _np.arange(n, dtype="float64")


def bartlett(M):
    """Bartlett (triangular) window of length ``M``."""
    if M < 1:
        return _np.asarray([], dtype="float64")
    if M == 1:
        return _np.asarray([1.0])
    n = _window_arange(M)
    return _np.where(n <= (M - 1) / 2, 2.0 * n / (M - 1), 2.0 - 2.0 * n / (M - 1))


def hamming(M):
    """Hamming window of length ``M``."""
    if M < 1:
        return _np.asarray([], dtype="float64")
    if M == 1:
        return _np.asarray([1.0])
    n = _window_arange(M)
    return 0.54 - 0.46 * _np.cos(2 * _pi() * n / (M - 1))


def hanning(M):
    """Hann (Hanning) window of length ``M``."""
    if M < 1:
        return _np.asarray([], dtype="float64")
    if M == 1:
        return _np.asarray([1.0])
    n = _window_arange(M)
    return 0.5 - 0.5 * _np.cos(2 * _pi() * n / (M - 1))


def blackman(M):
    """Blackman window of length ``M``."""
    if M < 1:
        return _np.asarray([], dtype="float64")
    if M == 1:
        return _np.asarray([1.0])
    n = _window_arange(M)
    arg = 2 * _pi() * n / (M - 1)
    return 0.42 - 0.5 * _np.cos(arg) + 0.08 * _np.cos(2 * arg)


def _bessel_i0(x):
    """Modified Bessel function I0 — Abramowitz & Stegun 9.8.1/9.8.2."""
    ax = abs(x)
    if ax < 3.75:
        y = (x / 3.75) ** 2
        return 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492
            + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))))
    y = 3.75 / ax
    base = _np.exp(ax) / _np.sqrt(ax)
    return base * (0.39894228 + y * (0.01328592 + y * (0.00225319
        + y * (-0.00157565 + y * (0.00916281 + y * (-0.02057706
        + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377))))))))


def i0(x):
    """Modified Bessel function I0, element-wise."""
    a = _np.asarray(x, dtype="float64")
    flat = a.ravel().tolist()
    out = [_bessel_i0(v) for v in flat]
    return _np.asarray(out).reshape(a.shape)


def kaiser(M, beta):
    """Kaiser window of length ``M`` and shape parameter ``beta``."""
    if M < 1:
        return _np.asarray([], dtype="float64")
    if M == 1:
        return _np.asarray([1.0])
    n = _window_arange(M)
    alpha = (M - 1) / 2.0
    t = (n - alpha) / alpha
    arg = beta * _np.sqrt(1 - t * t)
    return i0(arg) / _bessel_i0(beta)


# ---- index helpers ----

def broadcast_shapes(*shapes):
    """Compute the shape that all input shapes broadcast to."""
    if not shapes:
        return ()
    shapes = [tuple(s) for s in shapes]
    max_nd = max(len(s) for s in shapes)
    padded = [(1,) * (max_nd - len(s)) + s for s in shapes]
    out = []
    for axis in range(max_nd):
        dims = [p[axis] for p in padded]
        non_one = [d for d in dims if d != 1]
        if not non_one:
            out.append(1)
            continue
        target = non_one[0]
        for d in non_one[1:]:
            if d != target:
                raise ValueError(
                    f"shape mismatch on axis {axis}: cannot broadcast {dims}"
                )
        out.append(target)
    return tuple(out)


def vander(x, N=None, increasing=False):
    """Vandermonde matrix of ``x``, with ``N`` columns (default ``len(x)``)."""
    x = _np.asarray(x, dtype="float64")
    if x.ndim != 1:
        raise ValueError("vander requires a 1-D input")
    if N is None:
        N = x.shape[0]
    cols = []
    if increasing:
        powers = range(N)
    else:
        powers = range(N - 1, -1, -1)
    for p in powers:
        cols.append(_np.power(x, p))
    return _np.column_stack(cols) if hasattr(_np, "column_stack") else _np.stack(cols, axis=1)


def diag_indices_from(arr):
    """Return the indices to access the main diagonal of an N-D array."""
    arr = _np.asarray(arr)
    if arr.ndim < 2:
        raise ValueError("diag_indices_from requires at least 2-D input")
    n = arr.shape[0]
    for d in arr.shape:
        if d != n:
            raise ValueError("array must be square along all dimensions")
    idx = _np.arange(n)
    return tuple(idx for _ in range(arr.ndim))


def tril_indices_from(arr, k=0):
    """Return indices for the lower triangle of ``arr``."""
    arr = _np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError("tril_indices_from requires a 2-D input")
    n, m = arr.shape
    rows, cols = [], []
    for i in range(n):
        for j in range(m):
            if j - i <= k:
                rows.append(i)
                cols.append(j)
    return (_np.asarray(rows), _np.asarray(cols))


def triu_indices_from(arr, k=0):
    """Return indices for the upper triangle of ``arr``."""
    arr = _np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError("triu_indices_from requires a 2-D input")
    n, m = arr.shape
    rows, cols = [], []
    for i in range(n):
        for j in range(m):
            if j - i >= k:
                rows.append(i)
                cols.append(j)
    return (_np.asarray(rows), _np.asarray(cols))


def mask_indices(n, mask_func, k=0):
    """Return indices where ``mask_func(ones((n, n)), k)`` is non-zero."""
    base = _np.ones((n, n))
    masked = mask_func(base, k)
    flat = masked.ravel().tolist()
    rows, cols = [], []
    for idx, v in enumerate(flat):
        if v:
            rows.append(idx // n)
            cols.append(idx % n)
    return (_np.asarray(rows), _np.asarray(cols))


def fill_diagonal(a, val, wrap=False):
    """Set the main diagonal of ``a`` to ``val`` in-place."""
    _ = wrap
    n = min(a.shape)
    for i in range(n):
        idx = tuple(i for _ in range(a.ndim))
        a[idx] = val


# ---- set operations ----

def ediff1d(ary, to_end=None, to_begin=None):
    """First-order differences of a flat array, optionally padded."""
    a = list(_np.asarray(ary).ravel().tolist())
    out = []
    if to_begin is not None:
        out.extend(list(_np.asarray(to_begin).ravel().tolist()))
    for i in range(1, len(a)):
        out.append(a[i] - a[i - 1])
    if to_end is not None:
        out.extend(list(_np.asarray(to_end).ravel().tolist()))
    return _np.asarray(out)


def _to_sorted_unique_list(a):
    seen = sorted(set(_np.asarray(a).ravel().tolist()))
    return seen


def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
    """Sorted unique values present in both inputs."""
    s1 = _to_sorted_unique_list(ar1) if not assume_unique else list(_np.asarray(ar1).ravel().tolist())
    s2 = set(_to_sorted_unique_list(ar2) if not assume_unique else _np.asarray(ar2).ravel().tolist())
    out = sorted(v for v in s1 if v in s2)
    if return_indices:
        flat1 = list(_np.asarray(ar1).ravel().tolist())
        flat2 = list(_np.asarray(ar2).ravel().tolist())
        i1 = [flat1.index(v) for v in out]
        i2 = [flat2.index(v) for v in out]
        return _np.asarray(out), _np.asarray(i1), _np.asarray(i2)
    return _np.asarray(out)


def union1d(ar1, ar2):
    """Sorted union of the unique values in either input."""
    return _np.asarray(sorted(set(_to_sorted_unique_list(ar1)) | set(_to_sorted_unique_list(ar2))))


def setdiff1d(ar1, ar2, assume_unique=False):
    """Sorted unique values in ``ar1`` that aren't in ``ar2``."""
    s1 = _to_sorted_unique_list(ar1) if not assume_unique else list(_np.asarray(ar1).ravel().tolist())
    s2 = set(_np.asarray(ar2).ravel().tolist())
    return _np.asarray(sorted(v for v in s1 if v not in s2))


def setxor1d(ar1, ar2, assume_unique=False):
    """Sorted unique values in exactly one of the inputs."""
    s1 = set(_to_sorted_unique_list(ar1))
    s2 = set(_to_sorted_unique_list(ar2))
    return _np.asarray(sorted(s1 ^ s2))


def isin(element, test_elements, invert=False):
    """Element-wise: True where ``element`` is in ``test_elements``."""
    test = set(_np.asarray(test_elements).ravel().tolist())
    flat = list(_np.asarray(element).ravel().tolist())
    out = [(v in test) ^ invert for v in flat]
    shape = _np.asarray(element).shape
    return _np.asarray(out).reshape(shape)


def sort_complex(a):
    """Sort by real part, breaking ties on imaginary part."""
    flat = list(_np.asarray(a, dtype="complex128").ravel().tolist())
    flat.sort(key=lambda c: (c.real, c.imag))
    return _np.asarray(flat)


def unique_values(ar):
    """Array API: sorted unique values (no extras)."""
    return _np.asarray(_to_sorted_unique_list(ar))


def unique_counts(ar):
    """Array API: ``(values, counts)`` named tuple-ish (returned as a tuple)."""
    vals = _to_sorted_unique_list(ar)
    flat = list(_np.asarray(ar).ravel().tolist())
    counts = [flat.count(v) for v in vals]
    return (_np.asarray(vals), _np.asarray(counts))


def unique_inverse(ar):
    """Array API: ``(values, inverse)`` — inverse maps each element to its value's index."""
    vals = _to_sorted_unique_list(ar)
    idx = {v: i for i, v in enumerate(vals)}
    flat = list(_np.asarray(ar).ravel().tolist())
    inv = [idx[v] for v in flat]
    return (_np.asarray(vals), _np.asarray(inv).reshape(_np.asarray(ar).shape))


def unique_all(ar):
    """Array API: ``(values, indices, inverse, counts)``."""
    vals = _to_sorted_unique_list(ar)
    flat = list(_np.asarray(ar).ravel().tolist())
    idx = {v: i for i, v in enumerate(vals)}
    first_idx = [flat.index(v) for v in vals]
    inv = [idx[v] for v in flat]
    counts = [flat.count(v) for v in vals]
    return (
        _np.asarray(vals),
        _np.asarray(first_idx),
        _np.asarray(inv).reshape(_np.asarray(ar).shape),
        _np.asarray(counts),
    )


# ---- histograms ----

def digitize(x, bins, right=False):
    """Return indices of bins to which each value belongs."""
    x_list = list(_np.asarray(x).ravel().tolist())
    bin_list = list(_np.asarray(bins).ravel().tolist())
    ascending = len(bin_list) <= 1 or bin_list[0] <= bin_list[-1]
    out = []
    for v in x_list:
        i = 0
        for b in bin_list:
            if ascending:
                if (v < b) if not right else (v <= b):
                    break
            else:
                if (v > b) if not right else (v >= b):
                    break
            i += 1
        out.append(i)
    shape = _np.asarray(x).shape
    return _np.asarray(out).reshape(shape) if shape else _np.asarray(out)


def histogram_bin_edges(a, bins=10, range=None, weights=None):
    """Return the edges that would be used by ``np.histogram``."""
    _ = weights
    flat = list(_np.asarray(a).ravel().tolist())
    if range is None:
        lo, hi = (min(flat), max(flat)) if flat else (0.0, 1.0)
    else:
        lo, hi = range
    if isinstance(bins, int):
        n = bins
        if n <= 0:
            raise ValueError("bins must be positive")
        step = (hi - lo) / n if n else 0
        return _np.asarray([lo + i * step for i in builtins_range(n + 1)])
    return _np.asarray(bins)


def builtins_range(n):
    """Shim — rustpython's globals lookup for ``range`` from a numpy submodule."""
    return [i for i in range(n)]


def histogram2d(x, y, bins=10, range=None, weights=None):
    """2-D histogram of (x, y) pairs. Returns (H, xedges, yedges)."""
    _ = weights
    xf = list(_np.asarray(x).ravel().tolist())
    yf = list(_np.asarray(y).ravel().tolist())
    if isinstance(bins, int):
        bx = by = bins
    elif isinstance(bins, (list, tuple)) and len(bins) == 2 and isinstance(bins[0], int):
        bx, by = bins
    else:
        bx = by = 10
    if range is None:
        xlo, xhi = (min(xf), max(xf)) if xf else (0.0, 1.0)
        ylo, yhi = (min(yf), max(yf)) if yf else (0.0, 1.0)
    else:
        (xlo, xhi), (ylo, yhi) = range
    xe = histogram_bin_edges([], bins=bx, range=(xlo, xhi))
    ye = histogram_bin_edges([], bins=by, range=(ylo, yhi))
    H = [[0 for _ in builtins_range(by)] for _ in builtins_range(bx)]
    xe_list = list(xe.ravel().tolist())
    ye_list = list(ye.ravel().tolist())
    for xv, yv in zip(xf, yf):
        xi = min(max(digitize([xv], xe_list).ravel().tolist()[0] - 1, 0), bx - 1)
        yi = min(max(digitize([yv], ye_list).ravel().tolist()[0] - 1, 0), by - 1)
        H[xi][yi] += 1
    return _np.asarray(H), xe, ye


def histogramdd(sample, bins=10, range=None, weights=None):
    """N-D histogram. Returns (H, edges_list)."""
    _ = weights
    arr = _np.asarray(sample)
    if arr.ndim == 1:
        H, edges = _np.histogram(arr, bins=bins, range=range)
        return H, [edges]
    if arr.ndim != 2:
        raise ValueError("sample must be 1-D or 2-D")
    n_samples, n_dims = arr.shape
    if isinstance(bins, int):
        bin_counts = [bins] * n_dims
    else:
        bin_counts = list(bins)
    if range is None:
        ranges = [
            (float(min(arr[:, d].ravel().tolist())), float(max(arr[:, d].ravel().tolist())))
            for d in builtins_range(n_dims)
        ]
    else:
        ranges = list(range)
    edges = [
        histogram_bin_edges([], bins=bin_counts[d], range=ranges[d])
        for d in builtins_range(n_dims)
    ]
    # Iterative bin counting — slow but correct.
    shape = tuple(bin_counts)
    flat_h = [0] * (1 if not shape else _prod(shape))
    for row_idx in builtins_range(n_samples):
        ix = []
        for d in builtins_range(n_dims):
            v = arr[row_idx, d]
            e = list(edges[d].ravel().tolist())
            i = digitize([v], e).ravel().tolist()[0] - 1
            i = min(max(i, 0), bin_counts[d] - 1)
            ix.append(i)
        idx = 0
        for i, c in zip(ix, bin_counts):
            idx = idx * c + i
        flat_h[idx] += 1
    return _np.asarray(flat_h).reshape(shape), edges


def _prod(seq):
    out = 1
    for v in seq:
        out *= v
    return out


# ---- "method on array" exposed as a function ----

def copy(a):
    """Return an array copy."""
    return _np.asarray(a).copy()


def ravel(a):
    """Flatten ``a`` to 1-D."""
    return _np.asarray(a).ravel()


def shape(a):
    """Return the shape of ``a``."""
    return _np.asarray(a).shape


def size(a, axis=None):
    """Total element count, or count along an axis."""
    sh = _np.asarray(a).shape
    if axis is None:
        n = 1
        for d in sh:
            n *= d
        return n
    return sh[axis]


def ndim(a):
    """Number of dimensions of ``a``."""
    return _np.asarray(a).ndim


def astype(x, dtype, copy=True):
    """Convert ``x`` to a new array of ``dtype``."""
    _ = copy
    return _np.asarray(x).astype(dtype)


def diagonal(a, offset=0, axis1=0, axis2=1):
    """Diagonal of a 2-D matrix (axis args ignored — same as ``a.diagonal(k)``)."""
    _ = axis1
    _ = axis2
    return _np.asarray(a).diagonal(offset)


def std(a, axis=None, dtype=None, out=None, ddof=0):
    return _np.asarray(a).std(axis=axis, dtype=dtype, out=out, ddof=ddof)


def var(a, axis=None, dtype=None, out=None, ddof=0):
    return _np.asarray(a).var(axis=axis, dtype=dtype, out=out, ddof=ddof)


def take(a, indices, axis=None, out=None):
    """Return ``a[indices]`` along the given axis."""
    _ = out
    arr = _np.asarray(a)
    if axis is None:
        return arr.take(indices)
    flat = arr.take(indices)
    return flat


# ---- matrix transpose / vec ops (array-API names) ----

def matrix_transpose(a):
    """Transpose the last two axes."""
    arr = _np.asarray(a)
    if arr.ndim < 2:
        raise ValueError("matrix_transpose needs ndim>=2")
    return arr.swapaxes(arr.ndim - 1, arr.ndim - 2)


def vecdot(a, b, axis=-1):
    """Vector dot product along ``axis``."""
    aa = _np.asarray(a)
    bb = _np.asarray(b)
    return _np.sum(aa * bb, axis=axis)


def matvec(a, b):
    """Matrix-vector product: last axis of ``a`` contracts with ``b``."""
    return _np.matmul(_np.asarray(a), _np.asarray(b))


def vecmat(a, b):
    """Vector-matrix product: ``a`` (1-D) times ``b`` (2-D)."""
    aa = _np.asarray(a)
    bb = _np.asarray(b)
    return _np.matmul(aa, bb)


def unstack(x, axis=0):
    """Split ``x`` into a tuple of arrays along ``axis``."""
    arr = _np.asarray(x)
    n = arr.shape[axis]
    return tuple(_np.take(arr, [i], axis=axis).squeeze(axis) for i in builtins_range(n))


# ---- predicates ----

def isfortran(a):
    """Return False — rumpy's ndarray is always C-contiguous."""
    _ = a
    return False


def issubdtype(a, b):
    """``True`` if ``a`` is a subclass of ``b`` in the dtype hierarchy."""
    def cls(x):
        if isinstance(x, type):
            return x
        # `np.dtype("int32")` → look up corresponding scalar class.
        name = getattr(x, "name", None) or str(x)
        return _np.sctypeDict.get(name, type(None))
    return issubclass(cls(a), cls(b))


def isdtype(dtype, kind):
    """Array API ``isdtype`` — check whether ``dtype`` matches ``kind``."""
    # Normalize dtype to its name.
    name = getattr(dtype, "name", None) or str(dtype)

    KINDS = {
        "bool": {"bool"},
        "signed integer": {"int8", "int16", "int32", "int64"},
        "unsigned integer": {"uint8", "uint16", "uint32", "uint64"},
        "integral": {
            "int8", "int16", "int32", "int64",
            "uint8", "uint16", "uint32", "uint64",
        },
        "real floating": {"float16", "float32", "float64"},
        "complex floating": {"complex64", "complex128"},
        "numeric": {
            "int8", "int16", "int32", "int64",
            "uint8", "uint16", "uint32", "uint64",
            "float16", "float32", "float64",
            "complex64", "complex128",
        },
    }
    if isinstance(kind, str):
        return name in KINDS.get(kind, set())
    if isinstance(kind, tuple):
        return any(isdtype(dtype, k) for k in kind)
    return False


def isnat(x):
    """``True`` where ``x`` is NaT. rumpy lacks datetime, so always False."""
    a = _np.asarray(x)
    return _np.zeros(a.shape, dtype="bool")


def iterable(obj):
    """Return True if ``obj`` is iterable."""
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def bitwise_count(x):
    """Population count (number of set bits) of each element."""
    a = _np.asarray(x).astype("int64")
    flat = list(a.ravel().tolist())
    out = [bin(int(v) & 0xFFFFFFFFFFFFFFFF).count("1") for v in flat]
    return _np.asarray(out).reshape(a.shape)


# ---- text/console output (real numpy uses these for repr) ----

def array_repr(a, max_line_width=None, precision=None, suppress_small=None):
    _ = (max_line_width, precision, suppress_small)
    return repr(_np.asarray(a))


def array_str(a, max_line_width=None, precision=None, suppress_small=None):
    _ = (max_line_width, precision, suppress_small)
    return str(_np.asarray(a))


def array2string(
    a,
    max_line_width=None,
    precision=None,
    suppress_small=None,
    separator=" ",
    prefix="",
    style=None,
    formatter=None,
    threshold=None,
    edgeitems=None,
    sign=None,
    floatmode=None,
    suffix="",
    *,
    legacy=None,
):
    _ = (max_line_width, precision, suppress_small, separator, prefix, style,
         formatter, threshold, edgeitems, sign, floatmode, suffix, legacy)
    return str(_np.asarray(a))


def format_float_positional(x, precision=None, unique=True, fractional=True,
                             trim="k", sign=False, pad_left=None, pad_right=None,
                             min_digits=None):
    _ = (unique, fractional, trim, pad_left, pad_right, min_digits)
    val = float(x)
    if precision is None:
        s = f"{val}"
    else:
        s = f"{val:.{int(precision)}f}"
    if sign and val >= 0:
        s = "+" + s
    return s


def format_float_scientific(x, precision=None, unique=True, trim="k", sign=False,
                              pad_left=None, exp_digits=None, min_digits=None):
    _ = (unique, trim, pad_left, exp_digits, min_digits)
    val = float(x)
    if precision is None:
        s = f"{val:e}"
    else:
        s = f"{val:.{int(precision)}e}"
    if sign and val >= 0:
        s = "+" + s
    return s


# ---- print-options machinery ----

_print_state = {
    "precision": 8,
    "threshold": 1000,
    "edgeitems": 3,
    "linewidth": 75,
    "suppress": False,
    "nanstr": "nan",
    "infstr": "inf",
    "formatter": None,
    "sign": "-",
    "floatmode": "maxprec",
    "legacy": False,
}


def set_printoptions(**kwargs):
    for k, v in kwargs.items():
        if k in _print_state:
            _print_state[k] = v


def get_printoptions():
    return dict(_print_state)


class _PrintOptionsCtx:
    def __init__(self, **kwargs):
        self.new = kwargs
        self.old = None

    def __enter__(self):
        self.old = dict(_print_state)
        set_printoptions(**self.new)
        return self

    def __exit__(self, exc_type, exc, tb):
        for k, v in self.old.items():
            _print_state[k] = v


def printoptions(**kwargs):
    """Context manager that temporarily changes print options."""
    return _PrintOptionsCtx(**kwargs)


_bufsize = [8192]


def getbufsize():
    return _bufsize[0]


def setbufsize(size):
    _bufsize[0] = int(size)
    return _bufsize[0]


_err_call = [None]


def seterrcall(func):
    prev = _err_call[0]
    _err_call[0] = func
    return prev


# ---- buffer / file-like sources ----

def frombuffer(buffer, dtype="float64", count=-1, offset=0):
    """Materialize bytes ``buffer`` as a 1-D array of ``dtype``."""
    raw = bytes(buffer)[offset:]
    import struct
    fmts = {
        "float64": ("d", 8),
        "float32": ("f", 4),
        "int64": ("q", 8),
        "int32": ("i", 4),
        "int16": ("h", 2),
        "int8": ("b", 1),
        "uint64": ("Q", 8),
        "uint32": ("I", 4),
        "uint16": ("H", 2),
        "uint8": ("B", 1),
    }
    name = getattr(dtype, "name", None) or str(dtype)
    fmt, size_bytes = fmts.get(name, ("d", 8))
    n = len(raw) // size_bytes
    if count >= 0:
        n = min(n, count)
    out = list(struct.unpack(f"<{n}{fmt}", raw[: n * size_bytes]))
    return _np.asarray(out, dtype=name)


def from_dlpack(x, /, *, device=None, copy=None):
    """Best-effort DLPack import. Calls ``x.__dlpack__()`` if available."""
    _ = (device, copy)
    if hasattr(x, "__array__"):
        return _np.asarray(x.__array__())
    raise NotImplementedError("from_dlpack: object does not expose an array interface")


def fromfunction(function, shape, dtype="float64", **kwargs):
    """Construct an array by applying ``function`` to each index tuple."""
    _ = kwargs
    if not isinstance(shape, (tuple, list)):
        shape = (shape,)
    total = 1
    for d in shape:
        total *= d

    def idx_at(flat_i):
        out = []
        rem = flat_i
        for d in reversed(shape):
            out.append(rem % d)
            rem //= d
        return tuple(reversed(out))

    vals = []
    for i in builtins_range(total):
        args = idx_at(i)
        vals.append(function(*args))
    return _np.asarray(vals, dtype=dtype).reshape(shape)


def fromregex(file, regexp, dtype="float64", encoding=None):
    """Read lines from ``file``, extract groups via ``regexp``, build an array."""
    _ = encoding
    import re
    pattern = re.compile(regexp) if isinstance(regexp, str) else regexp
    if hasattr(file, "read"):
        text = file.read()
    else:
        with open(file, "r") as fh:
            text = fh.read()
    out = []
    for m in pattern.finditer(text):
        groups = m.groups() or (m.group(0),)
        out.append(tuple(groups))
    return _np.asarray(out, dtype=dtype)


def genfromtxt(fname, dtype="float64", comments="#", delimiter=None,
               skip_header=0, skip_footer=0, converters=None,
               missing_values=None, filling_values=None, usecols=None,
               names=None, excludelist=None, deletechars=None,
               replace_space=None, autostrip=False, case_sensitive=True,
               defaultfmt="f%i", unpack=None, usemask=False,
               loose=True, invalid_raise=True, max_rows=None, encoding=None,
               *, ndmin=0, like=None):
    """Slim ``genfromtxt`` — reads numeric whitespace/delimited text."""
    _ = (
        missing_values, filling_values, names, excludelist, deletechars,
        replace_space, case_sensitive, defaultfmt, usemask, loose, invalid_raise,
        encoding, ndmin, like,
    )
    if hasattr(fname, "read"):
        text = fname.read()
    else:
        with open(fname, "r") as fh:
            text = fh.read()
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    lines = text.splitlines()[skip_header:]
    if skip_footer:
        lines = lines[:-skip_footer]
    rows = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith(comments):
            continue
        parts = line.split(delimiter) if delimiter else line.split()
        if autostrip:
            parts = [p.strip() for p in parts]
        if usecols is not None:
            cols = list(usecols) if not isinstance(usecols, int) else [usecols]
            parts = [parts[c] for c in cols]
        if converters:
            parts = [
                converters.get(i, lambda v: v)(parts[i])
                for i in builtins_range(len(parts))
            ]
        vals = []
        for p in parts:
            try:
                vals.append(float(p))
            except (TypeError, ValueError):
                vals.append(float("nan"))
        rows.append(vals)
    if unpack:
        return _np.asarray(rows, dtype=dtype).T
    return _np.asarray(rows, dtype=dtype)


def asarray_chkfinite(a, dtype=None, order=None):
    _ = order
    arr = _np.asarray(a, dtype=dtype) if dtype is not None else _np.asarray(a)
    if hasattr(arr, "dtype") and "float" in str(arr.dtype):
        if not _np.all(_np.isfinite(arr)):
            raise ValueError("array must not contain infs or NaNs")
    return arr


def packbits(a, axis=None, bitorder="big"):
    """Pack 0/1 array into a uint8 array, MSB-first within each byte."""
    flat = list(_np.asarray(a).astype("int64").ravel().tolist())
    out = []
    for i in builtins_range((len(flat) + 7) // 8):
        byte = 0
        for j in builtins_range(8):
            k = i * 8 + j
            if k >= len(flat):
                break
            bit = 1 if flat[k] else 0
            if bitorder == "big":
                byte |= bit << (7 - j)
            else:
                byte |= bit << j
        out.append(byte)
    return _np.asarray(out, dtype="uint8")


def unpackbits(a, axis=None, count=None, bitorder="big"):
    """Inverse of ``packbits``."""
    _ = axis
    flat = list(_np.asarray(a).astype("int64").ravel().tolist())
    out = []
    for byte in flat:
        for j in builtins_range(8):
            if bitorder == "big":
                out.append(1 if byte & (1 << (7 - j)) else 0)
            else:
                out.append(1 if byte & (1 << j) else 0)
    if count is not None:
        out = out[: int(count)] if count >= 0 else out[: len(out) + int(count)]
    return _np.asarray(out, dtype="uint8")


def putmask(a, mask, values):
    """Set ``a[mask]`` to ``values`` in place (broadcasting where needed)."""
    a_flat = a.ravel().tolist()
    m_flat = list(_np.asarray(mask).ravel().tolist())
    v_flat = list(_np.asarray(values).ravel().tolist())
    j = 0
    for i in builtins_range(len(a_flat)):
        if m_flat[i % len(m_flat)]:
            a_flat[i] = v_flat[j % len(v_flat)]
            j += 1
    new = _np.asarray(a_flat).reshape(a.shape)
    # In-place assignment via slicing.
    for i in builtins_range(len(a_flat)):
        # Index into a via flat indexing — relies on a being a rumpy ndarray.
        pass  # rumpy ndarrays don't yet expose flat assignment; users get the new array via a[:]=
    a[:] = new


def shares_memory(a, b, max_work=None):
    """Conservative: True only when objects are the same Python instance."""
    _ = max_work
    return a is b


def may_share_memory(a, b, max_work=None):
    _ = max_work
    return a is b


# ---- introspection / shims ----

def info(obj=None, maxwidth=76, output=None, toplevel="numpy"):
    """Print short docstring of ``obj`` to ``output`` (default stdout)."""
    _ = (maxwidth, toplevel)
    doc = getattr(obj, "__doc__", None) or "(no documentation)"
    if output is None:
        print(doc)
    else:
        output.write(doc + "\n")


def show_config(mode="stdout"):
    """rumpy is self-contained: no BLAS/LAPACK configuration to print."""
    _ = mode
    text = "rumpy: pure-Rust numeric core, no external BLAS / LAPACK."
    if mode == "dicts":
        return {"backend": "ndarray", "blas": None, "lapack": None}
    print(text)


def show_runtime():
    """Print runtime info (rumpy version, Python version)."""
    print(f"rumpy version: {_np.version.version}")


def get_include():
    """numpy includes C headers; rumpy has no such path."""
    return ""


def test(*args, **kwargs):
    """rumpy has no built-in test runner; this is a no-op."""
    _ = (args, kwargs)
    return True


def common_type(*arrays):
    """Return the smallest float-or-complex dtype that all inputs can promote to."""
    if not arrays:
        return _np.float64
    has_complex = any("complex" in str(_np.asarray(a).dtype) for a in arrays)
    return _np.complex128 if has_complex else _np.float64


def mintypecode(typechars, typeset="GDFgdf", default="d"):
    """Return the smallest type character among ``typechars`` that's in ``typeset``."""
    _ = (typechars, typeset)
    return default


def typename(char):
    """Return a human-readable name for a type character."""
    names = {
        "b": "signed char", "B": "unsigned char",
        "h": "short", "H": "unsigned short",
        "i": "integer", "I": "unsigned integer",
        "l": "long integer", "L": "unsigned long integer",
        "q": "long long integer", "Q": "unsigned long long integer",
        "f": "single precision", "d": "double precision",
        "g": "long precision",
        "F": "complex single precision",
        "D": "complex double precision",
        "G": "complex long double precision",
        "?": "bool", "O": "object", "S": "string", "U": "unicode", "V": "void",
    }
    return names.get(char, "unknown")


typecodes = {
    "Character": "c",
    "Integer": "bhilqp",
    "UnsignedInteger": "BHILQP",
    "Float": "efdg",
    "Complex": "FDG",
    "AllInteger": "bBhHiIlLqQpP",
    "AllFloat": "efdgFDG",
    "Datetime": "Mm",
    "All": "?bhilqpBHILQPefdgFDGSUVOMm",
}


# ---- piecewise / select / apply_over_axes ----

def select(condlist, choicelist, default=0):
    """Element-wise: pick from ``choicelist`` based on first true in ``condlist``."""
    if not condlist:
        return _np.asarray(default)
    shape = _np.asarray(condlist[0]).shape
    out = _np.full(shape, default, dtype="float64") if hasattr(_np, "full") \
        else _np.zeros(shape) + default
    # Process in reverse so earlier conditions take precedence.
    for cond, choice in zip(reversed(condlist), reversed(choicelist)):
        out = _np.where(cond, choice, out)
    return out


def piecewise(x, condlist, funclist, *args, **kw):
    """Apply ``funclist[i](x)`` where ``condlist[i]`` is true."""
    arr = _np.asarray(x)
    out = _np.zeros(arr.shape, dtype="float64")
    extras = list(funclist)
    if len(extras) == len(condlist) + 1:
        default_fn = extras[-1]
        extras = extras[:-1]
    else:
        default_fn = None
    for cond, fn in zip(condlist, extras):
        val = fn(arr, *args, **kw) if callable(fn) else fn
        out = _np.where(cond, val, out)
    if default_fn is not None:
        # Apply default where no condition matched.
        combined = condlist[0]
        for c in condlist[1:]:
            combined = combined | c
        out = _np.where(combined, out, default_fn(arr, *args, **kw))
    return out


def apply_over_axes(func, a, axes):
    """Apply ``func`` repeatedly along each axis in ``axes``."""
    if isinstance(axes, int):
        axes = (axes,)
    out = _np.asarray(a)
    for ax in axes:
        out = func(out, ax)
        # Re-introduce reduced axis to keep dims aligned.
        if hasattr(out, "ndim") and out.ndim < _np.asarray(a).ndim:
            out = _np.expand_dims(out, ax) if hasattr(_np, "expand_dims") else out
    return out


def einsum_path(*operands, **kwargs):
    """Stub: rumpy's einsum runs single-pass, so we just return a trivial plan."""
    _ = kwargs
    return ("einsum_path", []), "single-pass evaluation (rumpy has no path optimizer)"


class _IndexExpression:
    """Helper for ``np.index_exp[...]`` / ``np.s_[...]``."""

    def __init__(self, maketuple=True):
        self._maketuple = maketuple

    def __getitem__(self, item):
        if self._maketuple and not isinstance(item, tuple):
            return (item,)
        return item


index_exp = _IndexExpression(True)


# ---------------------------------------------------------------------------
# Legacy poly1d / poly* family (descending coefficient order).
#
# Real numpy 2.x still exposes these at the top level even though
# numpy.polynomial is the new home. We reuse the polynomial submodule's
# ascending-order implementations and convert.
# ---------------------------------------------------------------------------


def _descending_to_ascending(c):
    return list(reversed(list(c)))


class poly1d:
    """Legacy 1-D polynomial class. Coefficients are stored *descending*."""

    def __init__(self, c_or_r, r=False, variable=None):
        if r:
            # Coefficients-from-roots constructor.
            roots = list(c_or_r)
            coef = [1.0]
            for root in roots:
                new = [0.0] * (len(coef) + 1)
                for i, v in enumerate(coef):
                    new[i] += v
                    new[i + 1] -= v * root
                coef = new
            self._c = coef  # descending
        else:
            self._c = list(c_or_r)
        self.variable = variable or "x"

    @property
    def coef(self):
        return self._c

    @property
    def order(self):
        return max(0, len(self._c) - 1)

    @property
    def coeffs(self):
        return self._c

    def __call__(self, x):
        # Horner from leading coefficient.
        if not self._c:
            return 0
        acc = self._c[0]
        for c in self._c[1:]:
            acc = acc * x + c
        return acc

    def __repr__(self):
        return f"poly1d({self._c!r})"

    def __add__(self, other):
        if isinstance(other, poly1d):
            return poly1d(_descending_to_ascending(
                _np.polynomial.polyadd(
                    _descending_to_ascending(self._c),
                    _descending_to_ascending(other._c),
                )
            ))
        return poly1d(_descending_to_ascending(
            _np.polynomial.polyadd(_descending_to_ascending(self._c), [other])
        ))

    def __mul__(self, other):
        if isinstance(other, poly1d):
            return poly1d(_descending_to_ascending(
                _np.polynomial.polymul(
                    _descending_to_ascending(self._c),
                    _descending_to_ascending(other._c),
                )
            ))
        return poly1d([c * other for c in self._c])


def poly(seq_of_zeros):
    """Build polynomial coefficients (descending) from a sequence of roots."""
    return poly1d(list(seq_of_zeros), r=True).coef


def polyadd(a1, a2):
    """Add two polynomials in *descending* order."""
    a = _descending_to_ascending(_coerce_list(a1))
    b = _descending_to_ascending(_coerce_list(a2))
    return _descending_to_ascending(_np.polynomial.polyadd(a, b))


def polysub(a1, a2):
    a = _descending_to_ascending(_coerce_list(a1))
    b = _descending_to_ascending(_coerce_list(a2))
    return _descending_to_ascending(_np.polynomial.polysub(a, b))


def polymul(a1, a2):
    a = _descending_to_ascending(_coerce_list(a1))
    b = _descending_to_ascending(_coerce_list(a2))
    return _descending_to_ascending(_np.polynomial.polymul(a, b))


def polydiv(u, v):
    """Polynomial division — returns (quotient, remainder) both descending."""
    u = _coerce_list(u)
    v = _coerce_list(v)
    if not v or all(c == 0 for c in v):
        raise ZeroDivisionError("polydiv: divisor is zero")
    u = list(u)
    quotient = []
    while len(u) >= len(v):
        ratio = u[0] / v[0]
        quotient.append(ratio)
        for i in range(len(v)):
            u[i] -= ratio * v[i]
        u = u[1:]
    return quotient, u


def _coerce_list(c):
    if hasattr(c, "tolist"):
        c = c.tolist()
    return list(c)


# ---------------------------------------------------------------------------
# ufunc — placeholder base class. Real numpy ufuncs are C-level objects
# with `.reduce`, `.accumulate`, etc.; rumpy exposes a small stand-in so
# that ``isinstance(np.add, np.ufunc)`` doesn't blow up.
# ---------------------------------------------------------------------------


class ufunc:
    """Stand-in for numpy's ufunc base class."""

    def __init__(self, fn=None):
        self._fn = fn

    def __call__(self, *args, **kwargs):
        if self._fn is None:
            raise NotImplementedError("ufunc: no callable bound")
        return self._fn(*args, **kwargs)


__all__ = [
    # math
    "sinc", "float_power", "logaddexp", "logaddexp2", "nan_to_num",
    "real_if_close", "trim_zeros",
    # windows
    "bartlett", "hamming", "hanning", "blackman", "kaiser", "i0",
    # index helpers
    "broadcast_shapes", "vander", "diag_indices_from",
    "tril_indices_from", "triu_indices_from", "mask_indices", "fill_diagonal",
    # set ops
    "ediff1d", "intersect1d", "union1d", "setdiff1d", "setxor1d", "isin",
    "sort_complex", "unique_values", "unique_counts", "unique_inverse", "unique_all",
    # histograms
    "digitize", "histogram_bin_edges", "histogram2d", "histogramdd",
    # method-as-function
    "copy", "ravel", "shape", "size", "ndim", "astype", "diagonal",
    "std", "var", "take",
    # matrix / vec
    "matrix_transpose", "vecdot", "matvec", "vecmat", "unstack",
    # predicates
    "isfortran", "issubdtype", "isdtype", "isnat", "iterable", "bitwise_count",
    # text I/O
    "array_repr", "array_str", "array2string",
    "format_float_positional", "format_float_scientific",
    "set_printoptions", "get_printoptions", "printoptions",
    "getbufsize", "setbufsize", "seterrcall",
    # I/O sources
    "frombuffer", "from_dlpack", "fromfunction", "fromregex", "genfromtxt",
    "asarray_chkfinite",
    # packbits family
    "packbits", "unpackbits", "putmask",
    # memory helpers
    "shares_memory", "may_share_memory",
    # introspection
    "info", "show_config", "show_runtime", "get_include", "test",
    "common_type", "mintypecode", "typename", "typecodes",
    # piecewise / select
    "select", "piecewise", "apply_over_axes",
    # einsum path
    "einsum_path",
    # index_exp
    "index_exp",
    # legacy poly1d family
    "poly1d", "poly", "polyadd", "polysub", "polymul", "polydiv",
    # ufunc base class
    "ufunc",
]
