"""``numpy.matrixlib`` — the (legacy) ``matrix`` subclass.

Provides ``matrix``, ``asmatrix``/``mat``, and ``bmat``. Mirrors the
deprecated numpy.matrix API: a strictly 2-D wrapper where ``*`` is matrix
multiplication, ``**`` is matrix power, and ``.T`` / ``.H`` / ``.I`` /
``.A`` give transpose, conjugate-transpose, inverse, and the underlying
ndarray respectively.

We wrap an ndarray rather than subclassing it — rustpython doesn't permit
subclassing the built-in ``ndarray`` from Python. Intentionally bare-bones:
enough for legacy code paths, not a faithful drop-in replacement.
"""

# `_np` is the rumpy numpy module, injected from Rust.


def _parse_string(s):
    """Parse a matlab-style ``"1 2; 3 4"`` literal into a nested list."""
    rows = [r for r in (row.strip() for row in s.split(";")) if r]
    out = []
    for r in rows:
        parts = r.replace(",", " ").split()
        row = []
        for p in parts:
            if "." in p or "e" in p.lower() or "j" in p.lower():
                try:
                    row.append(float(p))
                except ValueError:
                    row.append(complex(p))
            else:
                row.append(int(p))
        out.append(row)
    return out


def _as_2d(arr):
    nd = getattr(arr, "ndim", None)
    if nd == 2:
        return arr
    if nd == 1:
        return arr.reshape(1, -1)
    if nd == 0:
        return arr.reshape(1, 1)
    raise ValueError("matrix must be 2-dimensional")


def _to_2d_array(data, dtype=None):
    if isinstance(data, matrix):
        return data._a if dtype is None else _np.asarray(data._a, dtype=dtype)
    if isinstance(data, str):
        return _as_2d(_np.array(_parse_string(data), dtype=dtype))
    return _as_2d(_np.array(data, dtype=dtype))


def _unwrap(other):
    return other._a if isinstance(other, matrix) else other


class matrix:
    """A 2-D matrix wrapper around an ndarray. ``*`` is matrix multiply."""

    __array_priority__ = 10.0

    def __init__(self, data, dtype=None, copy=True):
        _ = copy
        self._a = _to_2d_array(data, dtype)

    # ----- ndarray-style attributes -----
    @property
    def A(self):
        return self._a

    @property
    def A1(self):
        return self._a.ravel()

    @property
    def T(self):
        return matrix(self._a.T)

    @property
    def H(self):
        return matrix(_np.conjugate(self._a).T)

    @property
    def I(self):
        return matrix(_np.linalg.inv(self._a))

    @property
    def shape(self):
        return self._a.shape

    @property
    def ndim(self):
        return 2

    @property
    def size(self):
        return self._a.size

    @property
    def dtype(self):
        return self._a.dtype

    # ----- arithmetic -----
    def __mul__(self, other):
        o = _unwrap(other)
        if hasattr(o, "shape") or isinstance(o, (list, tuple)):
            return matrix(_np.matmul(self._a, _np.asarray(o)))
        return matrix(self._a * o)

    def __rmul__(self, other):
        o = _unwrap(other)
        if hasattr(o, "shape") or isinstance(o, (list, tuple)):
            return matrix(_np.matmul(_np.asarray(o), self._a))
        return matrix(o * self._a)

    def __matmul__(self, other):
        return matrix(_np.matmul(self._a, _unwrap(other)))

    def __rmatmul__(self, other):
        return matrix(_np.matmul(_unwrap(other), self._a))

    def __pow__(self, n):
        if not isinstance(n, int):
            raise TypeError("matrix power requires an integer exponent")
        if self._a.shape[0] != self._a.shape[1]:
            raise ValueError("matrix power requires a square matrix")
        if n < 0:
            base = _np.linalg.inv(self._a)
            n = -n
        else:
            base = self._a
        if n == 0:
            return matrix(_np.eye(self._a.shape[0], dtype=self._a.dtype))
        out = base
        for _ in range(n - 1):
            out = _np.matmul(out, base)
        return matrix(out)

    def __add__(self, other):
        return matrix(self._a + _unwrap(other))

    def __radd__(self, other):
        return matrix(_unwrap(other) + self._a)

    def __sub__(self, other):
        return matrix(self._a - _unwrap(other))

    def __rsub__(self, other):
        return matrix(_unwrap(other) - self._a)

    def __truediv__(self, other):
        return matrix(self._a / _unwrap(other))

    def __rtruediv__(self, other):
        return matrix(_unwrap(other) / self._a)

    def __neg__(self):
        return matrix(-self._a)

    def __pos__(self):
        return matrix(+self._a)

    def __eq__(self, other):
        return self._a == _unwrap(other)

    def __ne__(self, other):
        return self._a != _unwrap(other)

    # ----- shape / element-wise methods -----
    def transpose(self, *axes):
        if axes:
            return matrix(self._a.transpose(*axes))
        return matrix(self._a.T)

    def conjugate(self):
        return matrix(_np.conjugate(self._a))

    conj = conjugate

    def getA(self):
        return self._a

    def getA1(self):
        return self._a.ravel()

    def getT(self):
        return self.T

    def getH(self):
        return self.H

    def getI(self):
        return self.I

    def tolist(self):
        return self._a.tolist()

    def sum(self, axis=None, dtype=None, out=None):
        return self._a.sum(axis=axis, dtype=dtype, out=out)

    def mean(self, axis=None, dtype=None, out=None):
        return self._a.mean(axis=axis, dtype=dtype, out=out)

    def std(self, axis=None, dtype=None, out=None, ddof=0):
        return self._a.std(axis=axis, dtype=dtype, out=out, ddof=ddof)

    def var(self, axis=None, dtype=None, out=None, ddof=0):
        return self._a.var(axis=axis, dtype=dtype, out=out, ddof=ddof)

    def max(self, axis=None, out=None):
        return self._a.max(axis=axis, out=out)

    def min(self, axis=None, out=None):
        return self._a.min(axis=axis, out=out)

    def prod(self, axis=None, dtype=None, out=None):
        return self._a.prod(axis=axis, dtype=dtype, out=out)

    # ----- container protocol -----
    def __getitem__(self, key):
        v = self._a[key]
        nd = getattr(v, "ndim", None)
        if nd == 2:
            return matrix(v)
        if nd == 1:
            return matrix(v.reshape(1, -1))
        return v

    def __setitem__(self, key, value):
        self._a[key] = _unwrap(value)

    def __iter__(self):
        for i in range(self._a.shape[0]):
            yield matrix(self._a[i].reshape(1, -1))

    def __len__(self):
        return self._a.shape[0]

    def __repr__(self):
        return f"matrix({self._a.tolist()!r})"

    def __str__(self):
        return str(self._a)


def asmatrix(data, dtype=None):
    """Convert ``data`` to a ``matrix`` without copying when already one."""
    if isinstance(data, matrix) and dtype is None:
        return data
    return matrix(data, dtype=dtype)


mat = asmatrix


def bmat(obj, ldict=None, gdict=None):
    """Build a block matrix from a string, nested list, or array.

    String form uses ``';'`` to separate rows and ``','`` or whitespace to
    separate names within a row; each name is looked up in ``ldict`` then
    ``gdict``.
    """
    if isinstance(obj, str):
        env = {}
        if gdict:
            env.update(gdict)
        if ldict:
            env.update(ldict)
        rows = []
        for row_src in obj.split(";"):
            parts = [
                p for p in (q.strip() for q in row_src.replace(",", " ").split()) if p
            ]
            if not parts:
                continue
            blocks = []
            for name in parts:
                if name not in env:
                    raise NameError(f"bmat: name {name!r} not found")
                blocks.append(asmatrix(env[name])._a)
            rows.append(_np.concatenate(blocks, axis=1))
        return matrix(_np.concatenate(rows, axis=0))
    if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], (list, tuple)):
        rows = []
        for row in obj:
            blocks = [asmatrix(b)._a for b in row]
            rows.append(_np.concatenate(blocks, axis=1))
        return matrix(_np.concatenate(rows, axis=0))
    return asmatrix(obj)


__all__ = ["matrix", "asmatrix", "mat", "bmat"]
