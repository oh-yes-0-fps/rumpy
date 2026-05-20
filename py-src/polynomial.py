"""``numpy.polynomial`` — polynomial classes.

This is a small slice of numpy's ``polynomial`` package: a ``Polynomial``
class with arithmetic, evaluation, fitting, root-finding, derivative and
integral, plus the same coefficient-array level functions exposed at the
submodule's namespace (``polyval``, ``polyfit``, ``polyder``,
``polyint``, ``polymul``, ``polyadd``, ``polysub``).

Coefficients follow numpy.polynomial's *ascending* convention: index 0
is the constant term, index k is the coefficient of x**k. (This differs
from the legacy ``numpy.poly1d``/``numpy.polyval`` descending order.)
"""


def _coerce_seq(c):
    if hasattr(c, "tolist"):
        c = c.tolist()
    return list(c)


def _trim(coef, tol=0.0):
    """Drop trailing near-zero coefficients."""
    out = list(coef)
    while len(out) > 1 and abs(out[-1]) <= tol:
        out.pop()
    return out


def polyval(coef, x):
    """Evaluate polynomial with *ascending* coefficients at ``x``."""
    coef = _coerce_seq(coef)
    if not coef:
        return 0
    # Horner's rule from the high-order end.
    acc = coef[-1]
    for c in reversed(coef[:-1]):
        acc = acc * x + c
    return acc


def polyadd(c1, c2):
    a = _coerce_seq(c1)
    b = _coerce_seq(c2)
    n = max(len(a), len(b))
    a += [0] * (n - len(a))
    b += [0] * (n - len(b))
    return [a[i] + b[i] for i in range(n)]


def polysub(c1, c2):
    a = _coerce_seq(c1)
    b = _coerce_seq(c2)
    n = max(len(a), len(b))
    a += [0] * (n - len(a))
    b += [0] * (n - len(b))
    return [a[i] - b[i] for i in range(n)]


def polymul(c1, c2):
    a = _coerce_seq(c1)
    b = _coerce_seq(c2)
    out = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += ai * bj
    return out


def polyder(coef, m=1):
    """``m``-th derivative — ascending coefficients."""
    out = _coerce_seq(coef)
    for _ in range(m):
        if len(out) <= 1:
            return [0]
        out = [(k + 1) * out[k + 1] for k in range(len(out) - 1)]
    return out


def polyint(coef, m=1, k=0):
    """``m``-th antiderivative — ascending coefficients, constants ``k``.

    ``k`` may be a scalar (used for every integration step) or a list of
    length ``m`` (one constant per step, applied innermost first).
    """
    out = _coerce_seq(coef)
    if isinstance(k, (int, float, complex)):
        ks = [k] * m
    else:
        ks = list(k)
        if len(ks) != m:
            raise ValueError("polyint: k must have length m")
    for step in range(m):
        out = [ks[step]] + [out[i] / (i + 1) for i in range(len(out))]
    return out


def polyroots(coef):
    """Roots of the polynomial — Durand–Kerner iteration on the
    *ascending* coefficient vector. Returns ``len(coef) - 1`` roots.
    """
    c = _trim(_coerce_seq(coef))
    n = len(c) - 1
    if n <= 0:
        return []
    # Normalize so leading (highest-order) coefficient is 1.
    lead = c[-1]
    if lead == 0:
        return []
    monic = [ci / lead for ci in c]
    # Initial guesses spread on a circle. ``_pi``/``_cos``/``_sin`` are
    # injected from Rust to avoid an ``import math`` dependency.
    roots = [
        complex(
            0.4 * _cos(2 * _pi * k / n + 0.4),
            0.9 * _sin(2 * _pi * k / n + 0.4),
        )
        for k in range(n)
    ]
    for _ in range(200):
        new = []
        max_step = 0.0
        for i, r in enumerate(roots):
            denom = complex(1.0)
            for j, s in enumerate(roots):
                if i != j:
                    denom *= r - s
            if denom == 0:
                new.append(r)
                continue
            r2 = r - polyval(monic, r) / denom
            new.append(r2)
            d = abs(r2 - r)
            if d > max_step:
                max_step = d
        roots = new
        if max_step < 1e-14:
            break
    return roots


def polyfit(x, y, deg, rcond=None, full=False):
    """Least-squares polynomial fit — ascending coefficients.

    Solves the normal equations directly. Returns ``deg + 1`` coefficients.
    The ``rcond``/``full`` parameters are accepted for API parity but
    ignored.
    """
    _ = rcond  # API parity
    _ = full
    xs = _coerce_seq(x)
    ys = _coerce_seq(y)
    if len(xs) != len(ys):
        raise ValueError("polyfit: x and y must have the same length")
    n = deg + 1
    # Vandermonde-style normal equations: V^T V c = V^T y.
    # Build symmetric matrix sums[k] = sum(x_i ** k).
    sums = [sum(xv ** k for xv in xs) for k in range(2 * n - 1)]
    rhs = [sum(yv * xv ** k for xv, yv in zip(xs, ys)) for k in range(n)]
    mat = [[sums[i + j] for j in range(n)] for i in range(n)]
    return _gauss_solve(mat, rhs)


def _gauss_solve(a, b):
    """In-place Gaussian elimination with partial pivoting."""
    n = len(a)
    a = [row[:] for row in a]
    b = b[:]
    for k in range(n):
        # Pivot.
        piv = k
        for i in range(k + 1, n):
            if abs(a[i][k]) > abs(a[piv][k]):
                piv = i
        if piv != k:
            a[k], a[piv] = a[piv], a[k]
            b[k], b[piv] = b[piv], b[k]
        if a[k][k] == 0:
            raise ValueError("polyfit: singular normal-equations matrix")
        for i in range(k + 1, n):
            f = a[i][k] / a[k][k]
            for j in range(k, n):
                a[i][j] -= f * a[k][j]
            b[i] -= f * b[k]
    # Back-substitute.
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = b[i]
        for j in range(i + 1, n):
            s -= a[i][j] * x[j]
        x[i] = s / a[i][i]
    return x


class Polynomial:
    """A 1-D polynomial in *ascending* coefficient order.

    ``Polynomial([c0, c1, c2])`` represents ``c0 + c1*x + c2*x**2``.
    Supports arithmetic (``+ - *``), call evaluation, ``deriv``,
    ``integ``, ``roots``, and the convenience ``Polynomial.fit``
    constructor.
    """

    def __init__(self, coef):
        self.coef = _coerce_seq(coef)
        if not self.coef:
            self.coef = [0]

    def __call__(self, x):
        if isinstance(x, (list, tuple)):
            return [polyval(self.coef, v) for v in x]
        return polyval(self.coef, x)

    def __add__(self, other):
        if isinstance(other, Polynomial):
            return Polynomial(polyadd(self.coef, other.coef))
        return Polynomial(polyadd(self.coef, [other]))

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Polynomial):
            return Polynomial(polysub(self.coef, other.coef))
        return Polynomial(polysub(self.coef, [other]))

    def __rsub__(self, other):
        return Polynomial(polysub([other], self.coef))

    def __mul__(self, other):
        if isinstance(other, Polynomial):
            return Polynomial(polymul(self.coef, other.coef))
        return Polynomial([c * other for c in self.coef])

    __rmul__ = __mul__

    def __neg__(self):
        return Polynomial([-c for c in self.coef])

    def __eq__(self, other):
        if isinstance(other, Polynomial):
            return _trim(self.coef) == _trim(other.coef)
        return NotImplemented

    def __repr__(self):
        terms = ", ".join(repr(c) for c in self.coef)
        return f"Polynomial([{terms}])"

    @property
    def degree(self):
        c = _trim(self.coef)
        return max(0, len(c) - 1)

    def deriv(self, m=1):
        return Polynomial(polyder(self.coef, m))

    def integ(self, m=1, k=0):
        return Polynomial(polyint(self.coef, m, k))

    def roots(self):
        return polyroots(self.coef)

    @classmethod
    def fit(cls, x, y, deg):
        return cls(polyfit(x, y, deg))


__all__ = [
    "Polynomial",
    "polyval",
    "polyadd",
    "polysub",
    "polymul",
    "polyder",
    "polyint",
    "polyroots",
    "polyfit",
]
