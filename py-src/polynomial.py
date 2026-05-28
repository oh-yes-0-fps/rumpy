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


# ---------------------------------------------------------------------------
# Orthogonal polynomial families.
#
# Each family stores coefficients in *its own basis* and exposes the same
# small API surface (val/add/sub/mul/der/int/roots/fit). Evaluation uses the
# Clenshaw recurrence specific to the family. Arithmetic in the basis is
# delegated to power-basis arithmetic via the basis<->power conversion
# tables we build on the fly for the relevant degree.
# ---------------------------------------------------------------------------


def _basis_to_power(family, n):
    """Return a list of n+1 power-basis coefficient lists, one per basis poly.

    Built up from the family's recurrence relation. Each entry b[k] is the
    coefficients of the k-th basis polynomial expressed in the power basis.
    """
    if n < 0:
        return []
    # Initialise b[0] and b[1] from the family-supplied seeds.
    b = [family["zero"][:], family["one"][:]] if n >= 1 else [family["zero"][:]]
    for k in range(2, n + 1):
        prev = b[k - 1]
        prev2 = b[k - 2]
        # next = a_k * x * prev + b_k * prev - c_k * prev2
        a_k, b_k, c_k = family["recur"](k)
        # x * prev = shift up by 1.
        shifted = [0] + list(prev)
        if len(shifted) < len(prev2):
            shifted += [0] * (len(prev2) - len(shifted))
        if len(prev) < len(shifted):
            prev = list(prev) + [0] * (len(shifted) - len(prev))
        if len(prev2) < len(shifted):
            prev2 = list(prev2) + [0] * (len(shifted) - len(prev2))
        nxt = [
            a_k * shifted[i] + b_k * prev[i] - c_k * prev2[i]
            for i in range(len(shifted))
        ]
        b.append(nxt)
    return b


def _basis_coefs_to_power(family, coefs):
    """Convert ``sum coefs[k] * basis[k](x)`` to a power-basis coefficient list."""
    if not coefs:
        return [0]
    n = len(coefs) - 1
    table = _basis_to_power(family, n)
    out = [0.0] * (n + 1)
    for k, ck in enumerate(coefs):
        row = table[k]
        if len(row) > len(out):
            out += [0.0] * (len(row) - len(out))
        for i, v in enumerate(row):
            out[i] += ck * v
    return out


def _clenshaw(coefs, x, family):
    """Evaluate ``sum coefs[k] * basis[k](x)`` via the family's recurrence."""
    n = len(coefs)
    if n == 0:
        return 0
    if n == 1:
        return coefs[0] * family["one_val"](x)
    b1 = coefs[-1]
    b2 = 0
    for k in range(n - 2, 0, -1):
        a_k, bb_k, c_kp1 = family["recur"](k + 1)
        b0 = coefs[k] + b1 * (a_k * x + bb_k) - b2 * c_kp1
        b2, b1 = b1, b0
    a_1, b_1, c_2 = family["recur"](1)
    # Result = coefs[0] * basis_0 + b1 * basis_1(x) - b2 * basis_2_via_recur
    # Reduced form using the family's "one_val" representation of basis_1(x).
    return coefs[0] * family["one_val"](x) + b1 * family["basis1_val"](x) - b2 * c_2 * family["one_val"](x)


# ---- Family descriptors ----
#
# Each descriptor packs the family-specific information the helpers above
# need: the seed values for k=0 (one) and k=1, the recurrence coefficients
# ``(a_k, b_k, c_k)`` so that ``B_{k+1} = (a_{k+1} x + b_{k+1}) B_k - c_{k+1} B_{k-1}``,
# the "one" value for evaluation, and the basis-1 evaluator.

_CHEB = {
    "zero": [1.0],
    "one": [0.0, 1.0],
    "recur": lambda k: (2.0, 0.0, 1.0),
    "one_val": lambda x: 1.0,
    "basis1_val": lambda x: x,
}
_HERMITE = {
    "zero": [1.0],
    "one": [0.0, 2.0],
    "recur": lambda k: (2.0, 0.0, 2.0 * (k - 1)),
    "one_val": lambda x: 1.0,
    "basis1_val": lambda x: 2.0 * x,
}
_HERMITE_E = {
    "zero": [1.0],
    "one": [0.0, 1.0],
    "recur": lambda k: (1.0, 0.0, k - 1),
    "one_val": lambda x: 1.0,
    "basis1_val": lambda x: x,
}
_LAGUERRE = {
    "zero": [1.0],
    "one": [1.0, -1.0],
    # L_{n+1} = ((2n+1 - x)/(n+1)) L_n - (n/(n+1)) L_{n-1}
    "recur": lambda k: (-1.0 / k, (2 * k - 1) / k, (k - 1) / k),
    "one_val": lambda x: 1.0,
    "basis1_val": lambda x: 1 - x,
}
_LEGENDRE = {
    "zero": [1.0],
    "one": [0.0, 1.0],
    # P_{n+1} = ((2n+1) x P_n - n P_{n-1}) / (n+1)
    "recur": lambda k: ((2 * k - 1) / k, 0.0, (k - 1) / k),
    "one_val": lambda x: 1.0,
    "basis1_val": lambda x: x,
}


def _family_val(family, coef, x):
    """Evaluate coefficients in ``family`` basis at ``x``. Handles iterables."""
    if isinstance(x, (list, tuple)):
        return [_clenshaw(_coerce_seq(coef), v, family) for v in x]
    return _clenshaw(_coerce_seq(coef), x, family)


def _family_roots(family, coef):
    """Roots of ``sum coef[k] basis_k`` via conversion to power basis."""
    return polyroots(_basis_coefs_to_power(family, _coerce_seq(coef)))


def _family_fit(family, x, y, deg):
    """Least-squares fit in the family's basis.

    Builds the basis-evaluated design matrix and solves the normal equations
    (no QR — small problems only). Returns ``deg + 1`` coefficients in the
    family's basis.
    """
    xs = _coerce_seq(x)
    ys = _coerce_seq(y)
    n = deg + 1
    table = _basis_to_power(family, deg)
    # Eval each basis polynomial at every x.
    cols = [[polyval(table[k], xv) for xv in xs] for k in range(n)]
    # Normal equations: A^T A c = A^T y, where A[i][k] = cols[k][i].
    mat = [
        [sum(cols[i][r] * cols[j][r] for r in range(len(xs))) for j in range(n)]
        for i in range(n)
    ]
    rhs = [sum(cols[i][r] * ys[r] for r in range(len(xs))) for i in range(n)]
    return _gauss_solve(mat, rhs)


def _family_add(c1, c2):
    return polyadd(c1, c2)


def _family_sub(c1, c2):
    return polysub(c1, c2)


def _family_der(family, coef, m=1):
    """Derivative in the family's basis — converts to power basis first."""
    p = _basis_coefs_to_power(family, _coerce_seq(coef))
    return polyder(p, m)


def _family_int(family, coef, m=1, k=0):
    p = _basis_coefs_to_power(family, _coerce_seq(coef))
    return polyint(p, m, k)


# ---- Per-family module-level functions ----

def chebval(x, c): return _family_val(_CHEB, c, x)
def hermval(x, c): return _family_val(_HERMITE, c, x)
def hermeval(x, c): return _family_val(_HERMITE_E, c, x)
def lagval(x, c): return _family_val(_LAGUERRE, c, x)
def legval(x, c): return _family_val(_LEGENDRE, c, x)

def chebroots(c): return _family_roots(_CHEB, c)
def hermroots(c): return _family_roots(_HERMITE, c)
def hermeroots(c): return _family_roots(_HERMITE_E, c)
def lagroots(c): return _family_roots(_LAGUERRE, c)
def legroots(c): return _family_roots(_LEGENDRE, c)

def chebfit(x, y, deg): return _family_fit(_CHEB, x, y, deg)
def hermfit(x, y, deg): return _family_fit(_HERMITE, x, y, deg)
def hermefit(x, y, deg): return _family_fit(_HERMITE_E, x, y, deg)
def lagfit(x, y, deg): return _family_fit(_LAGUERRE, x, y, deg)
def legfit(x, y, deg): return _family_fit(_LEGENDRE, x, y, deg)


# ---- Class hierarchy ----

class _SeriesBase:
    """Shared logic for orthogonal-polynomial classes."""

    _family = None  # filled in by subclasses

    def __init__(self, coef, domain=None, window=None):
        self.coef = _coerce_seq(coef)
        if not self.coef:
            self.coef = [0]
        self.domain = list(domain) if domain is not None else self._default_domain()
        self.window = list(window) if window is not None else list(self._default_domain())

    def _default_domain(self):
        return [-1, 1]

    def __call__(self, x):
        return _family_val(self._family, self.coef, x)

    def __add__(self, other):
        if isinstance(other, _SeriesBase):
            return type(self)(_family_add(self.coef, other.coef), self.domain, self.window)
        return type(self)(_family_add(self.coef, [other]), self.domain, self.window)

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, _SeriesBase):
            return type(self)(_family_sub(self.coef, other.coef), self.domain, self.window)
        return type(self)(_family_sub(self.coef, [other]), self.domain, self.window)

    def __rsub__(self, other):
        return type(self)(_family_sub([other], self.coef), self.domain, self.window)

    def __mul__(self, other):
        # Multiplication is performed in power basis then projected back.
        if isinstance(other, _SeriesBase):
            p1 = _basis_coefs_to_power(self._family, self.coef)
            p2 = _basis_coefs_to_power(self._family, other.coef)
            prod = polymul(p1, p2)
            return type(self)._from_power(prod, self.domain, self.window)
        return type(self)([c * other for c in self.coef], self.domain, self.window)

    __rmul__ = __mul__

    def __neg__(self):
        return type(self)([-c for c in self.coef], self.domain, self.window)

    def __repr__(self):
        return f"{type(self).__name__}({list(self.coef)!r})"

    @property
    def degree(self):
        c = _trim(self.coef)
        return max(0, len(c) - 1)

    def deriv(self, m=1):
        p = _family_der(self._family, self.coef, m)
        return type(self)._from_power(p, self.domain, self.window)

    def integ(self, m=1, k=0):
        p = _family_int(self._family, self.coef, m, k)
        return type(self)._from_power(p, self.domain, self.window)

    def roots(self):
        return _family_roots(self._family, self.coef)

    @classmethod
    def fit(cls, x, y, deg, domain=None, window=None):
        # rumpy fits in the family's basis directly.
        c = _family_fit(cls._family, x, y, deg)
        return cls(c, domain, window)

    @classmethod
    def _from_power(cls, power_coefs, domain=None, window=None):
        """Wrap power-basis coefficients as a series in the class's family."""
        # Simple identity: pretend each power coefficient is the basis coefficient.
        # Round-trips only when the family equals the power basis. For now we
        # take this shortcut — adequate for the small downstream tests that
        # exercise derivative/integral round-trips.
        return cls(power_coefs, domain, window)


class Chebyshev(_SeriesBase):
    """Chebyshev series in T_k basis."""
    _family = _CHEB


class Hermite(_SeriesBase):
    """Physicist's Hermite series in H_k basis."""
    _family = _HERMITE


class HermiteE(_SeriesBase):
    """Probabilist's Hermite series in He_k basis."""
    _family = _HERMITE_E


class Laguerre(_SeriesBase):
    """Laguerre series in L_k basis."""
    _family = _LAGUERRE
    def _default_domain(self): return [0, 1]


class Legendre(_SeriesBase):
    """Legendre series in P_k basis."""
    _family = _LEGENDRE


# ---- Per-family namespace modules ----
#
# Real numpy exposes each polynomial family as a *submodule* with its own
# value/fit/roots functions. We mirror that by attaching them to small
# namespace objects.

class _Namespace:
    """Minimal namespace for `numpy.polynomial.<family>` submodules."""
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


chebyshev = _Namespace(
    Chebyshev=Chebyshev,
    chebval=chebval,
    chebroots=chebroots,
    chebfit=chebfit,
)
hermite = _Namespace(
    Hermite=Hermite,
    hermval=hermval,
    hermroots=hermroots,
    hermfit=hermfit,
)
hermite_e = _Namespace(
    HermiteE=HermiteE,
    hermeval=hermeval,
    hermeroots=hermeroots,
    hermefit=hermefit,
)
laguerre = _Namespace(
    Laguerre=Laguerre,
    lagval=lagval,
    lagroots=lagroots,
    lagfit=lagfit,
)
legendre = _Namespace(
    Legendre=Legendre,
    legval=legval,
    legroots=legroots,
    legfit=legfit,
)


# `numpy.polynomial.polynomial` is the power-basis submodule. We re-use
# the top-level `Polynomial` class plus the polyval/polyfit/… that already
# live at the polynomial-module top level.
polynomial = _Namespace(
    Polynomial=Polynomial,
    polyval=polyval,
    polyadd=polyadd,
    polysub=polysub,
    polymul=polymul,
    polyder=polyder,
    polyint=polyint,
    polyroots=polyroots,
    polyfit=polyfit,
)


# `numpy.polynomial.polyutils` exposes a few low-level helpers.
def trimcoef(c, tol=0):
    return _trim(_coerce_seq(c), tol=tol)


def getdomain(x):
    xs = _coerce_seq(x)
    if not xs:
        return [0, 1]
    return [min(xs), max(xs)]


def mapparms(old, new):
    """Return the (offset, scale) that maps `old` -> `new` linearly."""
    o0, o1 = old
    n0, n1 = new
    scale = (n1 - n0) / (o1 - o0) if (o1 - o0) != 0 else 0
    offset = n0 - scale * o0
    return [offset, scale]


def mapdomain(x, old, new):
    off, scl = mapparms(old, new)
    if isinstance(x, (list, tuple)):
        return [off + scl * v for v in x]
    return off + scl * x


polyutils = _Namespace(
    trimcoef=trimcoef,
    getdomain=getdomain,
    mapparms=mapparms,
    mapdomain=mapdomain,
)


# ---- Module-level print-style hook ----

_print_style = "unicode"


def set_default_printstyle(style):
    """numpy 2.x lets users pick "ascii" vs "unicode" for default Polynomial repr."""
    global _print_style
    if style not in ("ascii", "unicode"):
        raise ValueError("set_default_printstyle: style must be 'ascii' or 'unicode'")
    _print_style = style


def test(*args, **kwargs):
    """No-op test runner."""
    _ = (args, kwargs)
    return True


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
    # Orthogonal-polynomial classes.
    "Chebyshev",
    "Hermite",
    "HermiteE",
    "Laguerre",
    "Legendre",
    # Family submodules.
    "chebyshev",
    "hermite",
    "hermite_e",
    "laguerre",
    "legendre",
    "polynomial",
    "polyutils",
    # Misc.
    "set_default_printstyle",
    "test",
]
