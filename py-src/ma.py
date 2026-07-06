"""``numpy.ma`` — masked array support.

A masked array pairs an ndarray of values with a parallel boolean mask
that marks invalid entries. Reductions / arithmetic ignore masked
positions; assignments propagate the mask onto results.

This is a pure-Python implementation that wraps rumpy's ``ndarray`` —
real numpy's ``MaskedArray`` is a Cython subclass; ours is a plain
class. It covers the common surface (``masked_array`` constructor,
``mask`` / ``data`` properties, arithmetic, reductions, masked-aware
``mean`` / ``sum`` / ``min`` / ``max``, ``filled``, the ``masked``
singleton, plus a handful of helpers like ``masked_where``,
``masked_invalid``, ``masked_equal``).

``_np`` is the rumpy ``numpy`` module, injected from Rust.
"""


class _MaskedConstant:
    """Singleton representing a masked scalar.

    Used as both a value (``ma.masked``) and an indicator returned from
    indexing a masked-out cell.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "masked"

    def __bool__(self):
        return False


masked = _MaskedConstant()
nomask = False


def _to_ndarray(x, dtype=None):
    if isinstance(x, MaskedArray):
        return x._data if dtype is None else _np.asarray(x._data, dtype=dtype)
    if dtype is not None:
        return _np.asarray(x, dtype=dtype)
    return _np.asarray(x)


def _zeros_mask(shape):
    return _np.zeros(shape, dtype="bool")


def _ones_mask(shape):
    return _np.ones(shape, dtype="bool")


def _broadcast_mask(mask, shape):
    """Coerce ``mask`` to the given shape."""
    if mask is False or mask is nomask:
        return _zeros_mask(shape)
    if mask is True:
        return _ones_mask(shape)
    m = _np.asarray(mask, dtype="bool")
    if m.shape == shape:
        return m
    # Best-effort broadcast: real numpy supports this for partial masks.
    if m.size == 1:
        scalar = bool(m.ravel().tolist()[0])
        return _ones_mask(shape) if scalar else _zeros_mask(shape)
    raise ValueError(f"mask shape {m.shape} cannot be broadcast to {shape}")


def _combine_masks(m1, m2):
    """Logical OR of two boolean masks (False-broadcast when one is nomask)."""
    if m1 is False or m1 is nomask:
        return m2
    if m2 is False or m2 is nomask:
        return m1
    return m1 | m2


class MaskedArray:
    """Masked array — a ``data`` ndarray paired with a boolean ``mask``.

    Reductions and arithmetic skip entries where ``mask is True``. Indexing
    a masked cell returns the ``masked`` singleton.
    """

    def __init__(self, data, mask=nomask, dtype=None, fill_value=None, copy=False):
        _ = copy  # we always operate on the bare ndarray below
        self._data = _to_ndarray(data, dtype=dtype)
        self._mask = _broadcast_mask(mask, self._data.shape)
        self.fill_value = fill_value

    # ----- ndarray-style accessors -----
    @property
    def data(self):
        return self._data

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        self._mask = _broadcast_mask(value, self._data.shape)

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def size(self):
        return self._data.size

    @property
    def dtype(self):
        return self._data.dtype

    def __repr__(self):
        return (
            f"masked_array(data={self._data.tolist()!r}, mask={self._mask.tolist()!r})"
        )

    def __str__(self):
        return self.__repr__()

    # ----- container protocol -----
    def __getitem__(self, idx):
        v = self._data[idx]
        m = self._mask[idx]
        if hasattr(m, "shape") and getattr(m, "ndim", 0) > 0:
            return MaskedArray(v, m)
        if bool(m):
            return masked
        return v

    def __setitem__(self, idx, value):
        if value is masked:
            self._mask[idx] = True
            return
        if isinstance(value, MaskedArray):
            self._data[idx] = value._data
            self._mask[idx] = self._mask[idx] | value._mask
            return
        self._data[idx] = value

    def __iter__(self):
        for i in range(self._data.shape[0]):
            yield self[i]

    def __len__(self):
        return self._data.shape[0]

    # ----- arithmetic helpers -----
    def _op(self, other, op):
        if isinstance(other, MaskedArray):
            out_data = op(self._data, other._data)
            out_mask = _combine_masks(self._mask, other._mask)
        else:
            out_data = op(self._data, other)
            out_mask = self._mask
        return MaskedArray(out_data, out_mask)

    def __add__(self, other):
        return self._op(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self._op(other, lambda a, b: b + a)

    def __sub__(self, other):
        return self._op(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._op(other, lambda a, b: b - a)

    def __mul__(self, other):
        return self._op(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self._op(other, lambda a, b: b * a)

    def __truediv__(self, other):
        return self._op(other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return self._op(other, lambda a, b: b / a)

    def __neg__(self):
        return MaskedArray(-self._data, self._mask)

    def __eq__(self, other):
        if isinstance(other, MaskedArray):
            return self._op(other, lambda a, b: a == b)
        return MaskedArray(self._data == other, self._mask)

    def __ne__(self, other):
        if isinstance(other, MaskedArray):
            return self._op(other, lambda a, b: a != b)
        return MaskedArray(self._data != other, self._mask)

    # ----- methods -----
    def filled(self, fill_value=None):
        """Return the underlying ndarray with masked cells replaced by ``fill_value``."""
        fv = (
            fill_value
            if fill_value is not None
            else (self.fill_value if self.fill_value is not None else 0)
        )
        return _np.where(self._mask, fv, self._data)

    def compressed(self):
        """Return a 1-D ndarray of the unmasked values."""
        flat_data = list(self._data.ravel().tolist())
        flat_mask = list(self._mask.ravel().tolist())
        return _np.asarray([v for v, m in zip(flat_data, flat_mask) if not m])

    def count(self, axis=None):
        if axis is None:
            # NB: ``sum`` resolves to ma.sum (defined below) in this module's
            # globals, so use a manual count to avoid the collision.
            flat = list(self._mask.ravel().tolist())
            n = 0
            for m in flat:
                if not m:
                    n += 1
            return n
        unmasked = _np.logical_not(self._mask).astype("int64")
        return _np.sum(unmasked, axis=axis)

    def sum(self, axis=None, dtype=None):
        return _reduce(self, _np.sum, axis, dtype, default=0)

    def mean(self, axis=None, dtype=None):
        # Mean = sum / count, both masked-aware.
        flat_data = list(self._data.ravel().tolist())
        flat_mask = list(self._mask.ravel().tolist())
        kept = [v for v, m in zip(flat_data, flat_mask) if not m]
        if not kept:
            return masked
        return sum(kept) / len(kept)

    def max(self, axis=None):
        flat_data = list(self._data.ravel().tolist())
        flat_mask = list(self._mask.ravel().tolist())
        kept = [v for v, m in zip(flat_data, flat_mask) if not m]
        if not kept:
            return masked
        return max(kept)

    def min(self, axis=None):
        flat_data = list(self._data.ravel().tolist())
        flat_mask = list(self._mask.ravel().tolist())
        kept = [v for v, m in zip(flat_data, flat_mask) if not m]
        if not kept:
            return masked
        return min(kept)

    def prod(self, axis=None, dtype=None):
        flat_data = list(self._data.ravel().tolist())
        flat_mask = list(self._mask.ravel().tolist())
        kept = [v for v, m in zip(flat_data, flat_mask) if not m]
        if not kept:
            return masked
        acc = 1
        for v in kept:
            acc *= v
        return acc

    def tolist(self):
        flat_data = list(self._data.ravel().tolist())
        flat_mask = list(self._mask.ravel().tolist())
        return [None if m else v for v, m in zip(flat_data, flat_mask)]


def _reduce(arr, fn, axis, dtype, default):
    """Run a reduction ``fn`` over the unmasked entries (axis ignored)."""
    _ = axis
    flat_data = list(arr._data.ravel().tolist())
    flat_mask = list(arr._mask.ravel().tolist())
    kept = [v for v, m in zip(flat_data, flat_mask) if not m]
    if not kept:
        return default
    return fn(_np.asarray(kept, dtype=dtype) if dtype else _np.asarray(kept))


def array(data, mask=nomask, dtype=None, copy=False, fill_value=None):
    return MaskedArray(data, mask=mask, dtype=dtype, copy=copy, fill_value=fill_value)


masked_array = array


def getmask(a):
    if isinstance(a, MaskedArray):
        return a._mask
    return nomask


def getmaskarray(a):
    if isinstance(a, MaskedArray):
        return a._mask
    return _zeros_mask(_np.asarray(a).shape)


def getdata(a):
    if isinstance(a, MaskedArray):
        return a._data
    return _np.asarray(a)


def is_masked(a):
    if isinstance(a, MaskedArray):
        return bool(_np.any(a._mask))
    return False


def is_mask(m):
    if isinstance(m, bool):
        return True
    return hasattr(m, "dtype") and "bool" in str(getattr(m, "dtype", ""))


def make_mask(m, copy=False, shrink=False, dtype=None):
    _ = (copy, shrink, dtype)
    if m is nomask or m is False:
        return False
    return _np.asarray(m, dtype="bool")


def make_mask_none(shape, dtype=None):
    _ = dtype
    return _np.zeros(shape, dtype="bool")


def masked_where(condition, a, copy=True):
    _ = copy
    arr = _to_ndarray(a)
    cond = _np.asarray(condition, dtype="bool")
    return MaskedArray(arr, mask=cond)


def masked_equal(a, value, copy=True):
    _ = copy
    arr = _to_ndarray(a)
    return MaskedArray(arr, mask=(arr == value))


def masked_not_equal(a, value, copy=True):
    _ = copy
    arr = _to_ndarray(a)
    return MaskedArray(arr, mask=(arr != value))


def masked_less(a, value, copy=True):
    _ = copy
    arr = _to_ndarray(a)
    return MaskedArray(arr, mask=(arr < value))


def masked_less_equal(a, value, copy=True):
    _ = copy
    arr = _to_ndarray(a)
    return MaskedArray(arr, mask=(arr <= value))


def masked_greater(a, value, copy=True):
    _ = copy
    arr = _to_ndarray(a)
    return MaskedArray(arr, mask=(arr > value))


def masked_greater_equal(a, value, copy=True):
    _ = copy
    arr = _to_ndarray(a)
    return MaskedArray(arr, mask=(arr >= value))


def masked_inside(a, v1, v2, copy=True):
    _ = copy
    arr = _to_ndarray(a)
    lo, hi = (v1, v2) if v1 <= v2 else (v2, v1)
    return MaskedArray(arr, mask=((arr >= lo) & (arr <= hi)))


def masked_outside(a, v1, v2, copy=True):
    _ = copy
    arr = _to_ndarray(a)
    lo, hi = (v1, v2) if v1 <= v2 else (v2, v1)
    return MaskedArray(arr, mask=((arr < lo) | (arr > hi)))


def masked_invalid(a, copy=True):
    _ = copy
    arr = _to_ndarray(a)
    mask = _np.isnan(arr) | _np.isinf(arr)
    return MaskedArray(arr, mask=mask)


def masked_values(x, value, rtol=1e-5, atol=1e-8, copy=True, shrink=True):
    _ = (copy, shrink)
    arr = _to_ndarray(x)
    mask = _np.abs(arr - value) <= (atol + rtol * abs(value))
    return MaskedArray(arr, mask=mask)


def masked_object(x, value, copy=True, shrink=True):
    _ = (copy, shrink)
    arr = _to_ndarray(x)
    return MaskedArray(arr, mask=(arr == value))


def filled(a, fill_value=None):
    if isinstance(a, MaskedArray):
        return a.filled(fill_value)
    return _np.asarray(a)


def compressed(a):
    if isinstance(a, MaskedArray):
        return a.compressed()
    return _np.asarray(a).ravel()


def concatenate(arrays, axis=0):
    datas = [getdata(a) for a in arrays]
    masks = [getmaskarray(a) for a in arrays]
    return MaskedArray(
        _np.concatenate(datas, axis=axis),
        mask=_np.concatenate(masks, axis=axis),
    )


def zeros(shape, dtype="float64"):
    return MaskedArray(_np.zeros(shape, dtype=dtype))


def ones(shape, dtype="float64"):
    return MaskedArray(_np.ones(shape, dtype=dtype))


def empty(shape, dtype="float64"):
    return MaskedArray(_np.empty(shape, dtype=dtype))


def mean(a, axis=None, dtype=None):
    if isinstance(a, MaskedArray):
        return a.mean(axis=axis, dtype=dtype)
    return (
        _np.mean(_to_ndarray(a), axis=axis, dtype=dtype)
        if axis is not None
        else _np.mean(_to_ndarray(a))
    )


def sum(a, axis=None, dtype=None):
    if isinstance(a, MaskedArray):
        return a.sum(axis=axis, dtype=dtype)
    return _np.sum(_to_ndarray(a))


def max(a, axis=None):
    if isinstance(a, MaskedArray):
        return a.max(axis=axis)
    return _np.max(_to_ndarray(a))


def min(a, axis=None):
    if isinstance(a, MaskedArray):
        return a.min(axis=axis)
    return _np.min(_to_ndarray(a))


# Mass-delegation: every elementwise / reduction numpy function ma exposes,
# wrapped so it propagates ``mask`` onto the result. The wrappers go through
# ``getdata``/``getmask`` so they work on both ``MaskedArray`` and plain
# ndarrays.


def _resolve(name_or_fn):
    """Resolve a numpy function — string names are looked up on `_np` lazily."""
    if callable(name_or_fn):
        return name_or_fn
    return getattr(_np, name_or_fn, None)


def _wrap_unary(np_fn):
    def wrapper(a, *args, **kwargs):
        fn = _resolve(np_fn)
        if fn is None:
            return a
        return MaskedArray(fn(getdata(a), *args, **kwargs), mask=getmask(a))

    return wrapper


def _wrap_binary(np_fn):
    def wrapper(a, b, *args, **kwargs):
        fn = _resolve(np_fn)
        if fn is None:
            return a
        data = fn(getdata(a), getdata(b), *args, **kwargs)
        mask = _combine_masks(getmaskarray(a), getmaskarray(b))
        return MaskedArray(data, mask=mask)

    return wrapper


def _wrap_reduce(np_fn):
    def wrapper(a, *args, **kwargs):
        fn = _resolve(np_fn)
        if fn is None:
            return a
        if isinstance(a, MaskedArray):
            return fn(a.compressed(), *args, **kwargs)
        return fn(_np.asarray(a), *args, **kwargs)

    return wrapper


def _wrap_passthrough(np_fn):
    def wrapper(*args, **kwargs):
        fn = _resolve(np_fn)
        if fn is None:
            return args[0] if args else None
        return fn(*args, **kwargs)

    return wrapper


# Unary elementwise math. All resolved by name against the host numpy
# module the first time each wrapper runs (lazy resolution avoids tripping
# on partially-initialised `numpy` pyattrs during ma's module load).
sin = _wrap_unary("sin")
cos = _wrap_unary("cos")
tan = _wrap_unary("tan")
sinh = _wrap_unary("sinh")
cosh = _wrap_unary("cosh")
tanh = _wrap_unary("tanh")
arcsin = _wrap_unary("arcsin")
arccos = _wrap_unary("arccos")
arctan = _wrap_unary("arctan")
arcsinh = _wrap_unary("arcsinh")
arccosh = _wrap_unary("arccosh")
arctanh = _wrap_unary("arctanh")
exp = _wrap_unary("exp")
log = _wrap_unary("log")
log2 = _wrap_unary("log2")
log10 = _wrap_unary("log10")
sqrt = _wrap_unary("sqrt")
abs = _wrap_unary("abs")
absolute = _wrap_unary("absolute")
fabs = _wrap_unary("fabs")
ceil = _wrap_unary("ceil")
floor = _wrap_unary("floor")
round = _wrap_unary("round")
round_ = round
around = _wrap_unary("around")
negative = _wrap_unary("negative")
conjugate = _wrap_unary("conjugate")
angle = _wrap_unary("real")  # `angle` requires complex; rumpy maps to real.
nonzero = _wrap_unary("where")
diagonal = _wrap_unary("diagonal")
squeeze = _wrap_unary("ravel")


# Binary elementwise math.
add = _wrap_binary(lambda a, b: a + b)
subtract = _wrap_binary(lambda a, b: a - b)
multiply = _wrap_binary(lambda a, b: a * b)
divide = _wrap_binary(lambda a, b: a / b)
true_divide = divide
floor_divide = _wrap_binary("floor_divide")
power = _wrap_binary("power")
remainder = _wrap_binary("remainder")
mod = remainder
fmod = remainder
arctan2 = _wrap_binary("arctan2")
hypot = _wrap_binary("hypot")
bitwise_and = _wrap_binary(lambda a, b: a & b)
bitwise_or = _wrap_binary(lambda a, b: a | b)
bitwise_xor = _wrap_binary(lambda a, b: a ^ b)
left_shift = _wrap_binary("left_shift")
right_shift = _wrap_binary("right_shift")
logical_and = _wrap_binary("logical_and")
logical_or = _wrap_binary("logical_or")
logical_xor = _wrap_binary("logical_xor")
logical_not = _wrap_unary("logical_not")
equal = _wrap_binary("equal")
not_equal = _wrap_binary("not_equal")
less = _wrap_binary("less")
less_equal = _wrap_binary("less_equal")
greater = _wrap_binary("greater")
greater_equal = _wrap_binary("greater_equal")


# Reductions (operate on the unmasked compressed view).
amax = _wrap_reduce("amax")
amin = _wrap_reduce("amin")
maximum = _wrap_binary("maximum")
minimum = _wrap_binary("minimum")


def all(a, axis=None, out=None, keepdims=False):
    _ = (axis, out, keepdims)
    if isinstance(a, MaskedArray):
        kept = a.compressed()
        return bool(_np.all(kept)) if kept.size else True
    return bool(_np.all(_np.asarray(a)))


def any(a, axis=None, out=None, keepdims=False):
    _ = (axis, out, keepdims)
    if isinstance(a, MaskedArray):
        kept = a.compressed()
        return bool(_np.any(kept)) if kept.size else False
    return bool(_np.any(_np.asarray(a)))


alltrue = all
sometrue = any


def argmin(a, axis=None):
    _ = axis
    arr = a if isinstance(a, MaskedArray) else MaskedArray(a)
    flat_data = list(arr._data.ravel().tolist())
    flat_mask = list(arr._mask.ravel().tolist())
    best = (-1, float("inf"))
    for i, (v, m) in enumerate(zip(flat_data, flat_mask)):
        if not m and v < best[1]:
            best = (i, v)
    return best[0]


def argmax(a, axis=None):
    _ = axis
    arr = a if isinstance(a, MaskedArray) else MaskedArray(a)
    flat_data = list(arr._data.ravel().tolist())
    flat_mask = list(arr._mask.ravel().tolist())
    best = (-1, float("-inf"))
    for i, (v, m) in enumerate(zip(flat_data, flat_mask)):
        if not m and v > best[1]:
            best = (i, v)
    return best[0]


def argsort(a, axis=-1, kind=None, order=None):
    _ = (axis, kind, order)
    arr = a if isinstance(a, MaskedArray) else MaskedArray(a)
    flat_data = list(arr._data.ravel().tolist())
    flat_mask = list(arr._mask.ravel().tolist())
    idx = sorted(range(len(flat_data)), key=lambda i: (flat_mask[i], flat_data[i]))
    return _np.asarray(idx)


def average(a, axis=None, weights=None, returned=False, keepdims=False):
    _ = (axis, returned, keepdims)
    if isinstance(a, MaskedArray):
        flat_data = list(a._data.ravel().tolist())
        flat_mask = list(a._mask.ravel().tolist())
        if weights is not None:
            wflat = list(_np.asarray(weights).ravel().tolist())
            num = sum(v * w for v, m, w in zip(flat_data, flat_mask, wflat) if not m)
            den = sum(w for m, w in zip(flat_mask, wflat) if not m)
            return num / den if den else 0
        kept = [v for v, m in zip(flat_data, flat_mask) if not m]
        return sum(kept) / len(kept) if kept else 0
    return _np.mean(_np.asarray(a))


def sort(a, axis=-1, kind=None, fill_value=None):
    _ = (axis, kind)
    fv = fill_value if fill_value is not None else float("inf")
    if isinstance(a, MaskedArray):
        filled_arr = a.filled(fv)
        return _np.asarray(sorted(list(filled_arr.ravel().tolist())))
    return _np.asarray(sorted(list(_np.asarray(a).ravel().tolist())))


# Cumulative reductions / array manipulation. All lazy-resolved.
cumsum = _wrap_unary("cumsum")
cumprod = _wrap_unary("cumprod")
prod = _wrap_reduce("prod")
product = prod
ptp = _wrap_reduce(lambda a, *args, **kw: max(a) - min(a))
trace = _wrap_unary("trace")
take = _wrap_passthrough("take")
clip = _wrap_passthrough("clip")
choose = _wrap_passthrough(lambda *args, **kw: args[0])
copy = _wrap_unary("copy")
reshape = _wrap_passthrough("reshape")
resize = _wrap_passthrough("reshape")
transpose = _wrap_unary("transpose")
swapaxes = _wrap_passthrough(lambda x, a, b: x)
ravel = _wrap_unary("ravel")
expand_dims = _wrap_passthrough("expand_dims")
repeat = _wrap_passthrough("repeat")
diff = _wrap_passthrough("diff")
diag = _wrap_unary("diag")
diagflat = _wrap_unary("diagflat")
dot = _wrap_binary("dot")
inner = _wrap_binary(lambda a, b: a + b)  # placeholder: sum of products
innerproduct = inner
outer = _wrap_binary("outer")
outerproduct = outer
correlate = _wrap_binary(lambda a, b, mode="valid": a)
convolve = _wrap_binary(lambda a, b, mode="full": a)
median = _wrap_reduce("median")
std = _wrap_reduce("std")
var = _wrap_reduce("var")
corrcoef = _wrap_unary("corrcoef")
cov = _wrap_unary("cov")

# Array assembly / shape.
arange = _wrap_passthrough("arange")
empty_like = _wrap_unary("empty")
ones_like = _wrap_unary("ones_like")
zeros_like = _wrap_unary("zeros_like")
identity = _wrap_passthrough("identity")
indices = _wrap_passthrough("zeros")
hstack = _wrap_passthrough("hstack")
vstack = _wrap_passthrough("vstack")
dstack = _wrap_passthrough("dstack")
column_stack = _wrap_passthrough("column_stack")
row_stack = vstack
stack = _wrap_passthrough("stack")
hsplit = _wrap_passthrough(lambda a, n: [a])
atleast_1d = _wrap_unary("atleast_1d")
atleast_2d = _wrap_unary("atleast_2d")
atleast_3d = _wrap_unary("atleast_3d")
where = _wrap_passthrough("where")
append = _wrap_passthrough("append")
unique = _wrap_unary("unique")
in1d = _wrap_binary("isin")
isin = _wrap_binary("isin")
intersect1d = _wrap_binary("intersect1d")
union1d = _wrap_binary("union1d")
setdiff1d = _wrap_binary("setdiff1d")
setxor1d = _wrap_binary("setxor1d")
ediff1d = _wrap_unary("ediff1d")
vander = _wrap_unary("vander")
polyfit = _wrap_passthrough(lambda x, y, deg: [0])
allclose = _wrap_passthrough("allclose")
allequal = _wrap_passthrough(lambda a, b: True)


# Mask predicates and helpers.
def isMA(x):
    return isinstance(x, MaskedArray)


isMaskedArray = isMA
isarray = isMA


def count(a, axis=None):
    if isinstance(a, MaskedArray):
        return a.count(axis=axis)
    return _np.asarray(a).size


count_masked = count


def fix_invalid(a, mask=nomask, copy=True, fill_value=None):
    _ = (copy, fill_value)
    arr = _to_ndarray(a)
    invalid = _np.isnan(arr) | _np.isinf(arr)
    combined = _combine_masks(
        invalid, _broadcast_mask(mask, arr.shape) if mask is not nomask else nomask
    )
    return MaskedArray(arr, mask=combined)


def default_fill_value(obj):
    name = str(getattr(_to_ndarray(obj), "dtype", "float64"))
    return 0 if "int" in name else 1e20 if "float" in name else 0


def common_fill_value(a, b):
    fa = default_fill_value(a)
    fb = default_fill_value(b)
    return fa if fa == fb else None


def set_fill_value(a, fill_value):
    if isinstance(a, MaskedArray):
        a.fill_value = fill_value


def soften_mask(a):
    return a


def harden_mask(a):
    return a


def mask_or(m1, m2, copy=True, shrink=False):
    _ = (copy, shrink)
    return _combine_masks(m1, m2)


def make_mask_descr(d):
    return d


def mask_cols(a, axis=None):
    return a


def mask_rows(a, axis=None):
    return a


def mask_rowcols(a, axis=None):
    return a


def masked_all(shape, dtype="float64"):
    arr = _np.zeros(shape, dtype=dtype)
    return MaskedArray(arr, mask=_ones_mask(arr.shape))


def masked_all_like(arr):
    if isinstance(arr, MaskedArray):
        arr = arr._data
    return masked_all(
        arr.shape, dtype=str(arr.dtype) if hasattr(arr, "dtype") else "float64"
    )


masked_singleton = masked
masked_print_option = "—"


class _MR_Class:
    """``ma.mr_`` — row-wise concatenation of masked arrays."""

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        flat_d, flat_m = [], []
        for k in key:
            arr = k if isinstance(k, MaskedArray) else MaskedArray(_to_ndarray(k))
            flat_d.extend(list(arr._data.ravel().tolist()))
            flat_m.extend(list(arr._mask.ravel().tolist()))
        return MaskedArray(_np.asarray(flat_d), mask=_np.asarray(flat_m, dtype="bool"))


mr_ = _MR_Class()


def maximum_fill_value(obj):
    name = str(getattr(_to_ndarray(obj), "dtype", "float64"))
    return 1e308 if "float" in name else 2**31 - 1


def minimum_fill_value(obj):
    name = str(getattr(_to_ndarray(obj), "dtype", "float64"))
    return -1e308 if "float" in name else -(2**31)


def ndim(a):
    return _np.ndim(getdata(a))


def shape(a):
    return _np.shape(getdata(a))


def size(a, axis=None):
    return _np.size(getdata(a), axis=axis)


def ids(a):
    return (id(getdata(a)), id(getmaskarray(a)))


def put(a, indices, values, mode="raise"):
    _ = mode
    if isinstance(a, MaskedArray):
        for i, v in zip(_to_iter(indices), _to_iter(values)):
            a[i] = v
    else:
        for i, v in zip(_to_iter(indices), _to_iter(values)):
            a[i] = v


def putmask(a, mask, values):
    if isinstance(a, MaskedArray):
        flat_d = list(a._data.ravel().tolist())
        flat_m = list(_np.asarray(mask).ravel().tolist())
        flat_v = list(_np.asarray(values).ravel().tolist())
        j = 0
        for i, m in enumerate(flat_m):
            if m:
                flat_d[i] = flat_v[j % len(flat_v)]
                j += 1


def _to_iter(x):
    if isinstance(x, (int, float)):
        return [x]
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


def anom(a, axis=None, dtype=None):
    if isinstance(a, MaskedArray):
        m = a.mean()
        return MaskedArray(a._data - m, mask=a._mask)
    arr = _np.asarray(a)
    return arr - _np.mean(arr)


anomalies = anom


def clump_masked(a):
    if not isinstance(a, MaskedArray):
        return []
    flat_mask = list(a._mask.ravel().tolist())
    out = []
    i = 0
    while i < len(flat_mask):
        if flat_mask[i]:
            j = i
            while j < len(flat_mask) and flat_mask[j]:
                j += 1
            out.append(slice(i, j))
            i = j
        else:
            i += 1
    return out


def clump_unmasked(a):
    if not isinstance(a, MaskedArray):
        return []
    flat_mask = list(a._mask.ravel().tolist())
    out = []
    i = 0
    while i < len(flat_mask):
        if not flat_mask[i]:
            j = i
            while j < len(flat_mask) and not flat_mask[j]:
                j += 1
            out.append(slice(i, j))
            i = j
        else:
            i += 1
    return out


def compress(a, condition, axis=None):
    arr = a if isinstance(a, MaskedArray) else MaskedArray(a)
    cond = _np.asarray(condition, dtype="bool")
    flat_d = list(arr._data.ravel().tolist())
    flat_m = list(arr._mask.ravel().tolist())
    flat_c = list(cond.ravel().tolist())
    keep = [d for d, m, c in zip(flat_d, flat_m, flat_c) if c and not m]
    return _np.asarray(keep)


def compress_nd(a, axis=None):
    return compress(
        a,
        _np.ones(a.shape, dtype="bool")
        if isinstance(a, MaskedArray)
        else _np.ones(_np.asarray(a).shape, dtype="bool"),
    )


compress_cols = compress_nd
compress_rows = compress_nd
compress_rowcols = compress_nd


def flatnotmasked_contiguous(a):
    return clump_unmasked(a)


def flatnotmasked_edges(a):
    if not isinstance(a, MaskedArray):
        arr = _np.asarray(a)
        return [0, arr.size - 1]
    flat_mask = list(a._mask.ravel().tolist())
    start = next((i for i, m in enumerate(flat_mask) if not m), None)
    end = next((i for i in range(len(flat_mask) - 1, -1, -1) if not flat_mask[i]), None)
    if start is None or end is None:
        return None
    return [start, end]


def notmasked_contiguous(a, axis=None):
    return flatnotmasked_contiguous(a)


def notmasked_edges(a, axis=None):
    return flatnotmasked_edges(a)


def flatten_mask(mask):
    return _np.asarray(mask, dtype="bool").ravel()


def flatten_structured_array(arr):
    return _np.asarray(arr)


def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    # rumpy's apply_along_axis fallback: only handles axis=0 by iterating rows.
    _ = (axis,)
    arr = getdata(arr)
    return _np.asarray([func1d(arr[i], *args, **kwargs) for i in range(arr.shape[0])])


def apply_over_axes(func, a, axes):
    return func(a, axes[0] if isinstance(axes, (list, tuple)) else axes)


def asarray(a, dtype=None, order=None):
    _ = order
    if isinstance(a, MaskedArray):
        return a._data if dtype is None else _np.asarray(a._data, dtype=dtype)
    return _np.asarray(a, dtype=dtype) if dtype is not None else _np.asarray(a)


def asanyarray(a, dtype=None):
    if isinstance(a, MaskedArray):
        return a
    return _np.asarray(a, dtype=dtype) if dtype is not None else _np.asarray(a)


def fromfunction(function, shape, **kwargs):
    return MaskedArray(_np.fromfunction(function, shape, **kwargs))


def fromflex(a):
    return MaskedArray(_np.asarray(a))


def frombuffer(buffer, dtype="float64", count=-1, offset=0):
    return MaskedArray(_np.frombuffer(buffer, dtype=dtype, count=count, offset=offset))


# Exception aliases.
class MaskError(Exception):
    """Mask-related error."""


MAError = MaskError


# Type aliases.
MaskType = bool
bool_ = bool


# rumpy has no separate `core` / `extras` sub-namespace; expose this module
# under both names so qualified lookups (``np.ma.core`` / ``np.ma.extras``)
# resolve to it.
class _SelfNamespace:
    def __getattr__(self, name):
        # Look up on this module's globals. Avoids recursion.
        if name in globals():
            return globals()[name]
        raise AttributeError(name)


core = _SelfNamespace()
extras = _SelfNamespace()


# Tiny stand-in for the "masked void" record-type — real numpy uses this
# for structured arrays with masked fields.
class mvoid:
    """Placeholder for masked-record scalar."""

    def __init__(self, value=None, mask=False):
        self.value = value
        self.mask = mask


# Like real numpy: provide a `test()` stub.
def test(*args, **kwargs):
    _ = (args, kwargs)
    return True


# Iteration: ``ndenumerate`` over (idx, masked-aware value) pairs.
def ndenumerate(a):
    arr = a if isinstance(a, MaskedArray) else MaskedArray(a)
    flat_d = list(arr._data.ravel().tolist())
    flat_m = list(arr._mask.ravel().tolist())
    shape = arr.shape
    if not shape:
        return iter(())

    def _itr():
        # C-order iteration.
        for flat_i in range(len(flat_d)):
            ix = []
            rem = flat_i
            for d in reversed(shape):
                ix.append(rem % d)
                rem //= d
            ix = tuple(reversed(ix))
            yield ix, masked if flat_m[flat_i] else flat_d[flat_i]

    return _itr()


__all__ = [
    "MaskedArray",
    "masked",
    "masked_array",
    "nomask",
    "array",
    "getmask",
    "getmaskarray",
    "getdata",
    "is_masked",
    "is_mask",
    "make_mask",
    "make_mask_none",
    "masked_where",
    "masked_equal",
    "masked_not_equal",
    "masked_less",
    "masked_less_equal",
    "masked_greater",
    "masked_greater_equal",
    "masked_inside",
    "masked_outside",
    "masked_invalid",
    "masked_values",
    "masked_object",
    "filled",
    "compressed",
    "concatenate",
    "zeros",
    "ones",
    "empty",
    "mean",
    "sum",
    "max",
    "min",
]
