"""``numpy.char`` — element-wise string operations.

rumpy doesn't have a dedicated string-array dtype, so these helpers
accept any iterable of ``str``/``bytes`` and return a Python ``list`` of
the per-element results. The function names and semantics mirror the
canonical numpy.char API.
"""


def _map(x, f):
    return [f(v) for v in x]


def add(x, y):
    return [a + b for a, b in zip(x, y)]


def multiply(x, n):
    if hasattr(n, "__iter__"):
        return [a * b for a, b in zip(x, n)]
    return [a * n for a in x]


def mod(x, values):
    if isinstance(values, (list, tuple)):
        return [a % b for a, b in zip(x, values)]
    return [a % values for a in x]


def capitalize(x):
    return _map(x, str.capitalize)


def title(x):
    return _map(x, str.title)


def upper(x):
    return _map(x, str.upper)


def lower(x):
    return _map(x, str.lower)


def swapcase(x):
    return _map(x, str.swapcase)


def strip(x, chars=None):
    return [a.strip(chars) for a in x]


def lstrip(x, chars=None):
    return [a.lstrip(chars) for a in x]


def rstrip(x, chars=None):
    return [a.rstrip(chars) for a in x]


def split(x, sep=None, maxsplit=-1):
    return [a.split(sep, maxsplit) for a in x]


def rsplit(x, sep=None, maxsplit=-1):
    return [a.rsplit(sep, maxsplit) for a in x]


def splitlines(x, keepends=False):
    return [a.splitlines(keepends) for a in x]


def join(sep, seq):
    if isinstance(sep, str):
        return [sep.join(s) for s in seq]
    return [s.join(t) for s, t in zip(sep, seq)]


def replace(x, old, new, count=-1):
    return [a.replace(old, new, count) for a in x]


def startswith(x, prefix, start=0, end=None):
    return [a.startswith(prefix, start, end) if end is not None
            else a.startswith(prefix, start) for a in x]


def endswith(x, suffix, start=0, end=None):
    return [a.endswith(suffix, start, end) if end is not None
            else a.endswith(suffix, start) for a in x]


def count(x, sub, start=0, end=None):
    return [a.count(sub, start, end) if end is not None
            else a.count(sub, start) for a in x]


def find(x, sub, start=0, end=None):
    return [a.find(sub, start, end) if end is not None
            else a.find(sub, start) for a in x]


def rfind(x, sub, start=0, end=None):
    return [a.rfind(sub, start, end) if end is not None
            else a.rfind(sub, start) for a in x]


def index(x, sub, start=0, end=None):
    return [a.index(sub, start, end) if end is not None
            else a.index(sub, start) for a in x]


def rindex(x, sub, start=0, end=None):
    return [a.rindex(sub, start, end) if end is not None
            else a.rindex(sub, start) for a in x]


def isalpha(x):
    return _map(x, str.isalpha)


def isdigit(x):
    return _map(x, str.isdigit)


def isspace(x):
    return _map(x, str.isspace)


def isupper(x):
    return _map(x, str.isupper)


def islower(x):
    return _map(x, str.islower)


def isnumeric(x):
    return _map(x, str.isnumeric)


def isdecimal(x):
    return _map(x, str.isdecimal)


def str_len(x):
    return _map(x, len)


def encode(x, encoding="utf-8", errors="strict"):
    return [a.encode(encoding, errors) for a in x]


def decode(x, encoding="utf-8", errors="strict"):
    return [a.decode(encoding, errors) for a in x]


def zfill(x, width):
    return [a.zfill(width) for a in x]


def center(x, width, fillchar=" "):
    return [a.center(width, fillchar) for a in x]


def ljust(x, width, fillchar=" "):
    return [a.ljust(width, fillchar) for a in x]


def rjust(x, width, fillchar=" "):
    return [a.rjust(width, fillchar) for a in x]


def expandtabs(x, tabsize=8):
    return [a.expandtabs(tabsize) for a in x]


def equal(x, y):
    return [a == b for a, b in zip(x, y)]


def not_equal(x, y):
    return [a != b for a, b in zip(x, y)]


def greater(x, y):
    return [a > b for a, b in zip(x, y)]


def greater_equal(x, y):
    return [a >= b for a, b in zip(x, y)]


def less(x, y):
    return [a < b for a, b in zip(x, y)]


def less_equal(x, y):
    return [a <= b for a, b in zip(x, y)]


def compare_chararrays(a, b, cmp_op, _rstrip=True):
    op = {
        "==": lambda x, y: x == y,
        "!=": lambda x, y: x != y,
        "<": lambda x, y: x < y,
        "<=": lambda x, y: x <= y,
        ">": lambda x, y: x > y,
        ">=": lambda x, y: x >= y,
    }[cmp_op]
    return [op(x, y) for x, y in zip(a, b)]


def isalnum(x):
    return _map(x, str.isalnum)


def istitle(x):
    return _map(x, str.istitle)


def partition(x, sep):
    return [a.partition(sep) for a in x]


def rpartition(x, sep):
    return [a.rpartition(sep) for a in x]


def translate(x, table, deletechars=None):
    _ = deletechars
    return [a.translate(table) for a in x]


def array(obj, itemsize=None, copy=True, unicode=None, order=None):
    """Best-effort ``np.char.array`` — returns the input as a Python list."""
    _ = (itemsize, copy, unicode, order)
    if isinstance(obj, (list, tuple)):
        return list(obj)
    return [obj]


def asarray(obj, itemsize=None, unicode=None, order=None):
    """Best-effort ``np.char.asarray`` — equivalent to ``array(obj)``."""
    return array(obj, itemsize=itemsize, unicode=unicode, order=order)


class chararray(list):
    """A simple ``list``-backed string array.

    Real numpy's ``chararray`` is a string-typed ndarray subclass; rumpy
    has no string dtype, so we fall back to a ``list`` that supports the
    element-wise string operations declared above as methods.
    """

    def __new__(cls, shape, itemsize=1, unicode=True, buffer=None,
                offset=0, strides=None, order=None):
        _ = (itemsize, unicode, buffer, offset, strides, order)
        if isinstance(shape, int):
            n = shape
        else:
            n = 1
            for d in shape:
                n *= d
        return list.__new__(cls)

    def __init__(self, shape, itemsize=1, unicode=True, buffer=None,
                 offset=0, strides=None, order=None):
        _ = (itemsize, unicode, buffer, offset, strides, order)
        if isinstance(shape, int):
            n = shape
        else:
            n = 1
            for d in shape:
                n *= d
        super().__init__(["" for _ in range(n)])


__all__ = [
    "add", "multiply", "mod",
    "capitalize", "title", "upper", "lower", "swapcase",
    "strip", "lstrip", "rstrip",
    "split", "rsplit", "splitlines", "join", "replace",
    "startswith", "endswith",
    "count", "find", "rfind", "index", "rindex",
    "isalnum", "isalpha", "isdigit", "isspace", "isupper", "islower",
    "isnumeric", "isdecimal", "istitle",
    "str_len", "encode", "decode",
    "zfill", "center", "ljust", "rjust", "expandtabs",
    "equal", "not_equal", "greater", "greater_equal", "less", "less_equal",
    "compare_chararrays",
    "partition", "rpartition", "translate",
    "array", "asarray", "chararray",
]
