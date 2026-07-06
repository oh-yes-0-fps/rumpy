"""``numpy.strings`` — element-wise string operations (numpy 2.x).

This is the modern replacement for the legacy ``numpy.char`` namespace.
Same operations and semantics; the entire surface is implemented in
pure Python and accepts any iterable of ``str``/``bytes``, returning a
list of per-element results.
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


def replace(x, old, new, count=-1):
    return [a.replace(old, new, count) for a in x]


def split(x, sep=None, maxsplit=-1):
    return [a.split(sep, maxsplit) for a in x]


def rsplit(x, sep=None, maxsplit=-1):
    return [a.rsplit(sep, maxsplit) for a in x]


def startswith(x, prefix):
    return [a.startswith(prefix) for a in x]


def endswith(x, suffix):
    return [a.endswith(suffix) for a in x]


def count(x, sub):
    return [a.count(sub) for a in x]


def find(x, sub):
    return [a.find(sub) for a in x]


def rfind(x, sub):
    return [a.rfind(sub) for a in x]


def index(x, sub):
    return [a.index(sub) for a in x]


def rindex(x, sub):
    return [a.rindex(sub) for a in x]


def isalpha(x):
    return _map(x, str.isalpha)


def isalnum(x):
    return _map(x, str.isalnum)


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


def less(x, y):
    return [a < b for a, b in zip(x, y)]


def less_equal(x, y):
    return [a <= b for a, b in zip(x, y)]


def greater(x, y):
    return [a > b for a, b in zip(x, y)]


def greater_equal(x, y):
    return [a >= b for a, b in zip(x, y)]


def istitle(x):
    return _map(x, str.istitle)


def partition(x, sep):
    return [a.partition(sep) for a in x]


def rpartition(x, sep):
    return [a.rpartition(sep) for a in x]


def translate(x, table, deletechars=None):
    _ = deletechars
    return [a.translate(table) for a in x]


def slice(x, start=None, stop=None, step=None):
    """Element-wise Python slice on each string."""
    return [a[start:stop:step] for a in x]


__all__ = [
    "add",
    "multiply",
    "mod",
    "capitalize",
    "title",
    "upper",
    "lower",
    "swapcase",
    "strip",
    "lstrip",
    "rstrip",
    "replace",
    "split",
    "rsplit",
    "startswith",
    "endswith",
    "count",
    "find",
    "rfind",
    "index",
    "rindex",
    "isalpha",
    "isalnum",
    "isdigit",
    "isspace",
    "isupper",
    "islower",
    "isnumeric",
    "isdecimal",
    "istitle",
    "str_len",
    "encode",
    "decode",
    "zfill",
    "center",
    "ljust",
    "rjust",
    "expandtabs",
    "equal",
    "not_equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    "partition",
    "rpartition",
    "translate",
    "slice",
]
