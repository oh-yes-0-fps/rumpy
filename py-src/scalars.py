"""
NumPy scalar-type hierarchy.

The classes defined here mirror numpy's scalar type tree:

    generic
      number
        integer
          signedinteger: int8, int16, int32, int64
          unsignedinteger: uint8, uint16, uint32, uint64
        inexact
          floating: float16, float32, float64
          complexfloating: complex64, complex128
      bool_

Calling a leaf class (e.g. ``int32(5)``) builds a 0-D ndarray with that dtype,
matching the most common use of numpy scalars. ``isinstance`` chains and
``issubclass`` work for the abstract base classes.

This is a minimal compatibility shim — it does not implement the full
``np.generic`` scalar protocol (no per-instance arithmetic, no ``__int__``
beyond what the underlying 0-D ndarray provides).
"""

# `_np_array` and `_np_dtype` are injected from the Rust side so we can build
# 0-D ndarrays without re-importing the package (which would recurse).


class generic:
    """Base class for all numpy scalar types."""

    # Override per leaf class.
    _dtype_name = "object"

    def __new__(cls, value=0):
        if cls is generic or cls is number or cls is integer or \
           cls is signedinteger or cls is unsignedinteger or \
           cls is inexact or cls is floating or cls is complexfloating:
            raise TypeError(
                f"cannot instantiate abstract scalar type {cls.__name__}"
            )
        # Build a 0-D ndarray with the configured dtype.
        return _np_array(value, dtype=cls._dtype_name)


class number(generic):
    pass


class integer(number):
    pass


class signedinteger(integer):
    pass


class unsignedinteger(integer):
    pass


class inexact(number):
    pass


class floating(inexact):
    pass


class complexfloating(inexact):
    pass


# ---- concrete leaf classes ----


class bool_(generic):
    _dtype_name = "bool"


class int8(signedinteger):
    _dtype_name = "int8"


class int16(signedinteger):
    _dtype_name = "int16"


class int32(signedinteger):
    _dtype_name = "int32"


class int64(signedinteger):
    _dtype_name = "int64"


class uint8(unsignedinteger):
    _dtype_name = "uint8"


class uint16(unsignedinteger):
    _dtype_name = "uint16"


class uint32(unsignedinteger):
    _dtype_name = "uint32"


class uint64(unsignedinteger):
    _dtype_name = "uint64"


class float16(floating):
    _dtype_name = "float16"


class float32(floating):
    _dtype_name = "float32"


class float64(floating):
    _dtype_name = "float64"


class complex64(complexfloating):
    _dtype_name = "complex64"


class complex128(complexfloating):
    _dtype_name = "complex128"


# Numpy aliases for the platform-default integer/float widths.
intp = int64
uintp = uint64
intc = int32
uintc = uint32
short = int16
ushort = uint16
byte = int8
ubyte = uint8
longlong = int64
ulonglong = uint64
single = float32
double = float64
half = float16
csingle = complex64
cdouble = complex128
cfloat = complex128
complex_ = complex128
float_ = float64

# Numpy 2.x reintroduced these top-level python-builtin-shadowing aliases.
# Each is the same scalar class as its sibling, just under the bare name.
bool = bool_
int_ = int64
uint = uint64
long = int64
ulong = uint64

# Long-double has the same representation as double in rumpy — no extended
# precision in the underlying ndarray crate. We expose the names so that
# `isinstance(x, np.longdouble)` and dtype-name lookups round-trip.
class longdouble(floating):
    _dtype_name = "float64"


class clongdouble(complexfloating):
    _dtype_name = "complex128"


# numpy exposes float128/complex256 on Linux/x86-64 (80-bit extended precision
# padded to 16/32 bytes). The rumpy backend has no extended precision so we
# alias both to the same class as longdouble/clongdouble, matching numpy's
# behaviour on platforms without true long-double support. `np.float128 is
# np.longdouble` returns True in that mode (it does on real numpy too when
# they refer to the same internal type).
float128 = longdouble
complex256 = clongdouble


# Abstract bases that real numpy keeps in the scalar tree even though they're
# not numeric. Calling them raises (they're abstract).
class flexible(generic):
    """Abstract base for flexible-width scalar types (`character`, `void`)."""


class character(flexible):
    """Abstract base for string-like scalars (`str_`, `bytes_`)."""


class str_(character):
    """Unicode string scalar."""
    _dtype_name = "U"

    def __new__(cls, value=""):
        return str(value)


class bytes_(character):
    """Bytes string scalar."""
    _dtype_name = "S"

    def __new__(cls, value=b""):
        return bytes(value)


class object_(generic):
    """Object scalar — wraps any Python object verbatim."""
    _dtype_name = "O"

    def __new__(cls, value=None):
        return value


class void(flexible):
    """Void-type scalar — typically used as record-array element."""
    _dtype_name = "V"

    def __new__(cls, value=b""):
        return bytes(value)


# Mappings for introspection.
sctypeDict = {
    "bool": bool_,
    "int8": int8, "int16": int16, "int32": int32, "int64": int64,
    "uint8": uint8, "uint16": uint16, "uint32": uint32, "uint64": uint64,
    "float16": float16, "float32": float32, "float64": float64,
    "complex64": complex64, "complex128": complex128,
}

ScalarType = (
    int, float, complex, bool, bytes, str,
    bool_, int8, int16, int32, int64,
    uint8, uint16, uint32, uint64,
    float16, float32, float64,
    complex64, complex128,
)
