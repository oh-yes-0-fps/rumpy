"""``numpy.dtypes`` — concrete dtype classes (new in numpy 2.0).

Each scalar dtype is exposed as a dedicated class so callers can write
``isinstance(arr.dtype, Float64DType)``-style checks. rumpy's dtype tag
is a plain string, so these classes are lightweight wrappers around the
numpy dtype-string name.
"""

_NAMES = {
    "Bool": "bool",
    "Int8": "int8",
    "Int16": "int16",
    "Int32": "int32",
    "Int64": "int64",
    "UInt8": "uint8",
    "UInt16": "uint16",
    "UInt32": "uint32",
    "UInt64": "uint64",
    "Float16": "float16",
    "Float32": "float32",
    "Float64": "float64",
    "Complex64": "complex64",
    "Complex128": "complex128",
}


class _DTypeBase:
    """Common base for the concrete dtype classes."""

    name: str = ""

    def __init__(self, *args, **kwargs):
        # Accept arbitrary args to match numpy's permissive constructors.
        pass

    def __repr__(self):
        return f"dtype('{self.name}')"

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, _DTypeBase):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return NotImplemented

    def __hash__(self):
        return hash(self.name)


def _make(prefix, dtype_name):
    return type(f"{prefix}DType", (_DTypeBase,), {"name": dtype_name})


BoolDType = _make("Bool", _NAMES["Bool"])
Int8DType = _make("Int8", _NAMES["Int8"])
Int16DType = _make("Int16", _NAMES["Int16"])
Int32DType = _make("Int32", _NAMES["Int32"])
Int64DType = _make("Int64", _NAMES["Int64"])
UInt8DType = _make("UInt8", _NAMES["UInt8"])
UInt16DType = _make("UInt16", _NAMES["UInt16"])
UInt32DType = _make("UInt32", _NAMES["UInt32"])
UInt64DType = _make("UInt64", _NAMES["UInt64"])
Float16DType = _make("Float16", _NAMES["Float16"])
Float32DType = _make("Float32", _NAMES["Float32"])
Float64DType = _make("Float64", _NAMES["Float64"])
Complex64DType = _make("Complex64", _NAMES["Complex64"])
Complex128DType = _make("Complex128", _NAMES["Complex128"])


# numpy also exposes string/object dtype classes; provide aliases so the
# names exist (they all resolve to the same placeholder).
class StrDType(_DTypeBase):
    name = "str"


class BytesDType(_DTypeBase):
    name = "bytes"


class ObjectDType(_DTypeBase):
    name = "object"


class StringDType(_DTypeBase):
    """Variable-length unicode string dtype (numpy 2.x)."""

    name = "string"


class VoidDType(_DTypeBase):
    """Void / record dtype."""

    name = "void"


# numpy 2.x exposes the C-typedef-named variants of every integer width so
# downstream type-check code keeps working under platform-default widths.
# rumpy treats every C-named alias as its fixed-width sibling.

ByteDType = _make("Byte", "int8")  # signed char
UByteDType = _make("UByte", "uint8")  # unsigned char
ShortDType = _make("Short", "int16")
UShortDType = _make("UShort", "uint16")
IntDType = _make("Int", "int32")  # C `int`
UIntDType = _make("UInt", "uint32")
LongDType = _make("Long", "int64")  # C `long` (LP64 — Unix 64-bit)
ULongDType = _make("ULong", "uint64")
LongLongDType = _make("LongLong", "int64")
ULongLongDType = _make("ULongLong", "uint64")

# Extended-precision float types: rumpy's ndarray crate has no f80 / f128,
# so we alias long-double to float64 / complex128.
LongDoubleDType = _make("LongDouble", "float64")
CLongDoubleDType = _make("CLongDouble", "complex128")

DateTime64DType = _make("DateTime64", "datetime64")
TimeDelta64DType = _make("TimeDelta64", "timedelta64")


__all__ = [
    "BoolDType",
    "Int8DType",
    "Int16DType",
    "Int32DType",
    "Int64DType",
    "UInt8DType",
    "UInt16DType",
    "UInt32DType",
    "UInt64DType",
    "Float16DType",
    "Float32DType",
    "Float64DType",
    "Complex64DType",
    "Complex128DType",
    "StrDType",
    "BytesDType",
    "ObjectDType",
    "StringDType",
    "VoidDType",
    "ByteDType",
    "UByteDType",
    "ShortDType",
    "UShortDType",
    "IntDType",
    "UIntDType",
    "LongDType",
    "ULongDType",
    "LongLongDType",
    "ULongLongDType",
    "LongDoubleDType",
    "CLongDoubleDType",
    "DateTime64DType",
    "TimeDelta64DType",
]
